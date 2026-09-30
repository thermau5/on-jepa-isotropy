"""GPU-scale CNN regularizer experiment on MNIST 28x28."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import uniform_filter
from sklearn.model_selection import train_test_split
from torchvision import datasets
import yaml

from experiments.common import ROOT, expanded_grid, parse_args


MNIST_ROOT = ROOT / "data" / "mnist_raw"
FG_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Architecture blocks
# ---------------------------------------------------------------------------

class ConvNormGELU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1, padding: int = 0, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class CNNEncoder(nn.Module):
    """(B,1,28,14) -> (B, embedding_dim).  Spatial path: 28x14 -> 14x7 -> 7x4."""

    def __init__(self, channels: list[int], embedding_dim: int, hidden_dim: int, use_residual: bool = False):
        super().__init__()
        c1, c2, c3 = channels
        layers: list[nn.Module] = [
            ConvNormGELU(1, c1, 3, padding=1),
        ]
        if use_residual:
            layers.append(ResBlock(c1))
        layers += [
            ConvNormGELU(c1, c2, 3, stride=2, padding=1),
        ]
        if use_residual:
            layers.append(ResBlock(c2))
        layers += [
            ConvNormGELU(c2, c3, 3, stride=2, padding=1),
        ]
        if use_residual:
            layers.append(ResBlock(c3))
        self.conv = nn.Sequential(*layers)
        flat_dim = c3 * 7 * 4
        self.head = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        return self.head(feat.flatten(1))


class MLPPredictor(nn.Module):
    """(B, embedding_dim) -> (B, predictor_dim)."""

    def __init__(self, embedding_dim: int, predictor_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, predictor_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class CNNDecoder(nn.Module):
    """(B, predictor_dim) -> (B, 1, 28, 14).  Spatial path: 7x4 -> 14x7 -> 28x14."""

    def __init__(self, predictor_dim: int, channels: list[int], use_residual: bool = False):
        super().__init__()
        c3, c2, c1 = channels  # reverse order
        self.proj = nn.Linear(predictor_dim, c3 * 7 * 4)
        layers: list[nn.Module] = []
        if use_residual:
            layers.append(ResBlock(c3))
        layers += [
            nn.ConvTranspose2d(c3, c2, kernel_size=(4, 3), stride=2, padding=1),
            nn.GroupNorm(8, c2),
            nn.GELU(),
        ]
        if use_residual:
            layers.append(ResBlock(c2))
        layers += [
            nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, c1),
            nn.GELU(),
        ]
        layers.append(nn.Conv2d(c1, 1, kernel_size=1))
        self.up = nn.Sequential(*layers)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        x = self.proj(p).view(p.shape[0], -1, 7, 4)
        return self.up(x)


class MNISTGPUJepa(nn.Module):
    def __init__(
        self,
        channels: list[int],
        embedding_dim: int,
        predictor_dim: int,
        hidden_dim: int,
        use_residual: bool = False,
    ):
        super().__init__()
        self.encoder = CNNEncoder(channels, embedding_dim, hidden_dim, use_residual)
        self.predictor = MLPPredictor(embedding_dim, predictor_dim, hidden_dim)
        self.decoder = CNNDecoder(predictor_dim, channels, use_residual)

    def forward(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x_flat: (B, 392) -> y_hat: (B, 392), z: (B, emb), p: (B, pred)"""
        x = x_flat.view(-1, 1, 28, 14)
        z = self.encoder(x)
        p = self.predictor(z)
        y_hat = self.decoder(p).view(-1, 392)
        return y_hat, z, p


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_mnist_halves_gpu(
    seed: int,
    n_train: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_train, x_test, y_train, y_test) as float32 arrays in [0,1].

    Context = left half (392,), target = right half (392,).
    n_train=0 uses all 60k training examples.
    """
    MNIST_ROOT.mkdir(parents=True, exist_ok=True)
    train_ds = datasets.MNIST(root=str(MNIST_ROOT), train=True, download=True)
    test_ds = datasets.MNIST(root=str(MNIST_ROOT), train=False, download=True)

    x_all = train_ds.data.numpy().astype(np.float32) / 255.0   # (60000, 28, 28)
    labels_all = train_ds.targets.numpy()

    if n_train > 0 and n_train < len(x_all):
        idx = np.arange(len(x_all))
        train_idx, _ = train_test_split(idx, train_size=n_train, stratify=labels_all, random_state=seed)
        x_tr = x_all[train_idx]
    else:
        x_tr = x_all

    x_te = test_ds.data.numpy().astype(np.float32) / 255.0     # (10000, 28, 28)

    def halves(imgs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ctx = imgs[:, :, :14].reshape(len(imgs), -1)
        tgt = imgs[:, :, 14:].reshape(len(imgs), -1)
        return ctx, tgt

    ctx_tr, tgt_tr = halves(x_tr)
    ctx_te, tgt_te = halves(x_te)
    return ctx_tr, ctx_te, tgt_tr, tgt_te


# ---------------------------------------------------------------------------
# Regularizer losses
# ---------------------------------------------------------------------------

def fixed_directions(dim: int, count: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    rand = rng.normal(size=(count, dim)).astype(np.float32)
    rand /= np.linalg.norm(rand, axis=1, keepdims=True) + 1e-12
    return torch.from_numpy(np.vstack([np.eye(dim, dtype=np.float32), rand]))


def covariance_standard_normal_loss(samples: torch.Tensor) -> torch.Tensor:
    centered = samples - samples.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(samples.shape[0] - 1, 1)
    identity = torch.eye(samples.shape[1], dtype=samples.dtype, device=samples.device)
    return samples.mean(dim=0).square().mean() + (cov - identity).square().mean()


def projected_gaussianity_loss(samples: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    centered = samples - samples.mean(dim=0, keepdim=True)
    projected = centered @ directions.T
    t_values = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=samples.dtype, device=samples.device)
    target = torch.exp(-0.5 * t_values ** 2)
    angles = projected.unsqueeze(-1) * t_values
    real_gap = torch.cos(angles).mean(dim=0) - target
    imag_gap = torch.sin(angles).mean(dim=0)
    return (real_gap.square() + imag_gap.square()).mean()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def covariance_metrics(samples: np.ndarray) -> dict[str, float]:
    centered = samples - samples.mean(axis=0)
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    evals = np.clip(np.linalg.eigvalsh(0.5 * (cov + cov.T)), 1e-12, None)
    target_cov = np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    return {
        "effective_rank": float(np.sum(evals) / evals[-1]),
        "isotropy_error": float(
            np.linalg.norm(cov - target_cov, "fro") / max(np.linalg.norm(target_cov, "fro"), 1e-12)
        ),
    }


def projected_gaussianity_score(samples: np.ndarray, directions: np.ndarray) -> float:
    centered = samples - samples.mean(axis=0)
    projected = centered @ directions.T
    t_grid = np.array([0.5, 1.0, 1.5, 2.0])
    target = np.exp(-0.5 * t_grid ** 2)
    scores = []
    for col in range(projected.shape[1]):
        empirical = np.array([np.mean(np.exp(1j * t * projected[:, col])) for t in t_grid])
        scores.append(float(np.mean(np.abs(empirical - target) ** 2)))
    return float(np.mean(scores))


def _ssim_single(pred: np.ndarray, target: np.ndarray, win: int = 7) -> float:
    """Windowed SSIM on a single (H, W) image in [0, 1]."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    p, t = pred.astype(np.float64), target.astype(np.float64)
    mu_p = uniform_filter(p, win)
    mu_t = uniform_filter(t, win)
    s_p = uniform_filter(p * p, win) - mu_p ** 2
    s_t = uniform_filter(t * t, win) - mu_t ** 2
    s_pt = uniform_filter(p * t, win) - mu_p * mu_t
    num = (2 * mu_p * mu_t + C1) * (2 * s_pt + C2)
    den = (mu_p ** 2 + mu_t ** 2 + C1) * (s_p + s_t + C2)
    return float(np.mean(num / np.clip(den, 1e-12, None)))


def compute_all_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    z_np: np.ndarray,
    p_np: np.ndarray,
    z_dirs: np.ndarray,
    p_dirs: np.ndarray,
) -> dict[str, float]:
    """Compute all evaluation metrics from flat (N, 392) arrays."""
    preds_c = np.clip(preds, 0.0, 1.0)
    err2 = (preds_c - targets) ** 2
    fg = targets > FG_THRESHOLD

    test_mse = float(err2.mean())
    fg_mse = float(err2[fg].mean()) if fg.any() else float("nan")
    target_energy = float(np.mean(targets ** 2))

    # SSIM over right-half images
    pred_imgs = preds_c.reshape(-1, 28, 14)
    tgt_imgs = targets.reshape(-1, 28, 14)
    ssim_scores = [_ssim_single(pred_imgs[i], tgt_imgs[i]) for i in range(min(len(pred_imgs), 500))]

    z_m = covariance_metrics(z_np)
    p_m = covariance_metrics(p_np)

    return {
        "test_mse": test_mse,
        "test_rmse": float(np.sqrt(test_mse)),
        "fg_mse": fg_mse,
        "fg_rmse": float(np.sqrt(fg_mse)) if not np.isnan(fg_mse) else float("nan"),
        "relative_mse": test_mse / max(target_energy, 1e-12),
        "ssim": float(np.mean(ssim_scores)),
        "embedding_effective_rank": z_m["effective_rank"],
        "embedding_isotropy_error": z_m["isotropy_error"],
        "embedding_gaussianity_score": projected_gaussianity_score(z_np, z_dirs),
        "predictive_effective_rank": p_m["effective_rank"],
        "predictive_isotropy_error": p_m["isotropy_error"],
        "predictive_gaussianity_score": projected_gaussianity_score(p_np, p_dirs),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one(params: dict[str, Any]) -> dict[str, Any]:
    seed = int(params.get("seed", 0))
    n_train = int(params.get("n_train", 10000))
    channels = list(params.get("encoder_channels", [64, 128, 256]))
    embedding_dim = int(params.get("embedding_dim", 256))
    predictor_dim = int(params.get("predictor_dim", 256))
    hidden_dim = int(params.get("hidden_dim", 512))
    use_residual = bool(params.get("use_residual", False))
    epochs = int(params.get("epochs", 50))
    batch_size = int(params.get("batch_size", 256))
    lr = float(params.get("lr", 3e-4))
    weight_decay = float(params.get("weight_decay", 1e-4))
    use_amp = bool(params.get("use_amp", True))
    use_cosine_lr = bool(params.get("use_cosine_lr", True))
    reg_weight = float(params.get("reg_weight", 0.08))
    cov_weight = float(params.get("cov_weight", 1.0))
    n_directions = int(params.get("n_directions", 128))
    method = str(params.get("method", "baseline"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data
    x_tr, x_te, y_tr, y_te = load_mnist_halves_gpu(seed, n_train)
    x_tr_t = torch.from_numpy(x_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    x_te_t = torch.from_numpy(x_te).to(device)
    y_te_t = torch.from_numpy(y_te).to(device)

    # Model
    model = MNISTGPUJepa(channels, embedding_dim, predictor_dim, hidden_dim, use_residual).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if use_cosine_lr else None
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    # Regularizer directions (fixed for this seed/method)
    z_dirs = fixed_directions(embedding_dim, n_directions, seed + 101).to(device)
    p_dirs = fixed_directions(predictor_dim, n_directions, seed + 303).to(device)

    n = x_tr.shape[0]
    gen = torch.Generator().manual_seed(seed + 505)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, batch_size):
            batch = perm[start : start + batch_size]
            xb, yb = x_tr_t[batch], y_tr_t[batch]
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                pred, z, p = model(xb)
                loss = F.mse_loss(pred, yb)
                reg = torch.zeros((), device=device)
                if method in ("encoder_gaussian", "both"):
                    reg = reg + projected_gaussianity_loss(z, z_dirs)
                    reg = reg + cov_weight * covariance_standard_normal_loss(z)
                if method in ("predictive_gaussian", "both"):
                    reg = reg + projected_gaussianity_loss(p, p_dirs)
                    reg = reg + cov_weight * covariance_standard_normal_loss(p)
                total = loss + reg_weight * reg
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
        if scheduler is not None:
            scheduler.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - t_start

    # Batched evaluation to avoid OOM on large datasets
    def eval_batched(x_t: torch.Tensor, eval_bs: int = 512):
        model.eval()
        preds, zs, ps = [], [], []
        with torch.no_grad():
            for s in range(0, len(x_t), eval_bs):
                pred, z, p = model(x_t[s : s + eval_bs])
                preds.append(pred.cpu()); zs.append(z.cpu()); ps.append(p.cpu())
        return torch.cat(preds), torch.cat(zs), torch.cat(ps)

    tr_pred, _, _ = eval_batched(x_tr_t)
    train_mse = F.mse_loss(tr_pred, y_tr_t.cpu()).item()

    te_pred, te_z, te_p = eval_batched(x_te_t)
    preds_np = te_pred.numpy()
    z_np = te_z.numpy()
    p_np = te_p.numpy()
    z_dir_np = z_dirs.cpu().numpy()
    p_dir_np = p_dirs.cpu().numpy()

    metrics = compute_all_metrics(preds_np, y_te, z_np, p_np, z_dir_np, p_dir_np)
    metrics["gap"] = metrics["test_mse"] - train_mse

    return {
        "experiment": "regularizer_mnist_gpu",
        "method": method,
        "seed": seed,
        "n_train": n_train,
        "epochs": epochs,
        "batch_size": batch_size,
        "reg_weight": reg_weight,
        "cov_weight": cov_weight,
        "train_mse": train_mse,
        "train_seconds": train_seconds,
        **metrics,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(config_path: Path, output_path: Path) -> pd.DataFrame:
    config = load_config(config_path)
    rows = [train_one(p) for p in expanded_grid(config)]
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


if __name__ == "__main__":
    args = parse_args(
        ROOT / "configs" / "regularizer_mnist_gpu.yaml",
        ROOT / "results" / "exp_regularizer_mnist_gpu.csv",
    )
    run(Path(args.config), Path(args.output))
