"""GPU-scale CNN regularizer experiment on CIFAR-10/100 32x32 RGB.

Left-to-right image completion: context = left 16 columns, target = right
16 columns. The completion task never uses labels, so CIFAR-10 and
CIFAR-100 act as two natural-image distributions of different diversity.
Run CIFAR-10 with the defaults and CIFAR-100 via

    python -m experiments.exp_regularizer_cifar_gpu \
        --config configs/regularizer_cifar100.yaml \
        --output results/exp_regularizer_cifar100.csv
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torchvision import datasets
import yaml

from experiments.common import ROOT, expanded_grid, parse_args
from experiments.exp_regularizer_mnist_gpu import (
    ResBlock,
    ConvNormGELU,
    MLPPredictor,
    _ssim_single,
    covariance_metrics,
    covariance_standard_normal_loss,
    fixed_directions,
    projected_gaussianity_loss,
    projected_gaussianity_score,
)


CIFAR_ROOT = ROOT / "data" / "cifar_raw"
IMG_H, IMG_W, IMG_C = 32, 16, 3
FLAT_DIM = IMG_C * IMG_H * IMG_W  # 1536 per half


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class CIFAREncoder(nn.Module):
    """(B,3,32,16) -> (B, embedding_dim).  Spatial path: 32x16 -> 16x8 -> 8x4."""

    def __init__(self, channels: list[int], embedding_dim: int, hidden_dim: int, use_residual: bool = False):
        super().__init__()
        c1, c2, c3 = channels
        layers: list[nn.Module] = [
            ConvNormGELU(IMG_C, c1, 3, padding=1),
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
        flat_dim = c3 * 8 * 4
        self.head = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        return self.head(feat.flatten(1))


class CIFARDecoder(nn.Module):
    """(B, predictor_dim) -> (B, 3, 32, 16).  Spatial path: 8x4 -> 16x8 -> 32x16."""

    def __init__(self, predictor_dim: int, channels: list[int], use_residual: bool = False):
        super().__init__()
        c3, c2, c1 = channels  # reverse order
        self.proj = nn.Linear(predictor_dim, c3 * 8 * 4)
        layers: list[nn.Module] = []
        if use_residual:
            layers.append(ResBlock(c3))
        layers += [
            nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1),
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
        layers.append(nn.Conv2d(c1, IMG_C, kernel_size=1))
        self.up = nn.Sequential(*layers)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        x = self.proj(p).view(p.shape[0], -1, 8, 4)
        return self.up(x)


class CIFARGPUJepa(nn.Module):
    def __init__(
        self,
        channels: list[int],
        embedding_dim: int,
        predictor_dim: int,
        hidden_dim: int,
        use_residual: bool = False,
    ):
        super().__init__()
        self.encoder = CIFAREncoder(channels, embedding_dim, hidden_dim, use_residual)
        self.predictor = MLPPredictor(embedding_dim, predictor_dim, hidden_dim)
        self.decoder = CIFARDecoder(predictor_dim, channels, use_residual)

    def forward(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x_flat: (B, 1536) -> y_hat: (B, 1536), z: (B, emb), p: (B, pred)"""
        x = x_flat.view(-1, IMG_C, IMG_H, IMG_W)
        z = self.encoder(x)
        p = self.predictor(z)
        y_hat = self.decoder(p).view(-1, FLAT_DIM)
        return y_hat, z, p


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_cifar_halves_gpu(
    dataset: str,
    seed: int,
    n_train: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_train, x_test, y_train, y_test) as float32 arrays in [0,1].

    Context = left half (1536,), target = right half (1536,), both flattened
    channel-first. n_train=0 uses all 50k training examples.
    """
    CIFAR_ROOT.mkdir(parents=True, exist_ok=True)
    if dataset == "cifar10":
        ds_cls = datasets.CIFAR10
    elif dataset == "cifar100":
        ds_cls = datasets.CIFAR100
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    train_ds = ds_cls(root=str(CIFAR_ROOT), train=True, download=True)
    test_ds = ds_cls(root=str(CIFAR_ROOT), train=False, download=True)

    x_all = train_ds.data.astype(np.float32) / 255.0  # (50000, 32, 32, 3)
    labels_all = np.asarray(train_ds.targets)

    if n_train > 0 and n_train < len(x_all):
        idx = np.arange(len(x_all))
        train_idx, _ = train_test_split(idx, train_size=n_train, stratify=labels_all, random_state=seed)
        x_tr = x_all[train_idx]
    else:
        x_tr = x_all

    x_te = test_ds.data.astype(np.float32) / 255.0     # (10000, 32, 32, 3)

    def halves(imgs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        chw = imgs.transpose(0, 3, 1, 2)  # (N, 3, 32, 32) channel-first
        ctx = chw[:, :, :, :16].reshape(len(imgs), -1)
        tgt = chw[:, :, :, 16:].reshape(len(imgs), -1)
        return np.ascontiguousarray(ctx), np.ascontiguousarray(tgt)

    ctx_tr, tgt_tr = halves(x_tr)
    ctx_te, tgt_te = halves(x_te)
    return ctx_tr, ctx_te, tgt_tr, tgt_te


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_all_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    z_np: np.ndarray,
    p_np: np.ndarray,
    z_dirs: np.ndarray,
    p_dirs: np.ndarray,
) -> dict[str, float]:
    """Compute evaluation metrics from flat (N, 1536) arrays.

    Foreground metrics are omitted: natural images have no background/ink
    split, so the MNIST-style foreground mask is not meaningful here.
    """
    preds_c = np.clip(preds, 0.0, 1.0)
    err2 = (preds_c - targets) ** 2

    test_mse = float(err2.mean())
    target_energy = float(np.mean(targets ** 2))

    # SSIM averaged over RGB channels of the right-half images
    pred_imgs = preds_c.reshape(-1, IMG_C, IMG_H, IMG_W)
    tgt_imgs = targets.reshape(-1, IMG_C, IMG_H, IMG_W)
    n_ssim = min(len(pred_imgs), 500)
    ssim_scores = [
        np.mean([_ssim_single(pred_imgs[i, c], tgt_imgs[i, c]) for c in range(IMG_C)])
        for i in range(n_ssim)
    ]

    z_m = covariance_metrics(z_np)
    p_m = covariance_metrics(p_np)

    return {
        "test_mse": test_mse,
        "test_rmse": float(np.sqrt(test_mse)),
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
    dataset = str(params.get("dataset", "cifar10"))
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
    x_tr, x_te, y_tr, y_te = load_cifar_halves_gpu(dataset, seed, n_train)
    x_tr_t = torch.from_numpy(x_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    x_te_t = torch.from_numpy(x_te).to(device)
    y_te_t = torch.from_numpy(y_te).to(device)

    # Model
    model = CIFARGPUJepa(channels, embedding_dim, predictor_dim, hidden_dim, use_residual).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if use_cosine_lr else None
    )
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

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
            with torch.amp.autocast(device.type, enabled=amp_enabled):
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
        "experiment": f"regularizer_{dataset}",
        "dataset": dataset,
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
        ROOT / "configs" / "regularizer_cifar10.yaml",
        ROOT / "results" / "exp_regularizer_cifar10.csv",
    )
    run(Path(args.config), Path(args.output))
