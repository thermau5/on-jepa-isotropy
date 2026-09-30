"""JEPA-style regularizer experiment on MNIST 28x28."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torchvision import datasets
import yaml

from experiments.common import ROOT, expanded_grid, parse_args


STANDARD_NORMAL = NormalDist()
MNIST_ROOT = ROOT / "data" / "mnist_raw"
CONTEXT_DIM = 28 * 14   # 392
TARGET_DIM = 28 * 14    # 392


@dataclass
class MNISTConfig:
    epochs: int = 150
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    hidden_dim: int = 512
    embedding_dim: int = 64
    predictor_dim: int = 64
    reg_weight: float = 0.10
    cov_weight: float = 5.0
    n_directions: int = 64
    labeled_fraction: float = 0.08
    method: str = "baseline"
    seed: int = 0


class MNISTJepa(torch.nn.Module):
    def __init__(self, context_dim: int, target_dim: int, hidden_dim: int, embedding_dim: int, predictor_dim: int):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(context_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, embedding_dim),
        )
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, predictor_dim),
        )
        self.head = torch.nn.Linear(predictor_dim, target_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        p = self.predictor(z)
        y_hat = self.head(p)
        return y_hat, z, p


def load_mnist_halves(
    seed: int,
    labeled_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST, return (x_train, x_test, y_train, y_test, labels_train, labels_test).

    Context is left half (28x14=392), target is right half (28x14=392).
    Train set is a stratified sample of labeled_fraction of the 60k MNIST training split.
    Test set is the standard 10k MNIST test split.
    """
    MNIST_ROOT.mkdir(parents=True, exist_ok=True)
    train_full = datasets.MNIST(root=str(MNIST_ROOT), train=True, download=True)
    test_full = datasets.MNIST(root=str(MNIST_ROOT), train=False, download=True)

    x_all = train_full.data.numpy().astype(np.float32) / 255.0   # (60000, 28, 28)
    labels_all = train_full.targets.numpy()

    indices = np.arange(len(x_all))
    train_idx, _ = train_test_split(
        indices,
        train_size=labeled_fraction,
        stratify=labels_all,
        random_state=seed,
    )
    x_train_img = x_all[train_idx]          # (n_labeled, 28, 28)
    labels_train = labels_all[train_idx]

    x_test_img = test_full.data.numpy().astype(np.float32) / 255.0  # (10000, 28, 28)
    labels_test = test_full.targets.numpy()

    context_train = x_train_img[:, :, :14].reshape(len(x_train_img), -1)
    target_train = x_train_img[:, :, 14:].reshape(len(x_train_img), -1)
    context_test = x_test_img[:, :, :14].reshape(len(x_test_img), -1)
    target_test = x_test_img[:, :, 14:].reshape(len(x_test_img), -1)

    return context_train, context_test, target_train, target_test, labels_train, labels_test


def fixed_directions(dim: int, count: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    random = rng.normal(size=(count, dim)).astype(np.float32)
    random /= np.linalg.norm(random, axis=1, keepdims=True) + 1e-12
    directions = np.vstack([np.eye(dim, dtype=np.float32), random])
    return torch.from_numpy(directions)


def covariance_standard_normal_loss(samples: torch.Tensor) -> torch.Tensor:
    centered = samples - samples.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(samples.shape[0] - 1, 1)
    identity = torch.eye(samples.shape[1], dtype=samples.dtype, device=samples.device)
    mean_loss = samples.mean(dim=0).square().mean()
    cov_loss = (cov - identity).square().mean()
    return mean_loss + cov_loss


def projected_gaussianity_loss(samples: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    centered = samples - samples.mean(dim=0, keepdim=True)
    projected = centered @ directions.T
    t_values = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=samples.dtype, device=samples.device)
    target = torch.exp(-0.5 * t_values**2)
    angles = projected.unsqueeze(-1) * t_values
    real_gap = torch.cos(angles).mean(dim=0) - target
    imag_gap = torch.sin(angles).mean(dim=0)
    return (real_gap.square() + imag_gap.square()).mean()


def covariance_metrics(samples: np.ndarray) -> dict[str, float]:
    centered = samples - samples.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    evals = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    evals = np.clip(evals, 1e-12, None)
    target = np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    return {
        "kappa": float(evals[0] / evals[-1]),
        "effective_rank": float(np.sum(evals) / evals[-1]),
        "isotropy_error": float(np.linalg.norm(cov - target, ord="fro") / max(np.linalg.norm(target, ord="fro"), 1e-12)),
        "min_std": float(np.sqrt(evals[0])),
    }


def projected_gaussianity_score(samples: np.ndarray, directions: np.ndarray) -> float:
    centered = samples - samples.mean(axis=0, keepdims=True)
    projected = centered @ directions.T
    t_grid = np.array([0.5, 1.0, 1.5, 2.0])
    target = np.exp(-0.5 * t_grid**2)
    scores = []
    for column in range(projected.shape[1]):
        empirical = np.array([np.mean(np.exp(1j * t * projected[:, column])) for t in t_grid])
        scores.append(float(np.mean(np.abs(empirical - target) ** 2)))
    return float(np.mean(scores))


def train_one(params: dict[str, Any]) -> dict[str, Any]:
    cfg = MNISTConfig(**params)
    start_time = time.perf_counter()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, x_test, y_train, y_test, _labels_train, _labels_test = load_mnist_halves(
        cfg.seed, cfg.labeled_fraction
    )
    x_train_t = torch.from_numpy(x_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    x_test_t = torch.from_numpy(x_test).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)

    model = MNISTJepa(
        context_dim=x_train.shape[1],
        target_dim=y_train.shape[1],
        hidden_dim=cfg.hidden_dim,
        embedding_dim=cfg.embedding_dim,
        predictor_dim=cfg.predictor_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    z_dirs = fixed_directions(cfg.embedding_dim, cfg.n_directions, cfg.seed + 101).to(device)
    p_dirs = fixed_directions(cfg.predictor_dim, cfg.n_directions, cfg.seed + 303).to(device)
    n = x_train.shape[0]
    generator = torch.Generator().manual_seed(cfg.seed + 505)

    for _epoch in range(cfg.epochs):
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, cfg.batch_size):
            batch = perm[start : start + cfg.batch_size]
            pred, z, p = model(x_train_t[batch])
            pred_loss = torch.nn.functional.mse_loss(pred, y_train_t[batch])
            reg = torch.zeros((), dtype=pred_loss.dtype, device=device)
            if cfg.method in {"encoder_gaussian", "both"}:
                reg = reg + projected_gaussianity_loss(z, z_dirs)
                reg = reg + cfg.cov_weight * covariance_standard_normal_loss(z)
            if cfg.method in {"predictive_gaussian", "both"}:
                reg = reg + projected_gaussianity_loss(p, p_dirs)
                reg = reg + cfg.cov_weight * covariance_standard_normal_loss(p)
            loss = pred_loss + cfg.reg_weight * reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_pred, train_z, train_p = model(x_train_t)
        test_pred, test_z, test_p = model(x_test_t)
        train_mse = torch.nn.functional.mse_loss(train_pred, y_train_t).item()
        test_mse = torch.nn.functional.mse_loss(test_pred, y_test_t).item()
    train_seconds = time.perf_counter() - start_time

    z_np = test_z.cpu().numpy()
    p_np = test_p.cpu().numpy()
    z_dir_np = z_dirs.cpu().numpy()
    p_dir_np = p_dirs.cpu().numpy()
    z_metrics = covariance_metrics(z_np)
    p_metrics = covariance_metrics(p_np)

    return {
        "experiment": "regularizer_mnist",
        "method": cfg.method,
        "seed": cfg.seed,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "reg_weight": cfg.reg_weight,
        "cov_weight": cfg.cov_weight,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_seconds": train_seconds,
        "embedding_kappa": z_metrics["kappa"],
        "embedding_effective_rank": z_metrics["effective_rank"],
        "embedding_isotropy_error": z_metrics["isotropy_error"],
        "embedding_min_std": z_metrics["min_std"],
        "embedding_gaussianity_score": projected_gaussianity_score(z_np, z_dir_np),
        "predictive_kappa": p_metrics["kappa"],
        "predictive_effective_rank": p_metrics["effective_rank"],
        "predictive_isotropy_error": p_metrics["isotropy_error"],
        "predictive_min_std": p_metrics["min_std"],
        "predictive_gaussianity_score": projected_gaussianity_score(p_np, p_dir_np),
    }


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run(config_path: Path, output_path: Path) -> pd.DataFrame:
    config = load_config(config_path)
    rows = [train_one(params) for params in expanded_grid(config)]
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


if __name__ == "__main__":
    args = parse_args(
        ROOT / "configs" / "regularizer_mnist.yaml",
        ROOT / "results" / "exp_regularizer_mnist.csv",
    )
    run(Path(args.config), Path(args.output))
