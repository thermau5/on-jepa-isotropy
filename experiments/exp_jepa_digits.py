"""Latent-target JEPA on sklearn 8x8 digits with a frozen teacher.

The context encoder sees the left half; the predictor outputs the predicted teacher embedding of the
right half, and training minimizes the latent MSE. The teacher is frozen: PCA-whitening of the right
half onto its top `embedding_dim` components, fit on the training split. Because the predictor output
is the prediction itself, the prediction-side penalty acts on exactly the intrinsic predicted output
of Eq. 4.2. A separate pixel head decodes predictions into pixels; it trains on detached predictions,
so it never shapes the JEPA, and it reports pixel MSE. Validation images, for choosing strengths, come
from the held-out split; the rest of it is the test split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import yaml

from experiments.common import ROOT, expanded_grid, parse_args
from experiments.exp_regularizer_digits import (
    covariance_metrics,
    covariance_standard_normal_loss,
    fixed_directions,
    load_digit_halves,
    projected_gaussianity_loss,
    projected_gaussianity_score,
)


METHODS = ("baseline", "encoder_gaussian", "predictive_gaussian", "both")
# Scale of the prediction-side target: "unit" is Eq. 4.2 as written, N(0, I);
# "trace" rescales the predictions to trace d first, so the target is N(0, (tau/d) I) at their own trace tau.
PREDICTIVE_SCALES = ("unit", "trace")


@dataclass
class JEPADigitsConfig:
    epochs: int = 400
    batch_size: int = 32
    lr: float = 1e-3
    # Weight decay of the encoder and predictor; the pixel head keeps its own.
    weight_decay: float = 0.0
    pixel_weight_decay: float = 0.0
    hidden_dim: int = 512
    embedding_dim: int = 16
    reg_weight: float = 0.5
    cov_weight: float = 10.0
    n_directions: int = 32
    test_size: float = 0.92
    # Validation images, taken from the held-out split; the rest is the test split.
    n_val: int = 500
    # "pca": PCA-whitening of the right half; "cca": the predictively isotropic CCA teacher.
    teacher: str = "pca"
    teacher_dim: int = 16
    method: str = "baseline"
    predictive_scale: str = "unit"
    seed: int = 0


def trace_normalize(p: torch.Tensor) -> torch.Tensor:
    p = p - p.mean(dim=0, keepdim=True)
    scale = (p.square().sum(dim=1).mean() / p.shape[1]).sqrt()
    return p / scale.clamp_min(1e-12)


def mlp(in_dim: int, hidden_dim: int, out_dim: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden_dim),
        torch.nn.GELU(),
        torch.nn.Linear(hidden_dim, out_dim),
    )


class PCAWhiteningTeacher:
    """Frozen teacher: project onto the top-k principal components of the training targets, scaled to unit variance."""

    def __init__(self, y_train: np.ndarray, k: int):
        self.mean = y_train.mean(axis=0)
        cov = np.cov(y_train - self.mean, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1][:k]
        self.components = evecs[:, order]
        self.scale = np.sqrt(np.clip(evals[order], 1e-12, None))
        self.explained = float(evals[order].sum() / evals.sum())

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.mean) @ self.components / self.scale).astype(np.float32)


class CCATeacher:
    """Frozen predictively isotropic teacher: the top-k canonical variates of the target against the context.

    Linear CCA runs on the top `pca_dim` whitened principal components of each half. Each canonical variate
    is divided by its canonical correlation, so the best linear prediction of every teacher coordinate has
    unit variance, and the best linear predictor's output is isotropic, N(0, I_k) for Gaussian data.
    """

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, k: int, pca_dim: int = 128, ridge: float = 1e-3):
        def whitener(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            mean = a.mean(axis=0)
            evals, evecs = np.linalg.eigh(np.cov(a - mean, rowvar=False))
            order = np.argsort(evals)[::-1][:pca_dim]
            return mean, evecs[:, order] / np.sqrt(evals[order] + ridge * evals.max())

        self.x_mean, wx = whitener(x_train)
        self.mean, wy = whitener(y_train)
        xw, yw = (x_train - self.x_mean) @ wx, (y_train - self.mean) @ wy
        _, s, vt = np.linalg.svd(xw.T @ yw / (len(xw) - 1))
        self.corr = s[:k]
        self.projection = wy @ vt[:k].T / s[:k]
        self.explained = float("nan")

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.mean) @ self.projection).astype(np.float32)


def latent_metrics(
    teacher: PCAWhiteningTeacher | CCATeacher,
    p_tr: np.ndarray, t_tr: np.ndarray,
    p_val: np.ndarray, t_val: np.ndarray,
    p_te: np.ndarray, t_te: np.ndarray,
) -> dict[str, float]:
    """Latent risks against the frozen teacher in two weightings, plus a post-hoc shrinkage baseline.

    Whitened coordinates weight every teacher direction equally. Weighting each by its pixel variance
    instead gives the per-pixel error of decoding through the teacher's own linear inverse. The shrinkage
    baseline scales the predictions toward the teacher mean (zero) by the scalar fit on the validation split.
    The variance weighting needs orthogonal teacher directions, so it is NaN for the CCA teacher.
    """
    mse = lambda e: float(np.mean(e ** 2))
    if isinstance(teacher, PCAWhiteningTeacher):
        weights = teacher.scale ** 2 / len(teacher.mean)
        weighted = lambda e: float(np.mean(e ** 2 @ weights))
    else:
        weighted = lambda e: float("nan")
    alpha = float(np.sum(p_val * t_val) / max(float(np.sum(p_val ** 2)), 1e-12))
    return {
        "latent_train_mse": mse(p_tr - t_tr),
        "latent_val_mse": mse(p_val - t_val),
        "latent_test_mse": mse(p_te - t_te),
        "latent_gap": mse(p_te - t_te) - mse(p_tr - t_tr),
        "latent_trivial_mse": mse(t_te),
        "latent_val_mse_var_weighted": weighted(p_val - t_val),
        "latent_test_mse_var_weighted": weighted(p_te - t_te),
        "latent_trivial_mse_var_weighted": weighted(t_te),
        "shrink_factor": alpha,
        "latent_test_mse_shrunk": mse(alpha * p_te - t_te),
    }


def train_one(params: dict[str, Any]) -> dict[str, Any]:
    cfg = JEPADigitsConfig(**params)
    if cfg.method not in METHODS or cfg.predictive_scale not in PREDICTIVE_SCALES:
        raise ValueError(f"unknown method {cfg.method!r} or predictive_scale {cfg.predictive_scale!r}")
    start_time = time.perf_counter()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    x_train, x_held, y_train, y_held = load_digit_halves(cfg.seed, cfg.test_size)
    x_val, x_test, y_val, y_test = train_test_split(x_held, y_held, train_size=cfg.n_val, random_state=cfg.seed)
    if cfg.teacher == "pca":
        teacher = PCAWhiteningTeacher(y_train, cfg.embedding_dim)
    else:
        teacher = CCATeacher(x_train, y_train, cfg.teacher_dim)
    t_train, t_val, t_test = teacher(y_train), teacher(y_val), teacher(y_test)
    x_train_t, x_val_t, x_test_t = torch.from_numpy(x_train), torch.from_numpy(x_val), torch.from_numpy(x_test)
    y_train_t = torch.from_numpy(y_train)
    t_train_t = torch.from_numpy(t_train)

    d, h = cfg.embedding_dim, cfg.hidden_dim
    encoder = mlp(x_train.shape[1], h, d)
    k = t_train.shape[1]
    predictor = mlp(d, h, k)
    pixel_head = mlp(k, h, y_train.shape[1])
    optimizer = torch.optim.AdamW([
        {"params": list(encoder.parameters()) + list(predictor.parameters()), "weight_decay": cfg.weight_decay},
        {"params": list(pixel_head.parameters()), "weight_decay": cfg.pixel_weight_decay},
    ], lr=cfg.lr)
    z_dirs = fixed_directions(d, cfg.n_directions, cfg.seed + 101)
    p_dirs = fixed_directions(k, cfg.n_directions, cfg.seed + 303)
    n = x_train.shape[0]
    generator = torch.Generator().manual_seed(cfg.seed + 505)

    for _epoch in range(cfg.epochs):
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, cfg.batch_size):
            batch = perm[start : start + cfg.batch_size]
            z = encoder(x_train_t[batch])
            p = predictor(z)
            latent_loss = torch.nn.functional.mse_loss(p, t_train_t[batch])
            reg = torch.zeros((), dtype=latent_loss.dtype)
            if cfg.method in {"encoder_gaussian", "both"}:
                reg = reg + projected_gaussianity_loss(z, z_dirs) + cfg.cov_weight * covariance_standard_normal_loss(z)
            if cfg.method in {"predictive_gaussian", "both"}:
                pp = p if cfg.predictive_scale == "unit" else trace_normalize(p)
                reg = reg + projected_gaussianity_loss(pp, p_dirs) + cfg.cov_weight * covariance_standard_normal_loss(pp)
            # The pixel head sees detached predictions, so its loss never reaches the encoder or predictor.
            pixel_loss = torch.nn.functional.mse_loss(pixel_head(p.detach()), y_train_t[batch])
            loss = latent_loss + cfg.reg_weight * reg + pixel_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        z_te = encoder(x_test_t)
        p_tr, p_val, p_te = predictor(encoder(x_train_t)), predictor(encoder(x_val_t)), predictor(z_te)
        pix_tr, pix_val, pix_te = (pixel_head(p).numpy() for p in (p_tr, p_val, p_te))
    pixel_train, pixel_val, pixel_test = (float(np.mean((a - b) ** 2)) for a, b in
                                          ((pix_tr, y_train), (pix_val, y_val), (pix_te, y_test)))
    z_np, p_np = z_te.numpy(), p_te.numpy()
    z_m, p_m = covariance_metrics(z_np), covariance_metrics(p_np)
    return {
        "experiment": "jepa_digits",
        "method": cfg.method,
        "predictive_scale": cfg.predictive_scale,
        "teacher": cfg.teacher,
        "teacher_dim": k,
        "seed": cfg.seed,
        "reg_weight": cfg.reg_weight,
        "weight_decay": cfg.weight_decay,
        "cov_weight": cfg.cov_weight,
        "n_train": int(n),
        "n_val": cfg.n_val,
        "teacher_explained_variance": teacher.explained,
        **latent_metrics(teacher, p_tr.numpy(), t_train, p_val.numpy(), t_val, p_np, t_test),
        "pixel_train_mse": pixel_train,
        "pixel_val_mse": pixel_val,
        "pixel_test_mse": pixel_test,
        "pixel_gap": pixel_test - pixel_train,
        "train_seconds": time.perf_counter() - start_time,
        # Isotropy is defined up to scale (Cov = (tau/k) I), so the Gaussianity distances use unit average variance.
        "embedding_effective_rank": z_m["effective_rank"],
        "embedding_kappa": z_m["kappa"],
        "embedding_gaussianity_score": projected_gaussianity_score(trace_normalize(z_te).numpy(), z_dirs.numpy()),
        "predictive_effective_rank": p_m["effective_rank"],
        "predictive_kappa": p_m["kappa"],
        "predictive_trace": float(np.trace(np.cov(p_np, rowvar=False))),
        "predictive_gaussianity_score": projected_gaussianity_score(trace_normalize(p_te).numpy(), p_dirs.numpy()),
    }


def run(config_path: Path, output_path: Path) -> pd.DataFrame:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    rows = [train_one(params) for params in expanded_grid(config)]
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


if __name__ == "__main__":
    args = parse_args(ROOT / "configs" / "jepa_digits.yaml", ROOT / "results" / "exp_jepa_digits.csv")
    run(Path(args.config), Path(args.output))
