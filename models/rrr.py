"""Ordinary and reduced-rank regression solvers."""

from __future__ import annotations

import numpy as np


def covariance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return E[y x^T] using row-major samples."""

    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same sample count")
    return (y.T @ x) / x.shape[0]


def ols_fit(x: np.ndarray, y: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """Fit the linear predictor y_hat = x @ B.T and return B."""

    sigma_xx = covariance(x, x)
    sigma_yx = covariance(x, y)
    regularized = sigma_xx + ridge * np.eye(sigma_xx.shape[0])
    return sigma_yx @ np.linalg.pinv(regularized)


def reduced_rank_fit(
    x: np.ndarray,
    y: np.ndarray,
    rank: int,
    ridge: float = 1e-8,
) -> np.ndarray:
    """Closed-form reduced-rank regression via whitened OLS truncation."""

    if rank < 1:
        raise ValueError("rank must be positive")
    sigma_xx = covariance(x, x)
    sigma_yx = covariance(x, y)
    regularized = sigma_xx + ridge * np.eye(sigma_xx.shape[0])
    b_ols = sigma_yx @ np.linalg.pinv(regularized)

    evals, evecs = np.linalg.eigh(regularized)
    evals = np.clip(evals, ridge, None)
    sqrt_xx = (evecs * np.sqrt(evals)) @ evecs.T
    inv_sqrt_xx = (evecs * (1.0 / np.sqrt(evals))) @ evecs.T

    whitened = b_ols @ sqrt_xx
    u, s, vt = np.linalg.svd(whitened, full_matrices=False)
    kept = min(rank, s.size)
    truncated = (u[:, :kept] * s[:kept]) @ vt[:kept, :]
    return truncated @ inv_sqrt_xx


def predict(x: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
    """Predict row-major targets."""

    return x @ coefficient.T


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared prediction error per scalar target."""

    return float(np.mean((y_true - y_pred) ** 2))

