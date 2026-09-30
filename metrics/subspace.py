"""Rank, spectrum, and subspace recovery metrics."""

from __future__ import annotations

import numpy as np


def singular_values(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.svd(matrix, compute_uv=False)


def numerical_rank(matrix: np.ndarray, tol: float = 1e-6) -> int:
    values = singular_values(matrix)
    if values.size == 0:
        return 0
    threshold = tol * max(matrix.shape) * max(values[0], 1.0)
    return int(np.sum(values > threshold))


def right_subspace(matrix: np.ndarray, rank: int) -> np.ndarray:
    """Return top right singular vectors as columns."""

    if rank < 1:
        raise ValueError("rank must be positive")
    _, _, vt = np.linalg.svd(matrix, full_matrices=False)
    return vt[: min(rank, vt.shape[0]), :].T


def orthonormalize(basis: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(basis)
    return q[:, : basis.shape[1]]


def subspace_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Operator-norm sin(theta) distance between two column subspaces."""

    ref = orthonormalize(reference)
    est = orthonormalize(estimate)
    dim = min(ref.shape[1], est.shape[1])
    ref = ref[:, :dim]
    est = est[:, :dim]
    cosines = np.linalg.svd(ref.T @ est, compute_uv=False)
    cosines = np.clip(cosines, 0.0, 1.0)
    if np.min(cosines) > 1.0 - 1e-12:
        return 0.0
    return float(np.sqrt(max(0.0, 1.0 - np.min(cosines) ** 2)))


def smallest_nonzero_singular(matrix: np.ndarray, tol: float = 1e-6) -> float:
    values = singular_values(matrix)
    nonzero = values[values > tol]
    if nonzero.size == 0:
        return 0.0
    return float(nonzero[-1])


def spectral_tail(values: np.ndarray, rank: int) -> float:
    if rank >= values.size:
        return 0.0
    return float(np.sum(values[rank:] ** 2))


def lambda_het_proxy(stacked_cross_cov: np.ndarray, tol: float = 1e-8) -> float:
    """Smallest nonzero eigenvalue of G = Sigma_YX^T Sigma_YX."""

    gram = stacked_cross_cov.T @ stacked_cross_cov
    evals = np.linalg.eigvalsh(gram)
    positive = evals[evals > tol]
    if positive.size == 0:
        return 0.0
    return float(positive[0])
