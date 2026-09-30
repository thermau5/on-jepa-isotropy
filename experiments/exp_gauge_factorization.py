"""Experiment 8: gauge-factorization diagnostics for Section 3.6.1."""

from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from data.synthetic import SyntheticSpec, sample_problem
from experiments.common import ROOT, expanded_grid, inv_sqrt_psd, load_config, parse_args


def covariance(samples: np.ndarray) -> np.ndarray:
    centered = samples - samples.mean(axis=0, keepdims=True)
    return centered.T @ centered / max(samples.shape[0] - 1, 1)


def relative_covariance_isotropy_error(cov: np.ndarray) -> float:
    dim = cov.shape[0]
    target = np.trace(cov) / max(dim, 1) * np.eye(dim)
    return float(np.linalg.norm(cov - target, ord="fro") / max(np.linalg.norm(target, ord="fro"), 1e-12))


STANDARD_NORMAL = NormalDist()


def normal_cdf(values: np.ndarray) -> np.ndarray:
    return np.vectorize(STANDARD_NORMAL.cdf, otypes=[float])(values)


def normal_ppf(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-10, 1.0 - 1e-10)
    return np.vectorize(STANDARD_NORMAL.inv_cdf, otypes=[float])(clipped)


def standardized_laplace_cdf(values: np.ndarray) -> np.ndarray:
    scale = 1.0 / np.sqrt(2.0)
    cdf = np.where(
        values < 0.0,
        0.5 * np.exp(values / scale),
        1.0 - 0.5 * np.exp(-values / scale),
    )
    return np.clip(cdf, 1e-10, 1.0 - 1e-10)


def standardized_laplace_ppf(values: np.ndarray) -> np.ndarray:
    scale = 1.0 / np.sqrt(2.0)
    clipped = np.clip(values, 1e-10, 1.0 - 1e-10)
    return np.where(
        clipped < 0.5,
        scale * np.log(2.0 * clipped),
        -scale * np.log(2.0 * (1.0 - clipped)),
    )


def sample_svd_coordinates(
    n: int,
    dim: int,
    law: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if law == "gaussian":
        return rng.normal(size=(n, dim))
    if law == "laplace":
        return rng.laplace(scale=1.0 / np.sqrt(2.0), size=(n, dim))
    raise ValueError(f"unknown context_law: {law}")


def gaussianize_coordinates(values: np.ndarray, law: str) -> np.ndarray:
    if law == "gaussian":
        return values
    if law == "laplace":
        return normal_ppf(standardized_laplace_cdf(values))
    raise ValueError(f"unknown context_law: {law}")


def degaussianize_coordinates(values: np.ndarray, law: str) -> np.ndarray:
    if law == "gaussian":
        return values
    if law == "laplace":
        return standardized_laplace_ppf(normal_cdf(values))
    raise ValueError(f"unknown context_law: {law}")


def projection_gaussianity_score(
    samples: np.ndarray,
    rng: np.random.Generator,
    n_directions: int = 64,
) -> float:
    """Cramer-Wold-style standard-normal characteristic-function discrepancy."""

    dim = samples.shape[1]
    random_directions = rng.normal(size=(n_directions, dim))
    random_directions /= np.linalg.norm(random_directions, axis=1, keepdims=True) + 1e-12
    directions = np.vstack([np.eye(dim), random_directions])
    projected = samples @ directions.T
    t_grid = np.array([0.5, 1.0, 1.5, 2.0])
    target = np.exp(-0.5 * t_grid**2)
    scores = []
    for column in range(projected.shape[1]):
        values = projected[:, column]
        empirical = np.array([np.mean(np.exp(1j * t * values)) for t in t_grid])
        scores.append(float(np.mean(np.abs(empirical - target) ** 2)))
    return float(np.mean(scores))


def gauge_matrix(kind: str, dim: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "gaussianized":
        return np.eye(dim)
    if kind == "identity":
        return np.eye(dim)
    if kind == "orthogonal":
        q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
        return q
    if kind == "diagonal":
        scales = np.geomspace(2.5, 0.4, dim)
        scales *= np.sqrt(dim / np.sum(scales**2))
        return np.diag(scales)
    raise ValueError(f"unknown gauge_type: {kind}")


def evaluate(params: dict[str, object]) -> dict[str, object]:
    spec_params = {
        key: value
        for key, value in params.items()
        if key in SyntheticSpec.__dataclass_fields__
    }
    spec = SyntheticSpec(**spec_params)
    gauge_type = str(params["gauge_type"])
    context_law = str(params.get("context_law", "gaussian"))
    rng = np.random.default_rng(spec.seed + 17017)
    problem = sample_problem(spec)

    inv_sqrt_zz = inv_sqrt_psd(problem.population_zz_cov)
    t_operator = problem.population_cross_cov @ inv_sqrt_zz
    u, singular_values, vt = np.linalg.svd(t_operator, full_matrices=False)
    rank = int(params["d"])
    u_rank = u[:, :rank]
    s_rank = singular_values[:rank]
    v_rank = vt[:rank, :].T

    svd_coordinates = sample_svd_coordinates(spec.n_test, rank, context_law, rng)
    baseline_prediction = svd_coordinates @ np.diag(s_rank) @ u_rank.T

    gauge = gauge_matrix(gauge_type, rank, rng)
    gauge_inv = np.linalg.inv(gauge)
    if gauge_type == "gaussianized":
        embedding = gaussianize_coordinates(svd_coordinates, context_law)
        recovered_coordinates = degaussianize_coordinates(embedding, context_law)
    else:
        embedding = svd_coordinates @ gauge.T
        recovered_coordinates = embedding @ gauge_inv.T
    prediction = recovered_coordinates @ np.diag(s_rank) @ u_rank.T

    pred_norm = np.linalg.norm(baseline_prediction)
    relative_prediction_error = float(
        np.linalg.norm(prediction - baseline_prediction) / max(pred_norm, 1e-12)
    )
    prediction_mse = float(np.mean(np.sum((prediction - baseline_prediction) ** 2, axis=1)))

    emb_cov = covariance(embedding)
    expected_cov = gauge @ gauge.T
    expected_cov_error = float(
        np.linalg.norm(emb_cov - expected_cov, ord="fro")
        / max(np.linalg.norm(expected_cov, ord="fro"), 1e-12)
    )
    isotropy_error = relative_covariance_isotropy_error(emb_cov)
    expected_isotropy_error = relative_covariance_isotropy_error(expected_cov)
    gaussianity_score = projection_gaussianity_score(embedding, rng)

    return {
        "experiment": "gauge_factorization",
        "seed": spec.seed,
        "gauge_type": gauge_type,
        "context_law": context_law,
        "n_train": spec.n_train,
        "n_test": spec.n_test,
        "context_dim": spec.context_dim,
        "K": spec.K,
        "d": rank,
        "r_star": spec.r_star,
        "sigma_z": spec.sigma_z,
        "sigma_y": spec.sigma_y,
        "relative_prediction_error": relative_prediction_error,
        "prediction_mse": prediction_mse,
        "embedding_cov_isotropy_error": isotropy_error,
        "expected_embedding_cov_isotropy_error": expected_isotropy_error,
        "embedding_cov_expected_error": expected_cov_error,
        "projection_gaussianity_score": gaussianity_score,
    }


def run(config_path: Path, output_path: Path) -> pd.DataFrame:
    config = load_config(config_path)
    frame = pd.DataFrame([evaluate(params) for params in expanded_grid(config)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


if __name__ == "__main__":
    args = parse_args(
        ROOT / "configs" / "gauge_factorization.yaml",
        ROOT / "results" / "exp_gauge_factorization.csv",
    )
    run(Path(args.config), Path(args.output))
