"""Shared experiment runner utilities."""

from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from data.synthetic import SyntheticSpec, sample_problem
from metrics.subspace import (
    lambda_het_proxy,
    right_subspace,
    singular_values,
    spectral_tail,
    subspace_error,
)
from models.rrr import covariance, mse, ols_fit, predict, reduced_rank_fit


ROOT = Path(__file__).resolve().parents[1]
SPEC_FIELDS = {field.name for field in fields(SyntheticSpec)}
RANK_REL_TOL = 0.05
NUMERIC_RANK_TOL = 1e-10


def inv_sqrt_psd(matrix: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Symmetric inverse square root of a PSD matrix with eigenvalue flooring."""

    sym = 0.5 * (matrix + matrix.T)
    evals, evecs = np.linalg.eigh(sym)
    evals = np.clip(evals, floor, None)
    return (evecs * (1.0 / np.sqrt(evals))) @ evecs.T


def psd_condition_stats(matrix: np.ndarray, floor: float = 1e-12) -> tuple[float, float]:
    """Return smallest eigenvalue and floor-stabilized condition number."""

    sym = 0.5 * (matrix + matrix.T)
    evals = np.linalg.eigvalsh(sym)
    min_eval = float(np.min(evals))
    max_eval = float(np.max(evals))
    return min_eval, max_eval / max(min_eval, floor)


def effective_rank_and_min_sv(values: np.ndarray, rel_tol: float = RANK_REL_TOL) -> tuple[int, float]:
    """Effective rank and smallest retained singular value."""

    threshold = rel_tol * max(values[0], 1e-12) if values.size else 0.0
    retained = values[values > threshold]
    if retained.size == 0:
        return 0, 0.0
    return int(retained.size), float(retained[-1])


def population_rank_and_sv(values: np.ndarray, tol: float = NUMERIC_RANK_TOL) -> tuple[int, float]:
    """Population rank and sigma_rank using only numerical zero tolerance."""

    if values.size == 0:
        return 0, 0.0
    threshold = tol * max(values[0], 1.0)
    nonzero = values[values > threshold]
    if nonzero.size == 0:
        return 0, 0.0
    return int(nonzero.size), float(nonzero[-1])


def singular_value_at_rank(values: np.ndarray, rank: int) -> float:
    """Return the rank-th largest singular value using one-indexed rank notation."""

    if rank < 1 or values.size < rank:
        return 0.0
    return float(values[rank - 1])


def effective_rank(values: np.ndarray, rel_tol: float = RANK_REL_TOL) -> int:
    """Effective rank for an already-computed singular-value spectrum."""

    rank, _ = effective_rank_and_min_sv(values, rel_tol=rel_tol)
    return rank


def entropy_gap_from_singular_values(values: np.ndarray, rank: int) -> float:
    """Log-det isotropy score, zero for a flat positive spectrum."""

    if rank < 1 or values.size < rank:
        return float("nan")
    lambdas = np.maximum(values[:rank] ** 2, 1e-300)
    return float(np.mean(np.log(lambdas)) - np.log(np.mean(lambdas)))


def controlled_spectrum_scores(
    decay: float,
    rank: int,
    trace: float,
) -> tuple[float, float]:
    """Return relative conditioning and entropy gap for a geometric spectrum."""

    if rank < 1:
        return float("nan"), float("nan")
    if not 0.0 < decay <= 1.0:
        raise ValueError("decay must be in (0, 1]")
    if trace <= 0.0:
        raise ValueError("trace must be positive")
    weights = decay ** np.arange(rank, dtype=float)
    lambdas = trace * weights / np.sum(weights)
    relative_conditioning = float(np.min(lambdas) / np.max(lambdas))
    entropy_gap = float(np.mean(np.log(lambdas)) - np.log(np.mean(lambdas)))
    return relative_conditioning, entropy_gap


def sample_risk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean sum-squared error per sample on the raw stacked target."""

    return float(np.mean(np.sum((y_true - y_pred) ** 2, axis=1)))


def empirical_spectral_tail(
    x: np.ndarray,
    y: np.ndarray,
    rank: int,
) -> float:
    """Plug-in spectral tail from the empirical whitened OLS operator."""

    sigma_xx = covariance(x, x)
    sigma_yx = covariance(x, y)
    evals, evecs = np.linalg.eigh(sigma_xx + 1e-8 * np.eye(sigma_xx.shape[0]))
    evals = np.clip(evals, 1e-8, None)
    inv_xx = (evecs * (1.0 / evals)) @ evecs.T
    sqrt_xx = (evecs * np.sqrt(evals)) @ evecs.T
    b_ols = sigma_yx @ inv_xx
    whitened = b_ols @ sqrt_xx
    values = singular_values(whitened)
    return spectral_tail(values, rank)


def empirical_spectral_energy(x: np.ndarray, y: np.ndarray) -> float:
    """Total plug-in predictive spectral energy from empirical OLS."""

    return empirical_spectral_tail(x, y, rank=0)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def expanded_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    base = dict(config.get("base", {}))
    sweep = dict(config.get("sweep", {}))
    keys = list(sweep)
    values = [sweep[key] for key in keys]
    rows = []
    for combo in itertools.product(*values):
        row = dict(base)
        row.update(dict(zip(keys, combo)))
        rows.append(row)
    return rows


def evaluate_setting(experiment: str, params: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    spec_params = {key: value for key, value in params.items() if key in SPEC_FIELDS}
    spec = SyntheticSpec(**spec_params)
    problem = sample_problem(spec)

    b_ols = ols_fit(problem.z_train, problem.y_train)
    b_rrr = reduced_rank_fit(problem.z_train, problem.y_train, rank=params["d"])
    rrr_effective_rank = effective_rank(singular_values(b_rrr))
    model_effective_rank = min(rrr_effective_rank, params["d"], spec.r_star)

    ols_mse = mse(problem.y_test, predict(problem.z_test, b_ols))
    rrr_mse = mse(problem.y_test, predict(problem.z_test, b_rrr))
    oracle_test_mse = mse(problem.y_test, predict(problem.z_test, problem.population_b))
    ols_risk = sample_risk(problem.y_test, predict(problem.z_test, b_ols))
    rrr_risk = sample_risk(problem.y_test, predict(problem.z_test, b_rrr))
    excess_risk_over_ols = rrr_risk - ols_risk

    sigma_yz_hat = covariance(problem.z_train, problem.y_train)
    sigma_zz_hat = covariance(problem.z_train, problem.z_train)
    zz_hat_min_eig, zz_hat_condition = psd_condition_stats(sigma_zz_hat)
    zz_pop_min_eig, zz_pop_condition = psd_condition_stats(problem.population_zz_cov)
    subspace_operator = str(params.get("subspace_operator", "cross_cov"))
    if subspace_operator == "cross_cov":
        theory_operator_hat = sigma_yz_hat
        theory_operator_pop = problem.population_cross_cov
    elif subspace_operator == "whitened_raw":
        theory_operator_hat = sigma_yz_hat @ inv_sqrt_psd(sigma_zz_hat)
        theory_operator_pop = problem.population_cross_cov @ inv_sqrt_psd(
            problem.population_zz_cov
        )
    else:
        raise ValueError(f"unknown subspace_operator: {subspace_operator}")

    sigma_values = singular_values(theory_operator_hat)
    sigma_top = singular_value_at_rank(sigma_values, 1)
    lambda_top = sigma_top**2
    operator_error_op = singular_value_at_rank(
        singular_values(theory_operator_hat - theory_operator_pop), 1
    )
    cross_cov_error_op = operator_error_op
    rank_threshold = RANK_REL_TOL * max(sigma_values[0], 1e-12)
    recovered_rank, sigma_min = effective_rank_and_min_sv(sigma_values)
    population_cross_values = singular_values(theory_operator_pop)
    oracle_sigma_top = singular_value_at_rank(population_cross_values, 1)
    oracle_lambda_top = oracle_sigma_top**2
    oracle_rank, oracle_sigma_min = population_rank_and_sv(population_cross_values)
    oracle_effective_rank, oracle_effective_sigma_min = effective_rank_and_min_sv(
        population_cross_values
    )
    oracle_lambda = oracle_sigma_min**2
    sigma_at_population_rank = singular_value_at_rank(sigma_values, oracle_rank)
    lambda_at_population_rank = sigma_at_population_rank**2
    bottleneck_rank = min(params["d"], oracle_rank)
    sigma_at_bottleneck_rank = singular_value_at_rank(sigma_values, bottleneck_rank)
    sigma_after_bottleneck_rank = singular_value_at_rank(
        sigma_values,
        bottleneck_rank + 1,
    )
    oracle_sigma_at_bottleneck_rank = singular_value_at_rank(
        population_cross_values, bottleneck_rank
    )
    oracle_sigma_after_bottleneck_rank = singular_value_at_rank(
        population_cross_values, bottleneck_rank + 1
    )
    lambda_at_bottleneck_rank = sigma_at_bottleneck_rank**2
    oracle_lambda_at_bottleneck_rank = oracle_sigma_at_bottleneck_rank**2
    retained_gap_abs = max(
        oracle_sigma_at_bottleneck_rank - oracle_sigma_after_bottleneck_rank,
        0.0,
    )
    retained_relative_gap = retained_gap_abs / max(oracle_sigma_top, 1e-12)
    normalized_operator_error = operator_error_op / max(oracle_sigma_top, 1e-12)
    relative_conditioning = lambda_at_bottleneck_rank / max(lambda_top, 1e-12)
    oracle_relative_conditioning = oracle_lambda_at_bottleneck_rank / max(
        oracle_lambda_top, 1e-12
    )
    empirical_effdim = float(np.sum(sigma_values**2) / max(lambda_top, 1e-12))
    oracle_effdim = float(
        np.sum(population_cross_values**2) / max(oracle_lambda_top, 1e-12)
    )
    trace_h = float(np.sum(sigma_values**2))
    oracle_trace_h = float(np.sum(population_cross_values**2))
    entropy_gap = entropy_gap_from_singular_values(sigma_values, bottleneck_rank)
    oracle_entropy_gap = entropy_gap_from_singular_values(
        population_cross_values, bottleneck_rank
    )
    embedding_decay = float(params.get("embedding_decay", 1.0))
    embedding_trace = float(params.get("embedding_trace", max(params["d"], 1)))
    embedding_relative_conditioning, embedding_entropy_gap = controlled_spectrum_scores(
        embedding_decay,
        bottleneck_rank,
        embedding_trace,
    )
    rank_equality_target = min(params["d"], recovered_rank)
    rank_equality_gap = rrr_effective_rank - rank_equality_target
    sub_rank = min(params["d"], spec.r_star, recovered_rank, spec.context_dim)
    if sub_rank == 0:
        angle_error = 1.0
        fixed_eval_mse = float("nan")
        fixed_eval_relative_mse = float("nan")
    else:
        estimated = right_subspace(sigma_yz_hat, sub_rank)
        truth = right_subspace(problem.population_b, sub_rank)
        angle_error = subspace_error(truth, estimated)
        fixed_eval_operator = problem.eval_target_operator
        fixed_y_train = problem.latent_train @ fixed_eval_operator.T
        fixed_y_test = problem.latent_test @ fixed_eval_operator.T
        projected_train = problem.z_train @ estimated
        projected_test = problem.z_test @ estimated
        fixed_eval_coef = ols_fit(projected_train, fixed_y_train)
        fixed_eval_pred = predict(projected_test, fixed_eval_coef)
        fixed_eval_mse = mse(fixed_y_test, fixed_eval_pred)
        fixed_eval_relative_mse = fixed_eval_mse / max(float(np.mean(fixed_y_test**2)), 1e-12)

    oracle_sub_rank = min(params["d"], spec.r_star, oracle_effective_rank, spec.context_dim)
    if oracle_sub_rank == 0:
        oracle_fixed_eval_mse = float("nan")
        oracle_fixed_eval_relative_mse = float("nan")
    else:
        oracle_basis = right_subspace(problem.population_b, oracle_sub_rank)
        fixed_eval_operator = problem.eval_target_operator
        fixed_y_train = problem.latent_train @ fixed_eval_operator.T
        fixed_y_test = problem.latent_test @ fixed_eval_operator.T
        oracle_projected_train = problem.z_train @ oracle_basis
        oracle_projected_test = problem.z_test @ oracle_basis
        oracle_eval_coef = ols_fit(oracle_projected_train, fixed_y_train)
        oracle_eval_pred = predict(oracle_projected_test, oracle_eval_coef)
        oracle_fixed_eval_mse = mse(fixed_y_test, oracle_eval_pred)
        oracle_fixed_eval_relative_mse = oracle_fixed_eval_mse / max(
            float(np.mean(fixed_y_test**2)), 1e-12
        )

    theorem_sub_rank = min(oracle_rank, spec.r_star, spec.context_dim)
    if theorem_sub_rank == 0:
        theorem_subspace_error = float("nan")
        theorem_fixed_eval_mse = float("nan")
        theorem_oracle_fixed_eval_mse = float("nan")
        theorem_fixed_eval_relative_mse = float("nan")
    else:
        theorem_estimated = right_subspace(theory_operator_hat, theorem_sub_rank)
        theorem_truth = right_subspace(theory_operator_pop, theorem_sub_rank)
        theorem_subspace_error = subspace_error(theorem_truth, theorem_estimated)
        fixed_eval_operator = problem.eval_target_operator
        fixed_y_train = problem.latent_train @ fixed_eval_operator.T
        fixed_y_test = problem.latent_test @ fixed_eval_operator.T
        theorem_projected_train = problem.z_train @ theorem_estimated
        theorem_projected_test = problem.z_test @ theorem_estimated
        theorem_fixed_eval_coef = ols_fit(theorem_projected_train, fixed_y_train)
        theorem_fixed_eval_pred = predict(theorem_projected_test, theorem_fixed_eval_coef)
        theorem_fixed_eval_mse = mse(fixed_y_test, theorem_fixed_eval_pred)
        theorem_oracle_projected_train = problem.z_train @ theorem_truth
        theorem_oracle_projected_test = problem.z_test @ theorem_truth
        theorem_oracle_fixed_eval_coef = ols_fit(
            theorem_oracle_projected_train,
            fixed_y_train,
        )
        theorem_oracle_fixed_eval_pred = predict(
            theorem_oracle_projected_test,
            theorem_oracle_fixed_eval_coef,
        )
        theorem_oracle_fixed_eval_mse = mse(
            fixed_y_test,
            theorem_oracle_fixed_eval_pred,
        )
        theorem_fixed_eval_relative_mse = theorem_fixed_eval_mse / max(
            float(np.mean(fixed_y_test**2)),
            1e-12,
        )
    retained_sub_rank = min(params["d"], oracle_rank, spec.r_star, spec.context_dim)
    if retained_sub_rank == 0:
        retained_subspace_error = float("nan")
    else:
        retained_estimated = right_subspace(theory_operator_hat, retained_sub_rank)
        retained_truth = right_subspace(theory_operator_pop, retained_sub_rank)
        retained_subspace_error = subspace_error(retained_truth, retained_estimated)
    subspace_bound_denominator = oracle_sigma_min - cross_cov_error_op
    if subspace_bound_denominator > 0.0:
        empirical_subspace_bound = cross_cov_error_op / subspace_bound_denominator
        empirical_subspace_bound_clipped = min(1.0, empirical_subspace_bound)
        subspace_bound_valid = 1
    else:
        empirical_subspace_bound = float("nan")
        empirical_subspace_bound_clipped = 1.0
        subspace_bound_valid = 0
    retained_subspace_bound_denominator = retained_gap_abs - operator_error_op
    if retained_subspace_bound_denominator > 0.0:
        retained_subspace_bound = operator_error_op / retained_subspace_bound_denominator
        retained_subspace_bound_clipped = min(1.0, retained_subspace_bound)
        retained_subspace_bound_valid = 1
    else:
        retained_subspace_bound = float("nan")
        retained_subspace_bound_clipped = 1.0
        retained_subspace_bound_valid = 0

    pop_spectrum = problem.population_spectrum
    population_tail_per_target = spectral_tail(pop_spectrum, params["d"]) / max(
        spec.K * spec.target_dim,
        1,
    )
    population_rank_r_mse = oracle_test_mse + population_tail_per_target
    per_target_excess_mse = rrr_mse - oracle_test_mse
    rank_r_excess_mse = rrr_mse - population_rank_r_mse
    rank_r_risk_stability_ratio = max(rank_r_excess_mse, 0.0) / max(
        retained_subspace_error,
        1e-12,
    )
    per_target_risk_stability_ratio = max(per_target_excess_mse, 0.0) / max(
        theorem_subspace_error, 1e-12
    )
    fixed_eval_excess_mse = fixed_eval_mse - oracle_fixed_eval_mse
    risk_stability_ratio = max(fixed_eval_excess_mse, 0.0) / max(
        theorem_subspace_error, 1e-12
    )
    theorem_fixed_eval_excess_mse = (
        theorem_fixed_eval_mse - theorem_oracle_fixed_eval_mse
    )

    empirical_tail = empirical_spectral_tail(problem.z_train, problem.y_train, params["d"])
    population_spectral_energy = spectral_tail(pop_spectrum, rank=0)
    empirical_total_energy = empirical_spectral_energy(problem.z_train, problem.y_train)
    spectral_tail_ratio = spectral_tail(pop_spectrum, params["d"]) / max(
        population_spectral_energy, 1e-12
    )
    empirical_spectral_tail_ratio = empirical_tail / max(empirical_total_energy, 1e-12)
    excess_risk_ratio = excess_risk_over_ols / max(empirical_total_energy, 1e-12)
    target_rank = min(spec.K * spec.target_dim, spec.r_star, spec.context_dim)
    rank_fraction = recovered_rank / max(target_rank, 1)
    relative_rrr_to_ols = rrr_mse / max(ols_mse, 1e-12)
    return {
        "experiment": experiment,
        "seed": spec.seed,
        "condition_label": params.get("condition_label", ""),
        "n_train": spec.n_train,
        "n_test": spec.n_test,
        "context_dim": spec.context_dim,
        "target_dim": spec.target_dim,
        "K": spec.K,
        "d": params["d"],
        "r_star": spec.r_star,
        "alpha": spec.alpha,
        "overlap": spec.overlap,
        "sigma_z": spec.sigma_z,
        "sigma_y": spec.sigma_y,
        "spectrum_decay": spec.spectrum_decay,
        "spectrum_trace": spec.spectrum_trace,
        "spectrum_gap_rank": spec.spectrum_gap_rank,
        "embedding_decay": embedding_decay,
        "embedding_trace": embedding_trace,
        "subspace_operator": subspace_operator,
        "test_mse": rrr_mse,
        "ols_test_mse": ols_mse,
        "rrr_test_mse": rrr_mse,
        "oracle_test_mse": oracle_test_mse,
        "ols_test_risk": ols_risk,
        "rrr_test_risk": rrr_risk,
        "excess_risk_over_ols": excess_risk_over_ols,
        "recovered_rank": recovered_rank,
        "rrr_effective_rank": rrr_effective_rank,
        "model_effective_rank": model_effective_rank,
        "rank_equality_target": rank_equality_target,
        "rank_equality_gap": rank_equality_gap,
        "oracle_rank": oracle_rank,
        "oracle_effective_rank": oracle_effective_rank,
        "target_rank": target_rank,
        "rank_fraction": rank_fraction,
        "oracle_rank_fraction": oracle_rank / max(target_rank, 1),
        "subspace_error": angle_error,
        "theorem_subspace_error": theorem_subspace_error,
        "operator_error_op": operator_error_op,
        "cross_cov_error_op": cross_cov_error_op,
        "subspace_bound_denominator": subspace_bound_denominator,
        "empirical_subspace_bound": empirical_subspace_bound,
        "empirical_subspace_bound_clipped": empirical_subspace_bound_clipped,
        "subspace_bound_valid": subspace_bound_valid,
        "retained_subspace_error": retained_subspace_error,
        "retained_subspace_bound": retained_subspace_bound,
        "retained_subspace_bound_clipped": retained_subspace_bound_clipped,
        "retained_subspace_bound_valid": retained_subspace_bound_valid,
        "sigma_min_nonzero": sigma_min,
        "sigma_top": sigma_top,
        "lambda_top": lambda_top,
        "oracle_sigma_min_nonzero": oracle_sigma_min,
        "oracle_sigma_top": oracle_sigma_top,
        "oracle_lambda_top": oracle_lambda_top,
        "oracle_effective_sigma_min": oracle_effective_sigma_min,
        "lambda_het_proxy": lambda_het_proxy(theory_operator_hat, tol=rank_threshold**2),
        "oracle_lambda_het_proxy": oracle_lambda,
        "sigma_at_population_rank": sigma_at_population_rank,
        "lambda_at_population_rank": lambda_at_population_rank,
        "oracle_sigma_at_population_rank": oracle_sigma_min,
        "oracle_lambda_at_population_rank": oracle_lambda,
        "sigma_at_bottleneck_rank": sigma_at_bottleneck_rank,
        "sigma_after_bottleneck_rank": sigma_after_bottleneck_rank,
        "lambda_at_bottleneck_rank": lambda_at_bottleneck_rank,
        "oracle_sigma_at_bottleneck_rank": oracle_sigma_at_bottleneck_rank,
        "oracle_sigma_after_bottleneck_rank": oracle_sigma_after_bottleneck_rank,
        "oracle_lambda_at_bottleneck_rank": oracle_lambda_at_bottleneck_rank,
        "retained_gap_abs": retained_gap_abs,
        "retained_relative_gap": retained_relative_gap,
        "normalized_operator_error": normalized_operator_error,
        "relative_conditioning": relative_conditioning,
        "oracle_relative_conditioning": oracle_relative_conditioning,
        "trace_h": trace_h,
        "oracle_trace_h": oracle_trace_h,
        "entropy_gap": entropy_gap,
        "oracle_entropy_gap": oracle_entropy_gap,
        "embedding_relative_conditioning": embedding_relative_conditioning,
        "embedding_entropy_gap": embedding_entropy_gap,
        "empirical_effdim": empirical_effdim,
        "oracle_effdim": oracle_effdim,
        "zz_hat_min_eig": zz_hat_min_eig,
        "zz_hat_condition": zz_hat_condition,
        "zz_pop_min_eig": zz_pop_min_eig,
        "zz_pop_condition": zz_pop_condition,
        "spectral_tail_proxy": spectral_tail(pop_spectrum, params["d"]),
        "population_tail_per_target": population_tail_per_target,
        "population_rank_r_mse": population_rank_r_mse,
        "empirical_spectral_tail": empirical_tail,
        "population_spectral_energy": population_spectral_energy,
        "empirical_spectral_energy": empirical_total_energy,
        "spectral_tail_ratio": spectral_tail_ratio,
        "empirical_spectral_tail_ratio": empirical_spectral_tail_ratio,
        "excess_risk_ratio": excess_risk_ratio,
        "relative_rrr_to_ols": relative_rrr_to_ols,
        "per_target_excess_mse": per_target_excess_mse,
        "rank_r_excess_mse": rank_r_excess_mse,
        "rank_r_risk_stability_ratio": rank_r_risk_stability_ratio,
        "per_target_risk_stability_ratio": per_target_risk_stability_ratio,
        "fixed_eval_mse": fixed_eval_mse,
        "fixed_eval_excess_mse": fixed_eval_excess_mse,
        "risk_stability_ratio": risk_stability_ratio,
        "fixed_eval_relative_mse": fixed_eval_relative_mse,
        "oracle_fixed_eval_mse": oracle_fixed_eval_mse,
        "oracle_fixed_eval_relative_mse": oracle_fixed_eval_relative_mse,
        "theorem_fixed_eval_mse": theorem_fixed_eval_mse,
        "theorem_oracle_fixed_eval_mse": theorem_oracle_fixed_eval_mse,
        "theorem_fixed_eval_excess_mse": theorem_fixed_eval_excess_mse,
        "theorem_fixed_eval_relative_mse": theorem_fixed_eval_relative_mse,
        "runtime_sec": time.perf_counter() - start,
    }


def run_from_config(experiment: str, config_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    config = load_config(config_path)
    rows = [evaluate_setting(experiment, params) for params in expanded_grid(config)]
    frame = pd.DataFrame(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def parse_args(default_config: Path, default_output: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--output", default=str(default_output))
    return parser.parse_args()
