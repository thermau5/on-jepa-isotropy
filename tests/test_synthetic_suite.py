"""Smoke and unit tests for the synthetic BM-JEPA suite."""

from __future__ import annotations

import numpy as np

from data.synthetic import (
    SyntheticSpec,
    fixed_trace_spectrum_target_operator,
    relative_conditioning_target_operator,
    sample_problem,
    target_operator,
)
from experiments.common import evaluate_setting
from metrics.subspace import numerical_rank, subspace_error
from models.rrr import mse, ols_fit, predict, reduced_rank_fit


def test_generator_is_deterministic() -> None:
    spec = SyntheticSpec(seed=7, n_train=32, n_test=16, K=3)
    first = sample_problem(spec)
    second = sample_problem(spec)
    np.testing.assert_allclose(first.z_train, second.z_train)
    np.testing.assert_allclose(first.y_train, second.y_train)


def test_shapes_follow_k_and_target_dim() -> None:
    spec = SyntheticSpec(n_train=20, n_test=10, K=5, target_dim=2, context_dim=12, r_star=6)
    problem = sample_problem(spec)
    assert problem.z_train.shape == (20, 12)
    assert problem.y_train.shape == (20, 10)
    assert problem.target_operator.shape == (10, 6)


def test_target_heterogeneity_controls_dictionary_rank() -> None:
    collapsed = target_operator(K=8, target_dim=1, r_star=8, alpha=0.0, overlap=0.0, rng=None)
    spread = target_operator(K=8, target_dim=1, r_star=8, alpha=1.0, overlap=0.0, rng=None)
    assert numerical_rank(collapsed, tol=1e-8) == 1
    assert numerical_rank(spread, tol=1e-8) == 8


def test_target_overlap_concentrates_energy_without_reusing_alpha() -> None:
    no_overlap = target_operator(K=8, target_dim=1, r_star=8, alpha=1.0, overlap=0.0, rng=None)
    high_overlap = target_operator(K=8, target_dim=1, r_star=8, alpha=1.0, overlap=1.0, rng=None)
    mirror_alpha = target_operator(K=8, target_dim=1, r_star=8, alpha=0.0, overlap=0.0, rng=None)
    assert numerical_rank(high_overlap, tol=1e-8) == 1
    assert not np.allclose(np.linalg.svd(no_overlap, compute_uv=False), np.linalg.svd(mirror_alpha, compute_uv=False))


def test_relative_conditioning_bank_improves_kappa() -> None:
    rng = np.random.default_rng(0)
    initial = relative_conditioning_target_operator(
        K=6,
        target_dim=1,
        r_star=6,
        rng=rng,
        initial_imbalance_ratio=0.1,
        weak_target_strength=np.sqrt(0.1),
    )
    rng = np.random.default_rng(0)
    improved = relative_conditioning_target_operator(
        K=24,
        target_dim=1,
        r_star=6,
        rng=rng,
        initial_imbalance_ratio=0.1,
        weak_target_strength=np.sqrt(0.1),
    )

    def kappa_squared(operator: np.ndarray) -> float:
        evals = np.linalg.eigvalsh(operator.T @ operator)
        return float(evals[0] / evals[-1])

    assert numerical_rank(initial, tol=1e-8) == 6
    assert kappa_squared(improved) > kappa_squared(initial)


def test_fixed_trace_spectrum_controls_shape_not_trace() -> None:
    rng = np.random.default_rng(0)
    flat = fixed_trace_spectrum_target_operator(
        K=6,
        target_dim=1,
        r_star=6,
        rng=rng,
        spectrum_decay=1.0,
        spectrum_trace=6.0,
    )
    rng = np.random.default_rng(0)
    anisotropic = fixed_trace_spectrum_target_operator(
        K=6,
        target_dim=1,
        r_star=6,
        rng=rng,
        spectrum_decay=0.45,
        spectrum_trace=6.0,
    )

    flat_evals = np.linalg.eigvalsh(flat.T @ flat)
    aniso_evals = np.linalg.eigvalsh(anisotropic.T @ anisotropic)
    np.testing.assert_allclose(np.sum(flat_evals), np.sum(aniso_evals))
    assert numerical_rank(anisotropic, tol=1e-8) == 6
    assert aniso_evals[0] / aniso_evals[-1] < flat_evals[0] / flat_evals[-1]


def test_ols_recovers_nearly_noiseless_problem() -> None:
    spec = SyntheticSpec(n_train=2048, n_test=512, K=4, sigma_z=0.0, sigma_y=0.0, seed=3)
    problem = sample_problem(spec)
    coef = ols_fit(problem.z_train, problem.y_train, ridge=1e-10)
    assert mse(problem.y_test, predict(problem.z_test, coef)) < 1e-8


def test_rrr_respects_rank_constraint() -> None:
    spec = SyntheticSpec(n_train=256, n_test=64, K=8, seed=4)
    problem = sample_problem(spec)
    coef = reduced_rank_fit(problem.z_train, problem.y_train, rank=3)
    assert numerical_rank(coef, tol=1e-5) <= 3


def test_rank_equality_diagnostic_matches_rrr_solution() -> None:
    row = evaluate_setting(
        "unit",
        {
            "n_train": 512,
            "n_test": 128,
            "context_dim": 12,
            "target_dim": 1,
            "r_star": 6,
            "K": 6,
            "d": 3,
            "alpha": 1.0,
            "overlap": 0.0,
            "sigma_z": 0.02,
            "sigma_y": 0.02,
            "seed": 11,
        },
    )
    assert row["rrr_effective_rank"] == row["rank_equality_target"]
    assert row["rank_equality_gap"] == 0


def test_theory_diagnostic_columns_are_reported() -> None:
    row = evaluate_setting(
        "unit",
        {
            "n_train": 128,
            "n_test": 64,
            "context_dim": 10,
            "target_dim": 1,
            "r_star": 5,
            "K": 8,
            "d": 5,
            "alpha": 1.0,
            "overlap": 0.0,
            "sigma_z": 0.05,
            "sigma_y": 0.05,
            "seed": 13,
        },
    )
    for key in [
        "theorem_subspace_error",
        "lambda_at_bottleneck_rank",
        "oracle_lambda_at_bottleneck_rank",
        "fixed_eval_excess_mse",
        "theorem_fixed_eval_excess_mse",
        "model_effective_rank",
        "operator_error_op",
        "cross_cov_error_op",
        "empirical_subspace_bound_clipped",
        "subspace_bound_valid",
        "oracle_test_mse",
        "spectral_tail_ratio",
        "empirical_spectral_tail_ratio",
        "excess_risk_ratio",
        "lambda_top",
        "oracle_lambda_top",
        "relative_conditioning",
        "oracle_relative_conditioning",
        "trace_h",
        "oracle_trace_h",
        "entropy_gap",
        "oracle_entropy_gap",
        "embedding_relative_conditioning",
        "embedding_entropy_gap",
        "risk_stability_ratio",
        "per_target_excess_mse",
        "per_target_risk_stability_ratio",
    ]:
        assert key in row
        assert np.isfinite(row[key])


def test_subspace_error_zero_for_same_subspace() -> None:
    q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(8, 3)))
    assert subspace_error(q, q) < 1e-10
