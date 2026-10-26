"""Exact finite-sample diagnostics for the oracle inequality and the post-saturation risk decomposition.

The rank-r ERM depends on the data only through the sample second moments, and in the linear-Gaussian model the
joint second-moment matrix of n samples of (Z, Y) is Wishart with n degrees of freedom. We draw it exactly by the
Bartlett decomposition, so any n costs the same, and we score every fit by its population risk in closed form.
Within a seed, smaller target stacks use the leading block of the same draw, so adding targets keeps the samples.
"""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.synthetic import SyntheticSpec, sample_problem
from experiments.common import expanded_grid, load_config


SPEC_FIELDS = {field.name for field in fields(SyntheticSpec)}


def sym_power(matrix: np.ndarray, power: float) -> np.ndarray:
    evals, evecs = np.linalg.eigh(0.5 * (matrix + matrix.T))
    return (evecs * evals**power) @ evecs.T


def population_moments(spec: SyntheticSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Sigma_ZZ, Sigma_YZ and Sigma_YY. sample_problem builds the model before it draws samples."""

    problem = sample_problem(replace(spec, n_train=1, n_test=1))
    a = problem.target_operator
    sigma_yy = a @ a.T + spec.sigma_y**2 * np.eye(a.shape[0])
    return problem.population_zz_cov, problem.population_cross_cov, sigma_yy


def wishart_second_moment(sigma: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample second-moment matrix of n i.i.d. N(0, sigma) draws, by the Bartlett decomposition."""

    p = sigma.shape[0]
    bartlett = np.tril(rng.standard_normal((p, p)), -1)
    bartlett[np.diag_indices(p)] = np.sqrt(rng.chisquare(n - np.arange(p)))
    factor = np.linalg.cholesky(sigma) @ bartlett
    return factor @ factor.T / n


def evaluate_moments(
    sigma_zz: np.ndarray,
    sigma_yz: np.ndarray,
    sigma_yy: np.ndarray,
    sigma_zz_hat: np.ndarray,
    sigma_yz_hat: np.ndarray,
    r: int,
) -> dict[str, float]:
    """Theorem quantities for the rank-r ERM fitted on the given sample moments."""

    K, dx = sigma_yz.shape
    zz_inv_half, zz_half = sym_power(sigma_zz, -0.5), sym_power(sigma_zz, 0.5)
    t = sigma_yz @ zz_inv_half
    _, sv, t_vt = np.linalg.svd(t)
    t_hat = sigma_yz_hat @ zz_inv_half
    f_hat = zz_half @ np.linalg.inv(sigma_zz_hat) @ zz_half
    s_hat = t_hat @ sym_power(f_hat, 0.5)
    _, s_hat_sv, s_hat_vt = np.linalg.svd(s_hat)

    # Rank-r ERM on the sample moments, mapped to population-whitened coordinates.
    zz_hat_inv_half = sym_power(sigma_zz_hat, -0.5)
    u, s, vt = np.linalg.svd(sigma_yz_hat @ zz_hat_inv_half, full_matrices=False)
    c_hat = (u[:, :r] * s[:r]) @ vt[:r] @ zz_hat_inv_half @ zz_half

    sigma_top = float(sv[0])
    sigma_next = float(sv[r]) if r < sv.size else 0.0
    oracle_risk = float(np.trace(sigma_yy) - np.sum(sv[:r] ** 2)) / K
    excess_risk = float(np.linalg.norm(c_hat - t, "fro") ** 2 - np.sum(sv[r:] ** 2)) / K
    rho_hat = float(np.linalg.norm(t_hat - t, 2)) / sigma_top
    eps_hat = float(np.linalg.norm(f_hat - np.eye(dx), 2))
    sin_theta = float(np.linalg.norm(t_vt[:r] @ s_hat_vt[r:].T, 2))
    lipschitz = r * sigma_top**2 / K
    return {
        "K": K,
        "rank": r,
        "sigma_r": float(sv[r - 1]),
        "sigma_r_next": sigma_next,
        "sigma_r_hat": float(s_hat_sv[r - 1]),
        "sigma_r_next_hat": float(s_hat_sv[r]) if r < s_hat_sv.size else 0.0,
        "lambda_top": sigma_top**2,
        "lambda_r": float(sv[r - 1]) ** 2,
        "lambda_top_hat": float(s_hat_sv[0]) ** 2,
        "lambda_r_hat": float(s_hat_sv[r - 1]) ** 2,
        "kappa_sq": (float(sv[r - 1]) / sigma_top) ** 2,
        "kappa_sq_hat": (float(s_hat_sv[r - 1]) / float(s_hat_sv[0])) ** 2,
        "relative_gap": (float(sv[r - 1]) - sigma_next) / sigma_top,
        "lipschitz": lipschitz,
        "rho_hat": rho_hat,
        "eps_hat": eps_hat,
        "s_perturbation": float(np.linalg.norm(s_hat - t, 2)) / sigma_top,
        "sin_theta": sin_theta,
        "lipschitz_bound": lipschitz * ((rho_hat + 4 * eps_hat) ** 2 + (sin_theta + eps_hat) ** 2),
        "ols_floor": float(np.trace(sigma_yy) - np.sum(sv**2)) / K,
        "oracle_risk": oracle_risk,
        "excess_risk": excess_risk,
        "population_risk": oracle_risk + excess_risk,
        "operator_error_op": rho_hat * sigma_top,
        "oracle_sigma_top": sigma_top,
        "oracle_effdim": float(np.sum(sv**2)) / sigma_top**2,
    }


def run(experiment: str, config_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    config = load_config(config_path)
    rows: list[dict[str, Any]] = []
    for params in expanded_grid(config):
        spec = SyntheticSpec(**{key: value for key, value in params.items() if key in SPEC_FIELDS})
        target_counts = [int(k) for k in params.get("target_counts", [spec.K])]
        sigma_zz, sigma_yz, sigma_yy = population_moments(replace(spec, K=max(target_counts)))
        dx = sigma_zz.shape[0]
        joint = np.block([[sigma_zz, sigma_yz.T], [sigma_yz, sigma_yy]])
        moments = wishart_second_moment(joint, spec.n_train, np.random.default_rng([spec.seed, spec.n_train]))
        for K in target_counts:
            # The K-target model is the leading block of the largest one.
            assert np.allclose(population_moments(replace(spec, K=K))[1], sigma_yz[:K])
            rows.append({
                "experiment": experiment,
                "seed": spec.seed,
                "n_train": spec.n_train,
                "context_dim": dx,
                **evaluate_moments(
                    sigma_zz,
                    sigma_yz[:K],
                    sigma_yy[:K, :K],
                    moments[:dx, :dx],
                    moments[dx:dx + K, :dx],
                    int(params["d"]),
                ),
            })
    frame = pd.DataFrame(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame
