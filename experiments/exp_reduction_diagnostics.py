"""Direct diagnostics for the JEPA-to-RRR reduction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.synthetic import SyntheticSpec, sample_problem
from experiments.common import ROOT
from metrics.subspace import singular_values
from models.rrr import mse, predict


def population_rrr_coefficient(
    sigma_yz: np.ndarray,
    sigma_zz: np.ndarray,
    rank: int,
    ridge: float = 1e-10,
) -> np.ndarray:
    """Population reduced-rank coefficient from known second moments."""

    regularized = sigma_zz + ridge * np.eye(sigma_zz.shape[0])
    b_ols = sigma_yz @ np.linalg.pinv(regularized)
    evals, evecs = np.linalg.eigh(regularized)
    evals = np.clip(evals, ridge, None)
    sqrt_zz = (evecs * np.sqrt(evals)) @ evecs.T
    inv_sqrt_zz = (evecs * (1.0 / np.sqrt(evals))) @ evecs.T
    whitened = b_ols @ sqrt_zz
    u, s, vt = np.linalg.svd(whitened, full_matrices=False)
    kept = min(rank, s.size)
    truncated = (u[:, :kept] * s[:kept]) @ vt[:kept, :]
    return truncated @ inv_sqrt_zz


def factorize_coefficient(coefficient: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Return W, A with W @ A equal to the rank-r coefficient up to roundoff."""

    u, s, vt = np.linalg.svd(coefficient, full_matrices=False)
    kept = min(rank, s.size)
    w = u[:, :kept] * s[:kept]
    a = vt[:kept, :]
    return w, a


def numeric_rank(matrix: np.ndarray, rel_tol: float = 1e-8) -> int:
    values = singular_values(matrix)
    if values.size == 0:
        return 0
    return int(np.sum(values > rel_tol * max(values[0], 1.0)))


def run(output_path: str | Path = ROOT / "results" / "exp_reduction_diagnostics.csv") -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    ranks = [2, 4, 8, 16, 24]
    target_counts = [2, 4, 8, 16, 24, 32, 48]
    seeds = range(10)
    for seed in seeds:
        for k in target_counts:
            spec = SyntheticSpec(
                n_train=4096,
                n_test=8192,
                context_dim=64,
                target_dim=1,
                r_star=24,
                K=k,
                alpha=1.0,
                overlap=0.0,
                sigma_z=0.12,
                sigma_y=0.12,
                seed=seed,
            )
            problem = sample_problem(spec)
            cross_rank = numeric_rank(problem.population_cross_cov)
            ols_mse = mse(problem.y_test, predict(problem.z_test, problem.population_b))
            for rank in ranks:
                b_rrr = population_rrr_coefficient(
                    problem.population_cross_cov,
                    problem.population_zz_cov,
                    rank=rank,
                )
                w, a = factorize_coefficient(b_rrr, rank=rank)
                b_factored = w @ a
                rrr_mse = mse(problem.y_test, predict(problem.z_test, b_rrr))
                factored_mse = mse(problem.y_test, predict(problem.z_test, b_factored))
                coefficient_norm = max(float(np.linalg.norm(b_rrr, ord="fro")), 1e-12)
                rows.append(
                    {
                        "seed": seed,
                        "K": k,
                        "rank_constraint": rank,
                        "cross_cov_rank": cross_rank,
                        "rank_formula": min(rank, cross_rank),
                        "coefficient_rank": numeric_rank(b_rrr),
                        "factorized_rank": numeric_rank(b_factored),
                        "relative_factorization_error": float(
                            np.linalg.norm(b_factored - b_rrr, ord="fro")
                            / coefficient_norm
                        ),
                        "rrr_mse": rrr_mse,
                        "factored_mse": factored_mse,
                        "ols_mse": ols_mse,
                        "factorized_loss_gap": factored_mse - rrr_mse,
                        "rrr_ols_gap": rrr_mse - ols_mse,
                    }
                )
    frame = pd.DataFrame(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def main() -> None:
    run()


if __name__ == "__main__":
    main()
