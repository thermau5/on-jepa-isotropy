"""Analyze effective-rank threshold transitions in population spectra."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import RANK_REL_TOL, ROOT


def transition_table(frame: pd.DataFrame, x_col: str) -> pd.DataFrame:
    rows = []
    grouped = frame.groupby(x_col)
    summary = grouped[
        [
            "oracle_effective_rank",
            "oracle_lambda_at_population_rank",
            "oracle_sigma_at_population_rank",
        ]
    ].mean()

    previous_x = None
    previous = None
    for x, row in summary.iterrows():
        if (
            previous is not None
            and row["oracle_effective_rank"] != previous["oracle_effective_rank"]
        ):
            rows.append(
                {
                    "from_x": previous_x,
                    "to_x": x,
                    "from_effective_rank": previous["oracle_effective_rank"],
                    "to_effective_rank": row["oracle_effective_rank"],
                    "from_lambda_rank_g": previous["oracle_lambda_at_population_rank"],
                    "to_lambda_rank_g": row["oracle_lambda_at_population_rank"],
                    "lambda_ratio": row["oracle_lambda_at_population_rank"]
                    / max(previous["oracle_lambda_at_population_rank"], 1e-12),
                    "from_sigma_rank_g": previous["oracle_sigma_at_population_rank"],
                    "to_sigma_rank_g": row["oracle_sigma_at_population_rank"],
                    "sigma_ratio": row["oracle_sigma_at_population_rank"]
                    / max(previous["oracle_sigma_at_population_rank"], 1e-12),
                }
            )
        previous_x = x
        previous = row
    return pd.DataFrame(rows)


def main() -> None:
    results = pd.read_csv(ROOT / "results" / "results.csv")
    output_dir = ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    for experiment, x_col in [
        ("heterogeneity", "alpha"),
    ]:
        frame = results[results["experiment"] == experiment]
        table = transition_table(frame, x_col)
        output = output_dir / f"{experiment}_threshold_transitions.csv"
        table.to_csv(output, index=False)
        print(f"\n{experiment} theory threshold transitions (tau={RANK_REL_TOL}):")
        if table.empty:
            print("  no oracle rank transitions detected")
        else:
            print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
