"""Run all Phase 1 synthetic experiments and combine their CSV outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.common import ROOT, run_from_config
from experiments.exp_gauge_factorization import run as run_gauge_factorization
from experiments.exp_reduction_diagnostics import run as run_reduction_diagnostics
from experiments.exp_regularizer_digits import run as run_regularizer_digits


EXPERIMENTS = [
    ("k_saturation", "k_saturation.yaml", "exp_k_saturation.csv"),
    ("heterogeneity", "heterogeneity.yaml", "exp_heterogeneity.csv"),
    ("bottleneck", "bottleneck.yaml", "exp_bottleneck.csv"),
    ("unified_risk", "unified_risk.yaml", "exp_unified_risk.csv"),
    ("post_saturation", "post_saturation.yaml", "exp_post_saturation.csv"),
    ("predictive_isotropy", "predictive_isotropy.yaml", "exp_predictive_isotropy.csv"),
    ("gauge_factorization", "gauge_factorization.yaml", "exp_gauge_factorization.csv"),
    ("regularizer_digits", "regularizer_digits.yaml", "exp_regularizer_digits.csv"),
]


def main() -> None:
    frames = []
    for name, config_name, output_name in EXPERIMENTS:
        if name == "gauge_factorization":
            frame = run_gauge_factorization(
                ROOT / "configs" / config_name,
                ROOT / "results" / output_name,
            )
        elif name == "regularizer_digits":
            frame = run_regularizer_digits(
                ROOT / "configs" / config_name,
                ROOT / "results" / output_name,
            )
        else:
            frame = run_from_config(
                name,
                ROOT / "configs" / config_name,
                ROOT / "results" / output_name,
            )
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    output = ROOT / "results" / "results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)

    run_reduction_diagnostics(ROOT / "results" / "exp_reduction_diagnostics.csv")


if __name__ == "__main__":
    main()
