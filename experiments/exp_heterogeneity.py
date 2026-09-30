"""Experiment 2: target heterogeneity improves subspace recovery."""

from pathlib import Path

from experiments.common import ROOT, parse_args, run_from_config


if __name__ == "__main__":
    args = parse_args(
        ROOT / "configs" / "heterogeneity.yaml",
        ROOT / "results" / "exp_heterogeneity.csv",
    )
    run_from_config("heterogeneity", Path(args.config), Path(args.output))

