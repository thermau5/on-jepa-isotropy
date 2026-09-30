"""Experiment 6: fixed-trace predictive isotropy ablation."""

from pathlib import Path

from experiments.common import ROOT, parse_args, run_from_config


if __name__ == "__main__":
    args = parse_args(
        ROOT / "configs" / "predictive_isotropy.yaml",
        ROOT / "results" / "exp_predictive_isotropy.csv",
    )
    run_from_config("predictive_isotropy", Path(args.config), Path(args.output))
