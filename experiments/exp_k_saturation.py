"""Experiment 1: K-saturation under a fixed bottleneck."""

from pathlib import Path

from experiments.common import ROOT, parse_args, run_from_config


if __name__ == "__main__":
    args = parse_args(
        ROOT / "configs" / "k_saturation.yaml",
        ROOT / "results" / "exp_k_saturation.csv",
    )
    run_from_config("k_saturation", Path(args.config), Path(args.output))

