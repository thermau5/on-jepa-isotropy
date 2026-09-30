"""Experiment 4: bottleneck dimension imposes a spectral-tail floor."""

from pathlib import Path

from experiments.common import ROOT, parse_args, run_from_config


if __name__ == "__main__":
    args = parse_args(
        ROOT / "configs" / "bottleneck.yaml",
        ROOT / "results" / "exp_bottleneck.csv",
    )
    run_from_config("bottleneck", Path(args.config), Path(args.output))

