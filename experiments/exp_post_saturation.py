"""Experiment 5: post-saturation relative-conditioning diagnostics."""

from pathlib import Path

from experiments.common import ROOT, parse_args
from experiments.moment_bounds import run


if __name__ == "__main__":
    args = parse_args(
        ROOT / "configs" / "post_saturation.yaml",
        ROOT / "results" / "exp_post_saturation.csv",
    )
    run("post_saturation", Path(args.config), Path(args.output))
