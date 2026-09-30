"""Experiment 5: post-saturation relative-conditioning diagnostics."""

from pathlib import Path

from experiments.common import ROOT, parse_args, run_from_config


if __name__ == "__main__":
    args = parse_args(
        ROOT / "configs" / "post_saturation.yaml",
        ROOT / "results" / "exp_post_saturation.csv",
    )
    run_from_config("post_saturation", Path(args.config), Path(args.output))
