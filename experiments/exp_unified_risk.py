"""Run the independent Theorem 3.15 unified-risk diagnostic."""

from __future__ import annotations

from pathlib import Path

from experiments.common import ROOT, parse_args
from experiments.moment_bounds import run


def main() -> None:
    args = parse_args(
        ROOT / "configs" / "unified_risk.yaml",
        ROOT / "results" / "exp_unified_risk.csv",
    )
    run("unified_risk", Path(args.config), Path(args.output))


if __name__ == "__main__":
    main()
