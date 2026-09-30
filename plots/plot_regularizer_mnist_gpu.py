"""Publication figure for the GPU MNIST CNN regularizer experiment."""

from __future__ import annotations

from pathlib import Path
import sys

from matplotlib.patches import Patch
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import ROOT
from plots.plot_main import (
    BLUE,
    FIGURES,
    GRAY,
    RED,
    RESULTS,
    TEAL,
    make_figure as make_bar_figure,
    save_figure,
    style_axis,
)


CSV_PATH = RESULTS / "exp_regularizer_mnist_gpu.csv"
OUTPUT_PATH = FIGURES / "exp_regularizer_mnist_gpu.pdf"


def make_figure(csv_path: Path = CSV_PATH, output_path: Path = OUTPUT_PATH) -> None:
    frame = pd.read_csv(csv_path).copy()
    order = ["baseline", "encoder_gaussian", "predictive_gaussian", "both"]
    labels = ["No reg.", "Encoder", "Predictive", "Pred. + Enc."]
    metrics = [
        "test_mse",
        "fg_mse",
        "gap",
        "predictive_gaussianity_score",
        "predictive_effective_rank",
    ]
    summary = frame.groupby("method")[metrics].agg(["mean", "sem"]).reindex(order)
    x = np.arange(len(order))
    fig, axes = make_bar_figure(ncols=len(metrics), figsize=(7.1, 2.3))
    colors = [GRAY, TEAL, BLUE, RED]
    ylabel_by_metric = {
        "test_mse": "Test pixel MSE",
        "fg_mse": "Foreground pixel MSE",
        "gap": "Train-test MSE gap",
        "predictive_gaussianity_score": "Gaussianity distance",
        "predictive_effective_rank": "Predictive effective rank",
    }
    panels = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    for ax, metric, panel in zip(axes, metrics, panels):
        means = summary[(metric, "mean")].to_numpy()
        sems = summary[(metric, "sem")].fillna(0.0).to_numpy()
        ax.bar(
            x,
            means,
            yerr=sems,
            color=colors,
            edgecolor="white",
            linewidth=0.6,
            capsize=2.0,
            width=0.62,
        )
        ax.set_xticks([])
        style_axis(ax, "", ylabel_by_metric[metric], panel)
        if metric == "predictive_gaussianity_score":
            ax.set_yscale("log")
    handles = [Patch(facecolor=color, edgecolor="white", label=label) for color, label in zip(colors, labels)]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=4,
        frameon=False,
        handlelength=1.0,
        columnspacing=1.1,
        handletextpad=0.35,
    )
    save_figure(fig, output_path)


if __name__ == "__main__":
    make_figure()
