"""Publication figures for the JEPA regularizer experiment (Table 1), one figure per task.

Panels (b, c) measure the prediction and panels (d, e) the encoder embedding, so each figure shows that
encoder-side regularization evens out the embedding while prediction-side regularization evens out the prediction.
"""

from __future__ import annotations

from pathlib import Path
import sys

from matplotlib.patches import Patch
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


TASKS = ["digits", "mnist", "cifar10", "cifar100"]
ORDER = ["baseline", "encoder_gaussian", "predictive_gaussian", "both"]
LABELS = ["No reg.", "Encoder", "Predictive", "Pred. + Enc."]
YLABEL_BY_METRIC = {
    "latent_test_mse": "Latent test risk",
    "predictive_gaussianity_score": "Prediction\nGaussianity distance",
    "predictive_effective_rank": "Predictive effective rank",
    "embedding_effective_rank": "Embedding effective rank",
    "whitened_embedding_gaussianity_score": "Whitened embedding\nGaussianity distance",
}
# Gaussianity distances are plotted in units of 1e-3, as the captions state.
SCALE_BY_METRIC = {"predictive_gaussianity_score": 1e3, "whitened_embedding_gaussianity_score": 1e3}


def table1_runs(sweep: pd.DataFrame) -> pd.DataFrame:
    """Keep the baseline and, for each regularized objective, the weight with the lowest mean validation latent risk."""
    runs = [sweep[sweep["method"] == "baseline"]]
    for method in ORDER[1:]:
        sub = sweep[sweep["method"] == method]
        weight = sub.groupby("reg_weight")["latent_val_mse"].mean().idxmin()
        runs.append(sub[sub["reg_weight"] == weight])
    return pd.concat(runs, ignore_index=True)


def make_figure(task: str, output_path: Path) -> None:
    frame = table1_runs(pd.read_csv(RESULTS / f"exp_jepa_{task}_cca.csv"))
    shape = pd.read_csv(RESULTS / "exp_jepa_shape.csv")
    shape = shape[shape["task"] == task][["method", "seed", "whitened_embedding_gaussianity_score"]]
    frame = frame.merge(shape, on=["method", "seed"], how="left", validate="one_to_one")
    for metric, scale in SCALE_BY_METRIC.items():
        frame[metric] = frame[metric] * scale
    metrics = list(YLABEL_BY_METRIC)
    summary = frame.groupby("method")[metrics].agg(["mean", "sem"]).reindex(ORDER)
    x = np.arange(len(ORDER))
    fig, axes = make_bar_figure(ncols=len(metrics), figsize=(7.1, 2.3))
    colors = [GRAY, TEAL, BLUE, RED]
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
        style_axis(ax, "", YLABEL_BY_METRIC[metric], panel)
    handles = [Patch(facecolor=color, edgecolor="white", label=label) for color, label in zip(colors, LABELS)]
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
    for task in TASKS:
        make_figure(task, FIGURES / f"exp_jepa_{task}_diagnostics.pdf")
