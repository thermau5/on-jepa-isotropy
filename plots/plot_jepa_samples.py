"""Qualitative completions for the latent-target JEPA experiment (Table 1), one figure per task.

Each objective trains at seed 0 with its Table 1 weight (the lowest mean validation latent risk), and the pixel
head renders its predictions of the right half. Rows are the top two of visual_candidates, except the Figure 2
rows picked by eye from its top, and cell labels give each completion's target-half pixel MSE, as in Table 1. The
experiment scripts are not changed: the predictions, teacher targets and pixel outputs are captured where
train_one hands them to its metric functions, and the 8x8 pixel head where train_one builds it.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import exp_jepa_digits as digits
from experiments import exp_jepa_images as images
from experiments import exp_regularizer_cifar_gpu as cifar
from experiments import exp_regularizer_mnist_gpu as mnist
from experiments.common import ROOT
from experiments.exp_regularizer_digits import load_digit_halves
from plots.plot_main import FIGURES, GRAY, RESULTS, configure_matplotlib


TASKS = ["digits", "mnist", "cifar10", "cifar100"]
METHODS = [
    ("baseline",           "No reg."),
    ("encoder_gaussian",   "Enc. reg."),
    ("predictive_gaussian","Pred. reg."),
    ("both",               "Pred.+Enc."),
]
# (channels, height, width) of each image half.
HALF_SHAPES = {"digits": (1, 8, 4), "mnist": (1, 28, 14), "cifar10": (3, 32, 16), "cifar100": (3, 32, 16)}
# Figure 2 rows, picked by eye for a visible difference among the test images where prediction-side has the lowest
# latent risk and pixel MSE, ranked by its margin in stroke overlap (MNIST, pixels above 0.5) and in 4x4-pooled
# color error (CIFAR-100).
FIGURE2_ROWS = {"mnist": [1752, 7532], "cifar100": [8808, 6975]}

captured: dict[str, np.ndarray] = {}
built: list[torch.nn.Module] = []


def capture_latent(fn):
    def wrapped(teacher, p_tr, t_tr, p_val, t_val, p_te, t_te):
        captured.update(p_te=np.asarray(p_te), t_te=np.asarray(t_te))
        return fn(teacher, p_tr, t_tr, p_val, t_val, p_te, t_te)
    return wrapped


def capture_pixels(fn):
    def wrapped(preds, targets, z_np, p_np, z_dirs, p_dirs):
        captured["pixels"] = np.asarray(preds)
        return fn(preds, targets, z_np, p_np, z_dirs, p_dirs)
    return wrapped


def record_mlp(fn):
    def wrapped(*args):
        built.append(fn(*args))
        return built[-1]
    return wrapped


digits.latent_metrics = capture_latent(digits.latent_metrics)
images.latent_metrics = capture_latent(images.latent_metrics)
mnist.compute_all_metrics = capture_pixels(mnist.compute_all_metrics)
cifar.compute_all_metrics = capture_pixels(cifar.compute_all_metrics)
# train_one builds the 8x8 encoder, predictor and pixel head in that order.
digits.mlp = record_mlp(digits.mlp)


def table1_weights(task: str) -> dict[str, float]:
    sweep = pd.read_csv(RESULTS / f"exp_jepa_{task}_cca.csv")
    return {m: float(sweep[sweep["method"] == m].groupby("reg_weight")["latent_val_mse"].mean().idxmin())
            for m, _ in METHODS[1:]}


def train_task(task: str, seed: int = 0):
    """Return (x_test, y_test, teacher targets, latent predictions, pixel completions) for the four objectives."""
    base = yaml.safe_load(open(ROOT / "configs" / f"jepa_{task}_cca.yaml"))["base"]
    weights = table1_weights(task)
    latent, pixels = {}, {}
    for method, _ in METHODS:
        params = {**base, "method": method, "seed": seed}
        if task != "digits":
            params["dataset"] = task
        if method != "baseline":
            params.update(predictive_scale="unit", reg_weight=weights[method])
        captured.clear()
        built.clear()
        row = (digits if task == "digits" else images).train_one(params)
        latent[method] = captured["p_te"]
        if task == "digits":
            with torch.no_grad():
                pixels[method] = built[2](torch.from_numpy(captured["p_te"])).numpy()
        else:
            pixels[method] = captured["pixels"]
        print(f"  {method:20s}  latent_test_mse={row['latent_test_mse']:.4f}", flush=True)
    if task == "digits":
        _, x_held, _, y_held = load_digit_halves(seed, float(base["test_size"]))
        _, x_test, _, y_test = train_test_split(x_held, y_held, train_size=int(base["n_val"]), random_state=seed)
    else:
        _, x_test, _, y_test, _, _ = images.load(task, seed, int(base["n_train"]), int(base["n_val"]))
    return x_test, y_test, captured["t_te"], latent, pixels


def visual_candidates(risk: dict[str, np.ndarray], px: dict[str, np.ndarray]) -> np.ndarray:
    """Test indices where prediction-side regularization has both the lowest latent risk and the lowest pixel MSE,
    ranked by its pixel-MSE margin over the best other objective."""
    others = [m for m, _ in METHODS if m != "predictive_gaussian"]
    best = (np.all([risk["predictive_gaussian"] < risk[m] for m in others], axis=0)
            & np.all([px["predictive_gaussian"] < px[m] for m in others], axis=0))
    margin = np.min([px[m] for m in others], axis=0) - px["predictive_gaussian"]
    return np.argsort(-np.where(best, margin, -np.inf))[: int(best.sum())]


def to_image(half: np.ndarray, task: str) -> np.ndarray:
    """Flat image half -> (H, W, 3) display array, with grayscale digits drawn dark on light."""
    c, h, w = HALF_SHAPES[task]
    image = np.clip(half.reshape(c, h, w).transpose(1, 2, 0), 0.0, 1.0)
    return np.repeat(1.0 - image, 3, axis=2) if c == 1 else image


def plot_samples(task: str, output_path: Path, seed: int = 0) -> None:
    x_test, y_test, targets, latent, pixels = train_task(task, seed)
    risk = {m: np.mean((latent[m] - targets) ** 2, axis=1) for m, _ in METHODS}
    # Pixel MSE as in Table 1: completions clipped to [0, 1] for MNIST and CIFAR, raw for 8x8 digits.
    clip = (lambda a: a) if task == "digits" else (lambda a: np.clip(a, 0.0, 1.0))
    pixel_mse = {m: np.mean((clip(pixels[m]) - y_test) ** 2, axis=1) for m, _ in METHODS}
    chosen = FIGURE2_ROWS.get(task) or [int(i) for i in visual_candidates(risk, pixel_mse)[:2]]
    for row, idx in enumerate(chosen):
        print(f"  sample {row + 1} (test idx {idx}): "
              + "  ".join(f"{lbl}={pixel_mse[m][idx]:.4f} (risk {risk[m][idx]:.3f})" for m, lbl in METHODS))

    configure_matplotlib()
    plt.rcParams.update({"font.weight": "normal", "axes.titleweight": "normal"})

    column_specs = [("Context", None), ("Target", None)] + [(lbl, m) for m, lbl in METHODS]
    fig_w, fig_h = 6.2, 2.35
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = np.empty((len(chosen), len(column_specs)), dtype=object)

    _, half_h, half_w = HALF_SHAPES[task]
    full_h = 0.37
    full_w = full_h * fig_h / fig_w * (2 * half_w) / half_h
    context_w = 0.5 * full_w
    x_start, gap, task_gap = 0.095, 0.022, 0.044
    y_positions = [0.535, 0.085]

    x_positions, widths = [], []
    x = x_start
    for col in range(len(column_specs)):
        w = context_w if col == 0 else full_w
        x_positions.append(x); widths.append(w)
        x += w + (task_gap if col == 1 else gap)

    for col, (lbl, _) in enumerate(column_specs):
        fig.text(x_positions[col] + 0.5 * widths[col], 0.94, lbl,
                 ha="center", va="bottom", fontsize=9.0, fontweight="normal")

    for row, idx in enumerate(chosen):
        for col, (_, method) in enumerate(column_specs):
            ax = fig.add_axes([x_positions[col], y_positions[row], widths[col], full_h])
            axes[row, col] = ax
            if col == 0:
                image = to_image(x_test[idx], task)
            else:
                right = y_test[idx] if method is None else pixels[method][idx]
                image = np.concatenate([to_image(x_test[idx], task), to_image(right, task)], axis=1)
            ax.imshow(image, interpolation="nearest")
            ax.set_aspect("equal", adjustable="box", anchor="W")
            if col > 0:
                ax.axvline(half_w - 0.5, color="#BDBDBD", linewidth=0.45, alpha=0.95)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_linewidth(0.3); spine.set_edgecolor("#D4D4D4")
            if col == 0:
                ax.set_ylabel(f"sample {row + 1}", rotation=0, ha="right", va="center",
                              labelpad=9.0, color=GRAY, fontsize=8.8, fontweight="normal")
            if method is not None:
                ax.text(0.5, -0.075, f"MSE={pixel_mse[method][idx]:.4f}", transform=ax.transAxes,
                        ha="center", va="top", fontsize=7.0, fontweight="normal",
                        color=GRAY, clip_on=False)

    sep_x = 0.5 * (axes[0, 1].get_position().x1 + axes[0, 2].get_position().x0)
    sep_y0 = axes[-1, 0].get_position().y0
    sep_y1 = axes[0, 0].get_position().y1
    fig.add_artist(plt.Line2D([sep_x, sep_x], [sep_y0, sep_y1],
                              transform=fig.transFigure, color="#777777",
                              linewidth=0.85, alpha=0.9))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    for task in sys.argv[1:] or TASKS:
        print(f"=== {task} samples ===", flush=True)
        plot_samples(task, FIGURES / f"exp_jepa_{task}_samples.pdf")
