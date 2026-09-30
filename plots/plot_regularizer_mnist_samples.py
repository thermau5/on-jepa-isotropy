"""Qualitative reconstructions for the MNIST regularizer experiment."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import ROOT
from experiments.exp_regularizer_mnist import (
    MNISTJepa,
    MNISTConfig,
    covariance_standard_normal_loss,
    fixed_directions,
    load_config,
    load_mnist_halves,
    projected_gaussianity_loss,
)
from plots.plot_main import GRAY, configure_matplotlib


METHODS = [
    ("baseline", "No reg."),
    ("encoder_gaussian", "Enc. reg."),
    ("predictive_gaussian", "Pred. reg."),
    ("both", "Pred.+Enc."),
]


def train_predictions(
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    cfg = MNISTConfig(**params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    x_train, x_test, y_train, y_test, _labels_train, labels_test = load_mnist_halves(
        cfg.seed, cfg.labeled_fraction
    )
    x_train_t = torch.from_numpy(x_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    x_test_t = torch.from_numpy(x_test).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)

    model = MNISTJepa(
        context_dim=x_train.shape[1],
        target_dim=y_train.shape[1],
        hidden_dim=cfg.hidden_dim,
        embedding_dim=cfg.embedding_dim,
        predictor_dim=cfg.predictor_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    z_dirs = fixed_directions(cfg.embedding_dim, cfg.n_directions, cfg.seed + 101).to(device)
    p_dirs = fixed_directions(cfg.predictor_dim, cfg.n_directions, cfg.seed + 303).to(device)
    n = x_train.shape[0]
    generator = torch.Generator().manual_seed(cfg.seed + 505)

    for _epoch in range(cfg.epochs):
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, cfg.batch_size):
            batch = perm[start : start + cfg.batch_size]
            pred, z, p = model(x_train_t[batch])
            pred_loss = torch.nn.functional.mse_loss(pred, y_train_t[batch])
            reg = torch.zeros((), dtype=pred_loss.dtype, device=device)
            if cfg.method in {"encoder_gaussian", "both"}:
                reg = reg + projected_gaussianity_loss(z, z_dirs)
                reg = reg + cfg.cov_weight * covariance_standard_normal_loss(z)
            if cfg.method in {"predictive_gaussian", "both"}:
                reg = reg + projected_gaussianity_loss(p, p_dirs)
                reg = reg + cfg.cov_weight * covariance_standard_normal_loss(p)
            loss = pred_loss + cfg.reg_weight * reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_pred, _, _ = model(x_test_t)
        test_mse = torch.nn.functional.mse_loss(test_pred, y_test_t).item()
    return x_test, y_test, labels_test, test_pred.cpu().numpy(), test_mse


def combine_halves(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    image = np.zeros((28, 28), dtype=np.float32)
    image[:, :14] = left.reshape(28, 14)
    image[:, 14:] = right.reshape(28, 14)
    return image


def choose_examples(
    y_test: np.ndarray,
    labels: np.ndarray,
    baseline_pred: np.ndarray,
    predictive_pred: np.ndarray,
    digit_order: tuple[int, ...] = (4, 8),
) -> list[int]:
    baseline_error = np.mean((baseline_pred - y_test) ** 2, axis=1)
    predictive_error = np.mean((predictive_pred - y_test) ** 2, axis=1)
    improvement = baseline_error - predictive_error
    target_energy = np.mean(y_test**2, axis=1)
    chosen: list[int] = []
    for digit in digit_order:
        candidates = np.flatnonzero(labels == digit)
        if len(candidates) == 0:
            continue
        score = improvement[candidates] + 0.25 * target_energy[candidates]
        best = candidates[np.argmax(score)]
        chosen.append(int(best))
    return chosen


def display_tile(image: np.ndarray) -> np.ndarray:
    gray = 1.0 - np.clip(image, 0.0, 1.0)
    return np.repeat(gray[:, :, None], 3, axis=2)


def plot_samples(config_path: Path, output_path: Path, seed: int = 0) -> None:
    config = load_config(config_path)
    base = dict(config["base"])
    predictions: dict[str, np.ndarray] = {}
    mse: dict[str, float] = {}
    x_test = y_test = labels = None

    for method, _label in METHODS:
        params = dict(base)
        params.update({"method": method, "seed": seed})
        x_curr, y_curr, label_curr, pred_curr, mse_curr = train_predictions(params)
        if x_test is None:
            x_test, y_test, labels = x_curr, y_curr, label_curr
        predictions[method] = pred_curr
        mse[method] = mse_curr

    assert x_test is not None and y_test is not None and labels is not None
    chosen = choose_examples(
        y_test,
        labels,
        predictions["baseline"],
        predictions["predictive_gaussian"],
    )

    configure_matplotlib()
    plt.rcParams.update(
        {
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
        }
    )
    column_specs = [("Context", None), ("Target", None)] + [
        (label, method) for method, label in METHODS
    ]
    fig_w, fig_h = 6.2, 2.35
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = np.empty((len(chosen), len(column_specs)), dtype=object)
    full_h = 0.37
    full_w = full_h * fig_h / fig_w
    context_w = 0.5 * full_w
    x_start = 0.095
    y_positions = [0.535, 0.085]
    gap = 0.022
    task_gap = 0.044
    x_positions: list[float] = []
    widths: list[float] = []
    x = x_start
    for col in range(len(column_specs)):
        width = context_w if col == 0 else full_w
        x_positions.append(x)
        widths.append(width)
        x += width + (task_gap if col == 1 else gap)

    for col, (column_label, _method) in enumerate(column_specs):
        fig.text(
            x_positions[col] + 0.5 * widths[col],
            0.94,
            column_label,
            ha="center",
            va="bottom",
            fontsize=9.0,
            fontweight="normal",
            color="black",
        )

    for row, index in enumerate(chosen):
        for col, (_column_label, method) in enumerate(column_specs):
            ax = fig.add_axes([x_positions[col], y_positions[row], widths[col], full_h])
            axes[row, col] = ax
            if method is None and col == 0:
                image = x_test[index].reshape(28, 14)
                is_context = True
            elif method is None:
                image = combine_halves(x_test[index], y_test[index])
                is_context = False
            else:
                pred_right = np.clip(predictions[method][index], 0.0, 1.0)
                image = combine_halves(x_test[index], pred_right)
                is_context = False
                sample_mse = float(np.mean((predictions[method][index] - y_test[index]) ** 2))
            ax.imshow(display_tile(image), interpolation="nearest")
            ax.set_aspect("equal", adjustable="box", anchor="W")
            if not is_context:
                ax.axvline(13.5, color="#A8A8A8", linewidth=0.45, alpha=0.95)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.3)
                spine.set_edgecolor("#D4D4D4")
            if col == 0:
                ax.set_ylabel(
                    f"sample {int(labels[index])}",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=9.0,
                    color=GRAY,
                    fontsize=8.8,
                    fontweight="normal",
                )
            if method is not None:
                ax.text(
                    0.5,
                    -0.075,
                    f"MSE={sample_mse:.3f}",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=7.0,
                    fontweight="normal",
                    color=GRAY,
                    clip_on=False,
                )

    separator_x = 0.5 * (axes[0, 1].get_position().x1 + axes[0, 2].get_position().x0)
    separator_y0 = axes[-1, 0].get_position().y0
    separator_y1 = axes[0, 0].get_position().y1
    separator = plt.Line2D(
        [separator_x, separator_x],
        [separator_y0, separator_y1],
        transform=fig.transFigure,
        color="#777777",
        linewidth=0.85,
        alpha=0.9,
    )
    fig.add_artist(separator)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    plot_samples(
        ROOT / "configs" / "regularizer_mnist.yaml",
        ROOT / "figures" / "exp_regularizer_mnist_samples.pdf",
    )
