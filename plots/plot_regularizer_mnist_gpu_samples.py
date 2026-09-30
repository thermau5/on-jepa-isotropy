"""Qualitative reconstructions for the GPU MNIST CNN regularizer experiment."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import ROOT
from experiments.exp_regularizer_mnist_gpu import (
    MNISTGPUJepa,
    covariance_standard_normal_loss,
    fixed_directions,
    load_config,
    load_mnist_halves_gpu,
    projected_gaussianity_loss,
)
from plots.plot_main import GRAY, configure_matplotlib


METHODS = [
    ("baseline",           "No reg."),
    ("encoder_gaussian",   "Enc. reg."),
    ("predictive_gaussian","Pred. reg."),
    ("both",               "Pred.+Enc."),
]


def train_predictions(params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Train one model and return (x_test, y_test, labels_placeholder, preds, test_mse)."""
    seed          = int(params.get("seed", 0))
    n_train       = int(params.get("n_train", 10000))
    channels      = list(params.get("encoder_channels", [64, 128, 256]))
    embedding_dim = int(params.get("embedding_dim", 256))
    predictor_dim = int(params.get("predictor_dim", 256))
    hidden_dim    = int(params.get("hidden_dim", 512))
    use_residual  = bool(params.get("use_residual", False))
    epochs        = int(params.get("epochs", 50))
    batch_size    = int(params.get("batch_size", 256))
    lr            = float(params.get("lr", 3e-4))
    weight_decay  = float(params.get("weight_decay", 1e-4))
    use_amp       = bool(params.get("use_amp", True))
    use_cosine_lr = bool(params.get("use_cosine_lr", True))
    reg_weight    = float(params.get("reg_weight", 0.08))
    cov_weight    = float(params.get("cov_weight", 1.0))
    n_directions  = int(params.get("n_directions", 128))
    method        = str(params.get("method", "baseline"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)

    x_tr, x_te, y_tr, y_te = load_mnist_halves_gpu(seed, n_train)
    x_tr_t = torch.from_numpy(x_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    x_te_t = torch.from_numpy(x_te).to(device)
    y_te_t = torch.from_numpy(y_te).to(device)

    model = MNISTGPUJepa(channels, embedding_dim, predictor_dim, hidden_dim, use_residual).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if use_cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    z_dirs = fixed_directions(embedding_dim, n_directions, seed + 101).to(device)
    p_dirs = fixed_directions(predictor_dim, n_directions, seed + 303).to(device)
    n = x_tr.shape[0]
    gen = torch.Generator().manual_seed(seed + 505)

    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, batch_size):
            b = perm[start : start + batch_size]
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                pred, z, p = model(x_tr_t[b])
                loss = F.mse_loss(pred, y_tr_t[b])
                reg = torch.zeros((), device=device)
                if method in ("encoder_gaussian", "both"):
                    reg = reg + projected_gaussianity_loss(z, z_dirs) + cov_weight * covariance_standard_normal_loss(z)
                if method in ("predictive_gaussian", "both"):
                    reg = reg + projected_gaussianity_loss(p, p_dirs) + cov_weight * covariance_standard_normal_loss(p)
                total = loss + reg_weight * reg
            scaler.scale(total).backward()
            scaler.step(optimizer); scaler.update()
        if use_cosine_lr:
            scheduler.step()

    model.eval()
    preds_list = []
    with torch.no_grad():
        for s in range(0, len(x_te_t), 512):
            pred, _, _ = model(x_te_t[s : s + 512])
            preds_list.append(pred.cpu())
        all_preds = torch.cat(preds_list)
        test_mse = F.mse_loss(all_preds, torch.from_numpy(y_te)).item()

    return x_te, y_te, np.zeros(len(x_te), dtype=int), all_preds.numpy(), test_mse


def combine_halves(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    img = np.zeros((28, 28), dtype=np.float32)
    img[:, :14] = left.reshape(28, 14)
    img[:, 14:] = right.reshape(28, 14)
    return img


def choose_examples(
    y_test: np.ndarray,
    baseline_pred: np.ndarray,
    pred_pred: np.ndarray,
    n_rows: int = 2,
    seed: int = 42,
) -> list[int]:
    """Pick test images where pred reg clearly beats baseline."""
    rng = np.random.default_rng(seed)
    base_err = np.mean((baseline_pred - y_test) ** 2, axis=1)
    pred_err = np.mean((pred_pred - y_test) ** 2, axis=1)
    improvement = base_err - pred_err
    target_energy = np.mean(y_test ** 2, axis=1)
    score = improvement + 0.3 * target_energy
    top_idx = np.argsort(score)[::-1]
    # pick n_rows well-separated by position to get visual variety
    chosen = [int(top_idx[0])]
    for idx in top_idx[1:]:
        if len(chosen) >= n_rows:
            break
        if abs(idx - chosen[-1]) > 50:  # not too close in the test set
            chosen.append(int(idx))
    return chosen


def display_tile(image: np.ndarray) -> np.ndarray:
    return np.repeat((1.0 - np.clip(image, 0.0, 1.0))[:, :, None], 3, axis=2)


def plot_samples(config_path: Path, output_path: Path, seed: int = 0) -> None:
    config = load_config(config_path)
    base = dict(config["base"])

    predictions: dict[str, np.ndarray] = {}
    x_test = y_test = None

    for method, _label in METHODS:
        params = dict(base, method=method, seed=seed)
        x_cur, y_cur, _, pred_cur, mse_cur = train_predictions(params)
        if x_test is None:
            x_test, y_test = x_cur, y_cur
        predictions[method] = pred_cur
        print(f"  {method:25s}  test_mse={mse_cur:.4f}")

    assert x_test is not None and y_test is not None
    chosen = choose_examples(y_test, predictions["baseline"], predictions["predictive_gaussian"])

    configure_matplotlib()
    plt.rcParams.update({"font.weight": "normal", "axes.titleweight": "normal"})

    column_specs = [("Context", None), ("Target", None)] + [(lbl, m) for m, lbl in METHODS]
    fig_w, fig_h = 6.2, 2.35
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = np.empty((len(chosen), len(column_specs)), dtype=object)

    full_h = 0.37
    full_w = full_h * fig_h / fig_w
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
            if method is None and col == 0:
                image = x_test[idx].reshape(28, 14); is_ctx = True
            elif method is None:
                image = combine_halves(x_test[idx], y_test[idx]); is_ctx = False
            else:
                pred_right = np.clip(predictions[method][idx], 0.0, 1.0)
                image = combine_halves(x_test[idx], pred_right); is_ctx = False
                sample_mse = float(np.mean((predictions[method][idx] - y_test[idx]) ** 2))
            ax.imshow(display_tile(image), interpolation="nearest")
            ax.set_aspect("equal", adjustable="box", anchor="W")
            if not is_ctx:
                ax.axvline(13.5, color="#A8A8A8", linewidth=0.45, alpha=0.95)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_linewidth(0.3); spine.set_edgecolor("#D4D4D4")
            if col == 0:
                ax.set_ylabel(f"sample {row + 1}", rotation=0, ha="right", va="center",
                              labelpad=9.0, color=GRAY, fontsize=8.8, fontweight="normal")
            if method is not None:
                ax.text(0.5, -0.075, f"MSE={sample_mse:.3f}", transform=ax.transAxes,
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
    plot_samples(
        ROOT / "configs" / "regularizer_mnist_gpu.yaml",
        ROOT / "figures" / "exp_regularizer_mnist_gpu_samples.pdf",
    )
