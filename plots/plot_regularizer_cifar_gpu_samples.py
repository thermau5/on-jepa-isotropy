"""Qualitative RGB completions for the GPU CIFAR-10/100 regularizer experiment."""

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
from experiments.exp_regularizer_cifar_gpu import (
    IMG_C,
    IMG_H,
    IMG_W,
    CIFARGPUJepa,
    covariance_standard_normal_loss,
    fixed_directions,
    load_cifar_halves_gpu,
    load_config,
    projected_gaussianity_loss,
)
from plots.plot_main import GRAY, configure_matplotlib


METHODS = [
    ("baseline",           "No reg."),
    ("encoder_gaussian",   "Enc. reg."),
    ("predictive_gaussian","Pred. reg."),
    ("both",               "Pred.+Enc."),
]


def train_predictions(params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Train one model and return (x_test, y_test, preds, test_mse)."""
    dataset       = str(params.get("dataset", "cifar10"))
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

    x_tr, x_te, y_tr, y_te = load_cifar_halves_gpu(dataset, seed, n_train)
    x_tr_t = torch.from_numpy(x_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    x_te_t = torch.from_numpy(x_te).to(device)

    model = CIFARGPUJepa(channels, embedding_dim, predictor_dim, hidden_dim, use_residual).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if use_cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

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
            with torch.amp.autocast(device.type, enabled=amp_enabled):
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

    return x_te, y_te, all_preds.numpy(), test_mse


def half_to_rgb(half: np.ndarray) -> np.ndarray:
    """Flat (1536,) channel-first half -> (32, 16, 3) display image."""
    return np.clip(half.reshape(IMG_C, IMG_H, IMG_W).transpose(1, 2, 0), 0.0, 1.0)


def combine_halves(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    img = np.zeros((IMG_H, 2 * IMG_W, 3), dtype=np.float32)
    img[:, :IMG_W] = half_to_rgb(left)
    img[:, IMG_W:] = half_to_rgb(right)
    return img


# Structural tolerances the chosen samples must satisfy, relative to the
# per-method mean over the test set. Tiers relax in order until enough samples
# qualify: (max |baseline MSE / mean baseline MSE - 1|, max relative encoder
# deviation from baseline, max relative combined deviation from predictive).
SELECTION_TIERS = [
    (0.20, 0.10, 0.20),
    (0.35, 0.20, 0.35),
    (0.60, 0.40, 0.60),
]


def choose_examples(
    y_test: np.ndarray,
    predictions: dict[str, np.ndarray],
    n_rows: int = 2,
) -> list[int]:
    """Pick test images of typical difficulty that show the aggregate pattern.

    Two failure modes bracket this choice. Ranking purely by improvement picks
    atypically hard images whose printed MSEs are several times the means in
    the results table. Ranking purely by closeness to the mean profile picks
    images where all four methods round to the same value, hiding the effect
    entirely. We therefore constrain the sample to reproduce the aggregate
    *structure* -- baseline difficulty near the dataset mean, encoder close to
    baseline, combined close to predictive -- and within that pool prefer the
    clearest predictive-versus-encoder margin, so the completions are visually
    comparable. The margin of a single image is necessarily larger than the
    dataset-averaged margin reported in the table; the table remains the
    quantitative claim, and these panels show what the difference looks like.
    """
    methods = [m for m, _ in METHODS]
    per_sample = np.stack(
        [np.mean((predictions[m] - y_test) ** 2, axis=1) for m in methods],
        axis=1,
    )  # (N, n_methods)
    mean_profile = per_sample.mean(axis=0)
    base, enc, pred, both = (per_sample[:, i] for i in range(4))

    level_error = np.abs(base / mean_profile[0] - 1.0)
    encoder_error = np.abs(enc - base) / base
    combined_error = np.abs(both - pred) / np.maximum(pred, 1e-12)
    margin = (enc - pred) / np.maximum(enc, 1e-12)

    eligible = np.zeros(len(per_sample), dtype=bool)
    for level_tol, encoder_tol, combined_tol in SELECTION_TIERS:
        eligible = (
            (level_error <= level_tol)
            & (encoder_error <= encoder_tol)
            & (combined_error <= combined_tol)
            & (pred < enc)
            & (pred < base)
        )
        if eligible.sum() >= n_rows:
            break

    ranked = [int(i) for i in np.argsort(np.where(eligible, -margin, 0.0)) if eligible[i]]
    if not ranked:  # nothing satisfies the structure; fall back to plain margin
        ranked = [int(i) for i in np.argsort(-margin)]

    chosen = ranked[:1]
    for idx in ranked[1:]:
        if len(chosen) >= n_rows:
            break
        if all(abs(idx - c) > 50 for c in chosen):
            chosen.append(idx)
    return chosen


def plot_samples(config_path: Path, output_path: Path, seed: int = 0) -> None:
    config = load_config(config_path)
    base = dict(config["base"])

    predictions: dict[str, np.ndarray] = {}
    x_test = y_test = None

    for method, _label in METHODS:
        params = dict(base, method=method, seed=seed)
        x_cur, y_cur, pred_cur, mse_cur = train_predictions(params)
        if x_test is None:
            x_test, y_test = x_cur, y_cur
        predictions[method] = pred_cur
        print(f"  {method:25s}  test_mse={mse_cur:.4f}")

    assert x_test is not None and y_test is not None
    chosen = choose_examples(y_test, predictions)
    for row, idx in enumerate(chosen):
        per_method = "  ".join(
            f"{lbl}={np.mean((predictions[m][idx] - y_test[idx]) ** 2):.3f}"
            for m, lbl in METHODS
        )
        print(f"  sample {row + 1} (test idx {idx}): {per_method}")

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
                image = half_to_rgb(x_test[idx]); is_ctx = True
            elif method is None:
                image = combine_halves(x_test[idx], y_test[idx]); is_ctx = False
            else:
                image = combine_halves(x_test[idx], predictions[method][idx]); is_ctx = False
                sample_mse = float(np.mean((predictions[method][idx] - y_test[idx]) ** 2))
            ax.imshow(image, interpolation="nearest")
            ax.set_aspect("equal", adjustable="box", anchor="W")
            if not is_ctx:
                ax.axvline(IMG_W - 0.5, color="#F2F2F2", linewidth=0.45, alpha=0.95)
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
    for dataset in ["cifar10", "cifar100"]:
        config_path = ROOT / "configs" / f"regularizer_{dataset}.yaml"
        print(f"=== {dataset} samples ===")
        plot_samples(
            config_path,
            ROOT / "figures" / f"exp_regularizer_{dataset}_samples.pdf",
        )
