"""Latent-target JEPA on MNIST, CIFAR-10 and CIFAR-100 with a frozen teacher.

Same encoder, predictor and decoder classes and the same training recipe as the pixel experiments
(exp_regularizer_mnist_gpu.py, exp_regularizer_cifar_gpu.py), but trained as a JEPA: the context
encoder sees the left half, and the predictor outputs the predicted teacher embedding of the right
half, trained with the latent MSE. The teacher is frozen PCA-whitening of the right half onto its top
`embedding_dim` components (see exp_jepa_digits.py). The predictor output is the prediction itself,
so the prediction-side penalty acts on exactly the intrinsic predicted output of Eq. 4.2. The
convolutional decoder is a separate pixel head trained on detached predictions; it reports pixel
metrics and never shapes the JEPA. Validation images, for choosing strengths, come from the training
images outside the training split.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from experiments.common import ROOT, expanded_grid, parse_args
from experiments.exp_jepa_digits import PREDICTIVE_SCALES, CCATeacher, PCAWhiteningTeacher, latent_metrics, trace_normalize
from experiments.exp_regularizer_digits import intrinsic_output
from experiments.exp_regularizer_mnist_gpu import (
    CNNDecoder,
    CNNEncoder,
    MLPPredictor,
    covariance_standard_normal_loss,
    fixed_directions,
    load_mnist_halves_gpu,
    projected_gaussianity_loss,
)
from experiments import exp_regularizer_cifar_gpu as cifar
from experiments import exp_regularizer_mnist_gpu as mnist


METHODS = ("baseline", "encoder_gaussian", "predictive_gaussian", "both")
SHAPES = {"mnist": (1, 28, 14), "cifar10": (3, 32, 16), "cifar100": (3, 32, 16)}


def build(dataset: str, channels: list[int], d: int, hidden: int, k: int, predictor: str) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    if dataset == "mnist":
        encoder, pixel_head = CNNEncoder(channels, d, hidden), CNNDecoder(k, channels)
    else:
        encoder, pixel_head = cifar.CIFAREncoder(channels, d, hidden), cifar.CIFARDecoder(k, channels)
    return encoder, (MLPPredictor(d, k, hidden) if predictor == "mlp" else torch.nn.Linear(d, k)), pixel_head


def load(dataset: str, seed: int, n_train: int, n_val: int) -> tuple[np.ndarray, ...]:
    if dataset == "mnist":
        return load_mnist_halves_gpu(seed, n_train, n_val)
    return cifar.load_cifar_halves_gpu(dataset, seed, n_train, n_val)


def train_one(params: dict[str, Any]) -> dict[str, Any]:
    dataset = str(params["dataset"])
    seed = int(params.get("seed", 0))
    n_train = int(params.get("n_train", 10000))
    n_val = int(params.get("n_val", 5000))
    channels = list(params.get("encoder_channels", [64, 128, 256]))
    d = int(params.get("embedding_dim", 64))
    # Teacher (prediction) dimension k. With k > d and a linear predictor W, the prediction W z + b has
    # rank d, and the prediction-side penalty acts on its intrinsic output R z (W = QR), as in Eq. 4.2.
    k = int(params.get("teacher_dim", d))
    predictor_type = str(params.get("predictor", "mlp"))
    # "pca": PCA-whitening of the right half; "cca": the predictively isotropic CCA teacher.
    teacher_type = str(params.get("teacher", "pca"))
    hidden = int(params.get("hidden_dim", 512))
    epochs = int(params.get("epochs", 50))
    batch_size = int(params.get("batch_size", 256))
    lr = float(params.get("lr", 3e-4))
    # Weight decay of the encoder and predictor; the pixel head keeps its own.
    weight_decay = float(params.get("weight_decay", 1e-4))
    pixel_weight_decay = float(params.get("pixel_weight_decay", 1e-4))
    use_amp = bool(params.get("use_amp", True))
    reg_weight = float(params.get("reg_weight", 0.08))
    cov_weight = float(params.get("cov_weight", 1.0))
    n_directions = int(params.get("n_directions", 128))
    method = str(params.get("method", "baseline"))
    predictive_scale = str(params.get("predictive_scale", "unit"))
    if predictor_type not in ("mlp", "linear") or teacher_type not in ("pca", "cca"):
        raise ValueError(f"unknown predictor {predictor_type!r} or teacher {teacher_type!r}")
    if dataset not in SHAPES or method not in METHODS or predictive_scale not in PREDICTIVE_SCALES:
        raise ValueError(f"unknown dataset {dataset!r}, method {method!r} or predictive_scale {predictive_scale!r}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = use_amp and device.type == "cuda"
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_tr, x_te, y_tr, y_te, x_va, y_va = load(dataset, seed, n_train, n_val)
    teacher = PCAWhiteningTeacher(y_tr, k) if teacher_type == "pca" else CCATeacher(x_tr, y_tr, k)
    t_tr, t_va, t_te = teacher(y_tr), teacher(y_va), teacher(y_te)
    shape = SHAPES[dataset]
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    x_tr_t, x_va_t, x_te_t, y_tr_t, t_tr_t = to(x_tr), to(x_va), to(x_te), to(y_tr), to(t_tr)

    encoder, predictor, pixel_head = (m.to(device) for m in build(dataset, channels, d, hidden, k, predictor_type))
    # Adam's weight_decay adds the L2 gradient to the loss gradient (coupled L2), the penalty the weight-decay proposition analyzes.
    optimizer_name = params.get("optimizer", "adamw")
    optimizer = (torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW)([
        {"params": list(encoder.parameters()) + list(predictor.parameters()), "weight_decay": weight_decay},
        {"params": list(pixel_head.parameters()), "weight_decay": pixel_weight_decay},
    ], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    z_dirs = fixed_directions(d, n_directions, seed + 101).to(device)
    p_dirs = fixed_directions(k, n_directions, seed + 303).to(device)
    q_dirs = fixed_directions(d, n_directions, seed + 303).to(device) if predictor_type == "linear" else p_dirs

    n = len(x_tr)
    gen = torch.Generator().manual_seed(seed + 505)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    for _epoch in range(epochs):
        encoder.train(); predictor.train(); pixel_head.train()
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, batch_size):
            batch = perm[start : start + batch_size]
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=amp):
                z = encoder(x_tr_t[batch].view(-1, *shape))
                p = predictor(z)
                latent_loss = F.mse_loss(p.float(), t_tr_t[batch])
                reg = torch.zeros((), device=device)
                if method in ("encoder_gaussian", "both"):
                    reg = reg + projected_gaussianity_loss(z, z_dirs) + cov_weight * covariance_standard_normal_loss(z)
                if method in ("predictive_gaussian", "both"):
                    if predictor_type == "linear":
                        with torch.autocast(device_type=device.type, enabled=False):
                            q = intrinsic_output(predictor, z.float())
                    else:
                        q = p
                    pp = q if predictive_scale == "unit" else trace_normalize(q.float())
                    reg = reg + projected_gaussianity_loss(pp, q_dirs) + cov_weight * covariance_standard_normal_loss(pp)
                # The pixel head sees detached predictions, so its loss never reaches the encoder or predictor.
                pixels = pixel_head(p.detach()).reshape(len(batch), -1)
                pixel_loss = F.mse_loss(pixels.float(), y_tr_t[batch])
                total = latent_loss + reg_weight * reg + pixel_loss
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - t_start

    def evaluate(x_t: torch.Tensor, eval_bs: int = 1024) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        encoder.eval(); predictor.eval(); pixel_head.eval()
        zs, ps, pix = [], [], []
        with torch.no_grad():
            for s in range(0, len(x_t), eval_bs):
                z = encoder(x_t[s : s + eval_bs].view(-1, *shape))
                p = predictor(z)
                zs.append(z.float().cpu()); ps.append(p.float().cpu())
                pix.append(pixel_head(p).reshape(len(p), -1).float().cpu())
        return torch.cat(zs).numpy(), torch.cat(ps).numpy(), torch.cat(pix).numpy()

    _, p_tr, pix_tr = evaluate(x_tr_t)
    _, p_va, pix_va = evaluate(x_va_t)
    z_te, p_te, pix_te = evaluate(x_te_t)
    metrics_fn = mnist.compute_all_metrics if dataset == "mnist" else cifar.compute_all_metrics
    # Isotropy is defined up to scale (Cov = (tau/k) I), so the Gaussianity distances use unit average variance.
    unit = lambda a: trace_normalize(torch.from_numpy(a)).numpy()
    pixel_metrics = metrics_fn(pix_te, y_te, unit(z_te), unit(p_te), z_dirs.cpu().numpy(), p_dirs.cpu().numpy())
    pixel_train_mse = float(np.mean((np.clip(pix_tr, 0.0, 1.0) - y_tr) ** 2))

    return {
        "experiment": f"jepa_{dataset}",
        "dataset": dataset,
        "method": method,
        "predictive_scale": predictive_scale,
        "seed": seed,
        "n_train": n_train,
        "n_val": n_val,
        "embedding_dim": d,
        "teacher_dim": k,
        "teacher": teacher_type,
        "predictor": predictor_type,
        "reg_weight": reg_weight,
        "weight_decay": weight_decay,
        "optimizer": optimizer_name,
        "cov_weight": cov_weight,
        "teacher_explained_variance": teacher.explained,
        **latent_metrics(teacher, p_tr, t_tr, p_va, t_va, p_te, t_te),
        "pixel_train_mse": pixel_train_mse,
        "pixel_val_mse": float(np.mean((np.clip(pix_va, 0.0, 1.0) - y_va) ** 2)),
        "pixel_gap": pixel_metrics["test_mse"] - pixel_train_mse,
        "pixel_trivial_mse": float(np.mean((y_te - y_tr.mean(axis=0)) ** 2)),
        "predictive_trace": float(np.trace(np.cov(p_te, rowvar=False))),
        "train_seconds": train_seconds,
        **{("pixel_" + k if k in ("test_mse", "test_rmse", "relative_mse") else k): v for k, v in pixel_metrics.items()},
    }


def run(config_path: Path, output_path: Path) -> pd.DataFrame:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    rows = [train_one(params) for params in expanded_grid(config)]
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


if __name__ == "__main__":
    args = parse_args(ROOT / "configs" / "jepa_mnist.yaml", ROOT / "results" / "exp_jepa_mnist.csv")
    run(Path(args.config), Path(args.output))
