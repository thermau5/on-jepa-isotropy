"""Whitened encoder-embedding Gaussianity distance for the Table 1 runs.

Retrains each Table 1 configuration (no regularization and each regularized objective at the weight with the
lowest mean validation latent risk, ten seeds, all four tasks) and scores the test-set encoder embedding after
whitening it. Whitening is a linear gauge that the last encoder layer and the first predictor layer absorb, so the
predictions are unchanged and the score measures only how far the embedding's shape is from a Gaussian. It uses
the same score and directions as the Table 1 encoder Gaussianity distance. The experiment scripts are not changed:
the embedding is captured where train_one hands it to its metric function, and each rerun keeps its latent test
risk and Table 1 encoder score to check that it reproduces the sweep.

Usage: python -m experiments.exp_jepa_shape [task ...]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import yaml

from experiments import exp_jepa_digits as digits
from experiments import exp_jepa_images as images
from experiments import exp_regularizer_cifar_gpu as cifar
from experiments import exp_regularizer_mnist_gpu as mnist
from experiments.common import ROOT
from experiments.exp_regularizer_mnist_gpu import fixed_directions, projected_gaussianity_score

TASKS = ["digits", "mnist", "cifar10", "cifar100"]
METHODS = ["baseline", "encoder_gaussian", "predictive_gaussian", "both"]
OUTPUT = ROOT / "results" / "exp_jepa_shape.csv"

captured: list[np.ndarray] = []


def capture_digits(fn):
    def wrapped(a):
        captured.append(np.asarray(a))
        return fn(a)
    return wrapped


def capture_images(fn):
    def wrapped(preds, targets, z_np, p_np, z_dirs, p_dirs):
        captured.append(np.asarray(z_np))
        return fn(preds, targets, z_np, p_np, z_dirs, p_dirs)
    return wrapped


# train_one scores the embedding before the prediction, so the first captured array is the embedding.
digits.covariance_metrics = capture_digits(digits.covariance_metrics)
mnist.compute_all_metrics = capture_images(mnist.compute_all_metrics)
cifar.compute_all_metrics = capture_images(cifar.compute_all_metrics)


def whiten(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    a = a - a.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(a, rowvar=False))
    return a @ evecs / np.sqrt(evals)


def table1_weights(task: str) -> dict[str, float]:
    sweep = pd.read_csv(ROOT / "results" / f"exp_jepa_{task}_cca.csv")
    return {m: float(sweep[sweep["method"] == m].groupby("reg_weight")["latent_val_mse"].mean().idxmin())
            for m in METHODS[1:]}


def run(tasks: list[str]) -> pd.DataFrame:
    rows = []
    for task in tasks:
        base = yaml.safe_load(open(ROOT / "configs" / f"jepa_{task}_cca.yaml"))["base"]
        weights = table1_weights(task)
        for seed in range(10):
            for method in METHODS:
                params = {**base, "method": method, "seed": seed}
                if task != "digits":
                    params["dataset"] = task
                if method != "baseline":
                    params.update(predictive_scale="unit", reg_weight=weights[method])
                captured.clear()
                row = (digits if task == "digits" else images).train_one(params)
                z = captured[0]
                z_dirs = fixed_directions(z.shape[1], int(base["n_directions"]), seed + 101).numpy().astype(np.float64)
                rows.append({
                    "task": task,
                    "seed": seed,
                    "method": method,
                    "reg_weight": params.get("reg_weight"),
                    "latent_test_mse": row["latent_test_mse"],
                    "embedding_gaussianity_score": row["embedding_gaussianity_score"],
                    "whitened_embedding_gaussianity_score": projected_gaussianity_score(whiten(z), z_dirs),
                })
                print(rows[-1], flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    tasks = sys.argv[1:] or TASKS
    frame = run(tasks)
    if OUTPUT.exists():
        previous = pd.read_csv(OUTPUT)
        frame = pd.concat([previous[~previous["task"].isin(tasks)], frame], ignore_index=True)
    frame.to_csv(OUTPUT, index=False)
