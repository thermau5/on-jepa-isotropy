"""Shape and layout tests for the CIFAR regularizer experiment (no dataset needed)."""

from __future__ import annotations

import numpy as np
import torch

from experiments.exp_regularizer_cifar_gpu import (
    FLAT_DIM,
    IMG_C,
    IMG_H,
    IMG_W,
    CIFARGPUJepa,
)
from plots.plot_regularizer_cifar_gpu_samples import combine_halves, half_to_rgb


def test_model_forward_shapes() -> None:
    torch.manual_seed(0)
    model = CIFARGPUJepa(channels=[8, 16, 32], embedding_dim=8, predictor_dim=8, hidden_dim=16)
    x = torch.randn(2, FLAT_DIM)
    y_hat, z, p = model(x)
    assert y_hat.shape == (2, FLAT_DIM)
    assert z.shape == (2, 8)
    assert p.shape == (2, 8)


def test_half_layout_roundtrip() -> None:
    """The plot-side unflatten must invert the loader's channel-first flatten."""
    rng = np.random.default_rng(0)
    img = rng.random((IMG_H, 2 * IMG_W, IMG_C)).astype(np.float32)  # (32, 32, 3) HWC
    chw = img.transpose(2, 0, 1)  # channel-first, as in load_cifar_halves_gpu
    left_flat = chw[:, :, :IMG_W].reshape(-1)
    right_flat = chw[:, :, IMG_W:].reshape(-1)
    assert left_flat.shape == (FLAT_DIM,)

    np.testing.assert_allclose(half_to_rgb(left_flat), img[:, :IMG_W, :], rtol=0, atol=0)
    np.testing.assert_allclose(combine_halves(left_flat, right_flat), img, rtol=0, atol=0)


def test_model_view_matches_loader_layout() -> None:
    """model.forward views flat input as (C,H,W); loader flattens the same way."""
    rng = np.random.default_rng(1)
    img = rng.random((IMG_H, IMG_W, IMG_C)).astype(np.float32)
    flat = img.transpose(2, 0, 1).reshape(-1)
    viewed = torch.from_numpy(flat).view(IMG_C, IMG_H, IMG_W).numpy()
    np.testing.assert_allclose(viewed.transpose(1, 2, 0), img, rtol=0, atol=0)
