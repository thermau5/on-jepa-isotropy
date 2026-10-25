# On JEPA Isotropy

A finite-sample, empirical-risk-minimization (ERM) theory of Joint-Embedding
Predictive Architectures (JEPA), and the experiment suite that corroborates it.

## What this work shows

JEPA is a prominent self-supervised paradigm, but its isotropy regularizers
(isotropic-Gaussian embeddings, SIGReg-style penalties) have been heuristic,
targeting the encoder marginal rather than the prediction risk itself. We give
them a rigorous ERM foundation:

- **JEPA is reduced-rank regression.** Casting the embedding bottleneck as a
  rank constraint reformulates multi-target linear JEPA as reduced-rank
  regression, yielding an exact risk decomposition into an *irreducible error*
  and a *finite-sample excess-risk bound*.
- **The irreducible term.** The optimal predictor's rank grows with target
  count and heterogeneity, then saturates at the embedding bottleneck, which
  floors an irreducible spectral tail.
- **The excess-risk bound.** It is governed by the spectral conditioning of the
  whitened end-to-end map `T_K`, and is minimized at *predictive
  isotropy*.
- **A principled regularizer.** These conditions prescribe a prediction-side
  isotropic regularizer. Under it, JEPA isotropy *subsumes* the canonical
  encoder-embedding isotropy designs, so the empirical success of existing
  JEPA isotropy heuristics follows as a corollary of empirical risk
  minimization.

## This repository

The experiment suite backing every claim: controlled linear-Gaussian
diagnostics (with known population operators, so each assumption,
proposition, and theorem is checked against ground truth) plus real-image
experiments validating the prediction-aware regularizer inside learned
nonlinear networks.

| Path | Contents |
|------|----------|
| `data/` | Linear-Gaussian synthetic generators |
| `models/` | OLS and reduced-rank regression solvers |
| `metrics/` | Rank, spectrum, and subspace-recovery metrics |
| `experiments/` | One `exp_*.py` per claim, plus `run_all.py` |
| `configs/` | Per-experiment YAML configs |
| `plots/` | Figure generation |
| `tests/` | pytest suite |

## Requirements

Python 3.10+:

```bash
pip install numpy scipy pandas matplotlib scikit-learn pyyaml torch torchvision pytest
```

## Usage

```bash
python -m experiments.run_all                   # synthetic suite + reduction/gauge/digits
python -m experiments.exp_regularizer_mnist      # MNIST MLP regularizer diagnostic
python -m experiments.exp_regularizer_mnist_gpu  # scaled MNIST regularizer diagnostic
python -m experiments.exp_regularizer_cifar_gpu  # CIFAR-10 regularizer diagnostic
python -m experiments.exp_regularizer_cifar_gpu \
    --config configs/regularizer_cifar100.yaml \
    --output results/exp_regularizer_cifar100.csv  # CIFAR-100 variant
python -m experiments.exp_jepa_digits \
    --config configs/jepa_digits_cca.yaml \
    --output results/exp_jepa_digits_cca.csv     # latent-target JEPA on 8x8 digits
python -m experiments.exp_jepa_images \
    --config configs/jepa_mnist_cca.yaml \
    --output results/exp_jepa_mnist_cca.csv      # latent-target JEPA on MNIST (same for cifar10, cifar100)
python -m experiments.exp_jepa_shape            # whitened encoder-embedding Gaussianity of the selected runs
python -m plots.plot_main                        # synthetic and 8x8-digit figures
python -m plots.plot_jepa_diagnostics           # latent-target JEPA figures
python -m plots.plot_regularizer_mnist_gpu       # MNIST regularizer figure
python -m plots.plot_regularizer_cifar_gpu       # CIFAR-10/100 regularizer figures
python -m plots.plot_regularizer_digits_samples     # 8x8-digit completions
python -m plots.plot_regularizer_mnist_samples      # MNIST MLP completions
python -m plots.plot_regularizer_mnist_gpu_samples  # MNIST CNN completions
python -m plots.plot_regularizer_cifar_gpu_samples  # CIFAR-10/100 completions
pytest                                           # run tests
```

CSVs are written to `results/`, figures to `figures/`.
