# On JEPA Isotropy

Code for the NeurIPS paper *On JEPA Isotropy*.

## Abstract

We study the empirical risk minimization (ERM) of linear Joint-Embedding Predictive Architecture (JEPA) and establish its key conditions for optimality. By casting the embedding bottleneck as a rank constraint, we formulate JEPA as a reduced-rank regression problem and study ERM via standard oracle inequality. We show that JEPA's oracle risk decreases as target heterogeneity raises the rank until it saturates at the bottleneck. Moreover, JEPA's excess risk is governed by the spectral conditioning of the JEPA prediction. At any fixed oracle risk, it reaches optimality at a flat spectrum, termed *predictive isotropy*. We show that this condition, enforced by a regularizer, subsumes canonical JEPA isotropic embedding designs and hence justifies their empirical successes through ERM. Synthetic experiments corroborate our theory. Realistic experiments show our proposed regularizer lowers the test risk by 11% and the generalization gap by 57% from the existing paradigm.

## This repository

The synthetic experiments use linear-Gaussian data with known population operators. The real-image experiments train a JEPA on 8×8 digits, MNIST, CIFAR-10 and CIFAR-100.

| Path | Contents |
|------|----------|
| `data/` | Linear-Gaussian synthetic generators |
| `models/` | OLS and reduced-rank regression solvers |
| `metrics/` | Rank, spectrum, and subspace-recovery metrics |
| `experiments/` | Experiment scripts, plus `run_all.py` |
| `configs/` | Per-experiment YAML configs |
| `plots/` | Figure generation |
| `tests/` | pytest suite |

## Requirements

Python 3.10 or later.

```bash
pip install numpy scipy pandas matplotlib scikit-learn pyyaml torch torchvision pytest
```

## Reproducing the paper

The commands below produce every figure and Table 1 of the paper. Run them from the repository root and in order, because later steps read the CSVs of earlier ones. CSVs go to `results/` and figures to `figures/`. All sweeps use seeds 0 to 9, and the completion figures use seed 0.

```bash
# 1. Synthetic experiments (Figures 1 and 3 to 10)
python -m experiments.run_all
python -m plots.plot_main

# 2. Real-image sweeps behind Table 1
python -m experiments.exp_jepa_digits --config configs/jepa_digits_cca.yaml --output results/exp_jepa_digits_cca.csv
for task in mnist cifar10 cifar100; do
    python -m experiments.exp_jepa_images --config configs/jepa_${task}_cca.yaml --output results/exp_jepa_${task}_cca.csv
done

# 3. Whitened encoder-embedding Gaussianity of the Table 1 runs
python -m experiments.exp_jepa_shape

# 4. Real-image figures
python -m plots.plot_jepa_diagnostics  # Figures 14 to 17
python -m plots.plot_jepa_samples      # Figures 2 and 11 to 13
```

The synthetic experiments and the 8×8 digits sweep run on a CPU. The MNIST and CIFAR experiments download their data into `data/` on first use and run on a GPU. We ran them on one NVIDIA RTX 4090 with Python 3.13 and PyTorch 2.11. The 8×8 digits sweep trains 340 models. Each MNIST and CIFAR sweep trains 190 models (no regularization and three regularized objectives at six weights, over ten seeds) at 13 to 17 s per model, so it takes under an hour.

| Paper item | Data | Script |
|------------|------|--------|
| Figure 1, post-saturation conditioning | `results/exp_post_saturation.csv` | `plots.plot_main` |
| Table 1, regularizer comparison on real images | `results/exp_jepa_{task}_cca.csv` | see below |
| Figure 2, MNIST completions | retrains each objective at seed 0 | `plots.plot_jepa_samples` |
| Figure 3, reduction to RRR | `results/exp_reduction_diagnostics.csv` | `plots.plot_main` |
| Figure 4, rank saturation | `results/exp_k_saturation.csv` | `plots.plot_main` |
| Figure 5, target heterogeneity | `results/exp_heterogeneity.csv` | `plots.plot_main` |
| Figure 6, embedding bottleneck | `results/exp_bottleneck.csv` | `plots.plot_main` |
| Figure 7, oracle inequality | `results/exp_unified_risk.csv` | `plots.plot_main` |
| Figure 8, predictive isotropy | `results/exp_predictive_isotropy.csv` | `plots.plot_main` |
| Figure 9, concentration bounds | `results/exp_heterogeneity.csv`, `results/exp_post_saturation.csv`, 8×8 digits | `plots.plot_main` |
| Figure 10, gauge factorization | `results/exp_gauge_factorization.csv` | `plots.plot_main` |
| Figures 11 to 13, 8×8 digits, CIFAR-10 and CIFAR-100 completions | retrains each objective at seed 0 | `plots.plot_jepa_samples` |
| Figures 14 to 17, regularizer diagnostics | `results/exp_jepa_{task}_cca.csv`, `results/exp_jepa_shape.csv` | `plots.plot_jepa_diagnostics` |

Table 1 summarizes `results/exp_jepa_{task}_cca.csv`. For each regularized objective, we keep the weight with the lowest mean validation latent risk (`table1_runs` in `plots/plot_jepa_diagnostics.py`). We then report the mean and standard error over seeds of `latent_test_mse` (latent risk), `latent_gap` (test–train gap), `pixel_test_mse` and `fg_mse` (pixel and FG MSE, in units of 10⁻²) and `train_seconds` (time). Tables 2 and 3 list the settings in `configs/` and the models in `experiments/`.

## Other scripts

These scripts run the pixel-target regularizer experiments and the tests.

```bash
python -m experiments.exp_regularizer_mnist      # MNIST MLP regularizer diagnostic
python -m experiments.exp_regularizer_mnist_gpu  # MNIST CNN regularizer diagnostic
python -m experiments.exp_regularizer_cifar_gpu  # CIFAR-10 regularizer diagnostic
python -m experiments.exp_regularizer_cifar_gpu --config configs/regularizer_cifar100.yaml --output results/exp_regularizer_cifar100.csv  # CIFAR-100 variant
python -m plots.plot_regularizer_mnist_gpu          # MNIST regularizer figure
python -m plots.plot_regularizer_cifar_gpu          # CIFAR-10/100 regularizer figures
python -m plots.plot_regularizer_digits_samples     # 8x8-digit completions
python -m plots.plot_regularizer_mnist_samples      # MNIST MLP completions
python -m plots.plot_regularizer_mnist_gpu_samples  # MNIST CNN completions
python -m plots.plot_regularizer_cifar_gpu_samples  # CIFAR-10/100 completions
pytest                                              # run tests
```
