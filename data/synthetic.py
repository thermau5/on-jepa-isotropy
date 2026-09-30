"""Linear-Gaussian generators for controlled BM-JEPA experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticSpec:
    """Configuration for a linear-Gaussian multi-target prediction problem."""

    n_train: int = 512
    n_test: int = 4096
    context_dim: int = 16
    target_dim: int = 1
    r_star: int = 8
    K: int = 8
    alpha: float = 1.0
    overlap: float = 0.0
    sigma_z: float = 0.05
    sigma_y: float = 0.05
    target_mode: str = "default"
    target_strength_slope: float = 1.18
    initial_imbalance_ratio: float = 0.1
    weak_target_strength: float = 0.31622776601683794
    spectrum_decay: float = 1.0
    spectrum_trace: float = 16.0
    spectrum_gap_rank: int = 8
    seed: int = 0


@dataclass(frozen=True)
class SyntheticProblem:
    """A sampled train/test problem plus its population operators."""

    latent_train: np.ndarray
    latent_test: np.ndarray
    z_train: np.ndarray
    y_train: np.ndarray
    z_test: np.ndarray
    y_test: np.ndarray
    context_operator: np.ndarray
    target_operator: np.ndarray
    eval_target_operator: np.ndarray
    true_subspace: np.ndarray
    population_cross_cov: np.ndarray
    population_zz_cov: np.ndarray
    population_b: np.ndarray
    population_spectrum: np.ndarray


def orthonormal_matrix(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """Return a deterministic random matrix with orthonormal columns."""

    if rows < cols:
        raise ValueError("rows must be at least cols")
    q, _ = np.linalg.qr(rng.normal(size=(rows, cols)))
    return q[:, :cols]


def target_operator(
    K: int,
    target_dim: int,
    r_star: int,
    alpha: float,
    overlap: float,
    rng: np.random.Generator | None = None,
    strength_slope: float = 1.18,
) -> np.ndarray:
    """Build stacked target maps with controlled complementarity/redundancy.

    The construction separates two mechanisms. alpha controls the angular spread
    of the available target dictionary. overlap controls how concentrated the
    target-bank coverage is over that dictionary.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if not 0.0 <= overlap <= 1.0:
        raise ValueError("overlap must be in [0, 1]")

    rows = K * target_dim
    shared = np.zeros(r_star)
    shared[0] = 1.0

    dictionary = [shared]
    angle_exponents = np.linspace(0.75, 1.65, max(r_star - 1, 1))
    for basis_idx in range(1, r_star):
        unique_scale = alpha ** angle_exponents[basis_idx - 1]
        shared_scale = np.sqrt(max(0.0, 1.0 - unique_scale**2))
        unique = np.zeros(r_star)
        unique[basis_idx] = 1.0
        vec = shared_scale * shared + unique_scale * unique
        vec /= np.linalg.norm(vec) + 1e-12
        dictionary.append(vec)
    dictionary_op = np.vstack(dictionary)

    assignments = np.arange(rows) % r_star
    counts = np.bincount(assignments, minlength=r_star).astype(float)
    if overlap >= 1.0:
        atom_prob = np.zeros(r_star)
        atom_prob[0] = 1.0
    else:
        base_prob = np.zeros(r_star)
        nonempty = counts > 0
        base_prob[nonempty] = 1.0 / np.sum(nonempty)
        concentration = 4.0 * overlap**2 / max(1.0 - overlap, 1e-6)
        atom_index = np.arange(r_star, dtype=float) / max(r_star - 1, 1)
        logits = np.log(np.maximum(base_prob, 1e-12)) - concentration * atom_index
        logits -= np.max(logits)
        atom_prob = np.exp(logits)
        atom_prob /= np.sum(atom_prob)
    atom_weight = np.zeros(r_star)
    nonempty = counts > 0
    atom_weight[nonempty] = np.sqrt(atom_prob[nonempty] * rows / counts[nonempty])

    weights = []
    for basis_idx in assignments:
        vec = dictionary_op[basis_idx]
        strength = strength_slope ** (r_star - basis_idx - 1)
        weights.append(atom_weight[basis_idx] * strength * vec)
    return np.vstack(weights)


def relative_conditioning_target_operator(
    K: int,
    target_dim: int,
    r_star: int,
    rng: np.random.Generator,
    initial_imbalance_ratio: float = 0.1,
    weak_target_strength: float = 0.31622776601683794,
) -> np.ndarray:
    """Full-rank bank whose relative conditioning improves with K.

    The first r_star rows cover every latent direction with imbalanced energy.
    Extra rows are allocated greedily to the currently weakest direction. A
    random latent rotation keeps the construction from being coordinate-aligned
    while preserving the intended Gram spectrum.
    """

    if target_dim != 1:
        raise ValueError("relative_conditioning target mode requires target_dim=1")
    if K < r_star:
        raise ValueError("relative_conditioning target mode requires K >= r_star")
    if not 0.0 < initial_imbalance_ratio <= 1.0:
        raise ValueError("initial_imbalance_ratio must be in (0, 1]")
    if weak_target_strength <= 0.0:
        raise ValueError("weak_target_strength must be positive")

    rotation = orthonormal_matrix(r_star, r_star, rng)
    if r_star == 1:
        energy = np.ones(1)
    else:
        energy = np.geomspace(1.0, initial_imbalance_ratio, r_star)

    rows = []
    for idx in range(r_star):
        rows.append(np.sqrt(energy[idx]) * rotation[:, idx])

    step_energy = weak_target_strength**2
    for _ in range(K - r_star):
        idx = int(np.argmin(energy))
        rows.append(weak_target_strength * rotation[:, idx])
        energy[idx] += step_energy

    return np.vstack(rows)


def fixed_trace_spectrum_target_operator(
    K: int,
    target_dim: int,
    r_star: int,
    rng: np.random.Generator,
    spectrum_decay: float = 1.0,
    spectrum_trace: float = 16.0,
) -> np.ndarray:
    """Full-rank target bank with fixed trace and controlled spectrum.

    The nonzero eigenvalues of A^T A are proportional to
    spectrum_decay ** i and normalized to sum to spectrum_trace. A random latent
    rotation prevents coordinate-axis artifacts while preserving the spectrum.
    """

    if target_dim != 1:
        raise ValueError("fixed_trace_spectrum target mode requires target_dim=1")
    if K < r_star:
        raise ValueError("fixed_trace_spectrum target mode requires K >= r_star")
    if not 0.0 < spectrum_decay <= 1.0:
        raise ValueError("spectrum_decay must be in (0, 1]")
    if spectrum_trace <= 0.0:
        raise ValueError("spectrum_trace must be positive")

    rotation = orthonormal_matrix(r_star, r_star, rng)
    weights = spectrum_decay ** np.arange(r_star, dtype=float)
    eigenvalues = spectrum_trace * weights / np.sum(weights)

    assignments = np.arange(K) % r_star
    counts = np.bincount(assignments, minlength=r_star)
    rows = [
        np.sqrt(eigenvalues[idx] / counts[idx]) * rotation[:, idx]
        for idx in assignments
    ]
    return np.vstack(rows)


def rank_gap_spectrum_target_operator(
    K: int,
    target_dim: int,
    r_star: int,
    rng: np.random.Generator,
    gap_rank: int,
    tail_relative_energy: float = 0.35,
    spectrum_trace: float = 16.0,
) -> np.ndarray:
    """Full-rank target stack with a controlled retained-rank eigengap."""

    if target_dim != 1:
        raise ValueError("rank_gap_spectrum target mode requires target_dim=1")
    if K < r_star:
        raise ValueError("rank_gap_spectrum target mode requires K >= r_star")
    if not 1 <= gap_rank < r_star:
        raise ValueError("spectrum_gap_rank must lie in {1, ..., r_star - 1}")
    if not 0.0 < tail_relative_energy <= 1.0:
        raise ValueError("spectrum_decay must be in (0, 1] for rank_gap_spectrum")
    if spectrum_trace <= 0.0:
        raise ValueError("spectrum_trace must be positive")

    rotation = orthonormal_matrix(r_star, r_star, rng)
    eigenvalues = np.ones(r_star)
    eigenvalues[gap_rank:] = tail_relative_energy
    eigenvalues *= spectrum_trace / np.sum(eigenvalues)

    assignments = np.arange(K) % r_star
    counts = np.bincount(assignments, minlength=r_star)
    rows = [
        np.sqrt(eigenvalues[idx] / counts[idx]) * rotation[:, idx]
        for idx in assignments
    ]
    return np.vstack(rows)


def sample_problem(spec: SyntheticSpec) -> SyntheticProblem:
    """Sample a train/test problem from a known linear-Gaussian model."""

    if spec.r_star > spec.context_dim:
        raise ValueError("r_star must not exceed context_dim")
    rng = np.random.default_rng(spec.seed)
    context_operator = orthonormal_matrix(spec.context_dim, spec.r_star, rng)
    if spec.target_mode == "default":
        full_target_op = target_operator(
            max(spec.K, spec.r_star),
            spec.target_dim,
            spec.r_star,
            spec.alpha,
            spec.overlap,
            rng,
            strength_slope=spec.target_strength_slope,
        )
    elif spec.target_mode == "relative_conditioning":
        full_target_op = relative_conditioning_target_operator(
            max(spec.K, spec.r_star),
            spec.target_dim,
            spec.r_star,
            rng,
            initial_imbalance_ratio=spec.initial_imbalance_ratio,
            weak_target_strength=spec.weak_target_strength,
        )
    elif spec.target_mode == "fixed_trace_spectrum":
        full_target_op = fixed_trace_spectrum_target_operator(
            max(spec.K, spec.r_star),
            spec.target_dim,
            spec.r_star,
            rng,
            spectrum_decay=spec.spectrum_decay,
            spectrum_trace=spec.spectrum_trace,
        )
    elif spec.target_mode == "rank_gap_spectrum":
        full_target_op = rank_gap_spectrum_target_operator(
            max(spec.K, spec.r_star),
            spec.target_dim,
            spec.r_star,
            rng,
            gap_rank=spec.spectrum_gap_rank,
            tail_relative_energy=spec.spectrum_decay,
            spectrum_trace=spec.spectrum_trace,
        )
    else:
        raise ValueError(f"unknown target_mode: {spec.target_mode}")
    full_eval_target_op = target_operator(
        spec.r_star,
        spec.target_dim,
        spec.r_star,
        1.0,
        0.0,
        rng,
        strength_slope=spec.target_strength_slope,
    )
    target_op = full_target_op[: spec.K * spec.target_dim]
    eval_target_op = full_eval_target_op[: spec.r_star * spec.target_dim]

    def draw(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        latent = rng.normal(size=(n, spec.r_star))
        z = latent @ context_operator.T
        z += spec.sigma_z * rng.normal(size=(n, spec.context_dim))
        y = latent @ target_op.T
        y += spec.sigma_y * rng.normal(size=(n, spec.K * spec.target_dim))
        return latent, z, y

    latent_train, z_train, y_train = draw(spec.n_train)
    latent_test, z_test, y_test = draw(spec.n_test)

    # Population linear predictor from context to targets for the noiseless part.
    sigma_zz = context_operator @ context_operator.T + spec.sigma_z**2 * np.eye(
        spec.context_dim
    )
    sigma_yz = target_op @ context_operator.T
    population_b = sigma_yz @ np.linalg.pinv(sigma_zz)
    evals, evecs = np.linalg.eigh(sigma_zz)
    evals = np.clip(evals, 0.0, None)
    sqrt_sigma_zz = (evecs * np.sqrt(evals)) @ evecs.T
    population_spectrum = np.linalg.svd(population_b @ sqrt_sigma_zz, compute_uv=False)
    true_subspace = context_operator

    return SyntheticProblem(
        latent_train=latent_train,
        latent_test=latent_test,
        z_train=z_train,
        y_train=y_train,
        z_test=z_test,
        y_test=y_test,
        context_operator=context_operator,
        target_operator=target_op,
        eval_target_operator=eval_target_op,
        true_subspace=true_subspace,
        population_cross_cov=sigma_yz,
        population_zz_cov=sigma_zz,
        population_b=population_b,
        population_spectrum=population_spectrum,
    )
