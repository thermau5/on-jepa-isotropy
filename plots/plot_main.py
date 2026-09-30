"""Create publication-ready figures from experiment CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator, NullFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

BLUE = "#2B2142"
TEAL = "#5D4A7F"
GOLD = "#B8872D"
RED = "#9A5B86"
GRAY = "#4B4654"
THEORY = "#D8A2AE"
GRID = "#CFCFCF"
LIGHT_SHADE = "#F7F4F8"
TAIL_PLOT_FLOOR = 1e-1
EMPIRICAL_MARKER_SIZE = 3.1
EMPIRICAL_LINE_WIDTH = 1.75
THEORY_LINE_WIDTH = 1.75
EMPIRICAL_BAND_ALPHA = 0.16
THEORY_BAND_ALPHA = 0.10
GUIDE_STYLE = {
    "color": GRAY,
    "linestyle": ":",
    "linewidth": 0.8,
    "alpha": 0.55,
    "zorder": 0,
}


def mean_sem(frame: pd.DataFrame, x: str, metrics: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(x, as_index=False)[metrics].agg(["mean", "sem"])
    grouped.columns = [x] + [f"{name}_{stat}" for name, stat in grouped.columns[1:]]
    return grouped


def line_with_band(
    ax,
    data: pd.DataFrame,
    x: str,
    y: str,
    label: str,
    color: str = BLUE,
    marker: str = "o",
    zorder: int = 3,
) -> None:
    mean_col = f"{y}_mean"
    sem_col = f"{y}_sem"
    ax.plot(
        data[x],
        data[mean_col],
        color=color,
        marker=marker,
        markersize=EMPIRICAL_MARKER_SIZE,
        markeredgewidth=0.5,
        linewidth=EMPIRICAL_LINE_WIDTH,
        label=label,
        zorder=zorder,
    )
    sem = data[sem_col].fillna(0.0)
    ax.fill_between(
        data[x],
        data[mean_col] - sem,
        data[mean_col] + sem,
        color=color,
        alpha=EMPIRICAL_BAND_ALPHA,
        linewidth=0.0,
        zorder=max(1, zorder - 1),
    )


def dashed_theory(
    ax,
    data: pd.DataFrame,
    x: str,
    y: str,
    label: str = "Population",
    floor: float | None = None,
    linestyle: str = "--",
    zorder: int = 1,
    alpha: float = 0.9,
    color: str = THEORY,
) -> None:
    """Dashed population-value curve."""

    mean_col = f"{y}_mean"
    sem_col = f"{y}_sem"
    values = data[mean_col]
    if floor is not None:
        values = values.clip(lower=floor)
    ax.plot(
        data[x],
        values,
        color=color,
        linestyle=linestyle,
        linewidth=THEORY_LINE_WIDTH,
        alpha=alpha,
        label=label,
        zorder=zorder,
    )
    if sem_col in data:
        sem = data[sem_col].fillna(0.0)
        lower = data[mean_col] - sem
        upper = data[mean_col] + sem
        if floor is not None:
            lower = lower.clip(lower=floor)
            upper = upper.clip(lower=floor)
        ax.fill_between(
            data[x],
            lower,
            upper,
            color=color,
            alpha=THEORY_BAND_ALPHA,
            linewidth=0.0,
            zorder=max(0, zorder - 1),
        )


def line_with_band_logsafe(ax, data: pd.DataFrame, x: str, y: str, label: str) -> None:
    """Line plot for log axes where exact zeros mean the tail has vanished."""

    plot_data = data.copy()
    for suffix in ["mean", "sem"]:
        col = f"{y}_{suffix}"
        plot_data[col] = plot_data[col].where(plot_data[col] > 0.0, np.nan)
    line_with_band(ax, plot_data, x, y, label)


def line_with_band_floor(
    ax,
    data: pd.DataFrame,
    x: str,
    y: str,
    label: str,
    floor: float = TAIL_PLOT_FLOOR,
    color: str = BLUE,
    marker: str = "o",
    zorder: int = 3,
) -> None:
    """Line plot for log axes with sub-floor values shown on a visible floor."""

    plot_data = data.copy()
    mean_col = f"{y}_mean"
    sem_col = f"{y}_sem"
    plot_data[mean_col] = plot_data[mean_col].clip(lower=floor)
    lower = (data[mean_col] - data[sem_col].fillna(0.0)).clip(lower=floor)
    upper = (data[mean_col] + data[sem_col].fillna(0.0)).clip(lower=floor)
    ax.plot(
        plot_data[x],
        plot_data[mean_col],
        color=color,
        marker=marker,
        markersize=EMPIRICAL_MARKER_SIZE,
        markeredgewidth=0.5,
        linewidth=EMPIRICAL_LINE_WIDTH,
        label=label,
        zorder=zorder,
    )
    ax.fill_between(
        plot_data[x],
        lower,
        upper,
        color=color,
        alpha=EMPIRICAL_BAND_ALPHA,
        linewidth=0.0,
        zorder=max(1, zorder - 1),
    )


def style_axis(ax, xlabel: str, ylabel: str, panel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(panel, loc="left", fontsize=8.6, fontweight="normal", pad=1.5)
    ax.grid(axis="both", color=GRID, linewidth=0.45, alpha=0.9)
    ax.tick_params(axis="both", length=2.5, width=0.65, color=GRAY, pad=2.0)


def set_compact_log_tick_labels(ax, ticks: list[float]) -> None:
    """Show explicit scientific tick labels on narrow log-scale panels."""

    def formatter(value: float, _position: int) -> str:
        if value <= 0:
            return ""
        label = f"{value:.0e}"
        return label.replace("e-0", "e-").replace("e+0", "e")

    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(formatter))
    ax.yaxis.set_minor_formatter(NullFormatter())


def style_rank_axis(ax, max_rank: int) -> None:
    """Use coarse count-scale ticks for seed-averaged rank diagnostics."""

    step = max(2, int(np.ceil(max_rank / 8.0)))
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.set_ylim(-0.25, max_rank + 0.75)


def rank_reference(summary: pd.DataFrame, column: str = "oracle_effective_rank_mean") -> int:
    """Integer population-rank reference used for guide lines."""

    return int(round(float(summary[column].max())))


def add_figure_legend(fig: plt.Figure, ax) -> None:
    """Place a shared legend outside the plotting area."""

    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        frameon=True,
        fancybox=False,
        framealpha=0.92,
        edgecolor=GRID,
        facecolor="white",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=min(3, max(1, len(labels))),
        borderpad=0.2,
        handlelength=1.8,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    legend.get_frame().set_linewidth(0.6)


def add_left_figure_legend(fig: plt.Figure, ax) -> None:
    """Place a shared legend above the left panels only."""

    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        frameon=True,
        fancybox=False,
        framealpha=0.92,
        edgecolor=GRID,
        facecolor="white",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=min(2, max(1, len(labels))),
        borderpad=0.2,
        handlelength=1.8,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    legend.get_frame().set_linewidth(0.6)


def add_axis_legend(ax) -> None:
    legend = ax.legend(
        frameon=True,
        fancybox=False,
        framealpha=0.92,
        edgecolor=GRID,
        facecolor="white",
        loc="best",
        fontsize=7.0,
        borderpad=0.2,
        handlelength=1.55,
        handletextpad=0.35,
        labelspacing=0.25,
    )
    legend.get_frame().set_linewidth(0.6)


def add_combined_axis_legend(ax, other_ax) -> None:
    """Place one legend for artists from a twinned-axis panel."""

    handles, labels = ax.get_legend_handles_labels()
    other_handles, other_labels = other_ax.get_legend_handles_labels()
    legend = ax.legend(
        handles + other_handles,
        labels + other_labels,
        frameon=True,
        fancybox=False,
        framealpha=0.92,
        edgecolor=GRID,
        facecolor="white",
        loc="best",
        borderpad=0.2,
        handlelength=1.8,
        handletextpad=0.4,
    )
    legend.get_frame().set_linewidth(0.6)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def configure_matplotlib() -> None:
    """Apply the paper plotting style, including direct single-figure calls."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["DejaVu Sans"],
            "font.weight": "normal",
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "regular",
            "font.size": 8.6,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.0,
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "lines.color": BLUE,
            "patch.facecolor": BLUE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": GRAY,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.minor.width": 0.45,
            "ytick.minor.width": 0.45,
            "legend.edgecolor": GRID,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def make_figure(ncols: int = 3, figsize: tuple[float, float] = (7.1, 2.25)) -> tuple[plt.Figure, list[plt.Axes]]:
    configure_matplotlib()
    fig, axes = plt.subplots(1, ncols, figsize=figsize, constrained_layout=True)
    return fig, list(axes)


def plot_k_saturation() -> None:
    frame = pd.read_csv(RESULTS / "exp_k_saturation.csv")
    summary = mean_sem(
        frame,
        "K",
        [
            "test_mse",
            "oracle_test_mse",
            "recovered_rank",
            "oracle_effective_rank",
            "lambda_at_population_rank",
            "oracle_lambda_at_population_rank",
        ],
    )
    rank_ref = rank_reference(summary)
    fig, axes = make_figure()
    dashed_theory(axes[0], summary, "K", "oracle_test_mse")
    line_with_band(axes[0], summary, "K", "test_mse", "Finite sample")
    dashed_theory(axes[1], summary, "K", "oracle_effective_rank")
    line_with_band(axes[1], summary, "K", "recovered_rank", "Finite sample")
    dashed_theory(axes[2], summary, "K", "oracle_lambda_at_population_rank")
    line_with_band(axes[2], summary, "K", "lambda_at_population_rank", "Finite sample")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[1].set_xscale("log", base=2)
    axes[2].set_xscale("log", base=2)
    for ax in axes:
        ticks = [int(x) for x in summary["K"] if x in {1, 4, 16, 64, 256}]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(x) for x in ticks])
    axes[0].axvline(rank_ref, **GUIDE_STYLE)
    axes[1].axhline(rank_ref, **GUIDE_STYLE)
    axes[2].axvline(rank_ref, **GUIDE_STYLE)
    add_figure_legend(fig, axes[0])
    style_rank_axis(axes[1], rank_ref)
    style_axis(axes[0], "Target count $K$", "Per-target MSE", "(a)")
    style_axis(axes[1], "Target count $K$", "Recovered rank", "(b)")
    style_axis(axes[2], "Target count $K$", r"Heterogeneity $\lambda_{\mathrm{het}}$", "(c)")
    save_figure(fig, FIGURES / "exp_k_saturation.pdf")


def plot_heterogeneity() -> None:
    frame = pd.read_csv(RESULTS / "exp_heterogeneity.csv")
    summary = mean_sem(
        frame,
        "alpha",
        [
            "recovered_rank",
            "oracle_effective_rank",
            "test_mse",
            "oracle_test_mse",
            "fixed_eval_mse",
            "oracle_fixed_eval_mse",
            "lambda_at_population_rank",
            "oracle_lambda_at_population_rank",
        ],
    )
    summary["lambda_axis"] = summary["lambda_at_population_rank_mean"]
    summary = summary.sort_values("lambda_axis")
    rank_ref = rank_reference(summary)
    fig, axes = make_figure(ncols=3, figsize=(7.1, 2.25))
    dashed_theory(axes[0], summary, "lambda_axis", "oracle_test_mse")
    line_with_band(axes[0], summary, "lambda_axis", "test_mse", "Finite sample")
    dashed_theory(axes[1], summary, "lambda_axis", "oracle_fixed_eval_mse")
    line_with_band(axes[1], summary, "lambda_axis", "fixed_eval_mse", "Finite sample")
    dashed_theory(axes[2], summary, "lambda_axis", "oracle_effective_rank")
    line_with_band(axes[2], summary, "lambda_axis", "recovered_rank", "Finite sample")
    add_left_figure_legend(fig, axes[0])
    axes[2].axhline(rank_ref, **GUIDE_STYLE)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    style_rank_axis(axes[2], rank_ref)
    style_axis(
        axes[0],
        r"Empirical heterogeneity $\widehat{\lambda}_{\mathrm{het}}$",
        "Per-target MSE",
        "(a)",
    )
    style_axis(
        axes[1],
        r"Empirical heterogeneity $\widehat{\lambda}_{\mathrm{het}}$",
        "Reference-stack MSE",
        "(b)",
    )
    style_axis(
        axes[2],
        r"Empirical heterogeneity $\widehat{\lambda}_{\mathrm{het}}$",
        "Recovered rank",
        "(c)",
    )
    save_figure(fig, FIGURES / "exp_heterogeneity.pdf")


def plot_post_saturation() -> None:
    frame = pd.read_csv(RESULTS / "exp_post_saturation.csv")
    plot_post_saturation_frame(frame, FIGURES / "exp_post_saturation.pdf")


def plot_post_saturation_frame(
    frame: pd.DataFrame,
    output_path: Path,
    summary_path: Path = RESULTS / "post_saturation_calibrated_bounds.csv",
) -> None:
    configure_matplotlib()
    frame = frame.copy()
    delta = 0.1
    frame["normalized_operator_error"] = (
        frame["operator_error_op"] / frame["oracle_sigma_top"].clip(lower=1e-12)
    )
    empirical_rho_bound = empirical_quantile(
        frame["normalized_operator_error"],
        1.0 - delta,
    )
    relative_perturbation_coverage = float(
        np.mean(frame["normalized_operator_error"] <= empirical_rho_bound)
    )
    empirical_l_bound = frame["per_target_risk_stability_ratio"].max()
    kappa = np.sqrt(frame["oracle_relative_conditioning"].clip(lower=1e-12))
    frame["calibrated_excess_risk_rhs"] = (
        empirical_l_bound * empirical_rho_bound / (kappa - empirical_rho_bound)
    )
    frame["calibrated_total_risk_rhs"] = (
        frame["oracle_test_mse"] + frame["calibrated_excess_risk_rhs"]
    )
    subspace_operator = (
        str(frame["subspace_operator"].iloc[0])
        if "subspace_operator" in frame.columns
        else "cross_cov"
    )
    if subspace_operator == "whitened_raw":
        eigen_ylabel = r"Eigenvalues of $T_K^\top T_K$"
        kappa_ylabel = r"$\kappa_K^2(T_K)$"
        effdim_ylabel = r"$r(T_K)$"
        stability_label = r"Lipschitz bound"
    else:
        eigen_ylabel = r"Eigenvalues of $G_K$"
        kappa_ylabel = r"$\kappa_K^2=\lambda_d/\lambda_1$"
        effdim_ylabel = r"$r(\Sigma_{YX})$"
        stability_label = r"Lipschitz bound"
    summary = mean_sem(
        frame,
        "K",
        [
            "lambda_top",
            "oracle_lambda_top",
            "lambda_at_bottleneck_rank",
            "oracle_lambda_at_bottleneck_rank",
            "relative_conditioning",
            "oracle_relative_conditioning",
            "theorem_subspace_error",
            "per_target_excess_mse",
            "test_mse",
            "oracle_test_mse",
            "oracle_effdim",
            "empirical_effdim",
            "per_target_risk_stability_ratio",
            "calibrated_excess_risk_rhs",
            "calibrated_total_risk_rhs",
        ],
    )
    summary["per_target_excess_mse_mean"] = summary["per_target_excess_mse_mean"].clip(
        lower=1e-6
    )
    summary["per_target_excess_mse_sem"] = summary["per_target_excess_mse_sem"].fillna(0.0)
    summary["risk_stability_bound_mean"] = (
        empirical_l_bound * summary["theorem_subspace_error_mean"]
    ).clip(lower=1e-6)
    summary["risk_stability_bound_sem"] = (
        empirical_l_bound * summary["theorem_subspace_error_sem"].fillna(0.0)
    )
    summary["delta"] = delta
    summary["calibrated_rho"] = empirical_rho_bound
    summary["empirical_l"] = empirical_l_bound
    summary["relative_perturbation_coverage"] = relative_perturbation_coverage
    summary.to_csv(summary_path, index=False)
    fig, axes_grid = plt.subplots(2, 3, figsize=(7.1, 3.3), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.015, wspace=0.02, hspace=0.028)
    axes = list(axes_grid.ravel())
    dashed_theory(
        axes[0],
        summary,
        "K",
        "oracle_lambda_top",
        label=r"Pop. $\lambda_1$",
        zorder=1,
    )
    dashed_theory(
        axes[0],
        summary,
        "K",
        "oracle_lambda_at_bottleneck_rank",
        label=r"Pop. $\lambda_d$",
        zorder=1,
    )
    line_with_band(
        axes[0],
        summary,
        "K",
        "lambda_top",
        r"Finite $\lambda_1$",
        color=RED,
        marker="s",
        zorder=4,
    )
    line_with_band(
        axes[0],
        summary,
        "K",
        "lambda_at_bottleneck_rank",
        r"Finite $\lambda_d$",
        color=BLUE,
        marker="o",
        zorder=5,
    )
    dashed_theory(
        axes[1],
        summary,
        "K",
        "oracle_relative_conditioning",
        label="Population",
    )
    line_with_band(
        axes[1],
        summary,
        "K",
        "relative_conditioning",
        "Finite sample",
    )
    dashed_theory(
        axes[2],
        summary,
        "K",
        "oracle_effdim",
        label="Population",
    )
    line_with_band(
        axes[2],
        summary,
        "K",
        "empirical_effdim",
        "Finite sample",
    )
    line_with_band_floor(
        axes[3],
        summary,
        "K",
        "per_target_excess_mse",
        "Excess risk",
        floor=1e-6,
        color=BLUE,
        marker="o",
    )
    bound_lower = (
        summary["risk_stability_bound_mean"] - summary["risk_stability_bound_sem"]
    ).clip(lower=1e-6)
    bound_upper = (
        summary["risk_stability_bound_mean"] + summary["risk_stability_bound_sem"]
    ).clip(lower=1e-6)
    axes[3].plot(
        summary["K"],
        summary["risk_stability_bound_mean"],
        color=THEORY,
        linestyle="--",
        linewidth=THEORY_LINE_WIDTH,
        alpha=0.9,
        label=stability_label,
        zorder=1,
    )
    axes[3].errorbar(
        summary["K"],
        summary["risk_stability_bound_mean"],
        yerr=[
            (summary["risk_stability_bound_mean"] - bound_lower).clip(lower=0.0),
            (bound_upper - summary["risk_stability_bound_mean"]).clip(lower=0.0),
        ],
        fmt="none",
        ecolor=THEORY,
        elinewidth=0.65,
        capsize=0.0,
        alpha=0.35,
        zorder=1,
    )
    dashed_theory(
        axes[4],
        summary,
        "K",
        "calibrated_excess_risk_rhs",
        label="Excess-risk bound",
        floor=1e-6,
        zorder=1,
    )
    line_with_band_floor(
        axes[4],
        summary,
        "K",
        "per_target_excess_mse",
        "Excess risk",
        floor=1e-6,
        color=BLUE,
        marker="o",
        zorder=4,
    )
    dashed_theory(
        axes[5],
        summary,
        "K",
        "calibrated_total_risk_rhs",
        label="Risk-decomp. bound",
        zorder=1,
    )
    line_with_band(
        axes[5],
        summary,
        "K",
        "test_mse",
        "Finite-sample risk",
    )
    for ax in axes:
        ax.set_xscale("log", base=2)
        ticks = [int(x) for x in summary["K"] if x in {64, 128, 256, 512}]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(x) for x in ticks])
    for ax in axes[:3]:
        ax.tick_params(axis="x", which="major", bottom=True, labelbottom=True)
    axes[0].set_yscale("log")
    axes[3].set_yscale("log")
    axes[4].set_yscale("log")
    axes[5].set_yscale("log")
    compact_excess_ticks = [3e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    set_compact_log_tick_labels(axes[3], compact_excess_ticks)
    set_compact_log_tick_labels(axes[4], compact_excess_ticks)
    axes[1].set_ylim(-0.03, 1.03)
    add_axis_legend(axes[0])
    add_axis_legend(axes[1])
    add_axis_legend(axes[2])
    add_axis_legend(axes[3])
    add_axis_legend(axes[4])
    add_axis_legend(axes[5])
    style_axis(axes[0], "Target count $K$", eigen_ylabel, "(a)")
    style_axis(axes[1], "Target count $K$", kappa_ylabel, "(b)")
    style_axis(
        axes[2],
        "Target count $K$",
        effdim_ylabel,
        "(c)",
    )
    for ax in axes[:3]:
        ax.xaxis.labelpad = -0.5
    style_axis(
        axes[3],
        "Target count $K$",
        "Per-target excess risk",
        "(d)",
    )
    style_axis(
        axes[4],
        "Target count $K$",
        "Per-target excess MSE",
        "(e)",
    )
    style_axis(
        axes[5],
        "Target count $K$",
        "Per-target MSE",
        "(f)",
    )
    save_figure(fig, output_path)


def empirical_quantile(values: pd.Series, probability: float) -> float:
    """Return the order-statistic quantile with at least the requested coverage."""

    sorted_values = np.sort(values.to_numpy(dtype=float))
    index = int(np.ceil(probability * len(sorted_values))) - 1
    index = int(np.clip(index, 0, len(sorted_values) - 1))
    return float(sorted_values[index])


def concentration_lemma_summary(
    frame: pd.DataFrame,
    group: str,
    x_value: str,
    lhs_value: str,
    scale: pd.Series | np.ndarray | None = None,
    delta: float = 0.1,
) -> tuple[pd.DataFrame, float, float]:
    """Aggregate empirical LHS and calibrated RHS for concentration lemmas."""

    data = frame.copy()
    if scale is None:
        scale = data["oracle_sigma_top"] * np.sqrt(
            (data["oracle_effdim"] + np.log(1.0 / delta)) / data["n_train"]
        )
    scale = pd.Series(scale, index=data.index, dtype=float)
    c_hat = empirical_quantile(data[lhs_value] / scale, 1.0 - delta)
    data["concentration_lhs"] = data[lhs_value]
    data["concentration_rhs"] = c_hat * scale
    coverage = float(np.mean(data["concentration_lhs"] <= data["concentration_rhs"]))
    summary = (
        data.groupby(group)
        .agg(
            x=(x_value, "mean"),
            concentration_lhs_mean=("concentration_lhs", "mean"),
            concentration_lhs_sem=("concentration_lhs", "sem"),
            concentration_rhs_mean=("concentration_rhs", "mean"),
            concentration_rhs_sem=("concentration_rhs", "sem"),
        )
        .reset_index()
        .sort_values("x")
    )
    return summary, c_hat, coverage


def joint_covariance_concentration_scale(frame: pd.DataFrame, delta: float) -> pd.Series:
    """Joint-covariance scale for cross-covariance noise."""

    sigma_y2 = frame["sigma_y"] ** 2
    sigma_z2 = frame["sigma_z"] ** 2
    target_width = frame["K"] * frame["target_dim"]
    signal_top = frame["oracle_lambda_top"].clip(lower=0.0)
    signal_sv_top = np.sqrt(signal_top)
    two_by_two_trace = sigma_y2 + signal_top + sigma_z2 + 1.0
    two_by_two_gap = sigma_y2 + signal_top - (sigma_z2 + 1.0)
    joint_op = 0.5 * (
        two_by_two_trace + np.sqrt(two_by_two_gap**2 + 4.0 * signal_sv_top**2)
    )
    joint_op = np.maximum.reduce(
        [
            joint_op.to_numpy(dtype=float),
            sigma_y2.to_numpy(dtype=float),
            sigma_z2.to_numpy(dtype=float),
        ]
    )
    joint_op = pd.Series(joint_op, index=frame.index)
    joint_trace = (
        sigma_y2 * target_width
        + sigma_z2 * frame["context_dim"]
        + frame["oracle_trace_h"]
        + frame["r_star"]
    )
    joint_effdim = joint_trace / joint_op.clip(lower=1e-12)
    window = (joint_effdim + np.log(1.0 / delta)) / frame["n_train"]
    return joint_op * (np.sqrt(window) + window)


def operator_concentration_scale(frame: pd.DataFrame, delta: float) -> pd.Series:
    """Operator concentration scale with the additive linear term."""

    window = (frame["oracle_effdim"] + np.log(1.0 / delta)) / frame["n_train"]
    return frame["oracle_sigma_top"] * (np.sqrt(window) + window)


def inverse_sqrt_for_plot(matrix: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    sym = 0.5 * (matrix + matrix.T)
    evals, evecs = np.linalg.eigh(sym)
    evals = np.clip(evals, floor, None)
    return (evecs * (1.0 / np.sqrt(evals))) @ evecs.T


def row_covariance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (y.T @ x) / x.shape[0]


def real_digit_t_concentration_frame(
    sample_sizes: list[int] | None = None,
    seeds: range = range(20),
) -> pd.DataFrame:
    """Empirical real-data proxy for the relative T_K perturbation scale."""

    from sklearn.datasets import load_digits

    if sample_sizes is None:
        sample_sizes = [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1280, 1536]
    digits = load_digits()
    images = digits.images.astype(np.float64) / 16.0
    context = images[:, :, :4].reshape(len(images), -1)
    target = images[:, :, 4:].reshape(len(images), -1)
    sigma_yz_ref = row_covariance(context, target)
    sigma_zz_ref = row_covariance(context, context)
    t_ref = sigma_yz_ref @ inverse_sqrt_for_plot(sigma_zz_ref)
    ref_singular_values = np.linalg.svd(t_ref, compute_uv=False)
    sigma_top = float(ref_singular_values[0])
    effdim = float(np.sum(ref_singular_values**2) / max(sigma_top**2, 1e-12))
    rows = []
    for n_train in sample_sizes:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(context), size=n_train, replace=False)
            z_sub = context[indices]
            y_sub = target[indices]
            sigma_yz_hat = row_covariance(z_sub, y_sub)
            sigma_zz_hat = row_covariance(z_sub, z_sub)
            t_hat = sigma_yz_hat @ inverse_sqrt_for_plot(sigma_zz_hat)
            lhs = float(np.linalg.svd(t_hat - t_ref, compute_uv=False)[0])
            rows.append(
                {
                    "n_train": n_train,
                    "operator_error_op": lhs,
                    "oracle_sigma_top": sigma_top,
                    "oracle_effdim": effdim,
                }
            )
    return pd.DataFrame(rows)


def plot_concentration_lemmas() -> None:
    """LHS/RHS diagnostics for the cross-covariance and T_K perturbations."""

    delta = 0.1
    heterogeneity = pd.read_csv(RESULTS / "exp_heterogeneity.csv")
    post_saturation = pd.read_csv(RESULTS / "exp_post_saturation.csv")
    real_digits = real_digit_t_concentration_frame()
    het_summary, het_c, het_coverage = concentration_lemma_summary(
        heterogeneity,
        group="alpha",
        x_value="oracle_lambda_het_proxy",
        lhs_value="cross_cov_error_op",
        scale=joint_covariance_concentration_scale(heterogeneity, delta),
        delta=delta,
    )
    post_summary, post_c, post_coverage = concentration_lemma_summary(
        post_saturation,
        group="K",
        x_value="K",
        lhs_value="operator_error_op",
        scale=operator_concentration_scale(post_saturation, delta),
        delta=delta,
    )
    real_summary, real_c, real_coverage = concentration_lemma_summary(
        real_digits,
        group="n_train",
        x_value="n_train",
        lhs_value="operator_error_op",
        scale=operator_concentration_scale(real_digits, delta),
        delta=delta,
    )
    fig, axes = make_figure(ncols=3, figsize=(7.1, 2.05))
    for ax, summary in zip(axes, (het_summary, post_summary, real_summary)):
        line_with_band(
            ax,
            summary,
            "x",
            "concentration_lhs",
            "Empirical LHS",
            color=GRAY,
            marker="o",
            zorder=4,
        )
        dashed_theory(
            ax,
            summary,
            "x",
            "concentration_rhs",
            label="Concentration bound",
            zorder=3,
            color=BLUE,
        )
        add_axis_legend(ax)
    _ = (het_c, het_coverage, post_c, post_coverage, real_c, real_coverage)
    style_axis(
        axes[0],
        r"Population $\lambda_{\mathrm{het}}$",
        "Operator norm",
        "(a) Cross-covariance",
    )
    style_axis(axes[1], "Target count $K$", "Operator norm", r"(b) $T_K$ perturbation")
    style_axis(axes[2], "Subsample size $n$", "Operator norm", r"(c) $8{\times}8$ proxy")
    axes[2].set_xscale("log")
    save_figure(fig, FIGURES / "exp_concentration_lemmas.pdf")


def compact_scientific(value: float, precision: int = 1) -> str:
    if np.isclose(value, 1.0):
        return "1"
    if value <= 0:
        return "0"
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / (10**exponent)
    return rf"{mantissa:.{precision}f}\times 10^{{{exponent}}}"


def kappa_condition_label(value: float) -> str:
    return rf"$\kappa_H^2={compact_scientific(value)}$"


def plot_reduction_diagnostics() -> None:
    """Appendix diagnostics for Proposition 3.1 and Lemma 3.2."""

    frame = pd.read_csv(RESULTS / "exp_reduction_diagnostics.csv")
    metrics = [
        "rank_formula",
        "coefficient_rank",
        "factorized_rank",
        "relative_factorization_error",
        "factorized_loss_gap",
        "rrr_ols_gap",
    ]
    summary = frame.groupby(["K", "rank_constraint"], as_index=False)[metrics].agg(
        ["mean", "sem"]
    )
    summary.columns = ["K", "rank_constraint"] + [
        f"{name}_{stat}" for name, stat in summary.columns[2:]
    ]
    fig, axes = make_figure(ncols=3, figsize=(7.1, 2.05))
    rank_values = sorted(summary["rank_constraint"].unique())
    colors = [BLUE, RED, TEAL, GOLD, GRAY]
    for rank, color in zip(rank_values, colors):
        subset = summary[summary["rank_constraint"] == rank]
        axes[0].plot(
            subset["K"],
            subset["coefficient_rank_mean"],
            marker="o",
            markersize=EMPIRICAL_MARKER_SIZE,
            linewidth=EMPIRICAL_LINE_WIDTH,
            color=color,
            label=fr"$r={rank}$",
        )
        axes[0].plot(
            subset["K"],
            subset["rank_formula_mean"],
            linestyle="--",
            linewidth=THEORY_LINE_WIDTH,
            color=color,
            alpha=0.7,
            zorder=1,
        )
    summary_by_rank = mean_sem(
        frame,
        "rank_constraint",
        ["relative_factorization_error", "factorized_loss_gap", "rrr_ols_gap"],
    )
    line_with_band_floor(
        axes[1],
        summary_by_rank,
        "rank_constraint",
        "relative_factorization_error",
        "Coefficient error",
        floor=1e-16,
        color=BLUE,
        marker="o",
    )
    line_with_band_floor(
        axes[2],
        summary_by_rank,
        "rank_constraint",
        "factorized_loss_gap",
        "Factorized - RRR",
        floor=1e-16,
        color=BLUE,
        marker="o",
    )
    line_with_band_floor(
        axes[2],
        summary_by_rank,
        "rank_constraint",
        "rrr_ols_gap",
        "RRR - OLS",
        floor=1e-16,
        color=TEAL,
        marker="^",
    )
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    add_axis_legend(axes[0])
    add_axis_legend(axes[1])
    add_axis_legend(axes[2])
    style_axis(axes[0], "Target count $K$", "Rank", "(a)")
    style_axis(axes[1], "Rank constraint $r$", "Relative error", "(b)")
    style_axis(axes[2], "Rank constraint $r$", "MSE gap", "(c)")
    save_figure(fig, FIGURES / "exp_reduction_diagnostics.pdf")


def plot_predictive_isotropy() -> None:
    frame = pd.read_csv(RESULTS / "exp_predictive_isotropy.csv").copy()
    decays = sorted(frame["spectrum_decay"].unique(), reverse=True)
    order = {decay: idx for idx, decay in enumerate(decays)}
    frame["condition"] = frame["spectrum_decay"].map(order)
    summary = mean_sem(
        frame,
        "condition",
        [
            "spectrum_decay",
            "oracle_relative_conditioning",
            "relative_conditioning",
            "oracle_trace_h",
            "trace_h",
            "operator_error_op",
            "oracle_sigma_at_bottleneck_rank",
            "empirical_subspace_bound_clipped",
            "theorem_subspace_error",
            "theorem_fixed_eval_excess_mse",
            "per_target_excess_mse",
            "test_mse",
            "oracle_test_mse",
        ],
    )
    summary["kappa_x"] = summary["oracle_relative_conditioning_mean"]
    summary = summary.sort_values("kappa_x")

    fig, axes_grid = plt.subplots(2, 2, figsize=(5.6, 4.05), constrained_layout=True)
    axes = list(axes_grid.ravel())
    colors = [BLUE, TEAL, GOLD, RED]
    rank = int(frame["d"].iloc[0])
    eig_index = np.arange(1, rank + 1)
    representative_indices = np.unique(
        np.linspace(0, len(decays) - 1, min(6, len(decays))).round().astype(int)
    )
    representative_decays = [decays[idx] for idx in representative_indices]
    for idx, decay in enumerate(representative_decays):
        trace_h = float(frame.loc[frame["spectrum_decay"] == decay, "oracle_trace_h"].mean())
        weights = decay ** np.arange(rank, dtype=float)
        eigvals = trace_h * weights / np.sum(weights)
        axes[0].plot(
            eig_index,
            eigvals,
            color=colors[idx % len(colors)],
            marker="o",
            markersize=EMPIRICAL_MARKER_SIZE,
            linewidth=EMPIRICAL_LINE_WIDTH,
            label=kappa_condition_label(
                float(frame.loc[frame["spectrum_decay"] == decay, "oracle_relative_conditioning"].mean())
            ),
        )
    axes[0].set_yscale("log")

    dashed_theory(
        axes[1],
        summary,
        "kappa_x",
        "oracle_sigma_at_bottleneck_rank",
        label=r"Weak signal $\sigma_d(T_K)$",
    )
    line_with_band(
        axes[1],
        summary,
        "kappa_x",
        "operator_error_op",
        "Operator perturbation",
        color=BLUE,
    )
    line_with_band(
        axes[2],
        summary,
        "kappa_x",
        "theorem_subspace_error",
        r"Observed",
        color=BLUE,
    )
    line_with_band(
        axes[2],
        summary,
        "kappa_x",
        "empirical_subspace_bound_clipped",
        r"Wedin bound",
        color=RED,
        marker="^",
        zorder=2,
    )
    line_with_band_floor(
        axes[3],
        summary,
        "kappa_x",
        "theorem_fixed_eval_excess_mse",
        "Finite sample",
        floor=1e-5,
        color=BLUE,
        marker="o",
    )

    for ax in axes[1:]:
        ax.set_xscale("log")
    axes[1].set_yscale("log")
    axes[3].set_yscale("log")
    legend = axes[0].legend(
        frameon=True,
        framealpha=0.82,
        edgecolor=GRID,
        facecolor="white",
        loc="lower left",
        ncol=1,
        fontsize=5.1,
        borderpad=0.12,
        handlelength=0.95,
        handletextpad=0.2,
        labelspacing=0.1,
        borderaxespad=0.18,
    )
    legend.get_frame().set_linewidth(0.45)
    add_axis_legend(axes[1])
    add_axis_legend(axes[2])
    add_axis_legend(axes[3])
    style_axis(axes[0], "Eigenvalue index", r"Intrinsic covariance eigenvalues", "(a)")
    style_axis(axes[1], r"Population $\kappa_H^2$", "Perturbation vs. gap", "(b)")
    style_axis(axes[2], r"Population $\kappa_H^2$", "Subspace recovery", "(c)")
    style_axis(axes[3], r"Population $\kappa_H^2$", "Balanced-stack excess MSE", "(d)")
    save_figure(fig, FIGURES / "exp_predictive_isotropy.pdf")


def plot_gauge_factorization() -> None:
    frame = pd.read_csv(RESULTS / "exp_gauge_factorization.csv").copy()
    gauge_order = ["identity", "orthogonal", "diagonal", "gaussianized"]
    gauge_labels = ["Identity", "Orthogonal", "Diagonal", "Gaussianized"]
    law_order = ["gaussian", "laplace"]
    law_labels = {"gaussian": "Gaussian context", "laplace": "Laplace context"}
    metrics = [
        "relative_prediction_error",
        "embedding_cov_isotropy_error",
        "projection_gaussianity_score",
    ]
    summary = frame.groupby(["context_law", "gauge_type"])[metrics].agg(["mean", "sem"])
    x = np.arange(len(gauge_order))
    fig, axes = make_figure(ncols=3, figsize=(7.1, 2.25))
    colors = {"gaussian": BLUE, "laplace": RED}
    offsets = {"gaussian": -0.18, "laplace": 0.18}
    ylabels = [
        "End-to-end relative error",
        "Embedding covariance anisotropy",
        "Projected Gaussianity score",
    ]
    for ax, metric, ylabel, panel in zip(axes, metrics, ylabels, ["(a)", "(b)", "(c)"]):
        for law in law_order:
            means = [
                float(summary.loc[(law, gauge), (metric, "mean")])
                for gauge in gauge_order
            ]
            sems = [
                float(summary.loc[(law, gauge), (metric, "sem")])
                for gauge in gauge_order
            ]
            ax.bar(
                x + offsets[law],
                means,
                yerr=sems,
                color=colors[law],
                edgecolor="white",
                linewidth=0.6,
                capsize=2.0,
                width=0.34,
                label=law_labels[law],
            )
        ax.set_xticks(x, gauge_labels, rotation=25, ha="right")
        style_axis(ax, "Gauge", ylabel, panel)
    axes[0].set_yscale("log")
    axes[2].set_yscale("log")
    axes[0].axhline(1e-10, **GUIDE_STYLE)
    add_axis_legend(axes[0])
    save_figure(fig, FIGURES / "exp_gauge_factorization.pdf")


def plot_regularizer_digits() -> None:
    frame = pd.read_csv(RESULTS / "exp_regularizer_digits.csv").copy()
    frame["generalization_gap"] = frame["test_mse"] - frame["train_mse"]
    order = ["baseline", "encoder_gaussian", "predictive_gaussian", "both"]
    labels = ["No reg.", "Encoder", "Predictive", "Pred. + Enc."]
    metrics = [
        "test_mse",
        "generalization_gap",
    ]
    if "train_seconds" in frame.columns:
        metrics.append("train_seconds")
    metrics.extend(["predictive_gaussianity_score", "predictive_effective_rank"])
    summary = frame.groupby("method")[metrics].agg(["mean", "sem"]).reindex(order)
    x = np.arange(len(order))
    fig, axes = make_figure(ncols=len(metrics), figsize=(7.1, 2.3))
    colors = [GRAY, TEAL, BLUE, RED]
    ylabel_by_metric = {
        "test_mse": "Test pixel MSE",
        "generalization_gap": "Train-test MSE gap",
        "train_seconds": "Train time (s)",
        "predictive_gaussianity_score": "Gaussianity distance",
        "predictive_effective_rank": "Predictive effective rank",
    }
    panels = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    for ax, metric, panel in zip(axes, metrics, panels):
        means = summary[(metric, "mean")].to_numpy()
        sems = summary[(metric, "sem")].fillna(0.0).to_numpy()
        ax.bar(
            x,
            means,
            yerr=sems,
            color=colors,
            edgecolor="white",
            linewidth=0.6,
            capsize=2.0,
            width=0.62,
        )
        ax.set_xticks([])
        style_axis(ax, "", ylabel_by_metric[metric], panel)
        if metric == "predictive_gaussianity_score":
            ax.set_yscale("log")
    handles = [Patch(facecolor=color, edgecolor="white", label=label) for color, label in zip(colors, labels)]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=4,
        frameon=False,
        handlelength=1.0,
        columnspacing=1.1,
        handletextpad=0.35,
    )
    save_figure(fig, FIGURES / "exp_regularizer_digits.pdf")


def plot_bottleneck() -> None:
    frame = pd.read_csv(RESULTS / "exp_bottleneck.csv")
    summary = mean_sem(
        frame,
        "d",
        [
            "excess_risk_over_ols",
            "excess_risk_ratio",
            "rrr_test_mse",
            "spectral_tail_proxy",
            "spectral_tail_ratio",
            "empirical_spectral_tail",
            "empirical_spectral_tail_ratio",
            "subspace_error",
            "recovered_rank",
            "model_effective_rank",
            "oracle_effective_rank",
        ],
    )
    rank_ref = rank_reference(summary)
    fig, axes = make_figure()
    line_with_band(axes[0], summary, "d", "rrr_test_mse", "Finite sample")
    dashed_theory(axes[1], summary, "d", "oracle_effective_rank", label="Population rank")
    line_with_band(axes[1], summary, "d", "model_effective_rank", "Model rank")
    dashed_theory(
        axes[2],
        summary,
        "d",
        "spectral_tail_ratio",
        "Population tail ratio",
        floor=1e-4,
    )
    line_with_band_floor(
        axes[2],
        summary,
        "d",
        "empirical_spectral_tail_ratio",
        "Plug-in tail ratio",
        floor=1e-4,
        color=TEAL,
        marker="^",
        zorder=3,
    )
    line_with_band_floor(
        axes[2],
        summary,
        "d",
        "excess_risk_ratio",
        "Excess-risk ratio",
        floor=1e-4,
        color=BLUE,
        marker="o",
        zorder=5,
    )
    axes[0].axvline(rank_ref, **GUIDE_STYLE)
    axes[1].axvline(rank_ref, **GUIDE_STYLE)
    axes[2].axvline(rank_ref, **GUIDE_STYLE)
    axes[0].set_yscale("log")
    axes[2].set_yscale("log")
    axes[2].axhline(1e-4, **GUIDE_STYLE)
    axes[2].set_ylim(bottom=7.5e-5, top=1.2)
    style_rank_axis(axes[1], rank_ref)
    add_axis_legend(axes[0])
    add_axis_legend(axes[1])
    add_axis_legend(axes[2])
    style_axis(axes[0], "Bottleneck dimension $d$", "Per-target MSE", "(a)")
    style_axis(axes[1], "Bottleneck dimension $d$", "Model rank", "(b)")
    style_axis(axes[2], "Bottleneck dimension $d$", "Normalized excess risk / tail", "(c)")
    save_figure(fig, FIGURES / "exp_bottleneck.pdf")


def plot_unified_risk() -> None:
    frame = pd.read_csv(RESULTS / "exp_unified_risk.csv").copy()
    delta = 0.1
    rho_by_n = frame.groupby("n_train")["normalized_operator_error"].apply(
        lambda values: empirical_quantile(values, 1.0 - delta)
    )
    coverage_by_n = {
        n_train: float(
            np.mean(group["normalized_operator_error"] <= rho_by_n.loc[n_train])
        )
        for n_train, group in frame.groupby("n_train")
    }
    frame["rho_hat"] = frame["n_train"].map(rho_by_n)
    frame["rho_coverage"] = frame["n_train"].map(coverage_by_n)
    empirical_l_bound = frame["rank_r_risk_stability_ratio"].max()
    frame["retained_recovery_bound"] = (
        frame["rho_hat"]
        / (frame["retained_relative_gap"] - frame["rho_hat"]).clip(lower=1e-12)
    )
    frame["rank_r_lipschitz_bound"] = (
        empirical_l_bound * frame["retained_subspace_error"]
    )
    frame["unified_excess_bound"] = (
        empirical_l_bound * frame["retained_recovery_bound"]
    )
    frame["unified_risk_bound"] = (
        frame["population_rank_r_mse"] + frame["unified_excess_bound"]
    )
    metrics = [
        "test_mse",
        "oracle_test_mse",
        "population_tail_per_target",
        "population_rank_r_mse",
        "rank_r_excess_mse",
        "normalized_operator_error",
        "rho_hat",
        "sigma_at_bottleneck_rank",
        "sigma_after_bottleneck_rank",
        "oracle_sigma_at_bottleneck_rank",
        "oracle_sigma_after_bottleneck_rank",
        "retained_relative_gap",
        "retained_subspace_error",
        "retained_recovery_bound",
        "rank_r_lipschitz_bound",
        "unified_excess_bound",
        "unified_risk_bound",
        "rho_coverage",
    ]
    summary = mean_sem(frame, "n_train", metrics)
    summary.to_csv(RESULTS / "unified_risk_calibrated_bounds.csv", index=False)

    fig, axes_grid = plt.subplots(2, 3, figsize=(7.1, 3.3), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.015, wspace=0.02, hspace=0.028)
    axes = list(axes_grid.ravel())

    dashed_theory(
        axes[0],
        summary,
        "n_train",
        "oracle_sigma_at_bottleneck_rank",
        label=r"Pop. $\sigma_r$",
        zorder=1,
    )
    dashed_theory(
        axes[0],
        summary,
        "n_train",
        "oracle_sigma_after_bottleneck_rank",
        label=r"Pop. $\sigma_{r+1}$",
        zorder=1,
    )
    line_with_band(
        axes[0],
        summary,
        "n_train",
        "sigma_at_bottleneck_rank",
        r"Finite $\sigma_r$",
        color=RED,
        marker="s",
        zorder=4,
    )
    line_with_band(
        axes[0],
        summary,
        "n_train",
        "sigma_after_bottleneck_rank",
        r"Finite $\sigma_{r+1}$",
        color=BLUE,
        marker="o",
        zorder=5,
    )

    line_with_band(axes[1], summary, "n_train", "test_mse", "Finite sample")
    dashed_theory(
        axes[1],
        summary,
        "n_train",
        "population_rank_r_mse",
        label=r"Population rank-$r$",
    )
    dashed_theory(
        axes[1],
        summary,
        "n_train",
        "oracle_test_mse",
        label="Raw OLS floor",
        color=GRAY,
        linestyle=":",
        alpha=0.75,
    )

    line_with_band(
        axes[2],
        summary,
        "n_train",
        "normalized_operator_error",
        r"Observed $\hat\eta$",
        color=BLUE,
    )
    axes[2].plot(
        summary["n_train"],
        summary["rho_hat_mean"],
        color=TEAL,
        linestyle="--",
        linewidth=THEORY_LINE_WIDTH,
        label=r"$\hat\rho_{0.9}$",
    )
    axes[2].plot(
        summary["n_train"],
        summary["retained_relative_gap_mean"],
        color=THEORY,
        linestyle="--",
        linewidth=THEORY_LINE_WIDTH,
        label=r"Gap $\Delta_{K,r}$",
    )

    line_with_band(
        axes[3],
        summary,
        "n_train",
        "retained_subspace_error",
        r"Observed",
        color=BLUE,
    )
    dashed_theory(
        axes[3],
        summary,
        "n_train",
        "retained_recovery_bound",
        label="Wedin bound",
    )

    line_with_band_floor(
        axes[4],
        summary,
        "n_train",
        "rank_r_excess_mse",
        r"Excess over $R_K^\star(r)$",
        floor=1e-4,
        color=BLUE,
        marker="o",
    )
    dashed_theory(
        axes[4],
        summary,
        "n_train",
        "rank_r_lipschitz_bound",
        label="Lipschitz bound",
        floor=1e-4,
    )

    line_with_band(axes[5], summary, "n_train", "test_mse", "Finite sample")
    dashed_theory(
        axes[5],
        summary,
        "n_train",
        "unified_risk_bound",
        label="Calibrated bound",
    )
    dashed_theory(
        axes[5],
        summary,
        "n_train",
        "population_rank_r_mse",
        label=r"Population rank-$r$",
        color=GRAY,
        linestyle=":",
        alpha=0.75,
    )

    for ax in axes:
        ax.set_xscale("log", base=2)
        ticks = [512, 2048, 8192, 16384]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["512", "2k", "8k", "16k"])
    for ax in axes[:3]:
        ax.tick_params(axis="x", which="major", bottom=True, labelbottom=True)
    axes[2].set_yscale("log")
    axes[3].set_yscale("log")
    axes[4].set_yscale("log")
    axes[5].set_yscale("log")
    add_axis_legend(axes[0])
    add_axis_legend(axes[1])
    add_axis_legend(axes[2])
    add_axis_legend(axes[3])
    add_axis_legend(axes[4])
    add_axis_legend(axes[5])
    style_axis(axes[0], "Training samples $n$", "Gap SVs", "(a)")
    style_axis(axes[1], "Training samples $n$", "Per-target MSE", "(b)")
    style_axis(axes[2], "Training samples $n$", "Rel. perturbation", "(c)")
    style_axis(axes[3], "Training samples $n$", "Subspace error", "(d)")
    style_axis(axes[4], "Training samples $n$", "Excess MSE", "(e)")
    style_axis(axes[5], "Training samples $n$", "Per-target MSE", "(f)")
    save_figure(fig, FIGURES / "exp_unified_risk.pdf")


def main() -> None:
    configure_matplotlib()
    plot_k_saturation()
    plot_heterogeneity()
    plot_bottleneck()
    plot_unified_risk()
    plot_post_saturation()
    plot_concentration_lemmas()
    plot_reduction_diagnostics()
    plot_predictive_isotropy()
    plot_gauge_factorization()
    plot_regularizer_digits()


if __name__ == "__main__":
    main()
