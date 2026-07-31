"""Create submission-ready figures for the formal-v3 KGR-LDP experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "formal_v3"
FIGURE_ROOT = PROJECT_ROOT / "figures" / "formal_v3"
DATASETS = ["PIC", "NHANES_2021_2023"]
DISPLAY_DATASET = {"PIC": "PIC", "NHANES_2021_2023": "NHANES 2021--2023"}
EPSILONS = [0.1, 0.25, 0.5, 1, 2, 4, 8]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.4,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def _save(fig: plt.Figure, stem: str) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        "svg": {},
        "pdf": {},
        "tiff": {"dpi": 600},
        "png": {"dpi": 240},
    }.items():
        fig.savefig(FIGURE_ROOT / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def _full_cohort_summary(dataset: str, family: str) -> pd.DataFrame:
    path = RESULT_ROOT / dataset / family / "summary_bootstrap_ci.csv"
    frame = pd.read_csv(path)
    return frame[frame["sample_size"].eq(frame["sample_size"].max())].copy()


def _panel_grid() -> tuple[plt.Figure, object]:
    return plt.subplots(2, 2, figsize=(7.2, 5.15), sharex=True, sharey=True)


def plot_protocol_comparison() -> pd.DataFrame:
    colors = {"SemMed+Smp": "#356DB2", "SemMed+Spl": "#8C8C8C", "SemMed+RS+FD": "#D47A43"}
    markers = {"SemMed+Smp": "o", "SemMed+Spl": "s", "SemMed+RS+FD": "^"}
    labels = {
        "SemMed+Smp": "KGR-LDP+Smp",
        "SemMed+Spl": "KGR-LDP+Spl",
        "SemMed+RS+FD": "KGR-LDP+RS+FD",
    }
    fig, axes = _panel_grid()
    source = []
    for row, dataset in enumerate(DATASETS):
        frame = _full_cohort_summary(dataset, "protocol_comparison")
        source.append(frame.assign(figure="protocol_comparison"))
        for column, mechanism in enumerate(["GRR", "OUE"]):
            ax = axes[row, column]
            local = frame[frame["mechanism"].eq(mechanism)]
            for method in colors:
                values = local[local["method"].eq(method)].sort_values("epsilon")
                x = values["epsilon"].to_numpy(float)
                mean = values["MSE_mean"].to_numpy(float)
                low = values["MSE_ci_low"].to_numpy(float)
                high = values["MSE_ci_high"].to_numpy(float)
                ax.plot(
                    x,
                    mean,
                    color=colors[method],
                    marker=markers[method],
                    ms=3.2,
                    markeredgecolor="white",
                    markeredgewidth=0.35,
                    label=labels[method],
                )
                ax.fill_between(x, low, high, color=colors[method], alpha=0.12, linewidth=0)
            ax.set_xscale("log", base=2)
            ax.set_xticks(EPSILONS, [str(value) for value in EPSILONS])
            ax.set_yscale("log")
            ax.grid(axis="y", color="#D9D9D9", lw=0.45, alpha=0.8)
            ax.set_title(f"{DISPLAY_DATASET[dataset]} | {mechanism}")
            ax.text(
                -0.14,
                1.03,
                chr(ord("a") + row * 2 + column),
                transform=ax.transAxes,
                fontsize=8,
                fontweight="bold",
                va="bottom",
            )
            if row == 1:
                ax.set_xlabel(r"Privacy budget $\epsilon$")
            if column == 0:
                ax.set_ylabel("Mean squared error")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(top=0.89, hspace=0.30, wspace=0.22)
    _save(fig, "fig_protocol_comparison")
    return pd.concat(source, ignore_index=True)


def plot_lopub_common_query_comparison() -> pd.DataFrame:
    """Compare KGR-LDP and a LoPub-derived baseline under paired Smp."""
    path = RESULT_ROOT / "lopub_common_query_summary.csv"
    frame = pd.read_csv(path)
    styles = {
        ("SemMed-LDP", "GRR"): {
            "label": "KGR-LDP + GRR",
            "color": "#356DB2",
            "marker": "o",
            "linestyle": "-",
        },
        ("SemMed-LDP", "OUE"): {
            "label": "KGR-LDP + OUE",
            "color": "#6F96C8",
            "marker": "^",
            "linestyle": "--",
        },
        ("LoPub-derived-Bloom-EM", "Bloom-RR"): {
            "label": "LoPub-derived Bloom-EM",
            "color": "#D47A43",
            "marker": "D",
            "linestyle": "-.",
        },
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), sharex=True, sharey=False)
    for panel, dataset in enumerate(DATASETS):
        ax = axes[panel]
        local = frame[frame["dataset_id"].eq(dataset)]
        for key, style in styles.items():
            method, mechanism = key
            values = local[
                local["method"].eq(method) & local["mechanism"].eq(mechanism)
            ].sort_values("epsilon")
            x = values["epsilon"].to_numpy(float)
            mean = values["MSE_mean"].to_numpy(float)
            low = values["MSE_ci_low"].to_numpy(float)
            high = values["MSE_ci_high"].to_numpy(float)
            ax.plot(
                x,
                mean,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                ms=3.4,
                markeredgecolor="white",
                markeredgewidth=0.35,
                label=style["label"],
            )
            ax.fill_between(x, low, high, color=style["color"], alpha=0.12, linewidth=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(EPSILONS, [str(value) for value in EPSILONS])
        ax.set_yscale("log")
        ax.grid(axis="y", color="#D9D9D9", lw=0.45, alpha=0.8)
        ax.set_title(f"{DISPLAY_DATASET[dataset]} | full cohort")
        ax.set_xlabel(r"Privacy budget $\epsilon$")
        ax.set_ylabel("Common-query mean squared error")
        ax.text(
            -0.12,
            1.04,
            chr(ord("a") + panel),
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(top=0.82, wspace=0.28)
    _save(fig, "fig_lopub_common_query")
    return frame.assign(figure="lopub_common_query")


def _paired_bootstrap_summary(
    baseline: np.ndarray,
    semmed: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, float]:
    """Summarize paired run-level MSE contrasts by paired resampling."""
    if baseline.shape != semmed.shape or baseline.ndim != 1:
        raise ValueError("paired MSE arrays must have the same one-dimensional shape")
    draws = rng.integers(0, baseline.size, size=(n_bootstrap, baseline.size))
    baseline_draws = baseline[draws]
    semmed_draws = semmed[draws]
    mean_difference = (baseline_draws - semmed_draws).mean(axis=1)
    relative_improvement = (
        100.0
        * (baseline_draws.mean(axis=1) - semmed_draws.mean(axis=1))
        / baseline_draws.mean(axis=1)
    )
    return {
        "paired_mean_difference_ci_low": float(np.quantile(mean_difference, 0.025)),
        "paired_mean_difference_ci_high": float(np.quantile(mean_difference, 0.975)),
        "relative_MSE_improvement_ci_low": float(np.quantile(relative_improvement, 0.025)),
        "relative_MSE_improvement_ci_high": float(np.quantile(relative_improvement, 0.975)),
    }


def _holm_adjust_within_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply Holm correction to the 14 LoPub comparisons within each cohort."""
    adjusted_frames = []
    for _, local in frame.groupby("dataset_id", sort=False):
        local = local.copy()
        order = np.argsort(local["wilcoxon_p_value"].to_numpy(float))
        adjusted = np.ones(len(local), dtype=float)
        running = 0.0
        for rank, index in enumerate(order):
            candidate = min(
                1.0,
                float(local.iloc[index]["wilcoxon_p_value"]) * (len(local) - rank),
            )
            running = max(running, candidate)
            adjusted[index] = running
        local["holm_adjusted_p"] = adjusted
        local["holm_significant_0_05"] = local["holm_adjusted_p"] < 0.05
        adjusted_frames.append(local)
    return pd.concat(adjusted_frames, ignore_index=True)


def build_lopub_paired_statistics(
    *,
    seed: int = 2026,
    n_bootstrap: int = 20_000,
) -> pd.DataFrame:
    """Compare each KGR-LDP randomizer with Bloom-EM over paired runs."""
    rows: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(seed)
    for dataset_id in DATASETS:
        path = RESULT_ROOT / dataset_id / "lopub_joint_distribution" / "run_metrics.csv"
        metrics = pd.read_csv(path)
        baseline = metrics[
            metrics["method"].eq("LoPub-derived-Bloom-EM")
            & metrics["mechanism"].eq("Bloom-RR")
        ][["epsilon", "run", "seed", "MSE"]].rename(columns={"MSE": "MSE_baseline"})
        for mechanism in ["GRR", "OUE"]:
            semmed = metrics[
                metrics["method"].eq("SemMed-LDP")
                & metrics["mechanism"].eq(mechanism)
            ][["epsilon", "run", "seed", "MSE"]].rename(columns={"MSE": "MSE_semmed"})
            paired = semmed.merge(
                baseline,
                on=["epsilon", "run", "seed"],
                how="inner",
                validate="one_to_one",
            )
            for epsilon, local in paired.groupby("epsilon", sort=True):
                semmed_values = local["MSE_semmed"].to_numpy(float)
                baseline_values = local["MSE_baseline"].to_numpy(float)
                differences = baseline_values - semmed_values
                statistic, p_value = wilcoxon(
                    semmed_values,
                    baseline_values,
                    alternative="two-sided",
                )
                ranks = rankdata(np.abs(differences))
                positive = ranks[differences > 0].sum()
                negative = ranks[differences < 0].sum()
                rank_biserial = (
                    float((positive - negative) / (positive + negative))
                    if positive + negative
                    else 0.0
                )
                baseline_mean = float(baseline_values.mean())
                semmed_mean = float(semmed_values.mean())
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "semmed_mechanism": mechanism,
                        "epsilon": float(epsilon),
                        "n_paired_runs": len(local),
                        "MSE_semmed_mean": semmed_mean,
                        "MSE_bloom_em_mean": baseline_mean,
                        "bloom_em_to_semmed_MSE_ratio": baseline_mean / semmed_mean,
                        "relative_MSE_improvement_percent": (
                            100.0 * (baseline_mean - semmed_mean) / baseline_mean
                        ),
                        "paired_mean_difference": float(differences.mean()),
                        "paired_median_difference": float(np.median(differences)),
                        "rank_biserial_effect": rank_biserial,
                        "wilcoxon_statistic": float(statistic),
                        "wilcoxon_p_value": float(p_value),
                        **_paired_bootstrap_summary(
                            baseline_values,
                            semmed_values,
                            rng=rng,
                            n_bootstrap=n_bootstrap,
                        ),
                        "bootstrap_replicates": n_bootstrap,
                    }
                )
    return _holm_adjust_within_dataset(pd.DataFrame(rows)).sort_values(
        ["dataset_id", "semmed_mechanism", "epsilon"]
    ).reset_index(drop=True)


def plot_representation_comparison() -> pd.DataFrame:
    colors = {
        "SemMed-LDP": "#3B6FB6",
        "Matched-Random-4D": "#A7A9AC",
        "External-Correlation-4D": "#6C9A76",
    }
    labels = {
        "SemMed-LDP": "KGR-LDP 4D",
        "Matched-Random-4D": "100 matched random 4D",
        "External-Correlation-4D": "external-correlation 4D",
    }
    markers = {"SemMed-LDP": "o", "Matched-Random-4D": "s", "External-Correlation-4D": "D"}
    fig, axes = _panel_grid()
    source = []
    for row, dataset in enumerate(DATASETS):
        frame = _full_cohort_summary(dataset, "representation_comparison")
        source.append(frame.assign(figure="representation_comparison"))
        for column, mechanism in enumerate(["GRR", "OUE"]):
            ax = axes[row, column]
            local = frame[frame["mechanism"].eq(mechanism)]
            for method in colors:
                values = local[local["method"].eq(method)].sort_values("epsilon")
                x = values["epsilon"].to_numpy(float)
                mean = values["MSE_mean"].to_numpy(float)
                low = values["MSE_ci_low"].to_numpy(float)
                high = values["MSE_ci_high"].to_numpy(float)
                ax.plot(x, mean, color=colors[method], marker=markers[method], ms=3, label=labels[method])
                ax.fill_between(x, low, high, color=colors[method], alpha=0.12, linewidth=0)
            ax.set_xscale("log", base=2)
            ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 4, 8], ["0.1", "0.25", "0.5", "1", "2", "4", "8"])
            ax.set_yscale("log")
            ax.grid(axis="y", color="#D9D9D9", lw=0.45, alpha=0.8)
            ax.set_title(f"{DISPLAY_DATASET[dataset]} | {mechanism}")
            if row == 1:
                ax.set_xlabel(r"Privacy budget $\epsilon$")
            if column == 0:
                ax.set_ylabel("Mean squared error")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(top=0.89, hspace=0.30, wspace=0.22)
    _save(fig, "fig_representation_comparison")
    return pd.concat(source, ignore_index=True)


def plot_resolution_tradeoff() -> pd.DataFrame:
    colors = {
        "Raw-Attribute-LDP": "#7F7F7F",
        "LOINC-Group-LDP": "#B3A16B",
        "SemMed-LDP": "#3B6FB6",
        "Source-Schema-LDP": "#6C9A76",
        "LOINC-Class-LDP": "#B96B72",
    }
    labels = {
        "Raw-Attribute-LDP": "Raw 10D",
        "LOINC-Group-LDP": "LOINC Group 10D",
        "SemMed-LDP": "KGR-LDP 4D",
        "Source-Schema-LDP": "Source-Schema 2--3D",
        "LOINC-Class-LDP": "LOINC Class 2D",
    }
    markers = {method: marker for method, marker in zip(colors, ["o", "s", "^", "D", "P"])}
    fig, axes = _panel_grid()
    source = []
    for row, dataset in enumerate(DATASETS):
        frame = _full_cohort_summary(dataset, "resolution_tradeoff")
        source.append(frame.assign(figure="resolution_tradeoff"))
        for column, mechanism in enumerate(["GRR", "OUE"]):
            ax = axes[row, column]
            local = frame[frame["mechanism"].eq(mechanism)]
            for method in colors:
                values = local[local["method"].eq(method)].sort_values("epsilon")
                ax.plot(
                    values["epsilon"],
                    values["MSE_mean"],
                    color=colors[method],
                    marker=markers[method],
                    ms=3,
                    label=labels[method],
                )
            ax.set_xscale("log", base=2)
            ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 4, 8], ["0.1", "0.25", "0.5", "1", "2", "4", "8"])
            ax.set_yscale("log")
            ax.grid(axis="y", color="#D9D9D9", lw=0.45, alpha=0.8)
            ax.set_title(f"{DISPLAY_DATASET[dataset]} | {mechanism}")
            if row == 1:
                ax.set_xlabel(r"Privacy budget $\epsilon$")
            if column == 0:
                ax.set_ylabel("Mean squared error")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(top=0.88, hspace=0.30, wspace=0.22)
    _save(fig, "fig_resolution_tradeoff")
    return pd.concat(source, ignore_index=True)


def main() -> None:
    source_root = FIGURE_ROOT / "source_data"
    source_root.mkdir(parents=True, exist_ok=True)
    plot_protocol_comparison().to_csv(source_root / "fig_protocol_comparison.csv", index=False)
    plot_lopub_common_query_comparison().to_csv(
        source_root / "fig_lopub_common_query.csv", index=False
    )
    build_lopub_paired_statistics().to_csv(
        source_root / "fig_lopub_common_query_paired_statistics.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
