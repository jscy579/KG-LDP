"""Plot KGR-LDP relative MSE improvement over the two fixed-4D controls."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semmed_ldp.formal_experiment import sha256_file


RESULT_ROOT = PROJECT_ROOT / "results" / "formal_v3"
FIGURE_ROOT = PROJECT_ROOT / "figures" / "formal_v3"
DATASETS = ["PIC", "NHANES_2021_2023"]
DISPLAY_DATASET = {"PIC": "PIC", "NHANES_2021_2023": "NHANES 2021--2023"}
BASELINES = ["Matched-Random-4D", "External-Correlation-4D"]
LABELS = {
    "Matched-Random-4D": "Random grouping",
    "External-Correlation-4D": "Cross-cohort correlation grouping",
}
COLORS = {
    "Matched-Random-4D": "#D07A3E",
    "External-Correlation-4D": "#4F8871",
}
MARKERS = {"Matched-Random-4D": "o", "External-Correlation-4D": "D"}
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
        "lines.linewidth": 1.35,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def _input_path(dataset_id: str, filename: str) -> Path:
    return RESULT_ROOT / dataset_id / "representation_comparison" / filename


def _relative_improvement(baseline: np.ndarray, proposed: np.ndarray) -> float:
    """Return 100 * (mean baseline - mean proposed) / mean baseline."""
    baseline_mean = float(np.mean(baseline))
    if baseline_mean <= 0:
        raise ValueError("baseline mean MSE must be positive")
    return 100.0 * (baseline_mean - float(np.mean(proposed))) / baseline_mean


def _paired_bootstrap_ci(
    baseline: np.ndarray,
    proposed: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float]:
    """Percentile CI for the ratio-of-means improvement using paired resampling."""
    if baseline.shape != proposed.shape or baseline.ndim != 1:
        raise ValueError("paired baseline and proposed arrays must have the same 1D shape")
    draws = rng.integers(0, baseline.size, size=(n_bootstrap, baseline.size))
    baseline_means = baseline[draws].mean(axis=1)
    proposed_means = proposed[draws].mean(axis=1)
    boot = 100.0 * (baseline_means - proposed_means) / baseline_means
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def build_source_data(*, seed: int = 2026, n_bootstrap: int = 20_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str | bool]] = []
    pair_keys = ["mechanism", "epsilon", "sample_size", "run", "seed"]

    for dataset_id in DATASETS:
        metrics = pd.read_csv(_input_path(dataset_id, "inferential_run_metrics.csv"))
        metrics = metrics[
            metrics["sample_size"].eq(metrics["sample_size"].max())
            & metrics["method"].isin(["SemMed-LDP", *BASELINES])
        ].copy()
        tests = pd.read_csv(_input_path(dataset_id, "paired_wilcoxon_holm.csv"))
        tests = tests[
            tests["sample_size"].eq(tests["sample_size"].max())
            & tests["metric"].eq("MSE")
            & tests["baseline"].isin(BASELINES)
        ].copy()

        semmed = metrics[metrics["method"].eq("SemMed-LDP")][pair_keys + ["MSE"]].rename(
            columns={"MSE": "MSE_semmed"}
        )
        for baseline in BASELINES:
            baseline_rows = metrics[metrics["method"].eq(baseline)][pair_keys + ["MSE"]].rename(
                columns={"MSE": "MSE_baseline"}
            )
            paired = semmed.merge(baseline_rows, on=pair_keys, how="inner", validate="one_to_one")
            for (mechanism, epsilon), group in paired.groupby(["mechanism", "epsilon"]):
                baseline_values = group["MSE_baseline"].to_numpy(float)
                semmed_values = group["MSE_semmed"].to_numpy(float)
                estimate = _relative_improvement(baseline_values, semmed_values)
                ci_low, ci_high = _paired_bootstrap_ci(
                    baseline_values,
                    semmed_values,
                    rng=rng,
                    n_bootstrap=n_bootstrap,
                )
                test = tests[
                    tests["mechanism"].eq(mechanism)
                    & np.isclose(tests["epsilon"], epsilon)
                    & tests["baseline"].eq(baseline)
                ]
                if len(test) != 1:
                    raise ValueError(
                        f"expected one Holm-adjusted test for {dataset_id}, "
                        f"{mechanism}, epsilon={epsilon}, {baseline}"
                    )
                test_row = test.iloc[0]
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "mechanism": mechanism,
                        "epsilon": float(epsilon),
                        "baseline": baseline,
                        "n_paired_runs": len(group),
                        "MSE_semmed_mean": float(semmed_values.mean()),
                        "MSE_baseline_mean": float(baseline_values.mean()),
                        "relative_MSE_improvement_percent": estimate,
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "bootstrap_replicates": n_bootstrap,
                        "wilcoxon_p_value": float(test_row["p_value"]),
                        "holm_adjusted_p": float(test_row["p_holm"]),
                        "holm_significant_0_05": bool(test_row["p_holm"] < 0.05),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["dataset_id", "mechanism", "baseline", "epsilon"]
    ).reset_index(drop=True)


def plot_relative_improvement(source: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.05), sharex=True, sharey=True)
    for row, dataset_id in enumerate(DATASETS):
        for column, mechanism in enumerate(["GRR", "OUE"]):
            ax = axes[row, column]
            local = source[
                source["dataset_id"].eq(dataset_id) & source["mechanism"].eq(mechanism)
            ]
            #ax.axhspan(0, 40, color="#E8F1EC", alpha=0.55, zorder=0)
            #ax.axhspan(-35, 0, color="#F2F2F2", alpha=0.7, zorder=0)
            ax.axhline(0, color="#444444", lw=0.8, zorder=1)

            for baseline in BASELINES:
                values = local[local["baseline"].eq(baseline)].sort_values("epsilon")
                x = values["epsilon"].to_numpy(float)
                estimate = values["relative_MSE_improvement_percent"].to_numpy(float)
                low = values["bootstrap_ci_low"].to_numpy(float)
                high = values["bootstrap_ci_high"].to_numpy(float)
                ax.plot(
                    x,
                    estimate,
                    color=COLORS[baseline],
                    marker=MARKERS[baseline],
                    ms=3.4,
                    label=LABELS[baseline],
                    zorder=3,
                )
                ax.fill_between(
                    x,
                    low,
                    high,
                    color=COLORS[baseline],
                    alpha=0.14,
                    linewidth=0,
                    zorder=2,
                )
            ax.set_xscale("log", base=2)
            ax.set_xticks(EPSILONS, ["0.1", "0.25", "0.5", "1", "2", "4", "8"])
            ax.set_ylim(-35, 40)
            ax.set_yticks([-30, -20, -10, 0, 10, 20, 30, 40])
            ax.grid(axis="y", color="#D7D7D7", lw=0.45, alpha=0.85)
            ax.set_title(f"{DISPLAY_DATASET[dataset_id]} | {mechanism}")
            ax.text(
                -0.12,
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
                ax.set_ylabel("Relative MSE improvement of KGR-LDP (%)")
            ax.text(
                0.98,
                0.965,
                "KGR-LDP better",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6.2,
                color="#446B56",
            )
            ax.text(
                0.98,
                0.04,
                "baseline better",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.2,
                color="#666666",
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(top=0.89, bottom=0.12, hspace=0.30, wspace=0.20)
    return fig


def _save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, kwargs in {
        "svg": {},
        "pdf": {},
        "tiff": {"dpi": 600},
        "png": {"dpi": 240},
    }.items():
        path = FIGURE_ROOT / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    seed = 2026
    n_bootstrap = 20_000
    stem = "fig_relative_mse_improvement"
    source = build_source_data(seed=seed, n_bootstrap=n_bootstrap)
    source_root = FIGURE_ROOT / "source_data"
    source_root.mkdir(parents=True, exist_ok=True)
    source_path = source_root / f"{stem}.csv"
    source.to_csv(source_path, index=False)
    outputs = _save_figure(plot_relative_improvement(source), stem)

    inputs = [
        _input_path(dataset_id, filename)
        for dataset_id in DATASETS
        for filename in ["inferential_run_metrics.csv", "paired_wilcoxon_holm.csv"]
    ]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "backend": "Python/matplotlib",
        "metric": "100 * (mean MSE_baseline - mean MSE_KGR-LDP) / mean MSE_baseline",
        "interval": "paired percentile 95% Monte Carlo interval over 30 randomized-run pairs",
        "bootstrap_seed": seed,
        "bootstrap_replicates": n_bootstrap,
        "significance": "not encoded in the figure",
        "inputs": [
            {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(path)}
            for path in inputs
        ],
        "outputs": [
            {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(path)}
            for path in [source_path, *outputs]
        ],
    }
    manifest_path = FIGURE_ROOT / f"{stem}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    qa_text = f"""# Figure QA: relative MSE improvement

- Core conclusion: KGR-LDP has conditional, not universal, MSE advantages across privacy budgets.
- Archetype: quantitative 2 x 2 robustness grid.
- Backend: Python/matplotlib only.
- Inputs: full-cohort inferential run metrics; 30 paired repeat-level observations per comparison.
- Center: relative improvement computed from paired-run mean MSEs.
- Interval: paired percentile 95% Monte Carlo interval, {n_bootstrap:,} resamples, seed {seed}.
- Significance markers: none; inferential results are reported in the manuscript text.
- Positive values: KGR-LDP lower MSE. Negative values: named baseline lower MSE.
- Random control: each repeat is the mean of 100 matched random partitions; partitions are not treated as independent replicates.
- Export: editable SVG and PDF; 600 dpi TIFF; PNG preview.
- Review limitation: the two controls are structural controls, not published end-to-end competing methods.
"""
    (FIGURE_ROOT / f"{stem}_QA.md").write_text(qa_text, encoding="utf-8")


if __name__ == "__main__":
    main()
