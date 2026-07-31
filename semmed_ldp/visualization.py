"""Plotting utilities for pilot and formal SemMed-LDP manuscript results."""

from __future__ import annotations

import logging
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
except ModuleNotFoundError:  # pragma: no cover - optional plotting fallback.
    plt = None
    FancyBboxPatch = None
    FancyArrowPatch = None

try:
    import seaborn as sns
except ModuleNotFoundError:  # pragma: no cover - optional plotting fallback.
    sns = None

LOGGER = logging.getLogger(__name__)

METHOD_COLORS = {
    "SemMed-LDP": "#0F4D92",
    "Full-SemMed-LDP": "#0F4D92",
    "Raw-Attribute-LDP": "#8F8F8F",
    "LOINC-Groups-LDP": "#B4C0E4",
    "Random-Grouped-LDP": "#7884B4",
    "Correlation-Grouped-LDP-oracle": "#9A4D8E",
    "Type-Grouped-LDP": "#4D4D4D",
    "No-HPO-LOINC-Groups": "#B4C0E4",
    "Shuffled-Knowledge-Edges": "#7884B4",
    "Anchor-Count-3": "#42949E",
    "Anchor-Count-2": "#767676",
    "No-UBERON-Structural-Negative-Control": "#E9A6A1",
}

METHOD_LABELS = {
    "SemMed-LDP": "SemMed-LDP (4D)",
    "Full-SemMed-LDP": "SemMed-LDP (4D)",
    "Raw-Attribute-LDP": "Raw attributes (10D)",
    "LOINC-Groups-LDP": "LOINC groups (10D)",
    "Random-Grouped-LDP": "Random groups (4D)",
    "Correlation-Grouped-LDP-oracle": "Correlation oracle (4D)",
    "Type-Grouped-LDP": "Type groups (2--3D)",
    "No-HPO-LOINC-Groups": "LOINC groups (10D)",
    "Shuffled-Knowledge-Edges": "Shuffled groups (4D)",
    "Anchor-Count-3": "Three anchors",
    "Anchor-Count-2": "Two anchors",
    "No-UBERON-Structural-Negative-Control": "No UBERON (4D)",
}


def apply_publication_style() -> None:
    """Apply the project-wide editable-vector manuscript style."""
    if plt is None:
        raise RuntimeError("matplotlib is required for manuscript figures")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
        }
    )


def _save_publication_figure(fig, output_base: str | Path) -> list[Path]:
    """Export editable vectors, print raster, and a lightweight preview."""
    base = _prepare_output(output_base).with_suffix("")
    outputs = [base.with_suffix(ext) for ext in [".svg", ".pdf", ".tiff", ".png"]]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    fig.savefig(outputs[2], dpi=600, bbox_inches="tight")
    fig.savefig(outputs[3], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outputs


def _panel_label(ax, label: str) -> None:
    ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def _draw_box(ax, x: float, y: float, width: float, height: float, text: str, color: str) -> None:
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=0.8, edgecolor="#4D4D4D", facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=6.5)


def _draw_arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=0.9, color="#4D4D4D"))


def plot_formal_framework(mapping: pd.DataFrame, output_base: str | Path) -> list[Path]:
    """Create the manuscript workflow and real ontology-group example."""
    apply_publication_style()
    local = mapping.loc[mapping["dataset_id"].astype(str).eq("PIC")].copy()
    if local.empty:
        local = mapping.drop_duplicates("attribute_name").copy()
    fig = plt.figure(figsize=(7.2, 4.4))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.55], hspace=0.28)
    ax = fig.add_subplot(grid[0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _panel_label(ax, "a")
    steps = [
        (0.01, "Local medical\nobservations", "#F2F2F2"),
        (0.18, "Public ontology\nconstrained groups", "#E0F0F0"),
        (0.38, "Equal-weight\nthree-state encoding", "#DDEAF7"),
        (0.59, "Sample one public\ndimension", "#E8E1F1"),
        (0.77, "One $\\epsilon$-LDP\nreport", "#F6E4E2"),
        (0.91, "Server\ndistributions", "#F2F2F2"),
    ]
    widths = [0.12, 0.14, 0.15, 0.14, 0.11, 0.08]
    for index, ((x, text, color), width) in enumerate(zip(steps, widths)):
        _draw_box(ax, x, 0.30, width, 0.40, text, color)
        if index < len(steps) - 1:
            _draw_arrow(ax, x + width, 0.50, steps[index + 1][0], 0.50)
    ax.text(0.455, 0.18, "0: all missing   1: observed, none abnormal   2: any abnormal", ha="center", fontsize=6.5)
    ax.text(0.835, 0.78, "GRR or OUE shown; any $\\epsilon$-LDP\nmechanism can be substituted", ha="center", fontsize=6.2, color="#4D4D4D")

    ax2 = fig.add_subplot(grid[1])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    _panel_label(ax2, "b")
    anchors = sorted(local["semantic_anchor"].astype(str).unique())
    anchor_colors = ["#DCE8F5", "#E2F0E8", "#F3E4E7", "#EEE8D8"]
    ax2.text(0.02, 0.94, "Ten reviewed observations mapped to four public semantic dimensions", fontsize=7.2, fontweight="bold")
    for row_index, (anchor, color) in enumerate(zip(anchors, anchor_colors)):
        y = 0.74 - row_index * 0.21
        attributes = local.loc[local["semantic_anchor"].astype(str).eq(anchor), "attribute_name"].astype(str).tolist()
        label = anchor.replace("_", " ")
        _draw_box(ax2, 0.03, y, 0.30, 0.14, label, color)
        ax2.text(0.37, y + 0.07, "  ".join(attributes), va="center", fontsize=6.5)
        _draw_arrow(ax2, 0.84, y + 0.07, 0.90, y + 0.07)
        ax2.text(0.94, y + 0.07, "0 / 1 / 2", ha="center", va="center", fontsize=6.5, fontweight="bold")
    ax2.text(0.03, 0.02, "Group membership is frozen before patient values are processed; no graph-distance weight or fitted coefficient is used.", fontsize=6.3, color="#4D4D4D")
    return _save_publication_figure(fig, output_base)


def _load_formal_frames(results_root: str | Path, filename: str) -> pd.DataFrame:
    root = Path(results_root)
    frames = []
    for dataset in ["PIC", "NHANES_2021_2023"]:
        path = root / dataset / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def _plot_summary_grid(summary: pd.DataFrame, methods: list[str], output_base: str | Path, *, ablation: bool) -> list[Path]:
    apply_publication_style()
    datasets = [("PIC", "PIC (paediatric)"), ("NHANES_2021_2023", "NHANES (adult)")]
    mechanisms = ["GRR", "OUE"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True)
    for row, (dataset, dataset_label) in enumerate(datasets):
        maximum_n = int(summary.loc[summary["dataset_id"].eq(dataset), "sample_size"].max())
        for column, mechanism in enumerate(mechanisms):
            ax = axes[row, column]
            subset = summary.loc[
                summary["dataset_id"].eq(dataset)
                & summary["mechanism"].eq(mechanism)
                & summary["sample_size"].eq(maximum_n)
            ]
            for method in methods:
                part = subset.loc[subset["method"].eq(method)].sort_values("epsilon")
                if part.empty:
                    continue
                color = METHOD_COLORS[method]
                linewidth = 2.0 if method in {"SemMed-LDP", "Full-SemMed-LDP"} else 1.15
                marker = "o" if method in {"SemMed-LDP", "Full-SemMed-LDP"} else None
                ax.plot(part["epsilon"], part["MSE_mean"], color=color, lw=linewidth, marker=marker, ms=3.2, label=METHOD_LABELS[method])
                ax.fill_between(
                    part["epsilon"].to_numpy(float),
                    part["MSE_ci_low"].to_numpy(float),
                    part["MSE_ci_high"].to_numpy(float),
                    color=color, alpha=0.10, linewidth=0,
                )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 4, 8])
            ax.set_xticklabels(["0.1", "0.25", "0.5", "1", "2", "4", "8"])
            ax.set_title(f"{dataset_label} | {mechanism} | n={maximum_n:,}")
            ax.set_xlabel("Privacy budget, $\\epsilon$")
            ax.set_ylabel("MSE (lower is better)")
            ax.grid(False)
            _panel_label(ax, chr(ord("a") + row * 2 + column))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3 if not ablation else 3, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.10, right=0.99, top=0.94, bottom=0.16, wspace=0.26, hspace=0.34)
    return _save_publication_figure(fig, output_base)


def plot_formal_main_performance(results_root: str | Path, output_base: str | Path) -> list[Path]:
    """Plot full-sample MSE curves with 95% bootstrap intervals."""
    summary = _load_formal_frames(results_root, "summary_bootstrap_ci.csv")
    methods = [
        "SemMed-LDP", "Raw-Attribute-LDP", "LOINC-Groups-LDP",
        "Random-Grouped-LDP", "Correlation-Grouped-LDP-oracle", "Type-Grouped-LDP",
    ]
    return _plot_summary_grid(summary, methods, output_base, ablation=False)


def plot_formal_ablation(results_root: str | Path, output_base: str | Path) -> list[Path]:
    """Plot grouping and anchor-count ablations under the full formal grid."""
    summary = _load_formal_frames(results_root, "ablations/summary_bootstrap_ci.csv")
    methods = [
        "Full-SemMed-LDP", "No-HPO-LOINC-Groups", "Shuffled-Knowledge-Edges",
        "Anchor-Count-3", "Anchor-Count-2", "No-UBERON-Structural-Negative-Control",
    ]
    return _plot_summary_grid(summary, methods, output_base, ablation=True)


def generate_formal_manuscript_figures(project_root: str | Path) -> dict[str, list[str]]:
    """Generate all formal manuscript figures from immutable formal-v2 outputs."""
    root = Path(project_root)
    results = root / "results/formal_v2"
    output = root / "figures/formal"
    mapping = pd.read_csv(root / "configs/formal_attribute_mapping.csv")
    generated = {
        "framework": plot_formal_framework(mapping, output / "figure1_framework"),
        "main_performance": plot_formal_main_performance(results, output / "figure2_main_performance"),
        "ablation": plot_formal_ablation(results, output / "figure3_ablation"),
    }
    source_paths = [
        root / "configs/formal_attribute_mapping.csv",
        root / "configs/formal_experiment.yaml",
        root / "semmed_ldp/visualization.py",
    ]
    for dataset in ["PIC", "NHANES_2021_2023"]:
        source_paths.extend(
            [
                results / dataset / "summary_bootstrap_ci.csv",
                results / dataset / "ablations/summary_bootstrap_ci.csv",
            ]
        )

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "Python/matplotlib",
        "statistical_display": "30-run mean with percentile 95% bootstrap confidence interval",
        "sources": {str(path.relative_to(root)): sha256(path) for path in source_paths},
        "outputs": {
            str(path.relative_to(root)): sha256(path)
            for paths in generated.values()
            for path in paths
        },
    }
    manifest_path = output / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    returned = {key: [str(path) for path in paths] for key, paths in generated.items()}
    returned["manifest"] = [str(manifest_path)]
    return returned


def _prepare_output(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def plot_metric_by_epsilon(results: pd.DataFrame, metric: str, output_path: str | Path) -> None:
    """Plot method-level mean error for one metric across epsilon values."""
    if plt is None:
        LOGGER.warning("matplotlib is not installed; skipping %s plot.", metric)
        return
    if results.empty or metric not in results.columns:
        LOGGER.warning("Cannot plot %s because results are empty or the metric is missing.", metric)
        return
    summary = results.groupby(["method", "epsilon"], as_index=False)[metric].mean()
    plt.figure(figsize=(8, 5))
    if sns is not None:
        sns.lineplot(data=summary, x="epsilon", y=metric, hue="method", marker="o")
    else:
        for method, part in summary.groupby("method"):
            plt.plot(part["epsilon"], part[metric], marker="o", label=method)
        plt.legend()
    plt.title(f"{metric} by epsilon")
    plt.tight_layout()
    output = _prepare_output(output_path)
    plt.savefig(output, dpi=300)
    plt.close()
    LOGGER.info("Saved %s comparison plot to %s.", metric, output)


def plot_error_heatmap(results: pd.DataFrame, metric: str, output_path: str | Path, method: str = "UMLS-Grouped-LDP") -> None:
    """Plot a semantic-group-by-epsilon heatmap for one metric."""
    if plt is None:
        LOGGER.warning("matplotlib is not installed; skipping heatmap.")
        return
    if results.empty or metric not in results.columns:
        LOGGER.warning("Cannot plot heatmap because results are empty or metric '%s' is missing.", metric)
        return
    subset = results[results["method"] == method]
    if subset.empty:
        subset = results
    pivot = subset.groupby(["semantic_group", "epsilon"])[metric].mean().unstack("epsilon")
    plt.figure(figsize=(9, max(4, 0.35 * len(pivot))))
    if sns is not None:
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    else:
        plt.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
        plt.colorbar(label=metric)
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title(f"{metric} heatmap")
    plt.tight_layout()
    output = _prepare_output(output_path)
    plt.savefig(output, dpi=300)
    plt.close()
    LOGGER.info("Saved error heatmap to %s.", output)


def plot_group_size_summary(group_to_attributes: dict[str, list[str]], output_path: str | Path) -> None:
    """Plot the number of attributes assigned to each UMLS semantic group."""
    if plt is None:
        LOGGER.warning("matplotlib is not installed; skipping group size plot.")
        return
    if not group_to_attributes:
        LOGGER.warning("Cannot plot group sizes because no semantic groups were provided.")
        return
    frame = pd.DataFrame(
        [{"semantic_group": group, "n_attributes": len(attributes)} for group, attributes in group_to_attributes.items()]
    ).sort_values("n_attributes", ascending=False)
    plt.figure(figsize=(9, max(4, 0.35 * len(frame))))
    if sns is not None:
        sns.barplot(data=frame, y="semantic_group", x="n_attributes", color="#4C78A8")
    else:
        plt.barh(frame["semantic_group"], frame["n_attributes"], color="#4C78A8")
        plt.gca().invert_yaxis()
    plt.title("UMLS semantic group sizes")
    plt.tight_layout()
    output = _prepare_output(output_path)
    plt.savefig(output, dpi=300)
    plt.close()
    LOGGER.info("Saved group size plot to %s.", output)


def plot_true_vs_estimated_distribution(
    results: pd.DataFrame,
    output_path: str | Path,
    *,
    method: str = "UMLS-Grouped-LDP",
    semantic_group: str | None = None,
    epsilon: float | None = None,
    mechanism: str | None = None,
) -> None:
    """Plot one true-vs-estimated distribution example."""
    if plt is None:
        LOGGER.warning("matplotlib is not installed; skipping true-vs-estimated distribution plot.")
        return
    if results.empty:
        LOGGER.warning("Cannot plot true-vs-estimated distribution because results are empty.")
        return
    subset = results[results["method"] == method]
    if subset.empty:
        subset = results
    if semantic_group is not None:
        subset = subset[subset["semantic_group"] == semantic_group]
    if epsilon is not None:
        subset = subset[subset["epsilon"] == epsilon]
    if mechanism is not None:
        subset = subset[subset["mechanism"] == mechanism.upper()]
    if subset.empty:
        LOGGER.warning("No matching rows for true-vs-estimated distribution plot.")
        return
    first_key = subset[["semantic_group", "epsilon", "mechanism"]].drop_duplicates().iloc[0]
    subset = subset[
        (subset["semantic_group"] == first_key["semantic_group"])
        & (subset["epsilon"] == first_key["epsilon"])
        & (subset["mechanism"] == first_key["mechanism"])
    ]
    long_frame = subset.melt(
        id_vars=["state"],
        value_vars=["true_frequency", "estimated_mean"],
        var_name="distribution",
        value_name="frequency",
    )
    plt.figure(figsize=(7, 4))
    if sns is not None:
        sns.barplot(data=long_frame, x="state", y="frequency", hue="distribution")
    else:
        true_values = subset.sort_values("state")["true_frequency"].to_numpy()
        estimated_values = subset.sort_values("state")["estimated_mean"].to_numpy()
        states = subset.sort_values("state")["state"].to_numpy()
        width = 0.35
        plt.bar(states - width / 2, true_values, width=width, label="true_frequency")
        plt.bar(states + width / 2, estimated_values, width=width, label="estimated_mean")
        plt.legend()
    plt.title(f"{first_key['semantic_group']} eps={first_key['epsilon']} {first_key['mechanism']}")
    plt.tight_layout()
    output = _prepare_output(output_path)
    plt.savefig(output, dpi=300)
    plt.close()
    LOGGER.info("Saved true-vs-estimated distribution plot to %s.", output)


def generate_basic_plots(
    results: pd.DataFrame,
    group_to_attributes: dict[str, list[str]],
    figures_dir: str | Path,
) -> None:
    """Generate the standard first-stage SemMed-LDP plots."""
    figures = Path(figures_dir)
    plot_metric_by_epsilon(results, "MSE", figures / "mse_by_epsilon.png")
    plot_metric_by_epsilon(results, "JS", figures / "js_divergence_by_epsilon.png")
    plot_error_heatmap(results, "MSE", figures / "semantic_group_mse_heatmap.png")
    plot_group_size_summary(group_to_attributes, figures / "umls_group_size_statistics.png")
    plot_true_vs_estimated_distribution(results, figures / "true_vs_estimated_distribution_example.png")


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[1]
    print(json.dumps(generate_formal_manuscript_figures(project), indent=2))
