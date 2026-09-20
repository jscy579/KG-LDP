"""Plot Supplementary Figure S2 from the NHANES schema-expansion MSE summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = PROJECT_ROOT / "figures" / "formal_v3"
RESULT_ROOT = PROJECT_ROOT / "results" / "schema_expansion"

PANEL_ORDER = ["D10", "D20", "D40"]
MECHANISM_ORDER = ["GRR", "OUE"]
METHOD_ORDER = ["KGR-LDP", "Random", "Raw"]
EPSILONS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
COLORS = {"KGR-LDP": "#2F6FA3", "Random": "#8B8F94", "Raw": "#D47A34"}
MARKERS = {"KGR-LDP": "o", "Random": "s", "Raw": "^"}
LINESTYLES = {"KGR-LDP": "-", "Random": "--", "Raw": "-."}
LABELS = {
    "KGR-LDP": "KGR-LDP",
    "Random": "Random",
    "Raw": "Raw (scaling reference)",
}

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
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.35,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def load_and_validate() -> pd.DataFrame:
    data = pd.read_csv(RESULT_ROOT / "mse_summary.csv")
    expected = pd.MultiIndex.from_product(
        [PANEL_ORDER, METHOD_ORDER, MECHANISM_ORDER, EPSILONS],
        names=["panel", "method", "mechanism", "epsilon"],
    )
    observed = pd.MultiIndex.from_frame(data[["panel", "method", "mechanism", "epsilon"]])
    if len(data) != 126 or set(expected) != set(observed):
        raise ValueError("MSE summary does not contain the complete 3 x 3 x 2 x 7 grid")
    if not data["n_runs"].eq(30).all():
        raise ValueError("Every MSE condition must contain 30 perturbation runs")
    if not (data["mean_MSE"] > 0).all():
        raise ValueError("All MSE values must be positive for logarithmic plotting")
    return data.sort_values(["mechanism", "panel", "method", "epsilon"])


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.15,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, kwargs in {
        "svg": {},
        "pdf": {},
        "tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
        "png": {"dpi": 300},
    }.items():
        path = FIGURE_ROOT / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    data = load_and_validate()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.75), sharex=True, sharey=True)
    for row, mechanism in enumerate(MECHANISM_ORDER):
        for column, panel in enumerate(PANEL_ORDER):
            ax = axes[row, column]
            local = data.loc[
                data["mechanism"].eq(mechanism) & data["panel"].eq(panel)
            ]
            for method in METHOD_ORDER:
                values = local.loc[local["method"].eq(method)].sort_values("epsilon")
                ax.plot(
                    values["epsilon"],
                    values["mean_MSE"],
                    color=COLORS[method],
                    marker=MARKERS[method],
                    linestyle=LINESTYLES[method],
                    markersize=3.2,
                    markeredgecolor="white",
                    markeredgewidth=0.35,
                    label=LABELS[method],
                )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(EPSILONS, ["0.1", "0.25", "0.5", "1", "2", "4", "8"])
            ax.grid(axis="y", which="both", color="#DEE1E3", linewidth=0.45)
            ax.set_axisbelow(True)
            ax.set_title(f"{panel} | {mechanism}", pad=6)
            panel_label(ax, chr(ord("a") + row * 3 + column))
            if row == 1:
                ax.set_xlabel(r"Privacy budget $\varepsilon$")
            if column == 0:
                ax.set_ylabel("Mean squared error")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.88, bottom=0.12, hspace=0.34, wspace=0.22)

    stem = "fig_supplementary_s2"
    outputs = save_figure(fig, stem)
    source_path = FIGURE_ROOT / f"{stem}.csv"
    data.to_csv(source_path, index=False)
    manifest = {
        "figure": "Supplementary Figure S2",
        "claim": "MSE scaling across D10, D20, and D40 is shown over the complete privacy-budget grid for KGR-LDP, dimension-matched Random, and Raw.",
        "input": (RESULT_ROOT / "mse_summary.csv").relative_to(PROJECT_ROOT).as_posix(),
        "source_data": source_path.relative_to(PROJECT_ROOT).as_posix(),
        "outputs": [path.relative_to(PROJECT_ROOT).as_posix() for path in outputs],
        "validation": "The complete 126-condition grid and 30 runs per condition were checked before plotting.",
        "note": "Raw is retained as a scaling reference rather than a dimension-matched semantic comparator.",
    }
    (FIGURE_ROOT / f"{stem}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
