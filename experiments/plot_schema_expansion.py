from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results" / "schema_expansion"
OUTPUT = PROJECT_ROOT / "figures" / "formal_v3"
OUTPUT.mkdir(parents=True, exist_ok=True)

PANEL_ORDER = ["D10", "D20", "D40"]
METHOD_ORDER = ["KGR-LDP", "Random", "Raw"]
MECHANISM_ORDER = ["GRR", "OUE"]
METHOD_LABELS = {
    "KGR-LDP": "KGR-LDP",
    "Random": "Random",
    "Raw": "Raw (scaling reference)",
}

COLORS = {
    "KGR-LDP": "#2F6FA3",
    "Random": "#8B8F94",
    "Raw": "#D47A34",
    "GRR": "#3B6EA8",
    "OUE": "#C46A3A",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def stable_seed(*parts: str) -> int:
    value = "|".join(parts).encode("utf-8")
    return int(hashlib.sha256(value).hexdigest()[:8], 16)


def bootstrap_mean_ci(values: np.ndarray, *, seed: int, n_bootstrap: int = 5000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_bootstrap, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def build_figure_data():
    panel = pd.read_csv(RESULTS / "panel_summary.csv")
    runs = pd.read_csv(RESULTS / "run_metrics_random_collapsed.csv")
    runtime_repeats = pd.read_csv(RESULTS / "runtime_repeats.csv")

    panel["panel"] = pd.Categorical(panel["panel"], PANEL_ORDER, ordered=True)
    panel = panel.sort_values("panel")

    eps1 = runs.loc[runs["epsilon"].eq(1.0)].copy()
    error_rows = []
    for keys, frame in eps1.groupby(["panel", "method", "mechanism"], sort=False):
        mean, low, high = bootstrap_mean_ci(
            frame["MSE"].to_numpy(), seed=stable_seed(*map(str, keys))
        )
        error_rows.append(
            {
                "panel": keys[0],
                "method": keys[1],
                "mechanism": keys[2],
                "mean_MSE": mean,
                "ci_low": low,
                "ci_high": high,
                "n_runs": len(frame),
                "interval": "95% percentile bootstrap CI of the mean (5,000 resamples)",
            }
        )
    error = pd.DataFrame(error_rows)
    error["panel"] = pd.Categorical(error["panel"], PANEL_ORDER, ordered=True)
    error["method"] = pd.Categorical(error["method"], METHOD_ORDER, ordered=True)
    error["mechanism"] = pd.Categorical(error["mechanism"], MECHANISM_ORDER, ordered=True)
    error = error.sort_values(["mechanism", "method", "panel"])

    runtime = (
        runtime_repeats.groupby("panel", as_index=False)
        .agg(
            median_client_seconds=("client_seconds", "median"),
            median_server_seconds=("server_seconds", "median"),
            n_timed_measurements=("repeat", "size"),
        )
    )
    runtime["panel"] = pd.Categorical(runtime["panel"], PANEL_ORDER, ordered=True)
    runtime = runtime.sort_values("panel")
    runtime["aggregation"] = "pooled across GRR and OUE; 10 timed repeats per mechanism"

    panel.to_csv(OUTPUT / "figure6a_source_data.csv", index=False)
    error.to_csv(OUTPUT / "figure6b_source_data.csv", index=False)
    runtime.to_csv(OUTPUT / "figure6c_source_data.csv", index=False)
    return panel, error, runtime


def panel_label(ax, label: str):
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
    )


def draw():
    panel, error, runtime = build_figure_data()
    x = np.arange(len(PANEL_ORDER), dtype=float)

    fig = plt.figure(figsize=(7.2, 2.95), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.00, 1.52, 1.22],
        left=0.075,
        right=0.985,
        bottom=0.20,
        top=0.83,
        wspace=0.38,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])

    # a | Release-space scaling
    width = 0.34
    raw = panel["n_attributes"].to_numpy()
    semantic = panel["n_semantic_dimensions"].to_numpy()
    bars_raw = ax_a.bar(
        x - width / 2,
        raw,
        width,
        color="#D9DDE1",
        edgecolor="#4B535A",
        linewidth=0.7,
        label="Raw dimensions",
    )
    bars_kgr = ax_a.bar(
        x + width / 2,
        semantic,
        width,
        color=COLORS["KGR-LDP"],
        edgecolor=COLORS["KGR-LDP"],
        linewidth=0.7,
        label="KGR semantic dimensions",
    )
    for bar, value in zip(bars_raw, raw):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.7,
            f"{int(value)}",
            ha="center",
            va="top",
            fontsize=6,
            color="#2F3439",
        )
    for bar, value in zip(bars_kgr, semantic):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.8,
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontsize=6,
            color=COLORS["KGR-LDP"],
        )
    for x_pos, raw_value, ratio in zip(x, raw, panel["compression_ratio_d_over_m"]):
        ax_a.text(
            x_pos,
            raw_value - 2.0,
            f"$R_{{comp}}={ratio:.2f}$",
            ha="center",
            va="top",
            fontsize=5.3,
            color=COLORS["KGR-LDP"],
        )
    ax_a.set_xticks(x, PANEL_ORDER)
    ax_a.set_ylabel("Number of release dimensions")
    ax_a.set_ylim(0, 48)
    ax_a.set_yticks([0, 10, 20, 30, 40])
    ax_a.grid(axis="y", color="#E7E9EB", linewidth=0.55)
    ax_a.set_axisbelow(True)
    ax_a.legend(loc="upper left", frameon=False, handlelength=1.2, borderaxespad=0)
    panel_label(ax_a, "a")
    ax_a.set_title("Release-space scaling", pad=7)

    # b | Estimation error at epsilon=1
    marker_map = {"GRR": "o", "OUE": "^"}
    line_map = {"GRR": "-", "OUE": "--"}
    offset_map = {"GRR": -0.035, "OUE": 0.035}
    for mechanism in MECHANISM_ORDER:
        for method in METHOD_ORDER:
            local = error.loc[
                error["mechanism"].eq(mechanism) & error["method"].eq(method)
            ].sort_values("panel")
            y = local["mean_MSE"].to_numpy()
            yerr = np.vstack(
                [y - local["ci_low"].to_numpy(), local["ci_high"].to_numpy() - y]
            )
            ax_b.errorbar(
                x + offset_map[mechanism],
                y,
                yerr=yerr,
                color=COLORS[method],
                marker=marker_map[mechanism],
                linestyle=line_map[mechanism],
                linewidth=1.05,
                markersize=3.8,
                markerfacecolor="white" if mechanism == "OUE" else COLORS[method],
                markeredgewidth=0.8,
                capsize=2,
                elinewidth=0.75,
                zorder=3,
            )
    ax_b.set_yscale("log")
    ax_b.set_xticks(x, PANEL_ORDER)
    ax_b.set_ylabel(r"MSE at $\varepsilon=1$ (log scale)")
    ax_b.grid(axis="y", which="both", color="#E7E9EB", linewidth=0.55)
    ax_b.set_axisbelow(True)
    method_handles = [
        Line2D([0], [0], color=COLORS[m], lw=1.5, label=METHOD_LABELS[m])
        for m in METHOD_ORDER
    ]
    mechanism_handles = [
        Line2D(
            [0], [0], color="#333333", lw=1.0, marker=marker_map[m],
            linestyle=line_map[m], markerfacecolor="white" if m == "OUE" else "#333333",
            markersize=3.8, label=m,
        )
        for m in MECHANISM_ORDER
    ]
    legend_methods = ax_b.legend(
        handles=method_handles,
        loc="upper left",
        frameon=False,
        handlelength=1.5,
        borderaxespad=0,
    )
    ax_b.add_artist(legend_methods)
    ax_b.legend(
        handles=mechanism_handles,
        loc="lower right",
        frameon=False,
        handlelength=1.6,
        borderaxespad=0.2,
    )
    panel_label(ax_b, "b")
    ax_b.set_title("Estimation error", pad=7)

    # c | Computational scaling
    runtime_specs = [
        ("median_client_seconds", "Client-side", "o", "-", "#3B6EA8"),
        ("median_server_seconds", "Server-side", "s", "--", "#C46A3A"),
    ]
    for column, label, marker, linestyle, color in runtime_specs:
        ax_c.plot(
            x,
            runtime[column],
            color=color,
            marker=marker,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.8,
            linestyle=linestyle,
            linewidth=1.05,
            markersize=4,
            label=label,
        )
    ax_c.set_yscale("log")
    ax_c.set_xticks(x, PANEL_ORDER)
    ax_c.set_ylabel("Median runtime (s, log scale)")
    ax_c.grid(axis="y", which="both", color="#E7E9EB", linewidth=0.55)
    ax_c.set_axisbelow(True)
    ax_c.legend(
        loc="center left",
        frameon=False,
        handlelength=1.8,
        borderaxespad=0,
        ncol=1,
    )
    panel_label(ax_c, "c")
    ax_c.set_title("Computational scaling", pad=7)

    fig.savefig(OUTPUT / "figure6_schema_expansion.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "figure6_schema_expansion.svg", bbox_inches="tight")
    fig.savefig(OUTPUT / "figure6_schema_expansion.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        OUTPUT / "figure6_schema_expansion.tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    manifest = {
        "figure": "Figure 6. Schema-expansion scalability of KGR-LDP on NHANES.",
        "panels": {
            "a": "Raw and semantic release dimensions for D10, D20, and D40.",
            "b": "Mean MSE at epsilon=1 with 95% percentile bootstrap intervals from 30 perturbation runs.",
            "c": "Median client-side and server-side runtime pooled across GRR and OUE, with ten timed repeats per mechanism after warm-up.",
        },
        "source_files": [
            (RESULTS / "panel_summary.csv").relative_to(PROJECT_ROOT).as_posix(),
            (RESULTS / "run_metrics_random_collapsed.csv").relative_to(PROJECT_ROOT).as_posix(),
            (RESULTS / "runtime_repeats.csv").relative_to(PROJECT_ROOT).as_posix(),
        ],
        "backend": f"Python / matplotlib {mpl.__version__}",
        "exports": ["PDF", "SVG", "PNG 300 dpi", "TIFF 600 dpi LZW"],
    }
    (OUTPUT / "figure6_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    draw()
