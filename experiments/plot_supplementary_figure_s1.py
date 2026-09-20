"""Plot Supplementary Figure S1 from the frozen NHANES schema-expansion outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = PROJECT_ROOT / "figures" / "formal_v3"
RESULT_ROOT = PROJECT_ROOT / "results" / "schema_expansion"

PANEL_ORDER = ["D10", "D20", "D40"]
PANEL_RANK = {panel: index for index, panel in enumerate(PANEL_ORDER)}
STATE_ORDER = [0, 1, 2]
STATE_LABELS = {
    0: "State 0: no abnormality",
    1: "State 1: abnormality present",
    2: "State 2: all missing",
}
STATE_COLORS = {0: "#4C78A8", 1: "#E28E45", 2: "#A9ADB2"}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def natural_group_key(value: str) -> int:
    match = re.search(r"(\d+)$", value)
    if match is None:
        raise ValueError(f"Cannot parse semantic group identifier: {value}")
    return int(match.group(1))


def included_in_panel(added_in: str, panel: str) -> bool:
    return PANEL_RANK[added_in] <= PANEL_RANK[panel]


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = pd.read_csv(RESULT_ROOT / "attribute_missingness_and_states.csv")
    mapping = pd.read_csv(RESULT_ROOT / "supplement_table_s1_mapping.csv")
    states = pd.read_csv(RESULT_ROOT / "state_concentration.csv")
    states = states.loc[states["method"].eq("KGR-LDP")].copy()

    if len(missing) != 40 or len(mapping) != 40:
        raise ValueError("Expected exactly 40 attributes in missingness and mapping files")
    if set(missing["attribute_id"]) != set(mapping["ID"]):
        raise ValueError("Attribute identifiers differ between missingness and frozen mapping")
    if not np.allclose(states.groupby(["panel", "dimension"])["proportion"].sum(), 1.0):
        raise ValueError("State 0/1/2 proportions do not sum to one for every semantic group")

    mapping = mapping.sort_values("ID").reset_index(drop=True)
    missing = missing.sort_values("attribute_id").reset_index(drop=True)
    if mapping["Attribute"].tolist() != missing["attribute"].tolist():
        raise ValueError("Attribute names differ between missingness and frozen mapping")

    for panel in PANEL_ORDER:
        included = mapping.loc[
            mapping["Added in"].map(lambda value: included_in_panel(value, panel))
        ]
        expected_sizes = included.groupby("Semantic group").size().to_dict()
        observed = states.loc[states["panel"].eq(panel)]
        observed_sizes = observed.groupby("dimension")["group_size"].first().to_dict()
        if expected_sizes != observed_sizes:
            raise ValueError(f"Frozen mapping and state composition disagree for {panel}")

    return missing, mapping, states


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.025,
        1.08,
        label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )


def draw_missingness(
    ax: plt.Axes, missing: pd.DataFrame, mapping: pd.DataFrame
) -> mpl.cm.ScalarMappable:
    matrix = np.full((len(PANEL_ORDER), len(mapping)), np.nan)
    missing_by_id = missing.set_index("attribute_id")["missing_rate"]
    for row, panel in enumerate(PANEL_ORDER):
        for column, record in mapping.iterrows():
            if included_in_panel(record["Added in"], panel):
                matrix[row, column] = missing_by_id.loc[record["ID"]]

    cmap = LinearSegmentedColormap.from_list(
        "missingness", ["#F4F7FA", "#BDD2E3", "#4C78A8"]
    )
    cmap.set_bad("#ECEDEF")
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.25, vmax=0.50)
    ax.set_yticks(range(len(PANEL_ORDER)), PANEL_ORDER)
    labels = [f"A{int(value):02d}" for value in mapping["ID"]]
    ax.set_xticks(range(len(labels)), labels)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Attribute-level missingness across nested panels", pad=8)
    ax.set_xlabel("Attribute ID (names and fixed assignments are listed in Supplementary Table S1)")
    for boundary in [9.5, 19.5]:
        ax.axvline(boundary, color="white", linewidth=1.4)
    panel_label(ax, "a")
    return image


def draw_state_composition(ax: plt.Axes, states: pd.DataFrame, panel: str, label: str) -> None:
    local = states.loc[states["panel"].eq(panel)].copy()
    dimensions = sorted(local["dimension"].unique(), key=natural_group_key)
    group_sizes = local.groupby("dimension")["group_size"].first()
    pivot = (
        local.pivot(index="dimension", columns="state", values="proportion")
        .reindex(index=dimensions, columns=STATE_ORDER)
        .fillna(0.0)
    )
    y = np.arange(len(dimensions))
    left = np.zeros(len(dimensions))
    for state in STATE_ORDER:
        values = pivot[state].to_numpy(float)
        ax.barh(
            y,
            values,
            left=left,
            height=0.70,
            color=STATE_COLORS[state],
            edgecolor="white",
            linewidth=0.35,
            label=STATE_LABELS[state],
        )
        for index, (start, value) in enumerate(zip(left, values)):
            if value >= 0.10:
                ax.text(
                    start + value / 2,
                    index,
                    f"{100 * value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=5.2,
                    color="white" if state in (0, 1) else "#333333",
                )
        left += values

    ax.set_yticks(
        y,
        [f"{dimension} (n={int(group_sizes.loc[dimension])})" for dimension in dimensions],
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0], ["0", "25", "50", "75", "100"])
    ax.set_xlabel("Records (%)")
    ax.set_title(f"{panel}: semantic-group state composition", pad=7)
    ax.grid(axis="x", color="#E3E5E7", linewidth=0.5)
    ax.set_axisbelow(True)
    panel_label(ax, label)


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
    missing, mapping, states = load_and_validate()
    fig = plt.figure(figsize=(11.2, 6.1))
    grid = fig.add_gridspec(2, 3, height_ratios=[0.72, 1.45], hspace=0.42, wspace=0.62)
    ax_missing = fig.add_subplot(grid[0, :])
    image = draw_missingness(ax_missing, missing, mapping)
    colorbar = fig.colorbar(image, ax=ax_missing, fraction=0.018, pad=0.012)
    colorbar.set_label("Missing records (%)")
    colorbar.set_ticks([0.25, 0.30, 0.40, 0.50])
    colorbar.set_ticklabels(["25", "30", "40", "50"])

    state_axes = [fig.add_subplot(grid[1, index]) for index in range(3)]
    for ax, panel, label in zip(state_axes, PANEL_ORDER, ["b", "c", "d"]):
        draw_state_composition(ax, states, panel, label)
    handles, labels = state_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.075, right=0.97, top=0.95, bottom=0.09)

    stem = "fig_supplementary_s1"
    outputs = save_figure(fig, stem)
    missing_source = FIGURE_ROOT / f"{stem}_missingness.csv"
    states_source = FIGURE_ROOT / f"{stem}_state_composition.csv"
    missing.merge(
        mapping[["ID", "Semantic group", "Added in"]],
        left_on="attribute_id",
        right_on="ID",
        validate="one_to_one",
    ).drop(columns="ID").rename(
        columns={"Semantic group": "semantic_group", "Added in": "added_in"}
    ).to_csv(missing_source, index=False)
    states.to_csv(states_source, index=False)

    manifest = {
        "figure": "Supplementary Figure S1",
        "claim": "The nested D10, D20, and D40 panels preserve the frozen attribute-to-group mapping while exposing attribute missingness and group-level semantic-state composition.",
        "inputs": [
            (RESULT_ROOT / "attribute_missingness_and_states.csv").relative_to(PROJECT_ROOT).as_posix(),
            (RESULT_ROOT / "state_concentration.csv").relative_to(PROJECT_ROOT).as_posix(),
            (RESULT_ROOT / "supplement_table_s1_mapping.csv").relative_to(PROJECT_ROOT).as_posix(),
        ],
        "source_data": [
            missing_source.relative_to(PROJECT_ROOT).as_posix(),
            states_source.relative_to(PROJECT_ROOT).as_posix(),
        ],
        "outputs": [path.relative_to(PROJECT_ROOT).as_posix() for path in outputs],
        "validation": "Attribute IDs, names, cumulative panel membership, group sizes, and state-proportion sums were checked before plotting.",
    }
    (FIGURE_ROOT / f"{stem}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
