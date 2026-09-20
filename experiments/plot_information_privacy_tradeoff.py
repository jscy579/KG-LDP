"""Plot raw-state information retention against existing formal-v3 LDP error."""

from __future__ import annotations

import argparse
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

from semmed_ldp.formal_experiment import (
    build_representation_method_states,
    build_tradeoff_method_states,
    sha256_file,
)
from semmed_ldp.formal_semantic_reduction import validate_formal_mapping
from semmed_ldp.information_retention import normalized_attribute_information_retention


RESULT_ROOT = PROJECT_ROOT / "results" / "formal_v3"
FIGURE_ROOT = PROJECT_ROOT / "figures" / "formal_v3"
MAPPING_PATH = PROJECT_ROOT / "configs" / "formal_attribute_mapping.csv"
INTERVAL_PATH = PROJECT_ROOT / "configs" / "reference_intervals.csv"
DATASET_CONFIG = {
    "PIC": {
        "features": PROJECT_ROOT / "data" / "processed" / "working" / "pic_feature_matrix.csv",
        "metadata": PROJECT_ROOT / "data" / "processed" / "working" / "pic_metadata.csv",
        "entity_id": "SUBJECT_ID",
        "external": "NHANES_2021_2023",
        "display": "PIC",
    },
    "NHANES_2021_2023": {
        "features": PROJECT_ROOT / "data" / "processed" / "working" / "nhanes_feature_matrix.csv",
        "metadata": PROJECT_ROOT / "data" / "processed" / "working" / "nhanes_metadata.csv",
        "entity_id": "SEQN",
        "external": "PIC",
        "display": "NHANES 2021--2023",
    },
}

COLORS = {
    "Raw-Attribute-LDP": "#565656",
    "LOINC-Group-LDP": "#B29A5B",
    "SemMed-LDP": "#356DB2",
    "Source-Schema-LDP": "#5D9270",
    "LOINC-Class-LDP": "#B7676C",
    "External-Correlation-4D": "#8064A2",
    "Matched-Random-4D": "#B8B8B8",
}
MARKERS = {
    "Raw-Attribute-LDP": "o",
    "LOINC-Group-LDP": "s",
    "SemMed-LDP": "*",
    "Source-Schema-LDP": "D",
    "LOINC-Class-LDP": "P",
    "External-Correlation-4D": "X",
    "Matched-Random-4D": ".",
}
LABELS = {
    "Raw-Attribute-LDP": "Raw 10D",
    "SemMed-LDP": "KGR-LDP 4D",
    "Source-Schema-LDP": "Source Schema 2--3D",
    "LOINC-Class-LDP": "LOINC Class 2D",
    "External-Correlation-4D": "Cross-cohort 4D",
    "Matched-Random-4D": "Random 4D",
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
        "legend.fontsize": 6.2,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--random-partitions", type=int, default=100)
    return parser.parse_args()


def _load_indexed(path: Path, entity_id: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if entity_id not in frame:
        raise ValueError(f"{path} lacks entity column {entity_id!r}")
    return frame.set_index(entity_id)


def _dataset_frames(dataset_id: str, mapping_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = DATASET_CONFIG[dataset_id]
    mapping = validate_formal_mapping(mapping_all, dataset_id=dataset_id)
    features = _load_indexed(config["features"], config["entity_id"])
    features = features[mapping["attribute_name"].tolist()]
    metadata = _load_indexed(config["metadata"], config["entity_id"]).loc[features.index]
    return features, metadata, mapping


def _retention_rows(
    *,
    dataset_id: str,
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    mapping: pd.DataFrame,
    external_features: pd.DataFrame,
    external_metadata: pd.DataFrame,
    external_mapping: pd.DataFrame,
    intervals: pd.DataFrame,
    seed: int,
    n_random_partitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tradeoff = build_tradeoff_method_states(
        features,
        mapping,
        metadata=metadata,
        reference_intervals=intervals,
    )
    raw_states = tradeoff["Raw-Attribute-LDP"]
    representations = build_representation_method_states(
        features,
        mapping,
        external_features=external_features,
        external_mapping=external_mapping,
        n_random_partitions=n_random_partitions,
        seed=seed,
        target_metadata=metadata,
        external_metadata=external_metadata,
        reference_intervals=intervals,
        external_dataset_id=DATASET_CONFIG[dataset_id]["external"],
    )

    rows: list[dict[str, float | int | str]] = []
    details: list[pd.DataFrame] = []

    def record(method: str, grouping_id: str, states: pd.DataFrame) -> None:
        score, attribute_details = normalized_attribute_information_retention(raw_states, states)
        rows.append(
            {
                "dataset_id": dataset_id,
                "method": method,
                "grouping_id": grouping_id,
                "n_dimensions": states.shape[1],
                "information_retention": score,
            }
        )
        details.append(
            attribute_details.assign(
                dataset_id=dataset_id,
                method=method,
                grouping_id=grouping_id,
                n_dimensions=states.shape[1],
            )
        )

    for method, states in tradeoff.items():
        record(method, method, states)
    for method_key, (states, grouping_id) in representations.items():
        if method_key == "SemMed-LDP":
            continue
        method = "Matched-Random-4D" if method_key.startswith("Matched-Random-4D::") else method_key
        record(method, grouping_id, states)
    return pd.DataFrame(rows), pd.concat(details, ignore_index=True)


def _mean_error_rows(dataset_id: str, epsilon: float) -> pd.DataFrame:
    trade_mean = pd.read_csv(
        RESULT_ROOT / dataset_id / "resolution_tradeoff" / "summary_bootstrap_ci.csv"
    )
    full_n = trade_mean["sample_size"].max()
    trade_mean = trade_mean[
        trade_mean["sample_size"].eq(full_n) & np.isclose(trade_mean["epsilon"], epsilon)
    ][
        ["dataset_id", "method", "mechanism", "n_dimensions", "MSE_mean", "MSE_ci_low", "MSE_ci_high"]
    ].copy()
    trade_mean["grouping_id"] = trade_mean["method"]

    representation_raw = pd.read_csv(
        RESULT_ROOT / dataset_id / "representation_comparison" / "run_metrics.csv"
    )
    representation_raw = representation_raw[
        representation_raw["sample_size"].eq(representation_raw["sample_size"].max())
        & np.isclose(representation_raw["epsilon"], epsilon)
        & representation_raw["method"].eq("Matched-Random-4D")
    ]
    random_mean = (
        representation_raw.groupby(
            ["dataset_id", "method", "grouping_id", "mechanism", "n_dimensions"],
            as_index=False,
        )["MSE"]
        .mean()
        .rename(columns={"MSE": "MSE_mean"})
    )
    random_mean["MSE_ci_low"] = np.nan
    random_mean["MSE_ci_high"] = np.nan

    representation_summary = pd.read_csv(
        RESULT_ROOT / dataset_id / "representation_comparison" / "summary_bootstrap_ci.csv"
    )
    representation_summary = representation_summary[
        representation_summary["sample_size"].eq(representation_summary["sample_size"].max())
        & np.isclose(representation_summary["epsilon"], epsilon)
        & representation_summary["method"].eq("External-Correlation-4D")
    ][
        ["dataset_id", "method", "mechanism", "n_dimensions", "MSE_mean", "MSE_ci_low", "MSE_ci_high"]
    ].copy()
    representation_summary["grouping_id"] = f"trained_on_{DATASET_CONFIG[dataset_id]['external']}"
    return pd.concat(
        [trade_mean, random_mean, representation_summary],
        ignore_index=True,
        sort=False,
    )


def build_source_data(epsilon: float, seed: int, n_random_partitions: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping_all = pd.read_csv(MAPPING_PATH)
    intervals = pd.read_csv(INTERVAL_PATH)
    retention_frames: list[pd.DataFrame] = []
    detail_frames: list[pd.DataFrame] = []
    for dataset_id, config in DATASET_CONFIG.items():
        features, metadata, mapping = _dataset_frames(dataset_id, mapping_all)
        external_features, external_metadata, external_mapping = _dataset_frames(
            config["external"], mapping_all
        )
        retention, details = _retention_rows(
            dataset_id=dataset_id,
            features=features,
            metadata=metadata,
            mapping=mapping,
            external_features=external_features,
            external_metadata=external_metadata,
            external_mapping=external_mapping,
            intervals=intervals,
            seed=seed,
            n_random_partitions=n_random_partitions,
        )
        retention_frames.append(retention)
        detail_frames.append(details)
    retention = pd.concat(retention_frames, ignore_index=True)
    errors = pd.concat(
        [_mean_error_rows(dataset_id, epsilon) for dataset_id in DATASET_CONFIG],
        ignore_index=True,
    )
    source = errors.merge(
        retention,
        on=["dataset_id", "method", "grouping_id", "n_dimensions"],
        how="left",
        validate="many_to_one",
    )
    if source["information_retention"].isna().any():
        missing = source.loc[source["information_retention"].isna(), ["dataset_id", "method", "grouping_id"]]
        raise ValueError(f"retention was not computed for:\n{missing.to_string(index=False)}")
    source["epsilon"] = epsilon
    source["retention_definition"] = (
        "mean I(X_a;Z)/H(X_a) over nonconstant raw attributes, equal attribute weights"
    )
    return source, pd.concat(detail_frames, ignore_index=True)


def plot_information_privacy_tradeoff(source: pd.DataFrame, epsilon: float) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25), sharex=True, sharey=True)
    for row, dataset_id in enumerate(DATASET_CONFIG):
        for column, mechanism in enumerate(["GRR", "OUE"]):
            ax = axes[row, column]
            local = source[source["dataset_id"].eq(dataset_id) & source["mechanism"].eq(mechanism)]
            random = local[local["method"].eq("Matched-Random-4D")]
            ax.scatter(
                random["MSE_mean"],
                random["information_retention"],
                s=14,
                color=COLORS["Matched-Random-4D"],
                alpha=0.55,
                linewidths=0,
                zorder=1,
                label=LABELS["Matched-Random-4D"],
            )
            for method in [
                "Raw-Attribute-LDP",
                "SemMed-LDP",
                "Source-Schema-LDP",
                "LOINC-Class-LDP",
                "External-Correlation-4D",
            ]:
                values = local[local["method"].eq(method)]
                if values.empty:
                    continue
                row_value = values.iloc[0]
                lower = max(row_value["MSE_mean"] - row_value["MSE_ci_low"], 0.0)
                upper = max(row_value["MSE_ci_high"] - row_value["MSE_mean"], 0.0)
                ax.errorbar(
                    row_value["MSE_mean"],
                    row_value["information_retention"],
                    xerr=np.asarray([[lower], [upper]]),
                    fmt=MARKERS[method],
                    ms=7 if method == "SemMed-LDP" else 5,
                    color=COLORS[method],
                    markeredgecolor="white" if method == "SemMed-LDP" else COLORS[method],
                    markeredgewidth=0.8 if method == "SemMed-LDP" else 0.6,
                    elinewidth=0.8,
                    capsize=1.8,
                    zorder=4 if method == "SemMed-LDP" else 3,
                    label=LABELS[method],
                )
            ax.set_xscale("log")
            ax.set_ylim(0.0, 1.05)
            ax.grid(color="#DDDDDD", lw=0.45, alpha=0.75)
            ax.set_title(f"{DATASET_CONFIG[dataset_id]['display']} | {mechanism}")
            ax.text(
                -0.12,
                1.04,
                chr(ord("a") + row * 2 + column),
                transform=ax.transAxes,
                fontsize=8,
                fontweight="bold",
                va="bottom",
            )
            if row == 1:
                ax.set_xlabel(rf"MSE at $\epsilon={epsilon:g}$")
            if column == 0:
                ax.set_ylabel(r"Raw-state information retention $R_{\mathrm{MI}}$")
            #ax.text(
            #    0.02,
            #    0.04,
            #    "higher retention / lower error",
            #    transform=ax.transAxes,
            #    fontsize=6.2,
            #    color="#5A5A5A",
            #)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.985),
        frameon=True,
    )
    fig.subplots_adjust(top=0.88, hspace=0.31, wspace=0.22)
    return fig


def _save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        "svg": {},
        "pdf": {},
        "tiff": {"dpi": 600},
        "png": {"dpi": 240},
    }.items():
        fig.savefig(FIGURE_ROOT / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    source, attribute_details = build_source_data(
        args.epsilon,
        args.seed,
        args.random_partitions,
    )
    source_root = FIGURE_ROOT / "source_data"
    source_root.mkdir(parents=True, exist_ok=True)
    source.to_csv(source_root / "fig_information_privacy_tradeoff.csv", index=False)
    attribute_details.to_csv(
        source_root / "fig_information_privacy_tradeoff_attribute_details.csv",
        index=False,
    )
    _save_figure(
        plot_information_privacy_tradeoff(source, args.epsilon),
        "fig_information_privacy_tradeoff",
    )
    inputs = [
        MAPPING_PATH,
        INTERVAL_PATH,
        Path(__file__),
        PROJECT_ROOT / "configs" / "formal_experiment.yaml",
    ]
    inputs.extend(config["features"] for config in DATASET_CONFIG.values())
    inputs.extend(config["metadata"] for config in DATASET_CONFIG.values())
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "epsilon": args.epsilon,
        "seed": args.seed,
        "random_partitions": args.random_partitions,
        "retention_definition": (
            "equal-weight mean of I(X_a;Z)/H(X_a) over nonconstant raw attributes"
        ),
        "inputs": [
            {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(path)}
            for path in inputs
        ],
    }
    (FIGURE_ROOT / "fig_information_privacy_tradeoff_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
