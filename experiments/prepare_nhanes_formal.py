"""Prepare an auditable NHANES 2021--2023 inventory and approved feature matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semmed_ldp.dataset_adapters import load_nhanes_xpt_directory, select_and_rename_features
from semmed_ldp.formal_semantic_reduction import validate_formal_mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/nhanes_2021_2023")
    parser.add_argument("--mapping", default="configs/formal_attribute_mapping.csv")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--working", action="store_true", help="Prepare an audit matrix from unapproved mappings")
    parser.add_argument("--minimum-age", type=float, default=18.0)
    args = parser.parse_args()
    raw_dir = PROJECT_ROOT / args.raw_dir
    features, metadata = load_nhanes_xpt_directory(raw_dir)
    interim = PROJECT_ROOT / "data/interim/nhanes_2021_2023"
    interim.mkdir(parents=True, exist_ok=True)
    inventory = pd.DataFrame(
        {
            "source_variable_id": features.columns,
            "n_participants": [int(features[column].notna().sum()) for column in features],
            "missing_rate": [float(features[column].isna().mean()) for column in features],
            "numeric_rate": [float(pd.to_numeric(features[column], errors="coerce").notna().mean()) for column in features],
            "reviewer_status": "needs_review",
        }
    )
    inventory.to_csv(interim / "variable_inventory.csv", index=False)
    normalized_metadata = pd.DataFrame(index=metadata.index)
    normalized_metadata["age_years"] = metadata["RIDAGEYR"]
    normalized_metadata["sex"] = metadata["RIAGENDR"].map({1.0: "male", 2.0: "female"})
    for column in ["WTMEC2YR", "SDMVSTRA", "SDMVPSU"]:
        if column in metadata:
            normalized_metadata[column] = metadata[column]
    normalized_metadata.to_csv(interim / "participant_metadata.csv", index_label="SEQN")
    if args.inventory_only:
        return
    mapping_path = PROJECT_ROOT / args.mapping
    mapping = pd.read_csv(mapping_path)
    if not args.working:
        mapping = validate_formal_mapping(mapping, dataset_id="NHANES_2021_2023")
    else:
        mapping = mapping.loc[mapping["dataset_id"].astype(str).eq("NHANES_2021_2023")].copy()
    selected = select_and_rename_features(features, mapping)
    eligible_index = normalized_metadata.index[normalized_metadata["age_years"].ge(args.minimum_age)]
    selected = selected.loc[selected.index.intersection(eligible_index)]
    normalized_metadata = normalized_metadata.loc[selected.index]
    output = PROJECT_ROOT / "data/processed" / ("working" if args.working else "formal_v1")
    output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / "nhanes_feature_matrix.csv", index_label="SEQN")
    normalized_metadata.to_csv(output / "nhanes_metadata.csv", index_label="SEQN")


if __name__ == "__main__":
    main()
