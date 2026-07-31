"""Audit unapproved evidence without promoting it to formal status."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semmed_ldp.reference_intervals import bounds_for_users, validate_reference_intervals



def cohort_audit(dataset_id: str, feature_name: str, metadata_name: str) -> dict[str, object]:
    base = ROOT / "data/processed/working"
    features = pd.read_csv(base / feature_name, index_col=0)
    metadata = pd.read_csv(base / metadata_name, index_col=0)
    mapping = pd.read_csv(ROOT / "configs/formal_attribute_mapping_working.csv")
    mapping = mapping.loc[
        mapping["dataset_id"].eq(dataset_id) & mapping["primary_analysis_status"].eq("included")
    ].copy()
    intervals = pd.read_csv(ROOT / "configs/reference_intervals_working.csv")

    uncovered: dict[str, str] = {}
    unit_mismatches: list[str] = []
    for row in mapping.itertuples(index=False):
        try:
            bounds_for_users(
                row.reference_interval_id,
                metadata.loc[features.index],
                intervals,
                require_approved=False,
            )
        except ValueError as exc:
            uncovered[str(row.attribute_name)] = str(exc)
        interval_units = set(
            intervals.loc[intervals["reference_interval_id"].eq(row.reference_interval_id), "unit"].astype(str)
        )
        if interval_units != {str(row.ucum_unit)}:
            unit_mismatches.append(str(row.attribute_name))

    features = features[mapping["attribute_name"].tolist()]
    missingness = features.isna().mean().rename("missing_rate").reset_index()
    missingness.columns = ["attribute_name", "missing_rate"]
    missingness.insert(0, "dataset_id", dataset_id)
    missingness.to_csv(ROOT / f"results/readiness/{dataset_id.lower()}_working_missingness.csv", index=False)
    return {
        "n_users": int(len(features)),
        "n_attributes": int(features.shape[1]),
        "age_min": float(metadata["age_years"].min()),
        "age_max": float(metadata["age_years"].max()),
        "sex_counts": {str(k): int(v) for k, v in metadata["sex"].value_counts(dropna=False).items()},
        "maximum_missing_rate": float(features.isna().mean().max()),
        "reference_interval_coverage_pass": not uncovered,
        "reference_interval_errors": uncovered,
        "unit_alignment_pass": not unit_mismatches,
        "unit_mismatches": unit_mismatches,
    }


def main() -> None:
    output = ROOT / "results/readiness"
    output.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(ROOT / "configs/formal_attribute_mapping_working.csv")
    intervals = pd.read_csv(ROOT / "configs/reference_intervals_working.csv")
    edges = pd.read_csv(ROOT / "configs/formal_graph_edges_working.csv")
    validate_reference_intervals(intervals, require_approved=False)

    required_mapping = [
        "loinc_code", "loinc_group_id", "hpo_low_id", "hpo_high_id", "uberon_id",
        "semantic_anchor", "graph_weight", "reference_interval_id", "threshold_source",
        "source_documentation_url", "reviewer_status",
    ]
    blank_counts = {
        column: int(mapping[column].fillna("").astype(str).str.strip().eq("").sum())
        for column in required_mapping
    }
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "structural audit only; no row is promoted or treated as clinically approved",
        "mapping_rows": int(len(mapping)),
        "primary_mapping_rows": int(mapping["primary_analysis_status"].eq("included").sum()),
        "excluded_mapping_rows": int(mapping["primary_analysis_status"].eq("excluded").sum()),
        "excluded_attributes": mapping.loc[
            mapping["primary_analysis_status"].eq("excluded"),
            ["dataset_id", "attribute_name", "primary_exclusion_reason"],
        ].to_dict("records"),
        "mapping_blank_counts": blank_counts,
        "all_mapping_rows_still_unapproved": bool(mapping["reviewer_status"].eq("needs_review").all()),
        "interval_strata": int(len(intervals)),
        "primary_interval_strata": int(
            intervals["reference_interval_id"].astype(str).isin(
                set(mapping.loc[mapping["primary_analysis_status"].eq("included"), "reference_interval_id"].astype(str))
            ).sum()
        ),
        "all_interval_rows_still_unapproved": bool(intervals["reviewer_status"].eq("needs_review").all()),
        "graph_edges": int(len(edges)),
        "graph_relation_counts": {str(k): int(v) for k, v in edges["relation"].value_counts().items()},
        "cohorts": {
            "PIC": cohort_audit("PIC", "pic_feature_matrix.csv", "pic_metadata.csv"),
            "NHANES_2021_2023": cohort_audit(
                "NHANES_2021_2023", "nhanes_feature_matrix.csv", "nhanes_metadata.csv"
            ),
        },
    }
    audit["structural_pass"] = bool(
        not any(blank_counts.values())
        and all(item["reference_interval_coverage_pass"] and item["unit_alignment_pass"] for item in audit["cohorts"].values())
    )
    (output / "working_evidence_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
