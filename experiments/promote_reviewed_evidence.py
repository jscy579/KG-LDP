"""Promote working evidence only after explicit, complete review decisions."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semmed_ldp.formal_semantic_reduction import validate_formal_mapping
from semmed_ldp.reference_intervals import validate_reference_intervals


def main() -> None:
    decisions = pd.read_csv(ROOT / "configs/clinical_review_decisions.csv", dtype=str).fillna("")
    mapping = pd.read_csv(ROOT / "configs/formal_attribute_mapping_working.csv")
    intervals = pd.read_csv(ROOT / "configs/reference_intervals_working.csv")
    edges = pd.read_csv(ROOT / "configs/formal_graph_edges_working.csv")
    keys = ["dataset_id", "attribute_name"]
    if decisions.duplicated(keys).any() or mapping.duplicated(keys).any():
        raise ValueError("review decisions and mappings must contain one row per dataset/attribute")
    expected = set(map(tuple, mapping[keys].astype(str).to_numpy()))
    received = set(map(tuple, decisions[keys].astype(str).to_numpy()))
    if expected != received:
        raise ValueError("review decision keys do not exactly match the working mapping")
    included_mapping = mapping.loc[mapping["primary_analysis_status"].eq("included")].copy()
    included_decisions = decisions.loc[decisions["primary_analysis_status"].eq("included")].copy()
    if len(included_mapping) != 20 or len(included_decisions) != 20:
        raise ValueError("Scheme A formal promotion requires exactly 20 included dataset-specific rows")
    decision_columns = ["mapping_decision", "reference_interval_decision", "graph_decision"]
    for column in decision_columns:
        if not included_decisions[column].str.lower().eq("approved").all():
            raise ValueError(f"every {column} must be approved before promotion")
    if included_decisions["reviewer_name"].str.strip().eq("").any():
        raise ValueError("every review row requires reviewer_name")
    parsed_dates = pd.to_datetime(included_decisions["review_date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("every review row requires a valid review_date")

    review_meta = included_decisions[keys + ["reviewer_name", "review_date", "comments"]]
    formal_mapping = included_mapping.merge(review_meta, on=keys, how="left", validate="one_to_one")
    formal_mapping["reviewer_status"] = "approved"
    allowed_interval_ids = set(formal_mapping["reference_interval_id"].astype(str))
    formal_intervals = intervals.loc[
        intervals["reference_interval_id"].astype(str).isin(allowed_interval_ids)
    ].copy()
    formal_intervals["reviewer_status"] = "approved"
    allowed_attributes = {
        f"ATTR:{row.dataset_id}:{row.attribute_name}" for row in formal_mapping.itertuples(index=False)
    }
    allowed_loinc = {f"LOINC:{value}" for value in formal_mapping["loinc_code"].astype(str)}
    allowed_groups = {f"LOINC_GROUP:{value}" for value in formal_mapping["loinc_group_id"].astype(str)}
    allowed_anchors = {f"ANCHOR:{value}" for value in formal_mapping["semantic_anchor"].astype(str)}
    source = edges["source_id"].astype(str)
    target = edges["target_id"].astype(str)
    relation = edges["relation"].astype(str)
    keep = (
        relation.eq("subclass_of")
        | (relation.eq("mapped_to_loinc") & source.isin(allowed_attributes))
        | (relation.isin(["member_of_group", "has_abnormal_phenotype", "has_anatomical_system"]) & source.isin(allowed_loinc))
        | (relation.eq("parent_group") & source.isin(allowed_groups))
        | (relation.eq("anchored_to") & target.isin(allowed_anchors))
    )
    formal_edges = edges.loc[keep].copy()
    formal_edges.loc[formal_edges["reviewer_status"].eq("needs_review"), "reviewer_status"] = "approved"

    for dataset_id in formal_mapping["dataset_id"].unique():
        validate_formal_mapping(formal_mapping, dataset_id=str(dataset_id))
    validate_reference_intervals(formal_intervals)
    formal_mapping.to_csv(ROOT / "configs/formal_attribute_mapping.csv", index=False)
    formal_intervals.to_csv(ROOT / "configs/reference_intervals.csv", index=False)
    formal_edges.to_csv(ROOT / "configs/formal_graph_edges.csv", index=False)
    print("Promoted reviewed mapping, reference intervals, and graph edges to formal files")


if __name__ == "__main__":
    main()
