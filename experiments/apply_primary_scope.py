"""Record the user-approved Scheme A primary-analysis scope.

Potassium and calcium remain in the audit artifacts and are excluded only from
the matched PIC/NHANES primary analysis. They may be reintroduced in explicitly
labelled supplementary sensitivity analyses.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DECISION_DATE = "2026-07-20"
EXCLUSIONS = {
    ("PIC", "potassium"): "PIC specimen is recorded as Other Body Fluid; whole-blood mapping and serum interval are not sufficiently aligned",
    ("NHANES_2021_2023", "potassium"): "paired exclusion preserves the same primary attribute set across PIC and NHANES",
    ("PIC", "ionized_calcium"): "PIC English/Chinese labels conflict and the available public interval is serum pH-adjusted rather than whole blood",
    ("NHANES_2021_2023", "total_calcium"): "paired exclusion preserves the same primary attribute set across PIC and NHANES",
}


def main() -> None:
    mapping_path = ROOT / "configs/formal_attribute_mapping_working.csv"
    decisions_path = ROOT / "configs/clinical_review_decisions.csv"
    mapping = pd.read_csv(mapping_path, dtype=str).fillna("")
    decisions = pd.read_csv(decisions_path, dtype=str).fillna("")
    keys = ["dataset_id", "attribute_name"]
    mapping["primary_analysis_status"] = "included"
    mapping["primary_exclusion_reason"] = ""
    mapping["supplementary_sensitivity_set"] = ""
    for (dataset_id, attribute_name), reason in EXCLUSIONS.items():
        match = mapping["dataset_id"].eq(dataset_id) & mapping["attribute_name"].eq(attribute_name)
        if int(match.sum()) != 1:
            raise ValueError(f"expected exactly one mapping row for {(dataset_id, attribute_name)}")
        mapping.loc[match, "primary_analysis_status"] = "excluded"
        mapping.loc[match, "primary_exclusion_reason"] = reason
        mapping.loc[match, "supplementary_sensitivity_set"] = (
            "add_back_potassium" if attribute_name == "potassium" else "add_back_calcium"
        )

    scope = mapping[
        keys + ["primary_analysis_status", "primary_exclusion_reason", "supplementary_sensitivity_set"]
    ].copy()
    scope["decision_date"] = DECISION_DATE
    scope["decision_authority"] = "user-approved Scheme A in the SemMed-LDP project task"
    scope.to_csv(ROOT / "configs/primary_analysis_scope.csv", index=False)
    mapping.to_csv(mapping_path, index=False)

    decisions = decisions.drop(
        columns=["primary_analysis_status", "decision_basis"], errors="ignore"
    ).merge(scope[keys + ["primary_analysis_status"]], on=keys, how="left", validate="one_to_one")
    decisions["decision_basis"] = ""
    excluded = decisions["primary_analysis_status"].eq("excluded")
    for column in ["mapping_decision", "reference_interval_decision", "graph_decision"]:
        decisions.loc[excluded, column] = "excluded"
    decisions.loc[excluded, "review_date"] = DECISION_DATE
    decisions.loc[excluded, "decision_basis"] = "user-approved Scheme A; retained only for supplementary sensitivity analysis"
    decisions.loc[excluded, "comments"] = decisions.loc[excluded, keys].apply(
        lambda row: EXCLUSIONS[(row["dataset_id"], row["attribute_name"])], axis=1
    )
    decisions.to_csv(decisions_path, index=False)
    print(
        f"Recorded {int((mapping['primary_analysis_status'] == 'included').sum())} primary rows "
        f"and {int((mapping['primary_analysis_status'] == 'excluded').sum())} paired exclusions"
    )


if __name__ == "__main__":
    main()
