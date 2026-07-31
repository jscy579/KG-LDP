"""Expand the shared-variable shortlist into a fail-closed mapping worksheet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    shortlist = pd.read_csv(PROJECT_ROOT / "configs/cross_dataset_variable_shortlist.csv", dtype=str)
    hpo_terms = {
        "sodium": ("Hyponatremia", "Hypernatremia"),
        "potassium": ("Hypokalemia", "Hyperkalemia"),
        "chloride": ("Hypochloremia", "Hyperchloremia"),
        "creatinine": ("Decreased serum creatinine", "Elevated circulating creatinine concentration"),
        "urea_nitrogen": ("Reduced blood urea nitrogen", "Increased blood urea nitrogen"),
        "glucose": ("Hypoglycemia", "Hyperglycemia"),
        "albumin": ("Hypoalbuminemia", "Hyperalbuminemia"),
        "calcium": ("Hypocalcemia", "Hypercalcemia"),
        "magnesium": ("Hypomagnesemia", "Hypermagnesemia"),
        "phosphate": ("Hypophosphatemia", "Hyperphosphatemia"),
        "hemoglobin": ("Decreased hemoglobin concentration", "Increased circulating hemoglobin concentration"),
        "platelet_count": ("Thrombocytopenia", "Thrombocytosis"),
    }
    pic_units = {
        "sodium": ("mmol/L", 1.0),
        "potassium": ("mmol/L", 1.0),
        "chloride": ("mmol/L", 1.0),
        "creatinine": ("umol/L", 1.0 / 88.4),
        "urea_nitrogen": ("mmol/L", 2.801),
        "glucose": ("mmol/L", 18.0182),
        "albumin": ("g/L", 0.1),
        "calcium": ("mmol/L", 4.008),
        "magnesium": ("mmol/L", 2.4305),
        "phosphate": ("mmol/L", 3.09598),
        "hemoglobin": ("g/L", 0.1),
        "platelet_count": ("10^9/L", 1.0),
    }
    rows: list[dict[str, object]] = []
    for item in shortlist.to_dict("records"):
        canonical = item["canonical_name"]
        for dataset_id, prefix, source_table in [
            ("PIC", "PIC", "labevents"),
            (
                "NHANES_2021_2023",
                "NHANES",
                "CBC_L" if item["NHANES_source_variable_id"] in {"LBXWBCSI", "LBXHGB", "LBXPLTSI"} else "BIOPRO_L",
            ),
        ]:
            source_label = item[f"{prefix}_source_label"]
            attribute_name = canonical
            canonical_name = canonical.replace("_", " ")
            target_unit = item["proposed_ucum_unit"]
            if canonical == "calcium" and dataset_id == "PIC":
                attribute_name = "ionized_calcium"
                canonical_name = "ionized calcium"
                target_unit = "mmol/L"
            elif canonical == "calcium":
                attribute_name = "total_calcium"
                canonical_name = "total calcium"
            source_unit, conversion_factor = (
                (("mmol/L", 1.0) if canonical == "calcium" else pic_units[canonical])
                if dataset_id == "PIC"
                else (target_unit, 1.0)
            )
            if source_table == "CBC_L" or canonical in {"hemoglobin", "platelet_count"}:
                system_hint = "whole blood"
            elif dataset_id == "PIC" and canonical in {"sodium", "potassium", "chloride", "calcium"}:
                system_hint = "whole blood"
            else:
                system_hint = "serum or plasma"
            anatomy_query = "blood"
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "attribute_name": attribute_name,
                    "source_variable_id": item[f"{prefix}_source_variable_id"],
                    "source_table": source_table,
                    "canonical_name": canonical_name,
                    "source_label": source_label,
                    "source_dictionary_label": "",
                    "source_dictionary_fluid": "",
                    "source_dictionary_category": "",
                    "source_dictionary_loinc": "",
                    "source_documentation_url": "",
                    "mapping_issue": "",
                    "query_text": (
                        "ionized calcium whole blood ion selective electrode"
                        if canonical == "calcium" and dataset_id == "PIC"
                        else f"{canonical_name} {system_hint}"
                    ),
                    "system_hint": system_hint,
                    "anatomy_query": anatomy_query,
                    "hpo_low_query": hpo_terms[canonical][0],
                    "hpo_high_query": hpo_terms[canonical][1],
                    "ucum_unit": target_unit,
                    "source_unit": source_unit,
                    "conversion_factor": conversion_factor,
                    "conversion_offset": 0.0,
                    "unit_conversion_source": "prespecified SI/conventional conversion; manual verification required",
                    "loinc_code": "",
                    "loinc_label": "",
                    "loinc_group_id": "",
                    "hpo_low_id": "",
                    "hpo_high_id": "",
                    "uberon_id": "",
                    "semantic_anchor": "",
                    "graph_weight": 1.0,
                    "reference_interval_id": "",
                    "lower_bound": "",
                    "upper_bound": "",
                    "direction": "both_abnormal",
                    "threshold_source": "",
                    "mapping_method": "candidate_generation_then_manual_review",
                    "mapping_confidence": 0.0,
                    "knowledge_version": "LOINC-2.82|HPO-2026-02-16|UBERON-2024-11-25",
                    "provenance": "",
                    "reviewer_status": "needs_review",
                    "primary_analysis_status": "included",
                    "primary_exclusion_reason": "",
                    "supplementary_sensitivity_set": "",
                }
            )
    output = PROJECT_ROOT / "configs/formal_attribute_mapping_working.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} review rows to {output}")


if __name__ == "__main__":
    main()
