"""Create a patient-free review packet for candidate knowledge evidence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def display(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    mapping = pd.read_csv(ROOT / "configs/formal_attribute_mapping_working.csv")
    intervals = pd.read_csv(ROOT / "configs/reference_intervals_working.csv")
    decisions = mapping[["dataset_id", "attribute_name", "primary_analysis_status"]].copy()
    decisions["mapping_decision"] = "needs_review"
    decisions["reference_interval_decision"] = "needs_review"
    decisions["graph_decision"] = "needs_review"
    decisions["reviewer_name"] = ""
    decisions["review_date"] = ""
    decisions["comments"] = ""
    decisions["decision_basis"] = ""
    decision_path = ROOT / "configs/clinical_review_decisions.csv"
    if not decision_path.exists():
        decisions.to_csv(decision_path, index=False)

    interval_notes = intervals.groupby("reference_interval_id")["applicability_note"].first().to_dict()
    lines = [
        "# SemMed-LDP clinical knowledge review packet",
        "",
        "This packet contains no patient-level values. It is a review aid, not evidence of clinical approval.",
        "Approve only after checking the source variable, specimen, unit conversion, LOINC concept, HPO directions, semantic anchor, and cited reference interval.",
        "",
        "## Known review priorities",
        "",
        "- PIC ionized calcium: the raw dictionary's English label says `Calcium, Total`, but its Chinese label and observed scale indicate ionized calcium; the cited Mayo interval is serum ionized calcium adjusted to pH 7.40 whereas PIC reports whole blood.",
        "- PIC chemistry specimen: the PIC dictionary often reports generic `Blood`; several cited intervals are serum-based, so transferability must be explicitly adjudicated.",
        "- PIC glucose: the selected LOINC represents the standardized post-conversion mg/dL blood value; confirm whether coding should instead preserve the source mmol/L observation.",
        "- Platelets: LOINC 777-3 is supported by the PIC dictionary and the automated NHANES CBC method.",
        "- Scheme A primary scope: potassium and calcium are excluded as paired PIC/NHANES attributes and retained only for supplementary sensitivity analyses.",
        "",
    ]
    columns = [
        "Dataset", "Attribute", "Primary status", "Source", "Source evidence", "Source-to-target unit", "LOINC",
        "HPO low/high", "UBERON", "Anchor", "Confidence", "Interval", "Applicability / issue",
    ]
    for dataset_id, group in mapping.groupby("dataset_id", sort=False):
        lines.extend([f"## {dataset_id}", "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)])
        for row in group.itertuples(index=False):
            source_evidence = "; ".join(
                display(value)
                for value in [
                    getattr(row, "source_dictionary_label", ""),
                    getattr(row, "source_dictionary_fluid", ""),
                    getattr(row, "source_documentation_url", ""),
                ]
                if pd.notna(value) and str(value).strip()
            )
            issue_parts = [interval_notes.get(str(row.reference_interval_id), ""), getattr(row, "mapping_issue", "")]
            values = [
                row.dataset_id,
                row.attribute_name,
                row.primary_analysis_status,
                f"{row.source_table}:{row.source_variable_id} ({row.source_label})",
                source_evidence,
                f"{row.source_unit} x {row.conversion_factor} + {row.conversion_offset} to {row.ucum_unit}",
                f"{row.loinc_code} {row.loinc_label}",
                f"{row.hpo_low_id} / {row.hpo_high_id}",
                row.uberon_id,
                row.semantic_anchor,
                f"{row.mapping_confidence}: {row.mapping_confidence_basis}",
                f"{row.reference_interval_id}; {row.threshold_source}",
                "; ".join(display(part) for part in issue_parts if pd.notna(part) and str(part).strip()),
            ]
            lines.append("| " + " | ".join(display(value) for value in values) + " |")
        lines.append("")
    lines.extend(
        [
            "## Decision procedure",
            "",
            "Record decisions in `configs/clinical_review_decisions.csv`. For included primary rows, each decision column must be `approved`, and reviewer name/date must be present, before formal files can be created. Excluded rows remain audit-only.",
            "Do not edit `reviewer_status` directly in the working mapping or interval files.",
            "",
        ]
    )
    output = ROOT / "results/readiness/clinical_review_packet.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output} and review decision worksheet")


if __name__ == "__main__":
    main()
