"""Build a source-audited but unapproved reference-interval worksheet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BMAMA = "https://pediatric.testcatalog.org/show/BMAMA"
ALBUMIN = "https://pediatric.testcatalog.org/show/ALB"
MAGNESIUM = "https://pediatric.testcatalog.org/show/MGS"
PHOSPHATE = "https://pediatric.testcatalog.org/show/PHOS"
CBC = "https://pediatric.testcatalog.org/show/CBC"
IONIZED_CALCIUM = "https://pediatric.testcatalog.org/show/CAI"


def add(rows: list[dict], interval_id: str, age_low: float, age_high: float, sex: str, low: float, high: float, unit: str, source: str) -> None:
    rows.append(
        {
            "reference_interval_id": interval_id,
            "min_age_years": age_low,
            "max_age_years": age_high,
            "sex": sex,
            "lower_bound": low,
            "upper_bound": high,
            "unit": unit,
            "source": source,
            "source_accessed": "2026-07-14",
            "reviewer_status": "needs_review",
        }
    )


def main() -> None:
    rows: list[dict] = []
    mapping = pd.read_csv(PROJECT_ROOT / "configs/formal_attribute_mapping_working.csv", dtype=str).fillna("")

    for attribute, low, high, unit, source in [
        ("sodium", 135, 145, "mmol/L", BMAMA),
        ("potassium", 3.6, 5.2, "mmol/L", BMAMA),
        ("chloride", 102, 112, "mmol/L", BMAMA),
        ("urea_nitrogen", 7, 20, "mg/dL", BMAMA),
        ("glucose", 70, 140, "mg/dL", BMAMA),
        ("albumin", 3.5, 5.0, "g/dL", ALBUMIN),
    ]:
        add(rows, f"PIC_{attribute}", 1, 18, "any", low, high, unit, source)

    for sex, partitions in {
        "male": [(1, 6, 0.19, 0.49), (6, 11, 0.26, 0.61), (11, 15, 0.35, 0.86), (15, 18, 0.74, 1.35)],
        "female": [(1, 6, 0.19, 0.49), (6, 11, 0.26, 0.61), (11, 16, 0.35, 0.86), (16, 18, 0.59, 1.04)],
    }.items():
        for age_low, age_high, low, high in partitions:
            add(rows, "PIC_creatinine", age_low, age_high, sex, low, high, "mg/dL", BMAMA)

    for age_low, age_high, low_mg, high_mg in [(1, 2, 5.04, 5.84), (2, 3, 4.87, 5.67), (3, 18, 4.83, 5.52)]:
        add(
            rows,
            "PIC_ionized_calcium",
            age_low,
            age_high,
            "any",
            low_mg / 4.008,
            high_mg / 4.008,
            "mmol/L",
            IONIZED_CALCIUM + " (mg/dL converted to mmol/L by division by 4.008)",
        )

    for age_low, age_high, low, high in [(1, 3, 1.6, 2.7), (3, 6, 1.6, 2.6), (6, 9, 1.6, 2.5), (9, 12, 1.6, 2.4), (12, 18, 1.6, 2.3)]:
        add(rows, "PIC_magnesium", age_low, age_high, "any", low, high, "mg/dL", MAGNESIUM)

    for sex, partitions in {
        "male": [(1, 5, 4.3, 5.4), (5, 14, 3.7, 5.4), (14, 16, 3.5, 5.3), (16, 18, 3.1, 4.7)],
        "female": [(1, 8, 4.3, 5.4), (8, 14, 4.0, 5.2), (14, 16, 3.5, 4.9), (16, 18, 3.1, 4.7)],
    }.items():
        for age_low, age_high, low, high in partitions:
            add(rows, "PIC_phosphate", age_low, age_high, sex, low, high, "mg/dL", PHOSPHATE)

    for sex, partitions in {
        "male": [(1, 2, 10.1, 12.5), (2, 3, 10.2, 12.7), (3, 6, 11.4, 14.3), (6, 9, 11.5, 14.3), (9, 11, 11.8, 14.7), (11, 15, 12.4, 15.7), (15, 18, 13.3, 16.9)],
        "female": [(1, 3, 10.2, 12.7), (3, 6, 11.4, 14.3), (6, 9, 11.5, 14.3), (9, 11, 11.8, 14.7), (11, 18, 11.9, 14.8)],
    }.items():
        for age_low, age_high, low, high in partitions:
            add(rows, "PIC_hemoglobin", age_low, age_high, sex, low, high, "g/dL", CBC)

    for sex, partitions in {
        "male": [(1, 2, 206, 445), (2, 3, 202, 403), (3, 6, 187, 445), (6, 10, 187, 400), (10, 14, 177, 381), (14, 18, 139, 320)],
        "female": [(1, 2, 214, 459), (2, 3, 189, 394), (3, 6, 187, 445), (6, 10, 187, 400), (10, 14, 177, 381), (14, 18, 158, 362)],
    }.items():
        for age_low, age_high, low, high in partitions:
            add(rows, "PIC_platelet_count", age_low, age_high, sex, low, high, "10*3/uL", CBC)

    for attribute, low, high, unit, source in [
        ("sodium", 135, 145, "mmol/L", BMAMA),
        ("potassium", 3.6, 5.2, "mmol/L", BMAMA),
        ("chloride", 98, 107, "mmol/L", BMAMA),
        ("glucose", 70, 140, "mg/dL", BMAMA),
        ("albumin", 3.5, 5.0, "g/dL", ALBUMIN),
        ("magnesium", 1.7, 2.3, "mg/dL", MAGNESIUM),
        ("phosphate", 2.5, 4.5, "mg/dL", PHOSPHATE),
    ]:
        add(rows, f"NHANES_{attribute}", 18, 151, "any", low, high, unit, source)
    for sex, low, high in [("male", 0.74, 1.35), ("female", 0.59, 1.04)]:
        add(rows, "NHANES_creatinine", 18, 151, sex, low, high, "mg/dL", BMAMA)
    for sex, low, high in [("male", 8, 24), ("female", 6, 21)]:
        add(rows, "NHANES_urea_nitrogen", 18, 151, sex, low, high, "mg/dL", BMAMA)
    add(rows, "NHANES_total_calcium", 18, 60, "any", 8.6, 10.0, "mg/dL", BMAMA)
    add(rows, "NHANES_total_calcium", 60, 91, "any", 8.8, 10.2, "mg/dL", BMAMA)
    add(rows, "NHANES_total_calcium", 91, 151, "any", 8.2, 9.6, "mg/dL", BMAMA)
    for sex, low, high in [("male", 13.2, 16.6), ("female", 11.6, 15.0)]:
        add(rows, "NHANES_hemoglobin", 18, 151, sex, low, high, "g/dL", CBC)
    for sex, low, high in [("male", 135, 317), ("female", 157, 371)]:
        add(rows, "NHANES_platelet_count", 18, 151, sex, low, high, "10*3/uL", CBC)

    intervals = pd.DataFrame(rows)
    intervals["applicability_note"] = "specimen and population aligned in the cited public catalog; manual confirmation required"
    whole_blood_serum = {
        "PIC_sodium", "PIC_potassium", "PIC_chloride", "PIC_creatinine",
        "PIC_urea_nitrogen", "PIC_glucose", "PIC_albumin", "PIC_magnesium", "PIC_phosphate",
    }
    intervals.loc[intervals["reference_interval_id"].isin(whole_blood_serum), "applicability_note"] = (
        "PIC dictionary reports Blood while the cited interval is serum-based; specimen/method transferability requires domain review"
    )
    intervals.loc[intervals["reference_interval_id"].eq("PIC_ionized_calcium"), "applicability_note"] = (
        "PIC is whole-blood ionized calcium; cited interval is serum ionized calcium adjusted to pH 7.40; domain review is mandatory"
    )
    intervals.to_csv(PROJECT_ROOT / "configs/reference_intervals_working.csv", index=False)
    for index, row in mapping.iterrows():
        prefix = "PIC" if row["dataset_id"] == "PIC" else "NHANES"
        interval_id = f'{prefix}_{row["attribute_name"]}'
        source = intervals.loc[intervals["reference_interval_id"].eq(interval_id), "source"].iloc[0]
        mapping.loc[index, "reference_interval_id"] = interval_id
        mapping.loc[index, "threshold_source"] = source
    mapping.to_csv(PROJECT_ROOT / "configs/formal_attribute_mapping_working.csv", index=False)
    print(f"Wrote {len(intervals)} unapproved interval strata and linked {len(mapping)} mapping rows")


if __name__ == "__main__":
    main()
