"""Build the first real SemMed-LDP attribute config from scanned PIC candidates."""

from __future__ import annotations

import difflib
import logging
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semmed_ldp.data_loader import load_yaml_config

LOGGER = logging.getLogger("build_initial_semmed_config")


TARGET_VARIABLES: list[dict[str, Any]] = [
    {
        "target_name": "Temperature",
        "source_table": "chartevents",
        "attribute_name": "temperature",
        "semantic_group": "VitalSign",
        "clinical_concepts": "Temperature;Fever;Hypothermia",
        "lower_bound": 36.0,
        "upper_bound": 38.0,
        "direction": "both_abnormal",
    },
    {
        "target_name": "Respiratory Rate",
        "source_table": "chartevents",
        "attribute_name": "respiratory_rate",
        "semantic_group": "Respiratory",
        "clinical_concepts": "Respiratory Rate;Tachypnea;Bradypnea",
        "lower_bound": "",
        "upper_bound": "",
        "direction": "both_abnormal",
    },
    {
        "target_name": "Heart Rate",
        "source_table": "chartevents",
        "attribute_name": "heart_rate",
        "semantic_group": "Cardiovascular",
        "clinical_concepts": "Heart Rate;Tachycardia;Bradycardia",
        "lower_bound": "",
        "upper_bound": "",
        "direction": "both_abnormal",
    },
    {
        "target_name": "Systolic Pressure",
        "source_table": "chartevents",
        "attribute_name": "systolic_pressure",
        "semantic_group": "Cardiovascular",
        "clinical_concepts": "Systolic Blood Pressure;Hypotension;Hypertension",
        "lower_bound": "",
        "upper_bound": "",
        "direction": "both_abnormal",
    },
    {
        "target_name": "Diastolic Pressure",
        "source_table": "chartevents",
        "attribute_name": "diastolic_pressure",
        "semantic_group": "Cardiovascular",
        "clinical_concepts": "Diastolic Blood Pressure;Hypotension;Hypertension",
        "lower_bound": "",
        "upper_bound": "",
        "direction": "both_abnormal",
    },
    {
        "target_name": "Oxygen saturation",
        "source_table": "chartevents",
        "attribute_name": "oxygen_saturation",
        "semantic_group": "Respiratory",
        "clinical_concepts": "Oxygen Saturation;Hypoxemia",
        "lower_bound": 90.0,
        "upper_bound": "",
        "direction": "low_abnormal",
    },
    {
        "target_name": "WBC Count",
        "source_table": "labevents",
        "attribute_name": "wbc_count",
        "semantic_group": "Infection",
        "clinical_concepts": "White Blood Cell Count;Leukocytosis;Leukopenia",
        "lower_bound": 4.0,
        "upper_bound": 12.0,
        "direction": "both_abnormal",
    },
    {
        "target_name": "Neutrophils %",
        "source_table": "labevents",
        "attribute_name": "neutrophils_percent",
        "semantic_group": "Infection",
        "clinical_concepts": "Neutrophils;Neutrophilia",
        "lower_bound": "",
        "upper_bound": 70.0,
        "direction": "high_abnormal",
    },
    {
        "target_name": "Lymphocytes, Percent",
        "source_table": "labevents",
        "attribute_name": "lymphocytes_percent",
        "semantic_group": "Hematology",
        "clinical_concepts": "Lymphocytes;Lymphopenia;Lymphocytosis",
        "lower_bound": 20.0,
        "upper_bound": 50.0,
        "direction": "both_abnormal",
    },
    {
        "target_name": "Platelet Count",
        "source_table": "labevents",
        "attribute_name": "platelet_count",
        "semantic_group": "Hematology",
        "clinical_concepts": "Platelet Count;Thrombocytopenia;Thrombocytosis",
        "lower_bound": 150.0,
        "upper_bound": 450.0,
        "direction": "both_abnormal",
    },
    {
        "target_name": "Hemoglobin",
        "source_table": "labevents",
        "attribute_name": "hemoglobin",
        "semantic_group": "Hematology",
        "clinical_concepts": "Hemoglobin;Anemia;Polycythemia",
        "lower_bound": "",
        "upper_bound": "",
        "direction": "both_abnormal",
    },
    {
        "target_name": "Hematocrit",
        "source_table": "labevents",
        "attribute_name": "hematocrit",
        "semantic_group": "Hematology",
        "clinical_concepts": "Hematocrit;Anemia;Hemoconcentration",
        "lower_bound": "",
        "upper_bound": "",
        "direction": "both_abnormal",
    },
    {
        "target_name": "Sodium, Whole Blood",
        "source_table": "labevents",
        "attribute_name": "sodium_whole_blood",
        "semantic_group": "Electrolyte",
        "clinical_concepts": "Sodium;Hyponatremia;Hypernatremia",
        "lower_bound": 135.0,
        "upper_bound": 145.0,
        "direction": "both_abnormal",
    },
    {
        "target_name": "Potassium",
        "source_table": "labevents",
        "attribute_name": "potassium",
        "semantic_group": "Electrolyte",
        "clinical_concepts": "Potassium;Hypokalemia;Hyperkalemia",
        "lower_bound": 3.5,
        "upper_bound": 5.5,
        "direction": "both_abnormal",
    },
    {
        "target_name": "Chloride, Whole Blood",
        "source_table": "labevents",
        "attribute_name": "chloride_whole_blood",
        "semantic_group": "Electrolyte",
        "clinical_concepts": "Chloride;Hypochloremia;Hyperchloremia",
        "lower_bound": 98.0,
        "upper_bound": 110.0,
        "direction": "both_abnormal",
    },
    {
        "target_name": "Calcium, Total",
        "source_table": "labevents",
        "attribute_name": "calcium_total",
        "semantic_group": "Electrolyte",
        "clinical_concepts": "Calcium;Hypocalcemia;Hypercalcemia",
        "lower_bound": 1.1,
        "upper_bound": 1.35,
        "direction": "both_abnormal",
    },
    {
        "target_name": "pH",
        "source_table": "labevents",
        "attribute_name": "ph",
        "semantic_group": "AcidBase",
        "clinical_concepts": "pH;Acidosis;Alkalosis",
        "lower_bound": 7.35,
        "upper_bound": 7.45,
        "direction": "both_abnormal",
    },
    {
        "target_name": "pCO2",
        "source_table": "labevents",
        "attribute_name": "pco2",
        "semantic_group": "AcidBase",
        "clinical_concepts": "Carbon Dioxide Partial Pressure;Hypercapnia;Hypocapnia",
        "lower_bound": 35.0,
        "upper_bound": 45.0,
        "direction": "both_abnormal",
    },
    {
        "target_name": "pO2",
        "source_table": "labevents",
        "attribute_name": "po2",
        "semantic_group": "Respiratory",
        "clinical_concepts": "Oxygen Partial Pressure;Hypoxemia",
        "lower_bound": 60.0,
        "upper_bound": "",
        "direction": "low_abnormal",
    },
    {
        "target_name": "Calculated Bicarbonate, Whole Blood",
        "source_table": "labevents",
        "attribute_name": "calculated_bicarbonate_whole_blood",
        "semantic_group": "AcidBase",
        "clinical_concepts": "Bicarbonate;Metabolic Acidosis;Metabolic Alkalosis",
        "lower_bound": 22.0,
        "upper_bound": 28.0,
        "direction": "both_abnormal",
    },
    {
        "target_name": "Lactate",
        "source_table": "labevents",
        "attribute_name": "lactate",
        "semantic_group": "Perfusion",
        "clinical_concepts": "Lactate;Hyperlactatemia;Tissue Hypoperfusion",
        "lower_bound": "",
        "upper_bound": 2.0,
        "direction": "high_abnormal",
    },
    {
        "target_name": "Glucose",
        "source_table": "labevents",
        "attribute_name": "glucose",
        "semantic_group": "Metabolic",
        "clinical_concepts": "Glucose;Hypoglycemia;Hyperglycemia",
        "lower_bound": 70.0,
        "upper_bound": 180.0,
        "direction": "both_abnormal",
    },
    {
        "target_name": "Creatinine",
        "source_table": "labevents",
        "attribute_name": "creatinine",
        "semantic_group": "Renal",
        "clinical_concepts": "Creatinine;Renal Dysfunction",
        "lower_bound": "",
        "upper_bound": "",
        "direction": "high_abnormal",
    },
    {
        "target_name": "Urea",
        "source_table": "labevents",
        "attribute_name": "urea",
        "semantic_group": "Renal",
        "clinical_concepts": "Urea;Blood Urea Nitrogen;Renal Dysfunction",
        "lower_bound": "",
        "upper_bound": "",
        "direction": "high_abnormal",
    },
    {
        "target_name": "Alanine Aminotransferase (ALT)",
        "source_table": "labevents",
        "attribute_name": "alanine_aminotransferase_alt",
        "semantic_group": "Liver",
        "clinical_concepts": "Alanine Aminotransferase;Liver Injury",
        "lower_bound": "",
        "upper_bound": "",
        "direction": "high_abnormal",
    },
    {
        "target_name": "Asparate Aminotransferase (AST)",
        "source_table": "labevents",
        "attribute_name": "aspartate_aminotransferase_ast",
        "semantic_group": "Liver",
        "clinical_concepts": "Aspartate Aminotransferase;Liver Injury",
        "lower_bound": "",
        "upper_bound": "",
        "direction": "high_abnormal",
    },
    {
        "target_name": "Bilirubin, Total",
        "source_table": "labevents",
        "attribute_name": "bilirubin_total",
        "semantic_group": "Liver",
        "clinical_concepts": "Bilirubin;Hyperbilirubinemia;Liver Dysfunction",
        "lower_bound": "",
        "upper_bound": "",
        "direction": "high_abnormal",
    },
    {
        "target_name": "Albumin",
        "source_table": "labevents",
        "attribute_name": "albumin",
        "semantic_group": "Nutrition_Liver",
        "clinical_concepts": "Albumin;Hypoalbuminemia;Nutritional Status",
        "lower_bound": "",
        "upper_bound": "",
        "direction": "low_abnormal",
    },
]


def normalize_name(value: str) -> str:
    """Normalize a clinical variable name for robust matching."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def setup_logging(log_dir: Path) -> None:
    """Configure config-builder logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "build_initial_semmed_config.log", encoding="utf-8")],
        force=True,
    )


def match_target(candidates: pd.DataFrame, target: dict[str, Any]) -> tuple[pd.Series | None, str, float]:
    """Match one target variable against PIC candidates with exact then fuzzy logic."""
    subset = candidates[candidates["source_table"].astype(str).str.lower() == target["source_table"]].copy()
    if subset.empty:
        return None, "source_table_missing", 0.0
    subset["name_norm"] = subset["variable_name"].fillna("").map(normalize_name)
    target_norm = normalize_name(target["target_name"])

    exact = subset[subset["name_norm"] == target_norm]
    if not exact.empty:
        exact = exact.sort_values(["patient_count", "total_count"], ascending=False)
        return exact.iloc[0], "exact", 1.0

    names = subset["name_norm"].tolist()
    match = difflib.get_close_matches(target_norm, names, n=1, cutoff=0.72)
    if match:
        fuzzy = subset[subset["name_norm"] == match[0]].sort_values(["patient_count", "total_count"], ascending=False)
        score = difflib.SequenceMatcher(None, target_norm, match[0]).ratio()
        return fuzzy.iloc[0], "fuzzy", score

    contains = subset[subset["name_norm"].str.contains(target_norm, regex=False) | subset["name_norm"].map(lambda name: name in target_norm)]
    if not contains.empty:
        contains = contains.sort_values(["patient_count", "total_count"], ascending=False)
        score = difflib.SequenceMatcher(None, target_norm, contains.iloc[0]["name_norm"]).ratio()
        return contains.iloc[0], "contains", score

    return None, "not_found", 0.0


def build_initial_config(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the initial SemMed attribute config and a matching report."""
    candidates = candidates[candidates["source_table"].isin(["labevents", "chartevents"])].copy()
    candidates = candidates[candidates["numeric_value_ratio"] >= 0.5].copy()
    rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    used_keys: set[tuple[str, str]] = set()

    for target in TARGET_VARIABLES:
        matched, status, score = match_target(candidates, target)
        if matched is None:
            report_rows.append(
                {
                    "target_name": target["target_name"],
                    "source_table": target["source_table"],
                    "match_status": status,
                    "match_score": score,
                    "variable_id": "",
                    "variable_name": "",
                    "patient_count": 0,
                    "total_count": 0,
                }
            )
            continue

        key = (str(matched["source_table"]), str(matched["variable_id"]))
        if key in used_keys:
            report_rows.append(
                {
                    "target_name": target["target_name"],
                    "source_table": target["source_table"],
                    "match_status": "duplicate_skipped",
                    "match_score": score,
                    "variable_id": matched["variable_id"],
                    "variable_name": matched["variable_name"],
                    "patient_count": matched["patient_count"],
                    "total_count": matched["total_count"],
                }
            )
            continue
        used_keys.add(key)

        rows.append(
            {
                "attribute_name": target["attribute_name"],
                "variable_id": str(matched["variable_id"]),
                "itemid": str(matched["variable_id"]),
                "source_table": matched["source_table"],
                "umls_cui": "",
                "umls_name": target["attribute_name"],
                "semantic_group": target["semantic_group"],
                "clinical_concepts": target["clinical_concepts"],
                "lower_bound": target["lower_bound"],
                "upper_bound": target["upper_bound"],
                "direction": target["direction"],
            }
        )
        report_rows.append(
            {
                "target_name": target["target_name"],
                "source_table": target["source_table"],
                "match_status": status,
                "match_score": score,
                "variable_id": matched["variable_id"],
                "variable_name": matched["variable_name"],
                "selected_attribute_name": target["attribute_name"],
                "semantic_group": target["semantic_group"],
                "patient_count": matched["patient_count"],
                "total_count": matched["total_count"],
                "numeric_value_ratio": matched["numeric_value_ratio"],
                "has_threshold": bool(str(target["lower_bound"]) or str(target["upper_bound"])),
            }
        )

    config = pd.DataFrame(rows)
    report = pd.DataFrame(report_rows)
    LOGGER.info("Selected %d SemMed attributes; %d targets unmatched.", len(config), int((report["match_status"] == "not_found").sum()) if not report.empty else 0)
    return config, report


def main(config_path: str | Path = PROJECT_ROOT / "configs" / "experiment_config.yaml") -> None:
    """Generate ``configs/semmed_attribute_config.csv`` from candidate variables."""
    experiment_config = load_yaml_config(config_path)
    paths = experiment_config.get("paths", {})
    tables_dir = resolve_project_path(paths.get("tables_dir", "results/tables"))
    logs_dir = resolve_project_path(paths.get("logs_dir", "results/logs"))
    setup_logging(logs_dir)

    candidates_path = tables_dir / "pic_variable_candidates.csv"
    if not candidates_path.exists():
        raise FileNotFoundError(f"Missing {candidates_path}. Run experiments/scan_pic_variables.py first.")
    candidates = pd.read_csv(candidates_path)
    config, report = build_initial_config(candidates)

    selected_path = tables_dir / "initial_semmed_selected_variables.csv"
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(selected_path, index=False)
    LOGGER.info("Saved matching report to %s.", selected_path)

    output_config = resolve_project_path(paths.get("attribute_config", "configs/semmed_attribute_config.csv"))
    output_config.parent.mkdir(parents=True, exist_ok=True)
    config.to_csv(output_config, index=False)
    LOGGER.info("Wrote initial SemMed attribute config to %s with %d rows.", output_config, len(config))

    failed = report[~report["match_status"].isin(["exact", "fuzzy", "contains"])]
    if not failed.empty:
        LOGGER.warning("Unmatched or skipped targets:\n%s", failed[["target_name", "source_table", "match_status"]].to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "configs" / "experiment_config.yaml")
