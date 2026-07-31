"""Scan real PIC CSV files and build SemMed-LDP variable inventories."""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semmed_ldp.data_loader import (
    ADMISSION_ID_CANDIDATES,
    LABEL_CANDIDATES,
    NUMERIC_VALUE_CANDIDATES,
    PATIENT_ID_CANDIDATES,
    VARIABLE_ID_CANDIDATES,
    PICDataLoader,
    infer_column,
    load_yaml_config,
)
from semmed_ldp.umls_mapping import load_attribute_config, split_multi_value

LOGGER = logging.getLogger("scan_pic_variables")

CORE_SCAN_TABLES = [
    "labevents",
    "chartevents",
    "diagnoses_icd",
    "prescriptions",
    "inputevents",
    "outputevents",
    "microbiologyevents",
]


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def setup_logging(log_dir: Path) -> None:
    """Configure scanner logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "scan_pic_variables.log", encoding="utf-8")],
        force=True,
    )


def _read_header(path: Path) -> pd.DataFrame:
    """Read only a CSV header."""
    return pd.read_csv(path, nrows=0, low_memory=False)


def _load_dictionary(
    loader: PICDataLoader,
    table_name: str,
    id_candidates: list[str],
    label_candidates: list[str],
) -> dict[str, str]:
    """Load a dictionary table as an ID-to-label mapping."""
    frame = loader.read_table(table_name, required=False)
    if frame.empty:
        return {}
    id_col = infer_column(frame, id_candidates)
    label_col = infer_column(frame, label_candidates)
    if id_col is None or label_col is None:
        LOGGER.warning("Dictionary %s lacks an ID or label column; dictionary matching disabled.", table_name)
        return {}
    mapping = frame[[id_col, label_col]].dropna(subset=[id_col]).drop_duplicates(subset=[id_col])
    return dict(zip(mapping[id_col].astype(str), mapping[label_col].astype(str)))


def _update_sets(target: dict[str, set[str]], grouped: pd.Series) -> None:
    """Merge grouped unique values into a per-variable set accumulator."""
    for variable_id, values in grouped.items():
        target[str(variable_id)].update(str(value) for value in values if pd.notna(value))


def scan_grouped_table(
    loader: PICDataLoader,
    table_name: str,
    *,
    variable_candidates: list[str],
    value_candidates: list[str] | None = None,
    dictionary: dict[str, str] | None = None,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Scan a grouped clinical table and return one row per variable."""
    path = loader.resolve_table_path(table_name)
    if not path.exists():
        LOGGER.warning("Configured table %s does not exist at %s; skipping.", table_name, path)
        return pd.DataFrame()

    header = _read_header(path)
    variable_col = infer_column(header, variable_candidates)
    patient_col = infer_column(header, PATIENT_ID_CANDIDATES)
    admission_col = infer_column(header, ADMISSION_ID_CANDIDATES)
    value_col = infer_column(header, value_candidates or NUMERIC_VALUE_CANDIDATES)
    label_col = infer_column(header, LABEL_CANDIDATES)
    if variable_col is None:
        LOGGER.warning("Table %s has no variable column among %s; skipping.", table_name, variable_candidates)
        return pd.DataFrame()

    usecols = [col for col in [variable_col, patient_col, admission_col, value_col, label_col] if col is not None]
    usecols = list(dict.fromkeys(usecols))
    total_counts: defaultdict[str, int] = defaultdict(int)
    non_missing_counts: defaultdict[str, int] = defaultdict(int)
    numeric_counts: defaultdict[str, int] = defaultdict(int)
    patient_sets: dict[str, set[str]] = defaultdict(set)
    admission_sets: dict[str, set[str]] = defaultdict(set)
    observed_names: dict[str, str] = {}

    LOGGER.info("Scanning %s from %s with variable column %s.", table_name, path, variable_col)
    for chunk_idx, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False), start=1):
        chunk = chunk[chunk[variable_col].notna()].copy()
        if chunk.empty:
            continue
        variable_values = chunk[variable_col].astype(str)
        counts = variable_values.value_counts(dropna=False)
        for variable_id, count in counts.items():
            total_counts[str(variable_id)] += int(count)

        if value_col is not None:
            non_missing = chunk[value_col].notna()
            non_counts = variable_values[non_missing].value_counts(dropna=False)
            numeric = pd.to_numeric(chunk[value_col], errors="coerce").notna()
            numeric_grouped = variable_values[numeric].value_counts(dropna=False)
            for variable_id, count in non_counts.items():
                non_missing_counts[str(variable_id)] += int(count)
            for variable_id, count in numeric_grouped.items():
                numeric_counts[str(variable_id)] += int(count)
        else:
            for variable_id, count in counts.items():
                non_missing_counts[str(variable_id)] += int(count)

        if patient_col is not None:
            _update_sets(patient_sets, chunk.groupby(variable_values)[patient_col].unique())
        if admission_col is not None:
            _update_sets(admission_sets, chunk.groupby(variable_values)[admission_col].unique())
        if label_col is not None:
            labels = chunk[[variable_col, label_col]].dropna().drop_duplicates(subset=[variable_col])
            for _, row in labels.iterrows():
                observed_names.setdefault(str(row[variable_col]), str(row[label_col]))

        if chunk_idx % 10 == 0:
            LOGGER.info("Scanned %s chunks for %s; observed %d variables.", chunk_idx, table_name, len(total_counts))

    rows: list[dict[str, Any]] = []
    dictionary = dictionary or {}
    for variable_id, total_count in sorted(total_counts.items(), key=lambda item: (-item[1], item[0])):
        non_missing_count = non_missing_counts.get(variable_id, total_count)
        numeric_value_count = numeric_counts.get(variable_id, 0)
        matched_name = dictionary.get(variable_id) or observed_names.get(variable_id, "")
        rows.append(
            {
                "source_table": table_name,
                "variable_id": variable_id,
                "variable_name": matched_name,
                "total_count": total_count,
                "patient_count": len(patient_sets.get(variable_id, set())),
                "admission_count": len(admission_sets.get(variable_id, set())),
                "non_missing_count": non_missing_count,
                "missing_rate": 1 - (non_missing_count / total_count if total_count else 0),
                "numeric_value_count": numeric_value_count,
                "numeric_value_ratio": numeric_value_count / total_count if total_count else 0,
                "dictionary_matched": bool(dictionary.get(variable_id)),
            }
        )
    LOGGER.info("Finished %s with %d variables.", table_name, len(rows))
    return pd.DataFrame(rows)


def scan_diagnoses(loader: PICDataLoader, dictionary: dict[str, str], chunksize: int = 500_000) -> pd.DataFrame:
    """Scan DIAGNOSES_ICD.csv by ICD code."""
    path = loader.resolve_table_path("diagnoses_icd")
    if not path.exists():
        LOGGER.warning("DIAGNOSES_ICD.csv not found at %s; skipping.", path)
        return pd.DataFrame()
    header = _read_header(path)
    code_col = infer_column(header, ["icd_code", "icd10_code_cn", "code"])
    patient_col = infer_column(header, PATIENT_ID_CANDIDATES)
    admission_col = infer_column(header, ADMISSION_ID_CANDIDATES)
    version_col = infer_column(header, ["icd_version"])
    if code_col is None:
        LOGGER.warning("diagnoses_icd has no ICD code column; skipping.")
        return pd.DataFrame()

    usecols = [col for col in [code_col, patient_col, admission_col, version_col] if col is not None]
    total_counts: defaultdict[str, int] = defaultdict(int)
    patient_sets: dict[str, set[str]] = defaultdict(set)
    admission_sets: dict[str, set[str]] = defaultdict(set)
    versions: dict[str, str] = {}
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk = chunk[chunk[code_col].notna()].copy()
        code_values = chunk[code_col].astype(str)
        for code, count in code_values.value_counts().items():
            total_counts[str(code)] += int(count)
        if patient_col is not None:
            _update_sets(patient_sets, chunk.groupby(code_values)[patient_col].unique())
        if admission_col is not None:
            _update_sets(admission_sets, chunk.groupby(code_values)[admission_col].unique())
        if version_col is not None:
            for _, row in chunk[[code_col, version_col]].dropna().drop_duplicates(subset=[code_col]).iterrows():
                versions.setdefault(str(row[code_col]), str(row[version_col]))

    rows = []
    for code, total_count in sorted(total_counts.items(), key=lambda item: (-item[1], item[0])):
        rows.append(
            {
                "source_table": "diagnoses_icd",
                "variable_id": code,
                "variable_name": dictionary.get(code, ""),
                "total_count": total_count,
                "patient_count": len(patient_sets.get(code, set())),
                "admission_count": len(admission_sets.get(code, set())),
                "non_missing_count": total_count,
                "missing_rate": 0.0,
                "numeric_value_count": 0,
                "numeric_value_ratio": 0.0,
                "dictionary_matched": bool(dictionary.get(code)),
                "icd_version": versions.get(code, ""),
            }
        )
    LOGGER.info("Finished diagnoses_icd with %d ICD variables.", len(rows))
    return pd.DataFrame(rows)


def scan_prescriptions(loader: PICDataLoader, chunksize: int = 500_000) -> pd.DataFrame:
    """Scan PRESCRIPTIONS.csv by drug name."""
    return scan_grouped_table(
        loader,
        "prescriptions",
        variable_candidates=["drug_name", "drug", "drug_name_en", "drug_name_generic"],
        value_candidates=["drug_name", "drug", "drug_name_en", "drug_name_generic"],
        dictionary=None,
        chunksize=chunksize,
    )


def scan_pic_variables(loader: PICDataLoader, chunksize: int = 500_000) -> pd.DataFrame:
    """Scan first-stage PIC clinical variable tables."""
    lab_dict = _load_dictionary(loader, "d_labitems", ["itemid", "item_id"], LABEL_CANDIDATES)
    item_dict = _load_dictionary(loader, "d_items", ["itemid", "item_id"], LABEL_CANDIDATES)
    icd_dict = _load_dictionary(loader, "d_icd_diagnoses", ["icd10_code_cn", "icd_code", "code"], ["title", "title_cn", "long_title"])

    frames = [
        scan_grouped_table(loader, "labevents", variable_candidates=["itemid", "item_id"], dictionary=lab_dict, chunksize=chunksize),
        scan_grouped_table(loader, "chartevents", variable_candidates=["itemid", "item_id"], dictionary=item_dict, chunksize=chunksize),
        scan_diagnoses(loader, icd_dict, chunksize=chunksize),
        scan_prescriptions(loader, chunksize=chunksize),
        scan_grouped_table(loader, "inputevents", variable_candidates=VARIABLE_ID_CANDIDATES, chunksize=chunksize),
        scan_grouped_table(loader, "outputevents", variable_candidates=["itemid", "item_id"], dictionary=item_dict, chunksize=chunksize),
        scan_grouped_table(
            loader,
            "microbiologyevents",
            variable_candidates=["org_name", "spec_type_desc", "ab_name", "org_itemid", "spec_itemid", "ab_itemid"],
            value_candidates=["interpretation", "dilution_value", "org_name", "spec_type_desc", "ab_name"],
            chunksize=chunksize,
        ),
    ]
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    inventory = pd.concat(frames, ignore_index=True, sort=False)
    for column in [
        "source_table",
        "variable_id",
        "variable_name",
        "total_count",
        "patient_count",
        "admission_count",
        "non_missing_count",
        "missing_rate",
        "numeric_value_count",
        "numeric_value_ratio",
        "dictionary_matched",
    ]:
        if column not in inventory.columns:
            inventory[column] = ""
    return inventory


def select_candidate_variables(inventory: pd.DataFrame, scan_config: dict[str, Any]) -> pd.DataFrame:
    """Filter variable inventory into first-stage candidate variables."""
    if inventory.empty:
        return pd.DataFrame()
    min_total = int(scan_config.get("candidate_min_total_count", 100))
    min_patients = int(scan_config.get("candidate_min_patient_count", 50))
    min_numeric_ratio = float(scan_config.get("candidate_min_numeric_value_ratio", 0.5))
    base = inventory[(inventory["total_count"] >= min_total) & (inventory["patient_count"] >= min_patients)].copy()
    numeric_tables = {"labevents", "chartevents"}
    numeric_mask = base["source_table"].isin(numeric_tables)
    candidates = base[(~numeric_mask) | (base["numeric_value_ratio"] >= min_numeric_ratio)].copy()
    LOGGER.info(
        "Selected %d candidate variables from %d inventory rows using total>=%d, patients>=%d, numeric_ratio>=%.3f.",
        len(candidates),
        len(inventory),
        min_total,
        min_patients,
        min_numeric_ratio,
    )
    return candidates


def validate_semmed_attribute_config(attribute_config: pd.DataFrame, inventory: pd.DataFrame, loader: PICDataLoader) -> pd.DataFrame:
    """Validate configured SemMed attributes against the scanned PIC inventory."""
    if inventory.empty:
        inventory_keys: set[tuple[str, str]] = set()
        name_lookup: dict[tuple[str, str], str] = {}
    else:
        inventory = inventory.copy()
        inventory["source_table_norm"] = inventory["source_table"].map(loader.normalize_table_name)
        inventory["variable_id_str"] = inventory["variable_id"].astype(str)
        inventory_keys = set(zip(inventory["source_table_norm"], inventory["variable_id_str"]))
        name_lookup = dict(zip(zip(inventory["source_table_norm"], inventory["variable_id_str"]), inventory["variable_name"].fillna("")))
    configured_tables = {loader.normalize_table_name(table) for table in loader.tables_config}

    rows: list[dict[str, Any]] = []
    for _, row in attribute_config.iterrows():
        attribute_name = str(row.get("attribute_name", "")).strip()
        source_table = loader.normalize_table_name(str(row.get("source_table", "")).strip())
        ids = split_multi_value(row.get("variable_id", "")) or split_multi_value(row.get("itemid", ""))
        has_threshold = bool(str(row.get("lower_bound", "")).strip() or str(row.get("upper_bound", "")).strip())
        has_umls_cui = bool(str(row.get("umls_cui", "")).strip())
        semantic_group = str(row.get("semantic_group", "")).strip()

        if not ids:
            exists = False
            matched_name = ""
            warning_parts = ["missing itemid/variable_id"]
        else:
            matches = [(source_table, variable_id) for variable_id in ids if (source_table, variable_id) in inventory_keys]
            exists = bool(matches)
            matched_name = "|".join(name_lookup.get(match, "") for match in matches)
            warning_parts = [] if exists else ["itemid/variable_id not found in scanned inventory"]

        if not attribute_name:
            warning_parts.append("attribute_name is empty")
        if source_table not in configured_tables:
            warning_parts.append("source_table is not configured")
        if not semantic_group:
            warning_parts.append("semantic_group is empty")
        if not has_threshold:
            warning_parts.append("missing lower_bound/upper_bound; fallback threshold required")

        rows.append(
            {
                "attribute_name": attribute_name,
                "source_table": source_table,
                "variable_id": "|".join(ids),
                "exists_in_pic": exists,
                "matched_name": matched_name,
                "semantic_group": semantic_group,
                "has_umls_cui": has_umls_cui,
                "has_threshold": has_threshold,
                "fallback_required": not has_threshold,
                "warning_message": "; ".join(warning_parts),
            }
        )
    validation = pd.DataFrame(rows)
    LOGGER.info("Validated %d SemMed attribute config rows.", len(validation))
    return validation


def main(config_path: str | Path = PROJECT_ROOT / "configs" / "experiment_config.yaml") -> None:
    """Run PIC variable scanning and config validation."""
    config = load_yaml_config(config_path)
    paths = config.get("paths", {})
    tables_dir = resolve_project_path(paths.get("tables_dir", "results/tables"))
    logs_dir = resolve_project_path(paths.get("logs_dir", "results/logs"))
    setup_logging(logs_dir)

    pic_paths = resolve_project_path(paths.get("pic_paths", "configs/pic_paths.yaml"))
    loader = PICDataLoader(pic_paths, project_root=PROJECT_ROOT)
    raw_dir = loader.raw_data_dir if loader.raw_data_dir.is_absolute() else PROJECT_ROOT / loader.raw_data_dir
    if not str(raw_dir).strip() or not raw_dir.exists():
        LOGGER.error("请先在 configs/pic_paths.yaml 中填写真实 PIC 解压目录: %s", raw_dir)
        print("请先在 configs/pic_paths.yaml 中填写真实 PIC 解压目录")
        return

    tables_dir.mkdir(parents=True, exist_ok=True)
    metadata = loader.check_table_paths()
    metadata.to_csv(tables_dir / "pic_table_path_check.csv", index=False)
    for _, row in metadata.iterrows():
        if not row["exists"]:
            LOGGER.warning("Configured PIC table %s is missing at %s.", row["table_name"], row["path"])

    inventory = scan_pic_variables(loader)
    inventory_path = tables_dir / "pic_variable_inventory.csv"
    inventory.to_csv(inventory_path, index=False)
    LOGGER.info("Saved PIC variable inventory to %s with %d rows.", inventory_path, len(inventory))

    candidates = select_candidate_variables(inventory, config.get("variable_scan", {}))
    candidates_path = tables_dir / "pic_variable_candidates.csv"
    candidates.to_csv(candidates_path, index=False)
    LOGGER.info("Saved PIC variable candidates to %s with %d rows.", candidates_path, len(candidates))

    attribute_config = load_attribute_config(resolve_project_path(paths.get("attribute_config", "configs/semmed_attribute_config.csv")))
    validation = validate_semmed_attribute_config(attribute_config, inventory, loader)
    validation_path = tables_dir / "semmed_config_validation.csv"
    validation.to_csv(validation_path, index=False)
    LOGGER.info("Saved SemMed config validation to %s with %d rows.", validation_path, len(validation))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "configs" / "experiment_config.yaml")
