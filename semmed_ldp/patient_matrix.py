"""Build patient-level feature matrices from PIC source tables."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


COMMON_ENTITY_COLUMNS = {
    "patient_id": ["patient_id", "subject_id", "pid", "hadm_subject_id", "SUBJECT_ID"],
    "admission_id": ["admission_id", "hadm_id", "adm_id", "visit_id", "stay_id", "encounter_id", "HADM_ID"],
}

COMMON_VALUE_COLUMNS = ["value_numeric", "valuenum", "value_num", "numeric_value", "value", "result_value"]
COMMON_ITEM_COLUMNS = ["itemid", "item_id", "label", "concept_id", "diagnosis_code", "drug_code", "code", "icd_code", "icd10_code_cn", "drug_name"]


def _first_existing(columns: pd.Index, candidates: list[str]) -> str | None:
    lower_to_original = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def _split_config_values(value: Any) -> list[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    text = str(value)
    for sep in ["|", ";"]:
        text = text.replace(sep, ",")
    return [item.strip() for item in text.split(",") if item.strip()]


def get_config_variable_ids(attr: pd.Series) -> list[str]:
    """Return configured variable IDs from ``variable_id`` or legacy ``itemid``."""
    variable_ids = _split_config_values(attr.get("variable_id", ""))
    if variable_ids:
        return variable_ids
    return _split_config_values(attr.get("itemid", ""))


def _coerce_optional_number(value: Any) -> float | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def infer_entity_column(frame: pd.DataFrame, entity_id: str) -> str | None:
    """Infer the patient or admission identifier column from a source table."""
    if entity_id in frame.columns:
        return entity_id
    return _first_existing(frame.columns, COMMON_ENTITY_COLUMNS.get(entity_id, [entity_id]))


def build_patient_feature_matrix(
    tables: dict[str, pd.DataFrame],
    attribute_config: pd.DataFrame,
    table_configs: dict[str, dict[str, Any]] | None = None,
    *,
    entity_id: str = "patient_id",
    aggregation: str = "median",
    max_missing_rate: float = 1.0,
    output_path: str | Path | None = None,
    missing_report_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct a patient-level high-dimensional feature matrix.

    Parameters
    ----------
    tables:
        Mapping from source table name to raw PIC table ``DataFrame``.
    attribute_config:
        SemMed attribute configuration. Each row defines one target attribute.
    table_configs:
        Optional path-config metadata that identifies table column names.
    entity_id:
        Identifier level to aggregate to, usually ``patient_id`` or
        ``admission_id``.
    aggregation:
        Aggregation method for repeated numeric values. The first version
        defaults to ``median``.
    max_missing_rate:
        Columns above this missing rate are dropped from the returned matrix.
    output_path:
        Optional CSV path for the completed feature matrix.
    missing_report_path:
        Optional CSV path for per-attribute missingness metadata.
    """
    if attribute_config.empty:
        LOGGER.warning("Attribute config is empty; returning an empty patient feature matrix.")
        return pd.DataFrame(), pd.DataFrame()

    table_configs = table_configs or {}
    feature_series: list[pd.Series] = []
    skipped_rows: list[dict[str, Any]] = []

    for _, attr in attribute_config.iterrows():
        attribute_name = str(attr.get("attribute_name", "")).strip()
        source_table = str(attr.get("source_table", "")).strip()
        if not attribute_name or not source_table:
            skipped_rows.append({"attribute_name": attribute_name, "source_table": source_table, "reason": "missing_name_or_source"})
            continue

        frame = tables.get(source_table)
        if frame is None or frame.empty:
            skipped_rows.append({"attribute_name": attribute_name, "source_table": source_table, "reason": "empty_or_missing_source_table"})
            continue

        table_cfg = table_configs.get(source_table, {})
        table_cfg = table_cfg if isinstance(table_cfg, dict) else {}
        entity_col = table_cfg.get(f"{entity_id}_col") or infer_entity_column(frame, entity_id)
        if entity_col not in frame.columns:
            skipped_rows.append({"attribute_name": attribute_name, "source_table": source_table, "reason": f"missing_{entity_id}_column"})
            continue

        item_col = table_cfg.get("itemid_col") or _first_existing(frame.columns, COMMON_ITEM_COLUMNS)
        attr_frame = frame
        item_values = get_config_variable_ids(attr)
        if item_values and item_col in frame.columns:
            attr_frame = frame[frame[item_col].astype(str).isin(item_values)]
        elif item_values and item_col not in frame.columns:
            skipped_rows.append({"attribute_name": attribute_name, "source_table": source_table, "reason": "configured_itemid_but_no_item_column"})
            continue

        if attr_frame.empty:
            skipped_rows.append({"attribute_name": attribute_name, "source_table": source_table, "reason": "no_rows_after_item_filter"})
            continue

        value_col = table_cfg.get("value_numeric_col")
        if value_col not in attr_frame.columns:
            value_col = table_cfg.get("value_col")
        if value_col not in attr_frame.columns:
            value_col = _first_existing(attr_frame.columns, COMMON_VALUE_COLUMNS)

        if value_col in attr_frame.columns:
            values = pd.to_numeric(attr_frame[value_col], errors="coerce")
            if values.notna().any():
                work = attr_frame[[entity_col]].copy()
                work["_value"] = values
                grouped = work.groupby(entity_col)["_value"].agg(aggregation)
            else:
                grouped = attr_frame.groupby(entity_col).size().astype(float)
        else:
            grouped = attr_frame.groupby(entity_col).size().astype(float)

        grouped.name = attribute_name
        feature_series.append(grouped)

    if not feature_series:
        LOGGER.warning("No attributes could be extracted from PIC tables.")
        missing_report = pd.DataFrame(skipped_rows)
        return pd.DataFrame(), missing_report

    matrix = pd.concat(feature_series, axis=1)
    matrix.index.name = entity_id
    raw_missing = matrix.isna().mean().rename("missing_rate")
    missing_report = raw_missing.reset_index().rename(columns={"index": "attribute_name"})
    if skipped_rows:
        skipped_report = pd.DataFrame(skipped_rows)
        for col in missing_report.columns:
            if col not in skipped_report.columns:
                skipped_report[col] = np.nan
        missing_report = pd.concat([missing_report, skipped_report[missing_report.columns]], ignore_index=True)

    kept_columns = raw_missing[raw_missing <= max_missing_rate].index.tolist()
    dropped_columns = sorted(set(matrix.columns) - set(kept_columns))
    if dropped_columns:
        LOGGER.warning("Dropping %d attributes above max_missing_rate %.3f: %s", len(dropped_columns), max_missing_rate, dropped_columns)
    matrix = matrix[kept_columns]

    for column in matrix.columns:
        if matrix[column].isna().all():
            matrix[column] = 0.0
        else:
            matrix[column] = matrix[column].fillna(matrix[column].median())

    LOGGER.info(
        "Built patient feature matrix with %d entities and %d attributes at %s level.",
        matrix.shape[0],
        matrix.shape[1],
        entity_id,
    )

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        matrix.to_csv(output)
        LOGGER.info("Saved patient feature matrix to %s.", output)
    if missing_report_path:
        output = Path(missing_report_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        missing_report.to_csv(output, index=False)
        LOGGER.info("Saved missingness report to %s.", output)

    return matrix, missing_report
