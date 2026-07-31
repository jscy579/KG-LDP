"""Convert high-dimensional patient attributes into semantic group states."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from semmed_ldp.umls_mapping import build_umls_mappings

LOGGER = logging.getLogger(__name__)


def _optional_float(value: Any) -> float | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _attribute_abnormal_state(values: pd.Series, config_row: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    direction = str(config_row.get("direction", "")).strip().lower()
    lower = _optional_float(config_row.get("lower_bound", ""))
    upper = _optional_float(config_row.get("upper_bound", ""))
    fallback_used = False

    numeric = pd.to_numeric(values, errors="coerce")
    if direction in {"presence_abnormal", "presence", "binary"}:
        abnormal = numeric.fillna(0) > 0
    elif lower is not None or upper is not None:
        abnormal = pd.Series(False, index=values.index)
        if direction == "low_abnormal":
            threshold = lower if lower is not None else numeric.quantile(0.1)
            abnormal = numeric < threshold
        elif direction == "high_abnormal":
            threshold = upper if upper is not None else numeric.quantile(0.9)
            abnormal = numeric > threshold
        elif direction == "both_abnormal":
            if lower is not None:
                abnormal = abnormal | (numeric < lower)
            if upper is not None:
                abnormal = abnormal | (numeric > upper)
        else:
            if lower is not None:
                abnormal = abnormal | (numeric < lower)
            if upper is not None:
                abnormal = abnormal | (numeric > upper)
    else:
        fallback_used = True
        q75 = numeric.quantile(0.75)
        q25 = numeric.quantile(0.25)
        if direction == "low_abnormal":
            abnormal = numeric < q25
        elif direction == "both_abnormal":
            abnormal = (numeric < q25) | (numeric > q75)
        else:
            abnormal = numeric > q75

    abnormal = abnormal.fillna(False).astype(int)
    info = {
        "attribute_name": config_row.get("attribute_name", ""),
        "fallback_quantile_threshold": fallback_used,
        "direction": direction,
        "lower_bound": lower,
        "upper_bound": upper,
        "abnormal_rate": float(abnormal.mean()) if len(abnormal) else 0.0,
    }
    return abnormal, info


def build_semantic_state_matrix(
    patient_feature_matrix: pd.DataFrame,
    attribute_config: pd.DataFrame,
    *,
    mode: str = "binary_abnormal",
    min_group_attributes: int = 1,
    output_path: str | Path | None = None,
    log_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce patient-level attributes to semantic group state variables.

    ``binary_abnormal`` returns group states in ``{0, 1}``.
    ``severity_level`` returns group states in ``{0, 1, 2, 3}`` based on the
    proportion of abnormal attributes in each group.
    """
    if mode not in {"binary_abnormal", "severity_level"}:
        raise ValueError("mode must be 'binary_abnormal' or 'severity_level'.")
    if patient_feature_matrix.empty:
        LOGGER.warning("Patient feature matrix is empty; semantic state matrix will also be empty.")
        return pd.DataFrame(index=patient_feature_matrix.index), pd.DataFrame()

    attribute_to_groups, group_to_attributes = build_umls_mappings(attribute_config)
    config_by_attr = attribute_config.set_index("attribute_name", drop=False)
    abnormal_matrix = pd.DataFrame(index=patient_feature_matrix.index)
    threshold_logs: list[dict[str, Any]] = []

    for attribute in patient_feature_matrix.columns:
        if attribute not in config_by_attr.index:
            LOGGER.warning("Feature '%s' is not present in the UMLS attribute config; skipping semantic reduction for it.", attribute)
            continue
        abnormal, info = _attribute_abnormal_state(patient_feature_matrix[attribute], config_by_attr.loc[attribute])
        abnormal_matrix[attribute] = abnormal
        threshold_logs.append(info)

    semantic_states = pd.DataFrame(index=patient_feature_matrix.index)
    for group, attributes in group_to_attributes.items():
        available = [attribute for attribute in attributes if attribute in abnormal_matrix.columns]
        if len(available) < min_group_attributes:
            LOGGER.warning("Skipping semantic group '%s' with only %d available attributes.", group, len(available))
            continue
        abnormal_ratio = abnormal_matrix[available].mean(axis=1)
        if mode == "binary_abnormal":
            semantic_states[group] = (abnormal_ratio > 0).astype(int)
        else:
            semantic_states[group] = np.select(
                [abnormal_ratio.eq(0), abnormal_ratio <= (1 / 3), abnormal_ratio <= (2 / 3), abnormal_ratio > (2 / 3)],
                [0, 1, 2, 3],
                default=0,
            ).astype(int)

    log_frame = pd.DataFrame(threshold_logs)
    fallback_count = int(log_frame.get("fallback_quantile_threshold", pd.Series(dtype=bool)).sum()) if not log_frame.empty else 0
    if fallback_count:
        LOGGER.warning("%d attributes used quantile fallback thresholds because medical bounds were not configured.", fallback_count)
    LOGGER.info(
        "Built semantic state matrix in %s mode with %d patients and %d semantic groups.",
        mode,
        semantic_states.shape[0],
        semantic_states.shape[1],
    )

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        semantic_states.to_csv(output)
        LOGGER.info("Saved semantic state matrix to %s.", output)
    if log_path:
        output = Path(log_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        log_frame.to_csv(output, index=False)
        LOGGER.info("Saved semantic threshold log to %s.", output)

    return semantic_states, log_frame
