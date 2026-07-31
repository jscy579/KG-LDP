"""Read UMLS-guided attribute mapping configuration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

REQUIRED_ATTRIBUTE_COLUMNS = [
    "attribute_name",
    "variable_id",
    "itemid",
    "source_table",
    "umls_cui",
    "umls_name",
    "semantic_group",
    "clinical_concepts",
    "lower_bound",
    "upper_bound",
    "direction",
]


def split_multi_value(value: Any) -> list[str]:
    """Split a multi-valued config field while preserving simple single values."""
    if pd.isna(value) or str(value).strip() == "":
        return []
    text = str(value)
    for separator in ["|", ";"]:
        text = text.replace(separator, ",")
    return [part.strip() for part in text.split(",") if part.strip()]


def load_attribute_config(config_path: str | Path) -> pd.DataFrame:
    """Load and validate ``semmed_attribute_config.csv``."""
    path = Path(config_path)
    frame = pd.read_csv(path, dtype=str).fillna("")
    if "variable_id" not in frame.columns and "itemid" in frame.columns:
        frame["variable_id"] = frame["itemid"]
    if "itemid" not in frame.columns and "variable_id" in frame.columns:
        frame["itemid"] = frame["variable_id"]
    missing = [column for column in REQUIRED_ATTRIBUTE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    frame = frame[REQUIRED_ATTRIBUTE_COLUMNS].copy()
    frame["attribute_name"] = frame["attribute_name"].str.strip()
    frame["source_table"] = frame["source_table"].str.strip()
    frame = frame[frame["attribute_name"] != ""].drop_duplicates(subset=["attribute_name"], keep="first")
    LOGGER.info("Loaded %d SemMed attributes from %s.", len(frame), path)
    return frame


def build_umls_mappings(attribute_config: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Build attribute-to-groups and group-to-attributes dictionaries."""
    attribute_to_groups: dict[str, list[str]] = {}
    group_to_attributes: dict[str, list[str]] = {}

    for _, row in attribute_config.iterrows():
        attribute = str(row["attribute_name"]).strip()
        groups = split_multi_value(row.get("semantic_group", ""))
        if not groups:
            LOGGER.warning("Attribute '%s' has no semantic_group and will not contribute to semantic reduction.", attribute)
            attribute_to_groups[attribute] = []
            continue

        attribute_to_groups[attribute] = groups
        for group in groups:
            group_to_attributes.setdefault(group, []).append(attribute)

    LOGGER.info(
        "Built UMLS mapping with %d attributes assigned to %d semantic groups.",
        len(attribute_to_groups),
        len(group_to_attributes),
    )
    return attribute_to_groups, group_to_attributes


def summarize_mapping(attribute_config: pd.DataFrame) -> pd.DataFrame:
    """Return one row per attribute-to-semantic-group assignment."""
    rows: list[dict[str, Any]] = []
    for _, row in attribute_config.iterrows():
        groups = split_multi_value(row.get("semantic_group", ""))
        concepts = split_multi_value(row.get("clinical_concepts", "")) or split_multi_value(row.get("umls_name", ""))
        if not groups:
            rows.append(
                {
                    "attribute_name": row.get("attribute_name", ""),
                    "semantic_group": "",
                    "clinical_concepts": "|".join(concepts),
                    "umls_cui": row.get("umls_cui", ""),
                    "source_table": row.get("source_table", ""),
                }
            )
        for group in groups:
            rows.append(
                {
                    "attribute_name": row.get("attribute_name", ""),
                    "semantic_group": group,
                    "clinical_concepts": "|".join(concepts),
                    "umls_cui": row.get("umls_cui", ""),
                    "source_table": row.get("source_table", ""),
                }
            )
    return pd.DataFrame(rows)
