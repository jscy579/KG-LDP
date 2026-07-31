"""Baseline grouping strategies for SemMed-LDP experiments."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from semmed_ldp.semantic_reduction import _attribute_abnormal_state
from semmed_ldp.umls_mapping import build_umls_mappings

LOGGER = logging.getLogger(__name__)


def build_attribute_abnormal_matrix(patient_feature_matrix: pd.DataFrame, attribute_config: pd.DataFrame) -> pd.DataFrame:
    """Convert each raw attribute into a binary abnormal-state variable."""
    if patient_feature_matrix.empty:
        return pd.DataFrame(index=patient_feature_matrix.index)
    config_by_attr = attribute_config.set_index("attribute_name", drop=False)
    abnormal = pd.DataFrame(index=patient_feature_matrix.index)
    for attribute in patient_feature_matrix.columns:
        if attribute not in config_by_attr.index:
            LOGGER.warning("Skipping baseline attribute '%s' because it is absent from attribute_config.", attribute)
            continue
        abnormal[attribute] = _attribute_abnormal_state(patient_feature_matrix[attribute], config_by_attr.loc[attribute])[0]
    LOGGER.info("Built raw abnormal attribute matrix with %d patients and %d attributes.", abnormal.shape[0], abnormal.shape[1])
    return abnormal


def raw_attribute_groups(attributes: list[str]) -> dict[str, list[str]]:
    """Create one group per raw attribute."""
    return {f"raw::{attribute}": [attribute] for attribute in attributes}


def type_grouped_attributes(attribute_config: pd.DataFrame, attributes: list[str]) -> dict[str, list[str]]:
    """Group attributes by source table, such as laboratory, vital, diagnosis, medication."""
    attr_set = set(attributes)
    groups: dict[str, list[str]] = {}
    for _, row in attribute_config.iterrows():
        attribute = row.get("attribute_name", "")
        if attribute not in attr_set:
            continue
        source = str(row.get("source_table", "unknown")).strip() or "unknown"
        groups.setdefault(f"type::{source}", []).append(attribute)
    return groups


def random_grouped_attributes(attributes: list[str], n_groups: int, seed: int = 2026) -> dict[str, list[str]]:
    """Randomly assign attributes to the same number of groups as UMLS groups."""
    if not attributes:
        return {}
    n_groups = max(1, min(n_groups, len(attributes)))
    rng = np.random.default_rng(seed)
    shuffled = list(attributes)
    rng.shuffle(shuffled)
    groups = {f"random_group_{idx + 1}": [] for idx in range(n_groups)}
    for idx, attribute in enumerate(shuffled):
        groups[f"random_group_{(idx % n_groups) + 1}"].append(attribute)
    return groups


def correlation_grouped_attributes(abnormal_matrix: pd.DataFrame, n_groups: int) -> dict[str, list[str]]:
    """Cluster attributes by empirical correlation in the PIC-derived matrix."""
    attributes = list(abnormal_matrix.columns)
    if not attributes:
        return {}
    n_groups = max(1, min(n_groups, len(attributes)))
    if n_groups == 1:
        return {"corr_group_1": attributes}

    working = abnormal_matrix.fillna(0)
    if working.shape[0] < 2 or working.nunique().le(1).all():
        LOGGER.warning("Correlation grouping fallback: not enough variation; assigning attributes round-robin.")
        return random_grouped_attributes(attributes, n_groups, seed=0)

    corr = working.corr().fillna(0).abs()
    distance = 1 - corr
    np.fill_diagonal(distance.values, 0)
    try:
        model = AgglomerativeClustering(n_clusters=n_groups, metric="precomputed", linkage="average")
    except TypeError:
        model = AgglomerativeClustering(n_clusters=n_groups, affinity="precomputed", linkage="average")
    labels = model.fit_predict(distance)
    groups: dict[str, list[str]] = {}
    for attribute, label in zip(attributes, labels):
        groups.setdefault(f"corr_group_{label + 1}", []).append(attribute)
    return groups


def grouped_state_matrix(
    abnormal_matrix: pd.DataFrame,
    group_to_attributes: dict[str, list[str]],
    *,
    mode: str = "binary_abnormal",
    min_group_attributes: int = 1,
) -> pd.DataFrame:
    """Aggregate binary attribute abnormal states into grouped state variables."""
    if mode not in {"binary_abnormal", "severity_level"}:
        raise ValueError("mode must be 'binary_abnormal' or 'severity_level'.")
    states = pd.DataFrame(index=abnormal_matrix.index)
    for group, attributes in group_to_attributes.items():
        available = [attribute for attribute in attributes if attribute in abnormal_matrix.columns]
        if len(available) < min_group_attributes:
            LOGGER.warning("Skipping baseline group '%s' with only %d available attributes.", group, len(available))
            continue
        ratio = abnormal_matrix[available].mean(axis=1)
        if mode == "binary_abnormal":
            states[group] = (ratio > 0).astype(int)
        else:
            states[group] = np.select(
                [ratio.eq(0), ratio <= (1 / 3), ratio <= (2 / 3), ratio > (2 / 3)],
                [0, 1, 2, 3],
                default=0,
            ).astype(int)
    return states


def build_baseline_state_matrices(
    patient_feature_matrix: pd.DataFrame,
    attribute_config: pd.DataFrame,
    *,
    mode: str = "binary_abnormal",
    seed: int = 2026,
    min_group_attributes: int = 1,
) -> dict[str, pd.DataFrame]:
    """Build state matrices for Raw, Type, Random, Correlation, and UMLS groupings."""
    abnormal_matrix = build_attribute_abnormal_matrix(patient_feature_matrix, attribute_config)
    _, umls_groups = build_umls_mappings(attribute_config)
    attributes = list(abnormal_matrix.columns)
    n_umls_groups = max(1, len(umls_groups))

    grouping_schemes: dict[str, dict[str, list[str]]] = {
        "Raw-Attribute-LDP": raw_attribute_groups(attributes),
        "Type-Grouped-LDP": type_grouped_attributes(attribute_config, attributes),
        "Random-Grouped-LDP": random_grouped_attributes(attributes, n_umls_groups, seed=seed),
        "Correlation-Grouped-LDP": correlation_grouped_attributes(abnormal_matrix, n_umls_groups),
        "UMLS-Grouped-LDP": {group: attrs for group, attrs in umls_groups.items()},
    }

    state_matrices: dict[str, pd.DataFrame] = {}
    for method, groups in grouping_schemes.items():
        state_matrices[method] = grouped_state_matrix(abnormal_matrix, groups, mode=mode, min_group_attributes=min_group_attributes)
        LOGGER.info(
            "%s produced %d groups from %d attributes.",
            method,
            state_matrices[method].shape[1],
            len(attributes),
        )
    return state_matrices


def group_size_summary(method_to_groups: dict[str, dict[str, list[str]]]) -> pd.DataFrame:
    """Summarize group sizes for reporting and visualization."""
    rows: list[dict[str, Any]] = []
    for method, groups in method_to_groups.items():
        for group, attributes in groups.items():
            rows.append({"method": method, "group": group, "n_attributes": len(attributes)})
    return pd.DataFrame(rows)
