"""Ontology-constrained equal-weight semantic reduction for SemMed-LDP."""

from __future__ import annotations

import pandas as pd

from semmed_ldp.reference_intervals import bounds_for_users


MISSING_STATE = 0
NO_ABNORMALITY_STATE = 1
ABNORMALITY_PRESENT_STATE = 2
DOMAIN_SIZE = 3


REQUIRED_MAPPING_COLUMNS = {
    "dataset_id",
    "attribute_name",
    "source_variable_id",
    "ucum_unit",
    "loinc_code",
    "semantic_anchor",
    "lower_bound",
    "upper_bound",
    "direction",
    "threshold_source",
    "reviewer_status",
}


def validate_formal_mapping(mapping: pd.DataFrame, *, dataset_id: str | None = None) -> pd.DataFrame:
    """Validate that a mapping is suitable for a formal, non-fallback run."""
    missing = REQUIRED_MAPPING_COLUMNS.difference(mapping.columns)
    if missing:
        raise ValueError(f"formal mapping is missing columns: {sorted(missing)}")
    frame = mapping.copy()
    if dataset_id is not None:
        frame = frame.loc[frame["dataset_id"].astype(str) == str(dataset_id)].copy()
    if frame.empty:
        raise ValueError(f"no mapping rows found for dataset {dataset_id!r}")
    text_required = ["attribute_name", "source_variable_id", "ucum_unit", "loinc_code", "semantic_anchor", "threshold_source"]
    for column in text_required:
        bad = frame[column].isna() | frame[column].astype(str).str.strip().eq("")
        if bad.any():
            raise ValueError(f"formal mapping contains {int(bad.sum())} blank values in {column}")
    if not frame["reviewer_status"].astype(str).str.lower().eq("approved").all():
        raise ValueError("every formal mapping row must have reviewer_status=approved")
    if frame["threshold_source"].astype(str).str.contains("quantile|data-derived", case=False, regex=True).any():
        raise ValueError("data-derived or quantile thresholds are forbidden in formal runs")
    direction = frame["direction"].astype(str).str.lower()
    allowed = {"low_abnormal", "high_abnormal", "both_abnormal"}
    if not direction.isin(allowed).all():
        raise ValueError(f"formal direction values must be one of {sorted(allowed)}")
    lower = pd.to_numeric(frame["lower_bound"], errors="coerce")
    upper = pd.to_numeric(frame["upper_bound"], errors="coerce")
    if "reference_interval_id" in frame.columns:
        has_stratified_interval = frame["reference_interval_id"].fillna("").astype(str).str.strip().ne("")
    else:
        has_stratified_interval = pd.Series(False, index=frame.index)
    invalid = (
        ~has_stratified_interval
        & (
            direction.eq("low_abnormal") & lower.isna()
            | direction.eq("high_abnormal") & upper.isna()
            | direction.eq("both_abnormal") & (lower.isna() | upper.isna())
        )
    )
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} formal rows lack required public bounds")
    if frame["attribute_name"].duplicated().any():
        raise ValueError("attribute_name must be unique within a dataset mapping")
    return frame


def attribute_abnormal_indicators(
    patient_features: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    metadata: pd.DataFrame | None = None,
    reference_intervals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create nullable binary abnormality indicators using public bounds only."""
    mapping = validate_formal_mapping(mapping)
    missing_attributes = sorted(set(mapping["attribute_name"]) - set(patient_features.columns))
    if missing_attributes:
        raise ValueError(f"patient matrix lacks mapped attributes: {missing_attributes}")
    result = pd.DataFrame(index=patient_features.index)
    for row in mapping.itertuples(index=False):
        values = pd.to_numeric(patient_features[row.attribute_name], errors="coerce")
        reference_id = str(getattr(row, "reference_interval_id", "")).strip()
        if reference_id and reference_id.lower() != "nan":
            if metadata is None or reference_intervals is None:
                raise ValueError(f"{row.attribute_name} requires metadata and a reference interval table")
            lower, upper = bounds_for_users(reference_id, metadata.loc[patient_features.index], reference_intervals)
        else:
            lower = pd.to_numeric(pd.Series([row.lower_bound]), errors="coerce").iloc[0]
            upper = pd.to_numeric(pd.Series([row.upper_bound]), errors="coerce").iloc[0]
        direction = str(row.direction).lower()
        if direction == "low_abnormal":
            abnormal = values < lower
        elif direction == "high_abnormal":
            abnormal = values > upper
        else:
            abnormal = (values < lower) | (values > upper)
        indicator = abnormal.astype("Int8")
        indicator.loc[values.isna()] = pd.NA
        result[row.attribute_name] = indicator
    return result


def build_ontology_group_states(
    patient_features: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    metadata: pd.DataFrame | None = None,
    reference_intervals: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce ontology-grouped attributes to equal-weight three-state variables.

    State 0 means that every attribute in the ontology-constrained group is
    missing. State 1 means that at least one attribute is observed and all
    observed attributes are within their public reference intervals. State 2
    means that at least one observed attribute is abnormal. Every attribute has
    the same logical contribution; graph-distance weights and burden cut points
    are deliberately excluded from the primary transformation.
    """
    mapping = validate_formal_mapping(mapping)
    indicators = attribute_abnormal_indicators(
        patient_features,
        mapping,
        metadata=metadata,
        reference_intervals=reference_intervals,
    )
    states = pd.DataFrame(index=patient_features.index)
    diagnostics: list[dict[str, float | int | str]] = []
    for anchor, rows in mapping.groupby("semantic_anchor", sort=True):
        attributes = rows["attribute_name"].tolist()
        values = indicators[attributes]
        observed = values.notna().any(axis=1)
        abnormal = values.eq(1).any(axis=1)
        anchor_states = pd.Series(MISSING_STATE, index=states.index, dtype="int8")
        anchor_states.loc[observed] = NO_ABNORMALITY_STATE
        anchor_states.loc[abnormal] = ABNORMALITY_PRESENT_STATE
        states[str(anchor)] = anchor_states
        diagnostics.append(
            {
                "semantic_anchor": str(anchor),
                "n_attributes": len(attributes),
                "n_users": len(states),
                "missing_state_rate": float(anchor_states.eq(MISSING_STATE).mean()),
                "abnormality_present_rate": float(anchor_states.eq(ABNORMALITY_PRESENT_STATE).mean()),
                "n_unique_states": int(anchor_states.nunique()),
            }
        )
    return states, pd.DataFrame(diagnostics)
