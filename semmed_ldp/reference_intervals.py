"""Public age- and sex-stratified reference interval application."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_INTERVAL_COLUMNS = {
    "reference_interval_id",
    "min_age_years",
    "max_age_years",
    "sex",
    "lower_bound",
    "upper_bound",
    "unit",
    "source",
    "reviewer_status",
}


def validate_reference_intervals(
    intervals: pd.DataFrame,
    *,
    require_approved: bool = True,
) -> pd.DataFrame:
    missing = REQUIRED_INTERVAL_COLUMNS.difference(intervals.columns)
    if missing:
        raise ValueError(f"reference interval table is missing columns: {sorted(missing)}")
    frame = intervals.copy()
    if require_approved and not frame["reviewer_status"].astype(str).str.lower().eq("approved").all():
        raise ValueError("all formal reference intervals must be approved")
    frame["min_age_years"] = pd.to_numeric(frame["min_age_years"], errors="raise")
    frame["max_age_years"] = pd.to_numeric(frame["max_age_years"], errors="raise")
    frame["lower_bound"] = pd.to_numeric(frame["lower_bound"], errors="coerce")
    frame["upper_bound"] = pd.to_numeric(frame["upper_bound"], errors="coerce")
    if (frame["min_age_years"] >= frame["max_age_years"]).any():
        raise ValueError("reference intervals require min_age_years < max_age_years")
    allowed_sex = {"any", "female", "male"}
    if not frame["sex"].astype(str).str.lower().isin(allowed_sex).all():
        raise ValueError(f"sex must be one of {sorted(allowed_sex)}")
    if frame["source"].isna().any() or frame["source"].astype(str).str.strip().eq("").any():
        raise ValueError("every reference interval requires a public source")
    return frame


def bounds_for_users(
    reference_interval_id: str,
    metadata: pd.DataFrame,
    intervals: pd.DataFrame,
    *,
    age_column: str = "age_years",
    sex_column: str = "sex",
    require_approved: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """Resolve one public interval ID to per-user lower and upper bounds."""
    intervals = validate_reference_intervals(intervals, require_approved=require_approved)
    if age_column not in metadata or sex_column not in metadata:
        raise ValueError(f"metadata must contain {age_column!r} and {sex_column!r}")
    candidates = intervals.loc[intervals["reference_interval_id"].astype(str) == str(reference_interval_id)]
    if candidates.empty:
        raise ValueError(f"reference interval {reference_interval_id!r} is absent")
    ages = pd.to_numeric(metadata[age_column], errors="coerce")
    sexes = metadata[sex_column].astype(str).str.lower()
    lower = pd.Series(np.nan, index=metadata.index, dtype=float)
    upper = pd.Series(np.nan, index=metadata.index, dtype=float)
    for row in candidates.itertuples(index=False):
        sex = str(row.sex).lower()
        matches = ages.ge(float(row.min_age_years)) & ages.lt(float(row.max_age_years))
        if sex != "any":
            matches &= sexes.eq(sex)
        lower.loc[matches] = row.lower_bound
        upper.loc[matches] = row.upper_bound
    unresolved = ages.notna() & (lower.isna() & upper.isna())
    if unresolved.any():
        raise ValueError(
            f"reference interval {reference_interval_id!r} does not cover {int(unresolved.sum())} users with known age"
        )
    return lower, upper
