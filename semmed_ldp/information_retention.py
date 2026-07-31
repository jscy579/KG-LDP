"""Information-retention metrics for deterministic semantic representations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score


def normalized_attribute_information_retention(
    raw_states: pd.DataFrame,
    reduced_states: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """Return equal-weight normalized mutual-information retention.

    For each raw three-state attribute ``X_a`` and the joint reduced state ``Z``,
    the retained fraction is ``I(X_a; Z) / H(X_a)``. The reported overall
    score is the unweighted mean across nonconstant raw attributes. This is a
    descriptive representation metric; it consumes no privacy budget and does
    not measure clinical semantic validity.
    """
    if raw_states.empty or reduced_states.empty:
        raise ValueError("raw_states and reduced_states must be nonempty")
    if not raw_states.index.equals(reduced_states.index):
        raise ValueError("raw_states and reduced_states must have identical ordered users")
    if raw_states.isna().any().any() or reduced_states.isna().any().any():
        raise ValueError("missingness must be encoded as an explicit state")

    joint_codes, _ = pd.factorize(
        pd.MultiIndex.from_frame(reduced_states.astype(int)),
        sort=False,
    )
    rows: list[dict[str, float | str]] = []
    for attribute in raw_states.columns:
        values = raw_states[attribute].astype(int).to_numpy()
        entropy = float(mutual_info_score(values, values))
        if np.isclose(entropy, 0.0):
            retention = np.nan
            mutual_information = 0.0
        else:
            mutual_information = float(mutual_info_score(values, joint_codes))
            retention = float(np.clip(mutual_information / entropy, 0.0, 1.0))
        rows.append(
            {
                "attribute_name": str(attribute),
                "entropy": entropy,
                "mutual_information": mutual_information,
                "normalized_retention": retention,
            }
        )
    details = pd.DataFrame(rows)
    valid = details["normalized_retention"].dropna()
    if valid.empty:
        raise ValueError("all raw attributes are constant; retention is undefined")
    return float(valid.mean()), details
