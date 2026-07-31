import pandas as pd

from semmed_ldp.reference_intervals import bounds_for_users, validate_reference_intervals


def test_age_and_sex_specific_bounds() -> None:
    intervals = pd.DataFrame(
        [
            ["hb", 0, 18, "any", 10, 16, "g/dL", "source", "approved"],
            ["hb", 18, 200, "female", 12, 16, "g/dL", "source", "approved"],
            ["hb", 18, 200, "male", 13, 17, "g/dL", "source", "approved"],
        ],
        columns=[
            "reference_interval_id",
            "min_age_years",
            "max_age_years",
            "sex",
            "lower_bound",
            "upper_bound",
            "unit",
            "source",
            "reviewer_status",
        ],
    )
    metadata = pd.DataFrame({"age_years": [10, 30, 30], "sex": ["female", "female", "male"]})
    lower, upper = bounds_for_users("hb", metadata, intervals)
    assert lower.tolist() == [10, 12, 13]
    assert upper.tolist() == [16, 16, 17]


def test_working_intervals_can_be_structurally_checked_without_approval() -> None:
    intervals = pd.DataFrame(
        [["x", 1, 18, "any", 0, 1, "mg/dL", "public source", "needs_review"]],
        columns=[
            "reference_interval_id", "min_age_years", "max_age_years", "sex",
            "lower_bound", "upper_bound", "unit", "source", "reviewer_status",
        ],
    )
    checked = validate_reference_intervals(intervals, require_approved=False)
    assert checked.loc[0, "reviewer_status"] == "needs_review"
