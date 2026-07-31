import pandas as pd
import pytest

from semmed_ldp.formal_experiment import (
    build_formal_ablation_states,
    build_formal_method_states,
    run_formal_distribution_experiment,
    semantic_state_diagnostics,
)


def _mapping() -> pd.DataFrame:
    rows = []
    for attribute, anchor, low, high in [
        ("sodium", "electrolyte", 135, 145),
        ("potassium", "electrolyte", 3.5, 5.0),
        ("creatinine", "renal", 0.5, 1.2),
        ("urea", "renal", 2.5, 7.5),
    ]:
        rows.append(
            {
                "dataset_id": "demo",
                "attribute_name": attribute,
                "source_variable_id": attribute,
                "source_table": "lab",
                "ucum_unit": "unit",
                "loinc_code": f"code-{attribute}",
                "loinc_group_id": anchor,
                "semantic_anchor": anchor,
                "lower_bound": low,
                "upper_bound": high,
                "direction": "both_abnormal",
                "threshold_source": "public-reference",
                "reviewer_status": "approved",
            }
        )
    return pd.DataFrame(rows)


def test_formal_run_has_one_report_per_user() -> None:
    features = pd.DataFrame(
        {
            "sodium": [140, 150, 130, 141] * 40,
            "potassium": [4.0, 6.0, 2.8, 4.2] * 40,
            "creatinine": [0.8, 2.0, 0.7, 1.0] * 40,
            "urea": [5.0, 10.0, 4.0, 6.0] * 40,
        }
    )
    results, diagnostics = run_formal_distribution_experiment(
        features,
        _mapping(),
        dataset_id="demo",
        epsilon_list=[1.0],
        mechanisms=["GRR"],
        sample_sizes=[100],
        repeat_runs=2,
        seed=5,
    )
    assert not results.empty
    assert diagnostics["reports_per_user"].eq(1.0).all()
    assert "SemMed-LDP" in set(results["method"])


def test_matched_random_group_count() -> None:
    features = pd.DataFrame({"sodium": [140], "potassium": [4], "creatinine": [1], "urea": [5]})
    states = build_formal_method_states(features, _mapping(), seed=3)
    assert states["SemMed-LDP"].shape[1] == states["Random-Grouped-LDP"].shape[1]


def test_age_sex_intervals_are_used_by_all_methods() -> None:
    mapping = _mapping().iloc[[0]].copy()
    mapping["reference_interval_id"] = "sodium_age_sex"
    mapping[["lower_bound", "upper_bound"]] = pd.NA
    features = pd.DataFrame({"sodium": [142.0, 142.0]}, index=[10, 11])
    metadata = pd.DataFrame(
        {"age_years": [8.0, 30.0], "sex": ["female", "female"]}, index=features.index
    )
    intervals = pd.DataFrame(
        {
            "reference_interval_id": ["sodium_age_sex", "sodium_age_sex"],
            "min_age_years": [0.0, 18.0],
            "max_age_years": [18.0, 120.0],
            "sex": ["any", "any"],
            "lower_bound": [135.0, 135.0],
            "upper_bound": [140.0, 145.0],
            "unit": ["unit", "unit"],
            "source": ["public", "public"],
            "reviewer_status": ["approved", "approved"],
        }
    )
    states = build_formal_method_states(
        features,
        mapping,
        seed=3,
        metadata=metadata,
        reference_intervals=intervals,
    )
    for state_frame in states.values():
        assert state_frame.iloc[:, 0].tolist() == [2, 1]


def test_type_baseline_prefers_dictionary_category() -> None:
    mapping = _mapping()
    mapping["source_dictionary_category"] = ["blood gas", "blood gas", "chemistry", "chemistry"]
    features = pd.DataFrame(
        {"sodium": [150], "potassium": [4], "creatinine": [2], "urea": [5]}
    )
    states = build_formal_method_states(features, mapping, seed=3)
    assert states["Type-Grouped-LDP"].shape[1] == 2


def test_formal_ablation_family_and_constant_guard() -> None:
    mapping = _mapping()
    features = pd.DataFrame(
        {
            "sodium": [140, 150, 130, 141],
            "potassium": [4.0, 6.0, 2.8, 4.2],
            "creatinine": [0.8, 2.0, 0.7, 1.0],
            "urea": [5.0, 10.0, 4.0, 6.0],
        }
    )
    ablations = build_formal_ablation_states(features, mapping, seed=7)
    assert "Full-SemMed-LDP" in ablations
    assert "Shuffled-Knowledge-Edges" in ablations
    assert "No-Graph-Distance-Weights" not in ablations
    with pytest.raises(ValueError, match="constant"):
        semantic_state_diagnostics(pd.DataFrame({"constant": [1, 1, 1]}))
