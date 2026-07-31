import pandas as pd
import pytest

from semmed_ldp.formal_semantic_reduction import build_ontology_group_states, validate_formal_mapping


def _mapping() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_id": "demo",
                "attribute_name": "sodium",
                "source_variable_id": "NA",
                "ucum_unit": "mmol/L",
                "loinc_code": "2951-2",
                "semantic_anchor": "electrolyte",
                "lower_bound": 135,
                "upper_bound": 145,
                "direction": "both_abnormal",
                "threshold_source": "public-reference-v1",
                "reviewer_status": "approved",
            },
            {
                "dataset_id": "demo",
                "attribute_name": "potassium",
                "source_variable_id": "K",
                "ucum_unit": "mmol/L",
                "loinc_code": "2823-3",
                "semantic_anchor": "electrolyte",
                "lower_bound": 3.5,
                "upper_bound": 5.0,
                "direction": "both_abnormal",
                "threshold_source": "public-reference-v1",
                "reviewer_status": "approved",
            },
        ]
    )


def test_missing_is_not_normal() -> None:
    features = pd.DataFrame({"sodium": [140.0, None, 150.0], "potassium": [4.0, None, None]})
    states, diagnostics = build_ontology_group_states(features, _mapping())
    assert states["electrolyte"].tolist() == [1, 0, 2]
    assert diagnostics.loc[0, "missing_state_rate"] == pytest.approx(1 / 3)


def test_group_attributes_have_equal_logical_contribution() -> None:
    features = pd.DataFrame(
        {"sodium": [150.0, 140.0], "potassium": [4.0, 6.0]}
    )
    mapping = _mapping()
    mapping["graph_weight"] = [1e-9, 1e9]
    states, _ = build_ontology_group_states(features, mapping)
    assert states["electrolyte"].tolist() == [2, 2]


def test_quantile_threshold_is_rejected() -> None:
    mapping = _mapping()
    mapping.loc[0, "threshold_source"] = "dataset quantile"
    with pytest.raises(ValueError, match="forbidden"):
        validate_formal_mapping(mapping)


def test_unapproved_mapping_is_rejected() -> None:
    mapping = _mapping()
    mapping.loc[0, "reviewer_status"] = "needs_review"
    with pytest.raises(ValueError, match="approved"):
        validate_formal_mapping(mapping)
