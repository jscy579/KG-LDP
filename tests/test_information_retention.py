import pandas as pd

from semmed_ldp.information_retention import normalized_attribute_information_retention


def test_identity_representation_retains_all_attribute_information():
    raw = pd.DataFrame(
        {
            "a": [0, 1, 2, 0, 1, 2],
            "b": [0, 0, 1, 1, 2, 2],
        }
    )
    score, details = normalized_attribute_information_retention(raw, raw)
    assert abs(score - 1.0) < 1e-12
    assert details["normalized_retention"].eq(1.0).all()


def test_coarse_representation_loses_attribute_information():
    raw = pd.DataFrame(
        {
            "a": [0, 0, 1, 1, 2, 2],
            "b": [0, 1, 0, 1, 0, 1],
        }
    )
    reduced = pd.DataFrame({"constant_for_b": raw["a"]})
    score, details = normalized_attribute_information_retention(raw, reduced)
    by_attribute = details.set_index("attribute_name")["normalized_retention"]
    assert by_attribute["a"] == 1.0
    assert by_attribute["b"] < 1e-12
    assert 0.0 < score < 1.0
