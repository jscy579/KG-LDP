from pathlib import Path

import pandas as pd
import pytest

from semmed_ldp.dataset_adapters import (
    load_pic_formal_matrix,
    prepare_pic_formal_from_raw,
    select_and_rename_features,
)


def test_pic_matrix_requires_unique_users(tmp_path: Path) -> None:
    path = tmp_path / "pic.csv"
    pd.DataFrame({"patient_id": [1, 1], "sodium": [140, 141]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="one row"):
        load_pic_formal_matrix(path)


def test_mapping_selects_and_renames() -> None:
    features = pd.DataFrame({"LBXSNASI": [140.0], "LBXSKSI": [4.0]})
    mapping = pd.DataFrame(
        {
            "source_variable_id": ["LBXSNASI", "LBXSKSI"],
            "attribute_name": ["sodium", "potassium"],
        }
    )
    selected = select_and_rename_features(features, mapping)
    assert list(selected.columns) == ["sodium", "potassium"]


def test_pic_adapter_uses_only_each_patients_earliest_admission(tmp_path: Path) -> None:
    pd.DataFrame(
        {"SUBJECT_ID": [1], "GENDER": ["F"], "DOB": ["2010-01-01"]}
    ).to_csv(tmp_path / "PATIENTS.csv", index=False)
    pd.DataFrame(
        {
            "SUBJECT_ID": [1, 1],
            "HADM_ID": [101, 102],
            "ADMITTIME": ["2015-01-01", "2018-01-01"],
        }
    ).to_csv(tmp_path / "ADMISSIONS.csv", index=False)
    pd.DataFrame(
        {
            "SUBJECT_ID": [1, 1, 1],
            "HADM_ID": [101, 101, 102],
            "ITEMID": [500, 500, 500],
            "VALUENUM": [10.0, 14.0, 100.0],
            "VALUEUOM": ["mg/dL", "mg/dL", "mg/dL"],
        }
    ).to_csv(tmp_path / "LABEVENTS.csv", index=False)
    mapping = pd.DataFrame(
        {
            "dataset_id": ["PIC"],
            "attribute_name": ["x"],
            "source_variable_id": ["500"],
            "source_unit": ["mg/dL"],
            "conversion_factor": [1.0],
            "conversion_offset": [0.0],
        }
    )
    features, metadata, _ = prepare_pic_formal_from_raw(tmp_path, mapping, chunksize=2)
    assert features.loc[1, "x"] == 12.0
    assert metadata.loc[1, "first_hadm_id"] == 101
