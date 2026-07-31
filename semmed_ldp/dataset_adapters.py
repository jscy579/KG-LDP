"""Independent PIC and NHANES adapters for formal SemMed-LDP experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _normalized_unit(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("μ", "u")
        .replace("µ", "u")
        .replace("×", "")
        .replace("*", "")
        .replace("^", "")
        .replace(" ", "")
    )


def prepare_pic_formal_from_raw(
    raw_dir: str | Path,
    mapping: pd.DataFrame,
    *,
    minimum_age_years: float = 1.0,
    maximum_age_years: float = 18.0,
    chunksize: int = 500_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build one PIC user row from each patient's earliest admission.

    Restricting observations to the earliest admission keeps the age used for
    public pediatric reference intervals aligned with the observations being
    aggregated, while retaining one reportable user row per patient.
    """
    root = Path(raw_dir)
    required = {"attribute_name", "source_variable_id", "source_unit", "conversion_factor", "conversion_offset"}
    missing = required.difference(mapping.columns)
    if missing:
        raise ValueError(f"PIC mapping lacks conversion columns: {sorted(missing)}")
    frame = mapping.loc[mapping["dataset_id"].astype(str).eq("PIC")].copy()
    if frame.empty:
        raise ValueError("PIC mapping has no rows")
    frame["source_variable_id"] = frame["source_variable_id"].astype(str)
    selected_ids = set(frame["source_variable_id"])

    admissions = pd.read_csv(root / "ADMISSIONS.csv", usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME"])
    admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"], errors="coerce")
    first_admission = (
        admissions.dropna(subset=["ADMITTIME"])
        .sort_values(["SUBJECT_ID", "ADMITTIME", "HADM_ID"])
        .drop_duplicates("SUBJECT_ID", keep="first")
        .set_index("SUBJECT_ID")[["HADM_ID", "ADMITTIME"]]
    )
    first_hadm_ids = set(first_admission["HADM_ID"])
    parts = []
    unit_rows = []
    for chunk in pd.read_csv(
        root / "LABEVENTS.csv",
        usecols=["SUBJECT_ID", "HADM_ID", "ITEMID", "VALUENUM", "VALUEUOM"],
        dtype={"ITEMID": str, "VALUEUOM": str},
        chunksize=chunksize,
        low_memory=False,
    ):
        selected = chunk.loc[
            chunk["ITEMID"].isin(selected_ids) & chunk["HADM_ID"].isin(first_hadm_ids)
        ].copy()
        if not selected.empty:
            expected_subject = selected["HADM_ID"].map(
                first_admission.reset_index().set_index("HADM_ID")["SUBJECT_ID"]
            )
            selected = selected.loc[selected["SUBJECT_ID"].eq(expected_subject)]
        if not selected.empty:
            parts.append(selected)
    if not parts:
        raise ValueError("no PIC laboratory rows matched the formal mapping")
    observations = pd.concat(parts, ignore_index=True)
    series_by_attribute = []
    for row in frame.itertuples(index=False):
        local = observations.loc[observations["ITEMID"].eq(str(row.source_variable_id))].copy()
        expected = _normalized_unit(row.source_unit)
        local["normalized_unit"] = local["VALUEUOM"].map(_normalized_unit)
        unit_counts = local.groupby(["VALUEUOM", "normalized_unit"], dropna=False).size()
        for (raw_unit, normalized), count in unit_counts.items():
            unit_rows.append(
                {
                    "attribute_name": row.attribute_name,
                    "source_variable_id": row.source_variable_id,
                    "raw_unit": raw_unit,
                    "normalized_unit": normalized,
                    "expected_normalized_unit": expected,
                    "n_rows": int(count),
                    "accepted": normalized == expected,
                }
            )
        local = local.loc[local["normalized_unit"].eq(expected)]
        values = pd.to_numeric(local["VALUENUM"], errors="coerce")
        converted = values * float(row.conversion_factor) + float(row.conversion_offset)
        grouped = converted.groupby(local["SUBJECT_ID"]).median()
        grouped.name = row.attribute_name
        series_by_attribute.append(grouped)
    features = pd.concat(series_by_attribute, axis=1)
    features.index.name = "SUBJECT_ID"

    patients = pd.read_csv(root / "PATIENTS.csv", usecols=["SUBJECT_ID", "GENDER", "DOB"])
    patients["DOB"] = pd.to_datetime(patients["DOB"], errors="coerce")
    metadata = patients.set_index("SUBJECT_ID").join(first_admission, how="inner")
    metadata = metadata.rename(columns={"ADMITTIME": "index_time", "HADM_ID": "first_hadm_id"})
    metadata["age_years"] = (metadata["index_time"] - metadata["DOB"]).dt.total_seconds() / (365.2425 * 86400)
    metadata["sex"] = metadata["GENDER"].astype(str).str.upper().map({"M": "male", "F": "female"})
    eligible = metadata["age_years"].ge(minimum_age_years) & metadata["age_years"].lt(maximum_age_years)
    metadata = metadata.loc[eligible, ["age_years", "sex", "index_time", "first_hadm_id"]]
    common = features.index.intersection(metadata.index)
    return features.loc[common], metadata.loc[common], pd.DataFrame(unit_rows)


def load_pic_formal_matrix(path: str | Path, *, entity_id: str = "patient_id") -> pd.DataFrame:
    """Load a project-generated PIC patient matrix without importing another workflow."""
    frame = pd.read_csv(path)
    if entity_id not in frame.columns:
        unnamed = [column for column in frame.columns if column.lower().startswith("unnamed")]
        if unnamed:
            frame = frame.rename(columns={unnamed[0]: entity_id})
    if entity_id not in frame.columns:
        raise ValueError(f"PIC matrix must contain {entity_id!r}")
    if frame[entity_id].duplicated().any():
        raise ValueError("PIC matrix must contain one row per project-defined user")
    return frame.set_index(entity_id)


def load_nhanes_xpt_directory(
    raw_dir: str | Path,
    *,
    files: Iterable[str] = ("DEMO_L.XPT", "CBC_L.XPT", "BIOPRO_L.XPT"),
    entity_id: str = "SEQN",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge selected NHANES 2021--2023 public-use XPT files on SEQN.

    The returned metadata contains age, sex, and the public survey weight when
    present.  Formal SemMed-LDP experiments estimate the empirical study-sample
    distribution; they do not label it as a US population prevalence estimate.
    """
    root = Path(raw_dir)
    paths = [root / name for name in files]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"NHANES files are missing: {missing}")
    frames = []
    for path in paths:
        frame = pd.read_sas(path, format="xport", encoding="utf-8")
        if entity_id not in frame.columns:
            raise ValueError(f"{path.name} lacks NHANES entity column {entity_id}")
        if frame[entity_id].duplicated().any():
            raise ValueError(f"{path.name} contains duplicate {entity_id} rows")
        frames.append(frame)
    merged = frames[0]
    for frame in frames[1:]:
        overlap = [column for column in frame.columns if column in merged.columns and column != entity_id]
        merged = merged.merge(frame.drop(columns=overlap), on=entity_id, how="outer", validate="one_to_one")
    metadata_columns = [column for column in [entity_id, "RIDAGEYR", "RIAGENDR", "WTMEC2YR", "SDMVSTRA", "SDMVPSU"] if column in merged]
    metadata = merged[metadata_columns].set_index(entity_id)
    feature_columns = [column for column in merged.columns if column not in metadata_columns]
    features = merged[[entity_id, *feature_columns]].set_index(entity_id)
    return features, metadata


def select_and_rename_features(
    features: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Select source variables and rename them to canonical attribute names."""
    required = {"source_variable_id", "attribute_name"}
    missing = required.difference(mapping.columns)
    if missing:
        raise ValueError(f"mapping is missing columns: {sorted(missing)}")
    source_to_attribute = dict(zip(mapping["source_variable_id"].astype(str), mapping["attribute_name"].astype(str)))
    absent = sorted(set(source_to_attribute) - set(features.columns.astype(str)))
    if absent:
        raise ValueError(f"source variables are absent from the dataset: {absent}")
    selected = features[list(source_to_attribute)].copy()
    selected.columns = [source_to_attribute[str(column)] for column in selected.columns]
    return selected


def write_dataset_provenance(
    output_path: str | Path,
    *,
    dataset_id: str,
    source_url: str,
    release: str,
    files: Iterable[str],
    population_claim: str,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"dataset_id: {dataset_id}",
        f"source_url: {source_url}",
        f"release: {release}",
        "files:",
        *[f"  - {value}" for value in files],
        f"population_claim: {population_claim}",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
