"""Migrate schema-expansion state labels to the manuscript encoding.

This is a label migration, not an experimental rerun. The original expansion
experiment used 0=normal, 1=abnormal, 2=all missing. The manuscript and formal
pipeline use 0=all missing, 1=normal, 2=abnormal. The deterministic permutation
is therefore 2->0, 0->1, 1->2. MSE and runtime files contain no state-indexed
columns and are intentionally left unchanged.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "schema_expansion"
STATE_FILE = RESULT_ROOT / "state_concentration.csv"
LEGACY_BACKUP = RESULT_ROOT / "state_concentration_legacy_encoding.csv"
MANIFEST_FILE = RESULT_ROOT / "manifest.json"

OLD_TO_NEW = {2: 0, 0: 1, 1: 2}
OLD_LABELS = {
    0: "no_abnormality",
    1: "abnormality_present",
    2: "all_missing",
}
NEW_LABELS = {
    0: "all_missing",
    1: "no_abnormality",
    2: "abnormality_present",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_distribution(frame: pd.DataFrame) -> None:
    required = {
        "panel",
        "method",
        "dimension",
        "group_size",
        "state",
        "state_label",
        "proportion",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {STATE_FILE.name}: {sorted(missing)}")
    if set(frame["state"].astype(int)) != {0, 1, 2}:
        raise ValueError("Expected exactly state codes 0, 1, and 2")
    totals = frame.groupby(["panel", "method", "dimension"])["proportion"].sum()
    if not np.allclose(totals.to_numpy(float), 1.0):
        raise ValueError("State proportions do not sum to one for every dimension")
    duplicates = frame.duplicated(["panel", "method", "dimension", "state"])
    if duplicates.any():
        raise ValueError("Duplicate panel/method/dimension/state rows detected")


def encoding_kind(frame: pd.DataFrame) -> str:
    observed = (
        frame[["state", "state_label"]]
        .drop_duplicates()
        .sort_values("state")
        .set_index("state")["state_label"]
        .astype(str)
        .to_dict()
    )
    if observed == OLD_LABELS:
        return "legacy"
    if observed == NEW_LABELS:
        return "canonical"
    raise ValueError(f"Unrecognized state-label mapping: {observed}")


def update_manifest(original_sha256: str) -> None:
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    manifest["state_encoding"] = {
        "0": "all attributes in the semantic dimension are missing",
        "1": (
            "at least one attribute is observed and all observed attributes are "
            "within their prespecified reference intervals"
        ),
        "2": (
            "at least one observed attribute is outside its prespecified "
            "reference interval"
        ),
    }
    manifest["state_encoding_migration"] = {
        "type": "deterministic label permutation",
        "mapping": {"2": 0, "0": 1, "1": 2},
        "legacy_backup": LEGACY_BACKUP.relative_to(PROJECT_ROOT).as_posix(),
        "legacy_sha256": original_sha256,
        "affected_output": STATE_FILE.relative_to(PROJECT_ROOT).as_posix(),
        "unchanged_outputs": "MSE and runtime outputs contain no state-indexed columns",
    }
    MANIFEST_FILE.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    frame = pd.read_csv(STATE_FILE)
    validate_distribution(frame)
    kind = encoding_kind(frame)
    if kind == "canonical":
        print(f"{STATE_FILE.name} already uses the canonical encoding; no changes made.")
        return

    original_sha256 = sha256(STATE_FILE)
    if not LEGACY_BACKUP.exists():
        shutil.copy2(STATE_FILE, LEGACY_BACKUP)
    elif sha256(LEGACY_BACKUP) != original_sha256:
        raise ValueError(
            f"Existing backup {LEGACY_BACKUP.name} does not match the current legacy file"
        )

    frame["state"] = frame["state"].astype(int).map(OLD_TO_NEW)
    frame["state_label"] = frame["state"].map(NEW_LABELS)
    frame = frame.sort_values(
        ["panel", "method", "dimension", "state"], kind="stable"
    ).reset_index(drop=True)
    validate_distribution(frame)
    if encoding_kind(frame) != "canonical":
        raise AssertionError("Canonical encoding validation failed")

    frame.to_csv(STATE_FILE, index=False)
    update_manifest(original_sha256)
    print(f"Migrated {len(frame)} rows in {STATE_FILE.name} using 2->0, 0->1, 1->2.")
    print(f"Preserved the original as {LEGACY_BACKUP.name}.")


if __name__ == "__main__":
    main()
