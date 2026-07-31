"""Fail-closed readiness audit for formal SemMed-LDP experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semmed_ldp.formal_semantic_reduction import validate_formal_mapping
from semmed_ldp.reference_intervals import validate_reference_intervals


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_file(path: Path, *, minimum_bytes: int = 1024) -> dict:
    exists = path.is_file()
    size = path.stat().st_size if exists else 0
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "passed": bool(exists and size > minimum_bytes),
        "bytes": size,
        "sha256": sha256(path) if exists and size else None,
    }


def check_mapping(path: Path, dataset_id: str) -> dict:
    if not path.is_file():
        return {"path": str(path.relative_to(PROJECT_ROOT)), "passed": False, "reason": "missing"}
    try:
        frame = validate_formal_mapping(pd.read_csv(path), dataset_id=dataset_id)
    except Exception as exc:  # the audit records all validation failures
        return {"path": str(path.relative_to(PROJECT_ROOT)), "passed": False, "reason": str(exc)}
    expected_attributes = 10
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "passed": len(frame) == expected_attributes,
        "n_attributes": len(frame),
        "expected_attributes": expected_attributes,
        "n_anchors": int(frame["semantic_anchor"].nunique()),
        "mapping_coverage": 1.0,
        "review_pass_rate": 1.0,
    }


def check_config(path: Path) -> dict:
    if not path.is_file():
        return {"path": str(path.relative_to(PROJECT_ROOT)), "passed": False, "reason": "missing"}
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    knowledge = config.get("knowledge", {})
    versions = {key: str(knowledge.get(key, "")) for key in ["loinc_version", "hpo_version"]}
    placeholders = {key: value for key, value in versions.items() if not value or "PIN_" in value.upper()}
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "passed": not placeholders,
        "knowledge_versions": versions,
        "placeholders": placeholders,
    }


def check_intervals(path: Path) -> dict:
    if not path.is_file():
        return {"path": str(path.relative_to(PROJECT_ROOT)), "passed": False, "reason": "missing"}
    try:
        frame = validate_reference_intervals(pd.read_csv(path))
    except Exception as exc:
        return {"path": str(path.relative_to(PROJECT_ROOT)), "passed": False, "reason": str(exc)}
    return {"path": str(path.relative_to(PROJECT_ROOT)), "passed": True, "n_rows": len(frame)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", default="configs/formal_attribute_mapping.csv")
    parser.add_argument("--reference-intervals", default="configs/reference_intervals.csv")
    parser.add_argument("--output", default="results/readiness/formal_readiness.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks = {
        "loinc_table": check_file(PROJECT_ROOT / "data/knowledge/loinc/2.82/LoincTable/Loinc.csv"),
        "loinc_groups": check_file(PROJECT_ROOT / "data/knowledge/loinc/2.82/AccessoryFiles/GroupFile/Group.csv"),
        "loinc_group_terms": check_file(PROJECT_ROOT / "data/knowledge/loinc/2.82/AccessoryFiles/GroupFile/GroupLoincTerms.csv"),
        "loinc_parent_groups": check_file(PROJECT_ROOT / "data/knowledge/loinc/2.82/AccessoryFiles/GroupFile/ParentGroup.csv"),
        "hpo": check_file(PROJECT_ROOT / "data/knowledge/hpo/hp.obo"),
        "nhanes_demo": check_file(PROJECT_ROOT / "data/raw/nhanes_2021_2023/DEMO_L.XPT"),
        "nhanes_cbc": check_file(PROJECT_ROOT / "data/raw/nhanes_2021_2023/CBC_L.XPT"),
        "nhanes_biopro": check_file(PROJECT_ROOT / "data/raw/nhanes_2021_2023/BIOPRO_L.XPT"),
        "formal_config": check_config(PROJECT_ROOT / "configs/formal_experiment.yaml"),
        "formal_graph": check_file(PROJECT_ROOT / "configs/formal_graph_edges.csv", minimum_bytes=100),
        "PIC_mapping": check_mapping(PROJECT_ROOT / args.mapping, "PIC"),
        "NHANES_mapping": check_mapping(PROJECT_ROOT / args.mapping, "NHANES_2021_2023"),
        "reference_intervals": check_intervals(PROJECT_ROOT / args.reference_intervals),
    }
    passed = all(value["passed"] for value in checks.values())
    report = {
        "project_id": "SemMed-LDP",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_execution_ready": passed,
        "checks": checks,
    }
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
