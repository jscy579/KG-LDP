"""Prepare the formal PIC matrix without invoking the pilot workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semmed_ldp.dataset_adapters import prepare_pic_formal_from_raw
from semmed_ldp.formal_semantic_reduction import validate_formal_mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="E:/research/data/pic_raw_data/paediatic-intensive-care-database-1.2.0")
    parser.add_argument("--mapping", default="configs/formal_attribute_mapping.csv")
    parser.add_argument("--working", action="store_true", help="Prepare an audit matrix from unapproved mappings")
    args = parser.parse_args()
    mapping = pd.read_csv(PROJECT_ROOT / args.mapping)
    if not args.working:
        mapping = validate_formal_mapping(mapping, dataset_id="PIC")
    features, metadata, units = prepare_pic_formal_from_raw(args.raw_dir, mapping)
    tier = "working" if args.working else "formal_v1"
    output = PROJECT_ROOT / "data/processed" / tier
    output.mkdir(parents=True, exist_ok=True)
    features.to_csv(output / "pic_feature_matrix.csv", index_label="SUBJECT_ID")
    metadata.to_csv(output / "pic_metadata.csv", index_label="SUBJECT_ID")
    units.to_csv(PROJECT_ROOT / "data/interim/pic_unit_audit.csv", index=False)
    print(f"Prepared {features.shape[0]} eligible PIC users and {features.shape[1]} attributes")


if __name__ == "__main__":
    main()
