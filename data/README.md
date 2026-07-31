# Data included in this package

## Local analysis inputs

`processed/working/` contains the four matrices used by the formal-v3 analysis:

- `pic_feature_matrix.csv`
- `pic_metadata.csv`
- `nhanes_feature_matrix.csv`
- `nhanes_metadata.csv`

They are included so that the formal experiment scripts and the
information--error analysis can be reproduced without changing paths. These
files contain record-level analysis rows and dataset identifiers. Keep them in
controlled local storage and do not upload them to a public repository without
checking the source-dataset access and redistribution terms.

## Public NHANES source files

`raw/nhanes_2021_2023/` contains the DEMO, CBC, and BIOPRO XPT files identified
in `knowledge/resource_manifest.json`. Their source URLs and hashes are recorded
in that manifest.

## Files deliberately not bundled

- PIC raw database tables: access-controlled source data; obtain them through
  the original data provider and use `experiments/prepare_pic_formal.py`.
- Complete LOINC 2.82 distribution: the local resource manifest marks it as
  `do_not_redistribute`.
- Complete HPO ontology: version, source, and hash are retained in the resource
  manifest, but the ontology file is not needed to rerun the frozen formal-v3
  experiments.

The frozen attribute mapping, semantic graph edges, and public reference
intervals required by the formal-v3 experiments are provided under `configs/`.
