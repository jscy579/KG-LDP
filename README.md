# KGR-LDP reproducibility package

This directory contains the code, frozen configurations, analysis inputs,
run-level results, and figure source data used for the current KGR-LDP
manuscript.

KGR-LDP is positioned as a knowledge-graph-guided representation method for
record-level local differential privacy (LDP). Public biomedical knowledge and
predefined semantic rules organize \(d\) medical attributes into
\(m_{\mathcal C}<d\) release dimensions before perturbation. Each record is
locally converted to an unweighted three-state representation and perturbed by
an LDP mechanism; the server estimates population semantic-state
distributions.

## Directory layout

- `semmed_ldp/`: core semantic reduction, LDP mechanisms, reporting protocols,
  estimators, metrics, information-retention analysis, and LoPub-derived
  comparison implementation.
- `experiments/`: data-preparation, formal-v3 experiment, statistical summary,
  and figure-generation scripts.
- `configs/`: frozen formal-v3 attribute mapping, graph edges, public reference
  intervals, analysis scope, and experiment settings.
- `data/processed/working/`: local analysis matrices used by the formal-v3
  experiments and the information--error figure.
- `data/raw/nhanes_2021_2023/`: public NHANES source files used by the included
  NHANES preparation script.
- `data/knowledge/`: resource versions, hashes, source URLs, and redistribution
  policies. The complete LOINC and HPO distributions are not bundled.
- `results/formal_v3/`: run-level metrics, paired statistical tests, bootstrap
  summaries, estimates, and provenance manifests used by the paper.
- `figures/formal_v3/source_data/`: clean source-data tables underlying the
  paper figures.
- `figures/formal_v3/`: reproducible PDF, PNG, and SVG outputs and figure
  manifests. TIFF duplicates are omitted to keep the package compact.
- `tests/`: unit and integration tests.

The Python package and several result tables retain the historical internal
identifier `SemMed-LDP`. These identifiers are preserved so that stored
manifests, method labels, and run-level results remain traceable; the manuscript
method name is KGR-LDP.

## Environment

Python 3.10 or later is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

## Regenerate the paper figures

From this directory, run:

```powershell
.\REPRODUCE_FIGURES.ps1
```

The script regenerates:

- `fig_protocol_comparison`
- `fig_lopub_common_query`
- `fig_information_privacy_tradeoff`
- `fig_relative_mse_improvement`
- the KGR-LDP framework figure

All numerical panels read the included formal-v3 results or the included local
analysis matrices. Figure values are not entered manually.

## Re-run the formal-v3 experiments

PIC:

```powershell
python experiments/run_formal_distribution_experiment.py `
  --dataset-id PIC `
  --features data/processed/working/pic_feature_matrix.csv `
  --entity-id SUBJECT_ID `
  --metadata data/processed/working/pic_metadata.csv `
  --external-dataset-id NHANES_2021_2023 `
  --external-features data/processed/working/nhanes_feature_matrix.csv `
  --external-entity-id SEQN `
  --external-metadata data/processed/working/nhanes_metadata.csv
```

NHANES 2021--2023:

```powershell
python experiments/run_formal_distribution_experiment.py `
  --dataset-id NHANES_2021_2023 `
  --features data/processed/working/nhanes_feature_matrix.csv `
  --entity-id SEQN `
  --metadata data/processed/working/nhanes_metadata.csv `
  --external-dataset-id PIC `
  --external-features data/processed/working/pic_feature_matrix.csv `
  --external-entity-id SUBJECT_ID `
  --external-metadata data/processed/working/pic_metadata.csv
```

LoPub-derived common-query comparison:

```powershell
python experiments/run_lopub_joint_distribution_experiment.py `
  --dataset-id PIC `
  --features data/processed/working/pic_feature_matrix.csv `
  --entity-id SUBJECT_ID `
  --metadata data/processed/working/pic_metadata.csv

python experiments/run_lopub_joint_distribution_experiment.py `
  --dataset-id NHANES_2021_2023 `
  --features data/processed/working/nhanes_feature_matrix.csv `
  --entity-id SEQN `
  --metadata data/processed/working/nhanes_metadata.csv

python experiments/summarize_lopub_joint_distribution.py
```

Use `--smoke` for a short execution check before launching the full 30-repeat
grid.

## Data and knowledge-resource boundaries

The included processed matrices are local reproducibility inputs, not files for
unrestricted public redistribution. PIC raw tables are not included. The
complete LOINC distribution is also excluded because its frozen manifest marks
it as `do_not_redistribute`. See `data/README.md` and
`data/knowledge/resource_manifest.json` before sharing this package.

## Integrity

`FILE_MANIFEST_SHA256.csv` records the relative path, byte size, and SHA-256
digest of every packaged file except the manifest itself. Recompute or verify
the manifest after any intentional modification.
