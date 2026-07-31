# SemMed-LDP Project Status

- Project ID: `SemMed-LDP`
- Current stage: corrected formal-v3 experiments, figures, and MDPI manuscript complete
- Status updated: 2026-07-23
- Source of truth: files and outputs inside this directory only

## Completed

- Reviewed PIC cohort: 5,349 patients aged 1--17 years with 10 primary observations.
- Reviewed NHANES 2021--2023 cohort: 8,153 adults with the same 10 canonical observations.
- Twenty dataset-specific mappings and the required public interval strata passed the named review gate.
- LOINC 2.82 and HPO 2026-02-16 are checksum-frozen and provenance-documented.
- UBERON was removed from the formal-v3 method because it did not discriminate the selected blood observations.
- Ontology-constrained equal-weight three-state reduction: all missing, observed with no abnormality, or at least one abnormality.
- User-level multidimensional protocols implemented for Smp, Spl, and RS+FD, each with replaceable GRR or OUE randomization.
- Main experiment 1 fixes the identical SemMed 4D truth and compares Smp, Spl, and RS+FD.
- Main experiment 2 fixes Smp and compares SemMed 4D, 100 size-matched random 4D partitions, and cross-cohort external-correlation 4D grouping.
- Secondary trade-off compares Raw 10D, LOINC Group 10D, SemMed 4D, Source-Schema 2--3D, and LOINC Class 2D.
- Both cohorts completed 30 repeats, seven privacy budgets, and both mechanisms.
- A corrected LoPub-derived component baseline completed 30 repeats and seven privacy budgets in both cohorts. Under the same Smp protocol as SemMed, every user reports one data-independently selected group with the complete user-level epsilon. The baseline applies group-local Bloom randomized response, estimates that group's raw joint distribution by EM, and post-processes it to the identical three-state query. It is not presented as the complete LoPub pipeline.
- The group-level Bloom relation is calibrated as `epsilon = 2*k_j*h*log((2-f_j)/f_j)`; all formal runs reconstructed the configured budget, completed at least 25 EM updates, converged within 500 updates, and produced finite metrics.
- Per cohort outputs: 6,300 protocol rows, 42,840 raw representation rows, and 10,500 trade-off rows.
- Random-partition inference averages 100 partitions within each repeat, preserving 30 independent repeat-level units.
- Protocol winners across 140 conditions: RS+FD 82, Smp 37, Spl 21.
- At fixed 4D, SemMed had lower MSE than matched random grouping in 18/28 conditions and external-correlation grouping in 16/28; the manuscript does not claim universal superiority.
- Six publication/diagnostic figures are available as editable SVG/PDF, 600 dpi TIFF, and PNG, including information--error and all-epsilon relative-MSE views with clean source data.
- The revised MDPI manuscript includes the literature comparison table, privacy derivations, all main figures, and bounded interpretation in the body.
- In the corrected paired-Smp common-query comparison, SemMed had lower MSE in 25/28 dataset--mechanism--epsilon conditions. At epsilon 1, Bloom-EM-to-SemMed ratios were 16.42/7.20 for PIC GRR/OUE and 149.47/66.24 for NHANES GRR/OUE.
- Test status: 28 passed.

## Formal evidence and outputs

- Formal mapping: `configs/formal_attribute_mapping.csv`.
- Public intervals: `configs/reference_intervals.csv`.
- Formal graph edges: `configs/formal_graph_edges.csv`.
- Experiment configuration: `configs/formal_experiment.yaml` (`formal-v3`).
- Aggregate outputs and manifests: `results/formal_v3/`.
- Figures and source data: `figures/formal_v3/`.
- Reproducible local MDPI build: `manuscript/mdpi_work/`.

## Next action

Complete author-level review and replace the remaining submission placeholders: authors, affiliations, correspondence, funding, institutional-review statement, informed-consent statement, and confirmed data/code availability wording.
