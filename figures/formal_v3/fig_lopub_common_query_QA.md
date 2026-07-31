# Figure 2 QA: paired Smp common-query comparison

- Core conclusion: under one sampled group, one report per user, the same full
  user-level epsilon, and the same four three-state queries, direct SemMed
  randomization has lower MSE than group-local Bloom-EM in 25 of 28 conditions.
- Archetype: two-panel quantitative comparison.
- Backend: Python/matplotlib only.
- Cohorts: PIC (n = 5,349) and NHANES 2021--2023 (n = 8,153).
- Repeats: 30 paired seeds per privacy budget.
- Center and interval: randomized-run mean and percentile 95% Monte Carlo
  interval over 30 paired runs on the fixed cohort.
- Paired inference: two-sided Wilcoxon signed-rank tests and rank-biserial
  effects, with Holm correction across the 14 mechanism--epsilon contrasts
  within each cohort.
- Source data: `source_data/fig_lopub_common_query.csv`.
- Paired statistics: `source_data/fig_lopub_common_query_paired_statistics.csv`.
- Formal result source: `results/formal_v3/lopub_common_query_summary.csv`.
- Baseline scope: LoPub-derived group-local Bloom randomized response plus EM
  joint estimation embedded in Smp; no Lasso or synthetic-data stage.
- Privacy check: each group reconstructs the configured epsilon using
  `epsilon = 2*k_j*h*log((2-f_j)/f_j)`.
- Numerical check: all 1,260 run-level metric rows are finite; all 420 Bloom-EM
  runs converged after 25--281 iterations.
- Export check: editable SVG and PDF, 600 dpi TIFF, and PNG preview generated.
- Visual check: log axes, labels, legend, confidence bands, and panel labels are
  readable at manuscript width; no clipping or overlap detected.
