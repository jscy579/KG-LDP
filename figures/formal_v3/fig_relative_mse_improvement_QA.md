# Figure QA: relative MSE improvement

- Core conclusion: KGR-LDP has conditional, not universal, MSE advantages across privacy budgets.
- Archetype: quantitative 2 x 2 robustness grid.
- Backend: Python/matplotlib only.
- Inputs: full-cohort inferential run metrics; 30 paired repeat-level observations per comparison.
- Center: relative improvement computed from paired-run mean MSEs.
- Interval: paired percentile 95% Monte Carlo interval, 20,000 resamples, seed 2026.
- Test: paired Wilcoxon signed-rank test; Holm correction within the existing formal-v3 comparison family.
- Positive values: KGR-LDP lower MSE. Negative values: named baseline lower MSE.
- Random control: each repeat is the mean of 100 matched random partitions; partitions are not treated as independent replicates.
- Export: editable SVG and PDF; 600 dpi TIFF; PNG preview.
- Review limitation: the two controls are structural controls, not published end-to-end competing methods.
