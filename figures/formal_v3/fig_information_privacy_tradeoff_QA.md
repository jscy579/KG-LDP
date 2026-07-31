# Information--privacy-error figure contract and QA

- Core conclusion: evaluate whether SemMed 4D occupies a favorable raw-state information-retention versus private-estimation-error region; do not presume it is universally optimal.
- Archetype: 2-by-2 quantitative comparison grid.
- Panels: PIC and NHANES 2021--2023, each under GRR and OUE.
- Privacy condition: full cohort, epsilon = 1, Smp reporting.
- X axis: mean MSE over 30 fixed-seed randomized runs. Named-method horizontal intervals are percentile 95% Monte Carlo intervals for the mean from the existing formal-v3 summaries. Each gray point is one random partition's mean over the same 30 runs.
- Y axis: equal-weight mean of normalized mutual information, mean_a I(X_a; Z)/H(X_a), over nonconstant raw attributes and calculated from the nonprivate three-state representation. It is descriptive and consumes no privacy budget.
- Random baseline: 100 partitions preserving the SemMed group-size multiset.
- User counts: PIC n = 5,349; NHANES n = 8,153.
- Visual encodings: method-specific color and marker; random partitions additionally encoded as a point cloud, so meaning is retained in grayscale.
- Export: editable SVG and PDF, 600 dpi TIFF, PNG preview, source-data CSVs, and hash manifest.
- Interpretation boundary: the Y axis measures retention of raw three-state statistical information, not clinical semantic validity or exact clinical-query preservation.
