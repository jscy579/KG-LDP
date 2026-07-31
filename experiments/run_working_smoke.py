"""Engineering evaluation using unapproved working evidence in memory.

Outputs are written under results/readiness and are never formal manuscript
evidence. Reviewer statuses in source files are not changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semmed_ldp.formal_experiment import (
    build_run_manifest,
    paired_wilcoxon_holm,
    run_formal_ablation_experiment,
    run_formal_distribution_experiment,
    sha256_file,
    summarize_with_bootstrap_ci,
    write_json,
)


DATASETS = {
    "PIC": ("pic_feature_matrix.csv", "pic_metadata.csv", "SUBJECT_ID"),
    "NHANES_2021_2023": ("nhanes_feature_matrix.csv", "nhanes_metadata.csv", "SEQN"),
}

PROVENANCE_CODE = [
    ROOT / "experiments/run_working_smoke.py",
    ROOT / "semmed_ldp/formal_experiment.py",
    ROOT / "semmed_ldp/formal_semantic_reduction.py",
    ROOT / "semmed_ldp/sampled_reporting.py",
    ROOT / "semmed_ldp/ldp_mechanisms.py",
    ROOT / "semmed_ldp/metrics.py",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the complete configured grid with working, unapproved evidence.",
    )
    args = parser.parse_args()
    config_path = ROOT / "configs/formal_experiment.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mapping = pd.read_csv(ROOT / "configs/formal_attribute_mapping_working.csv")
    mapping = mapping.loc[mapping["primary_analysis_status"].eq("included")].copy()
    intervals = pd.read_csv(ROOT / "configs/reference_intervals_working.csv")
    mapping["reviewer_status"] = "approved"
    intervals["reviewer_status"] = "approved"
    if args.full:
        epsilon_list = config["privacy"]["epsilon_list"]
        mechanisms = config["privacy"]["mechanisms"]
        sample_sizes = config["experiments"]["sample_sizes"]
        repeat_runs = int(config["privacy"]["repeat_runs"])
        output = ROOT / "results/readiness/working_experiment"
        purpose = "complete configured engineering evaluation; working evidence is unapproved and not formal"
    else:
        epsilon_list = [1.0]
        mechanisms = ["GRR", "OUE"]
        sample_sizes = [500]
        repeat_runs = 1
        output = ROOT / "results/readiness/working_smoke"
        purpose = "engineering-only smoke test; not formal evidence"
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "purpose": purpose,
        "source_reviewer_status": "needs_review",
        "formal_manuscript_evidence": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "conditions": {
            "epsilon": epsilon_list,
            "mechanism": mechanisms,
            "sample_size": sample_sizes,
            "repeat_runs": repeat_runs,
        },
        "datasets": {},
    }
    for dataset_id, (feature_name, metadata_name, entity_id) in DATASETS.items():
        features = pd.read_csv(ROOT / "data/processed/working" / feature_name, index_col=0)
        metadata = pd.read_csv(ROOT / "data/processed/working" / metadata_name, index_col=0)
        local_mapping = mapping.loc[mapping["dataset_id"].eq(dataset_id)].copy()
        main_results, main_diagnostics = run_formal_distribution_experiment(
            features,
            local_mapping,
            dataset_id=dataset_id,
            epsilon_list=epsilon_list,
            mechanisms=mechanisms,
            sample_sizes=sample_sizes,
            repeat_runs=repeat_runs,
            seed=2026,
            metadata=metadata,
            reference_intervals=intervals,
        )
        ablation_results, ablation_diagnostics, support = run_formal_ablation_experiment(
            features,
            local_mapping,
            dataset_id=dataset_id,
            epsilon_list=epsilon_list,
            mechanisms=mechanisms,
            sample_sizes=sample_sizes,
            repeat_runs=repeat_runs,
            seed=2026,
            metadata=metadata,
            reference_intervals=intervals,
        )
        dataset_output = output / dataset_id
        dataset_output.mkdir(parents=True, exist_ok=True)
        main_results.to_csv(dataset_output / "main_metrics.csv", index=False)
        main_diagnostics.to_csv(dataset_output / "main_diagnostics.csv", index=False)
        ablation_results.to_csv(dataset_output / "ablation_metrics.csv", index=False)
        ablation_diagnostics.to_csv(dataset_output / "ablation_diagnostics.csv", index=False)
        support.to_csv(dataset_output / "semantic_state_support.csv", index=False)
        if args.full:
            summarize_with_bootstrap_ci(main_results).to_csv(
                dataset_output / "main_summary_bootstrap_ci.csv", index=False
            )
            paired_wilcoxon_holm(main_results).to_csv(
                dataset_output / "main_paired_wilcoxon_holm.csv", index=False
            )
            summarize_with_bootstrap_ci(ablation_results).to_csv(
                dataset_output / "ablation_summary_bootstrap_ci.csv", index=False
            )
            manifest_config = dict(config)
            manifest_config["evidence_status"] = "working_unapproved"
            manifest_config["formal_manuscript_evidence"] = False
            manifest_inputs = [
                ROOT / "data/processed/working" / feature_name,
                ROOT / "data/processed/working" / metadata_name,
                ROOT / "configs/formal_attribute_mapping_working.csv",
                ROOT / "configs/reference_intervals_working.csv",
                config_path,
                ROOT / config["knowledge"]["edge_path"].replace(".csv", "_working.csv"),
                ROOT / config["knowledge"]["resource_manifest"],
                ROOT / config["primary_scope"]["scope_path"],
            ] + PROVENANCE_CODE
            output_files = [
                dataset_output / "main_metrics.csv",
                dataset_output / "main_diagnostics.csv",
                dataset_output / "main_summary_bootstrap_ci.csv",
                dataset_output / "main_paired_wilcoxon_holm.csv",
                dataset_output / "ablation_metrics.csv",
                dataset_output / "ablation_diagnostics.csv",
                dataset_output / "ablation_summary_bootstrap_ci.csv",
                dataset_output / "semantic_state_support.csv",
            ]
            manifest = build_run_manifest(manifest_config, manifest_inputs)
            manifest["outputs"] = [
                {"path": str(path), "sha256": sha256_file(path)} for path in output_files
            ]
            write_json(
                dataset_output / "manifest.json",
                manifest,
            )
        metric_columns = ["MSE", "MAE", "JS", "TVD", "Wasserstein"]
        finite_metrics = bool(
            np.isfinite(main_results[metric_columns].to_numpy(dtype=float)).all()
            and np.isfinite(ablation_results[metric_columns].to_numpy(dtype=float)).all()
        )
        report["datasets"][dataset_id] = {
            "entity_id": entity_id,
            "n_primary_attributes": int(len(local_mapping)),
            "main_methods": sorted(main_results["method"].unique().tolist()),
            "ablation_methods": sorted(ablation_results["method"].unique().tolist()),
            "one_report_per_user": bool(
                main_diagnostics["reports_per_user"].eq(1.0).all()
                and ablation_diagnostics["reports_per_user"].eq(1.0).all()
            ),
            "all_modules_nonconstant": bool(~support["constant"].any()),
            "all_metrics_finite": finite_metrics,
            "n_main_metric_rows": int(len(main_results)),
            "n_ablation_metric_rows": int(len(ablation_results)),
        }
    manifest_name = "working_experiment_manifest.json" if args.full else "smoke_manifest.json"
    (output / manifest_name).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
