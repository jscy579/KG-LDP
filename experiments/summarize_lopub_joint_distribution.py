"""Create a clean summary for the paired Smp common-query comparison."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semmed_ldp.formal_experiment import sha256_file


RESULT_ROOT = PROJECT_ROOT / "results" / "formal_v3"
DATASETS = ["PIC", "NHANES_2021_2023"]


def main() -> None:
    summary_frames = []
    ratio_frames = []
    inputs = []
    for dataset_id in DATASETS:
        lopub_path = (
            RESULT_ROOT
            / dataset_id
            / "lopub_joint_distribution"
            / "summary_bootstrap_ci.csv"
        )
        diagnostics_path = (
            RESULT_ROOT
            / dataset_id
            / "lopub_joint_distribution"
            / "run_diagnostics.csv"
        )
        inputs.extend([lopub_path, diagnostics_path])
        comparison = pd.read_csv(lopub_path)
        semmed = comparison[comparison["method"].eq("SemMed-LDP")].copy()
        lopub = comparison[
            comparison["method"].eq("LoPub-derived-Bloom-EM")
        ].copy()
        keep = [
            "dataset_id",
            "method",
            "mechanism",
            "epsilon",
            "sample_size",
            "MSE_mean",
            "MSE_ci_low",
            "MSE_ci_high",
            "JS_mean",
            "TVD_mean",
        ]
        summary_frames.extend([semmed[keep], lopub[keep]])

        semmed_mean = semmed[
            ["dataset_id", "mechanism", "epsilon", "MSE_mean"]
        ].rename(columns={"mechanism": "semmed_mechanism", "MSE_mean": "MSE_semmed"})
        lopub_mean = lopub[["dataset_id", "epsilon", "MSE_mean"]].rename(
            columns={"MSE_mean": "MSE_lopub_em_jd"}
        )
        ratio = semmed_mean.merge(
            lopub_mean,
            on=["dataset_id", "epsilon"],
            how="inner",
            validate="many_to_one",
        )
        ratio["lopub_to_semmed_MSE_ratio"] = (
            ratio["MSE_lopub_em_jd"] / ratio["MSE_semmed"]
        )
        ratio["lower_MSE_method"] = ratio.apply(
            lambda row: "LoPub-derived-Bloom-EM"
            if row["MSE_lopub_em_jd"] < row["MSE_semmed"]
            else f"SemMed-{row['semmed_mechanism']}",
            axis=1,
        )
        ratio_frames.append(ratio)

    summary = pd.concat(summary_frames, ignore_index=True).sort_values(
        ["dataset_id", "mechanism", "method", "epsilon"]
    )
    ratios = pd.concat(ratio_frames, ignore_index=True).sort_values(
        ["dataset_id", "semmed_mechanism", "epsilon"]
    )
    summary_path = RESULT_ROOT / "lopub_common_query_summary.csv"
    ratio_path = RESULT_ROOT / "lopub_vs_semmed_mse_ratio.csv"
    summary.to_csv(summary_path, index=False)
    ratios.to_csv(ratio_path, index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "comparison_target": "identical four ontology-group, three-state query distributions",
        "comparison_contract": (
            "same Smp-selected group, one report per user, same complete user-level epsilon"
        ),
        "lopub_scope": (
            "LoPub-derived group-local Bloom-RR plus EM joint estimation; "
            "not the complete LoPub pipeline"
        ),
        "inputs": [
            {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(path)}
            for path in inputs
        ],
        "outputs": [
            {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(path)}
            for path in [summary_path, ratio_path]
        ],
    }
    (RESULT_ROOT / "lopub_common_query_summary_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
