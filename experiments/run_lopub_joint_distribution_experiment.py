"""Run the paired Smp common-query comparison with a LoPub-derived baseline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semmed_ldp.formal_experiment import sha256_file, summarize_with_bootstrap_ci
from semmed_ldp.formal_semantic_reduction import (
    DOMAIN_SIZE,
    build_ontology_group_states,
    validate_formal_mapping,
)
from semmed_ldp.lopub_em import (
    LoPubParameters,
    build_raw_attribute_states,
    evaluate_lopub_smp_em_run,
    sample_group_indices,
)
from semmed_ldp.metrics import compute_all_metrics
from semmed_ldp.multidimensional_reporting import (
    MultidimensionalProtocol,
    run_multidimensional_protocol,
)
from semmed_ldp.sampled_reporting import empirical_state_distributions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True, choices=["PIC", "NHANES_2021_2023"])
    parser.add_argument("--features", required=True)
    parser.add_argument("--entity-id", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--mapping", default="configs/formal_attribute_mapping.csv")
    parser.add_argument("--reference-intervals", default="configs/reference_intervals.csv")
    parser.add_argument("--config", default="configs/formal_experiment.yaml")
    parser.add_argument("--output-dir", default="results/formal_v3")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _indexed(path: Path, entity_id: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if entity_id not in frame:
        raise ValueError(f"{path} lacks entity column {entity_id!r}")
    return frame.set_index(entity_id)


def _metrics_from_frames(truth: pd.DataFrame, estimate: pd.DataFrame) -> dict[str, float]:
    merged = truth.merge(
        estimate,
        on=["semantic_dimension", "state"],
        validate="one_to_one",
    )
    rows = [
        compute_all_metrics(
            group["true_frequency"].to_numpy(float),
            group["estimated_frequency"].to_numpy(float),
        )
        for _, group in merged.groupby("semantic_dimension", sort=True)
    ]
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _semmed_smp(
    semantic_states: pd.DataFrame,
    *,
    epsilon: float,
    mechanism: str,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame, dict[str, object]]:
    protocol = MultidimensionalProtocol(
        epsilon=epsilon,
        mechanism=mechanism,
        strategy="Smp",
        domain_size=DOMAIN_SIZE,
    )
    estimate, diagnostics = run_multidimensional_protocol(
        semantic_states,
        protocol,
        seed=seed,
        measure_communication=False,
    )
    truth = empirical_state_distributions(semantic_states, DOMAIN_SIZE)
    estimates = estimate.merge(
        truth,
        on=["semantic_dimension", "state"],
        validate="one_to_one",
    )
    return _metrics_from_frames(truth, estimate), estimates, diagnostics


def main() -> None:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    baseline_config = config["experiments"]["lopub_joint_distribution"]
    if str(baseline_config.get("strategy", "")).upper() != "SMP":
        raise ValueError("the common-query comparison requires strategy: Smp")

    mapping_path = PROJECT_ROOT / args.mapping
    intervals_path = PROJECT_ROOT / args.reference_intervals
    features_path = PROJECT_ROOT / args.features
    metadata_path = PROJECT_ROOT / args.metadata
    mapping_all = pd.read_csv(mapping_path)
    mapping = validate_formal_mapping(mapping_all, dataset_id=args.dataset_id)
    features = _indexed(features_path, args.entity_id)[mapping["attribute_name"].tolist()]
    metadata = _indexed(metadata_path, args.entity_id).loc[features.index]
    intervals = pd.read_csv(intervals_path)
    raw_states = build_raw_attribute_states(
        features,
        mapping,
        metadata=metadata,
        reference_intervals=intervals,
    )
    semantic_states, _ = build_ontology_group_states(
        features,
        mapping,
        metadata=metadata,
        reference_intervals=intervals,
    )
    if semantic_states.shape[1] != 4:
        raise ValueError("the common-query comparison requires four semantic groups")

    parameters = LoPubParameters(
        bloom_bits=int(baseline_config["bloom_bits"]),
        hash_functions=int(baseline_config["hash_functions"]),
        convergence_gap=float(baseline_config["convergence_gap"]),
        min_iterations=int(baseline_config["min_iterations"]),
        max_iterations=int(baseline_config["max_iterations"]),
    )
    epsilon_list = [1.0] if args.smoke else [float(v) for v in config["privacy"]["epsilon_list"]]
    repeat_runs = 1 if args.smoke else int(baseline_config["repeat_runs"])
    base_seed = int(config["experiments"]["seed"])
    metric_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    estimate_frames: list[pd.DataFrame] = []

    for run in range(repeat_runs):
        run_seed = base_seed + run
        selected_groups = sample_group_indices(
            len(raw_states),
            semantic_states.shape[1],
            seed=run_seed,
        )
        for epsilon in epsilon_list:
            for mechanism in ("GRR", "OUE"):
                started = time.perf_counter()
                metrics, estimates, diagnostics = _semmed_smp(
                    semantic_states,
                    epsilon=epsilon,
                    mechanism=mechanism,
                    seed=run_seed,
                )
                shared = {
                    "dataset_id": args.dataset_id,
                    "experiment_family": "smp_common_query_comparison",
                    "method": "SemMed-LDP",
                    "strategy": "Smp",
                    "mechanism": mechanism,
                    "epsilon": epsilon,
                    "sample_size": len(raw_states),
                    "run": run,
                    "seed": run_seed,
                    "n_input_attributes": raw_states.shape[1],
                    "n_dimensions": semantic_states.shape[1],
                }
                metric_rows.append({**shared, **metrics})
                diagnostic_rows.append(
                    {
                        **shared,
                        **diagnostics,
                        "paired_group_selection": True,
                        "wall_seconds": time.perf_counter() - started,
                    }
                )
                estimate_frames.append(estimates.assign(**shared))

            started = time.perf_counter()
            metrics, estimates, diagnostics = evaluate_lopub_smp_em_run(
                raw_states,
                mapping,
                epsilon=epsilon,
                parameters=parameters,
                seed=run_seed,
                selected_groups=selected_groups,
            )
            group_diagnostics = diagnostics.pop("group_diagnostics")
            shared = {
                "dataset_id": args.dataset_id,
                "experiment_family": "smp_common_query_comparison",
                "method": "LoPub-derived-Bloom-EM",
                "strategy": "Smp",
                "mechanism": "Bloom-RR",
                "epsilon": epsilon,
                "sample_size": len(raw_states),
                "run": run,
                "seed": run_seed,
                "n_input_attributes": raw_states.shape[1],
                "n_dimensions": semantic_states.shape[1],
            }
            metric_rows.append({**shared, **metrics})
            diagnostic_rows.append(
                {
                    **shared,
                    **diagnostics,
                    "paired_group_selection": True,
                    "group_diagnostics_json": json.dumps(group_diagnostics),
                    "bloom_bits": parameters.bloom_bits,
                    "hash_functions": parameters.hash_functions,
                    "convergence_gap": parameters.convergence_gap,
                    "min_iterations": parameters.min_iterations,
                    "max_iterations": parameters.max_iterations,
                    "wall_seconds": time.perf_counter() - started,
                }
            )
            estimate_frames.append(estimates.assign(**shared))
            print(
                f"{args.dataset_id} run={run + 1}/{repeat_runs} epsilon={epsilon:g} "
                f"Bloom-EM MSE={metrics['MSE']:.6g} "
                f"converged={diagnostics['all_groups_converged']}",
                flush=True,
            )

    output = PROJECT_ROOT / args.output_dir / args.dataset_id / "lopub_joint_distribution"
    output.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(metric_rows)
    diagnostics_frame = pd.DataFrame(diagnostic_rows)
    estimates_frame = pd.concat(estimate_frames, ignore_index=True)
    metrics_frame.to_csv(output / "run_metrics.csv", index=False)
    diagnostics_frame.to_csv(output / "run_diagnostics.csv", index=False)
    estimates_frame.to_csv(output / "run_estimates.csv", index=False)
    summarize_with_bootstrap_ci(metrics_frame).to_csv(
        output / "summary_bootstrap_ci.csv",
        index=False,
    )
    metrics_frame.to_csv(output / "common_query_comparison.csv", index=False)

    inputs = [features_path, metadata_path, mapping_path, intervals_path, config_path]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "paper": {
            "title": "LoPub: High-Dimensional Crowdsourced Data Publication With Local Differential Privacy",
            "doi": "10.1109/TIFS.2018.2812146",
            "implementation_scope": (
                "LoPub-derived Bloom randomized response and EM joint estimation "
                "embedded in Smp; no Lasso or synthetic-data generation"
            ),
        },
        "comparison_contract": {
            "target": "same four ontology-group, three-state query distributions",
            "strategy": "Smp",
            "reports_per_user": 1,
            "paired_group_selection": True,
            "user_level_epsilon": "same complete epsilon for every transmitted report",
        },
        "parameters": {
            "bloom_bits": parameters.bloom_bits,
            "hash_functions": parameters.hash_functions,
            "convergence_gap": parameters.convergence_gap,
            "min_iterations": parameters.min_iterations,
            "max_iterations": parameters.max_iterations,
            "group_privacy_relation": "epsilon = 2*k_j*h*ln((2-f_j)/f_j)",
        },
        "inputs": [
            {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(path)}
            for path in inputs
        ],
        "outputs": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(path),
            }
            for path in [
                output / "run_metrics.csv",
                output / "run_diagnostics.csv",
                output / "run_estimates.csv",
                output / "summary_bootstrap_ci.csv",
                output / "common_query_comparison.csv",
            ]
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
