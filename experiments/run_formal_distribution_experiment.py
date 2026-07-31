"""Run the three prespecified SemMed-LDP formal-v3 experiment families."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semmed_ldp.formal_experiment import (
    build_run_manifest,
    collapse_random_partition_metrics,
    paired_wilcoxon_holm,
    run_protocol_comparison_experiment,
    run_representation_comparison_experiment,
    run_tradeoff_experiment,
    summarize_with_bootstrap_ci,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--entity-id", default="patient_id")
    parser.add_argument("--metadata")
    parser.add_argument("--external-dataset-id", required=True)
    parser.add_argument("--external-features", required=True)
    parser.add_argument("--external-entity-id", required=True)
    parser.add_argument("--external-metadata")
    parser.add_argument("--mapping", default="configs/formal_attribute_mapping.csv")
    parser.add_argument("--reference-intervals", default="configs/reference_intervals.csv")
    parser.add_argument("--config", default="configs/formal_experiment.yaml")
    parser.add_argument("--output-dir", default="results/formal_v3")
    parser.add_argument("--smoke", action="store_true", help="Use one run, one partition, and epsilon=1")
    parser.add_argument(
        "--families",
        nargs="+",
        choices=["protocol", "representation", "tradeoff"],
        default=["protocol", "representation", "tradeoff"],
    )
    return parser.parse_args()


def _load_indexed(path: Path, entity_id: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if entity_id not in frame:
        raise ValueError(f"{path} lacks entity column {entity_id!r}")
    return frame.set_index(entity_id)


def _write_family(
    output: Path,
    results: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    inferential_results: pd.DataFrame | None = None,
    proposed: str = "SemMed-LDP",
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "run_metrics.csv", index=False)
    diagnostics.to_csv(output / "run_diagnostics.csv", index=False)
    inference = results if inferential_results is None else inferential_results
    inference.to_csv(output / "inferential_run_metrics.csv", index=False)
    summarize_with_bootstrap_ci(inference).to_csv(output / "summary_bootstrap_ci.csv", index=False)
    paired_wilcoxon_holm(inference, proposed=proposed).to_csv(
        output / "paired_wilcoxon_holm.csv", index=False
    )


def main() -> None:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mapping_path = PROJECT_ROOT / args.mapping
    mapping = pd.read_csv(mapping_path)
    feature_path = PROJECT_ROOT / args.features
    external_feature_path = PROJECT_ROOT / args.external_features
    features = _load_indexed(feature_path, args.entity_id)
    external_features = _load_indexed(external_feature_path, args.external_entity_id)
    metadata_path = PROJECT_ROOT / args.metadata if args.metadata else None
    external_metadata_path = PROJECT_ROOT / args.external_metadata if args.external_metadata else None
    metadata = _load_indexed(metadata_path, args.entity_id) if metadata_path else None
    external_metadata = (
        _load_indexed(external_metadata_path, args.external_entity_id)
        if external_metadata_path
        else None
    )
    interval_path = PROJECT_ROOT / args.reference_intervals
    reference_intervals = pd.read_csv(interval_path)

    privacy = config["privacy"]
    families = config["experiments"]
    epsilon_list = [1.0] if args.smoke else privacy["epsilon_list"]
    mechanisms = privacy["mechanisms"]
    seed = int(families["seed"])
    output = PROJECT_ROOT / args.output_dir / args.dataset_id

    if "protocol" in args.families:
        protocol_config = families["protocol_comparison"]
        protocol_results, protocol_diagnostics = run_protocol_comparison_experiment(
            features,
            mapping,
            dataset_id=args.dataset_id,
            epsilon_list=epsilon_list,
            mechanisms=mechanisms,
            strategies=privacy["strategies"],
            sample_sizes=["all"] if args.smoke else protocol_config["sample_sizes"],
            repeat_runs=1 if args.smoke else int(protocol_config["repeat_runs"]),
            seed=seed,
            metadata=metadata,
            reference_intervals=reference_intervals,
        )
        _write_family(
            output / "protocol_comparison",
            protocol_results,
            protocol_diagnostics,
            proposed="SemMed+Smp",
        )

    if "representation" in args.families:
        representation_config = families["representation_comparison"]
        representation_results, representation_diagnostics = run_representation_comparison_experiment(
            features,
            mapping,
            external_features=external_features,
            external_mapping=mapping,
            dataset_id=args.dataset_id,
            external_dataset_id=args.external_dataset_id,
            epsilon_list=epsilon_list,
            mechanisms=mechanisms,
            sample_sizes=representation_config["sample_sizes"],
            repeat_runs=1 if args.smoke else int(representation_config["repeat_runs"]),
            n_random_partitions=1 if args.smoke else int(representation_config["n_random_partitions"]),
            seed=seed,
            target_metadata=metadata,
            external_metadata=external_metadata,
            reference_intervals=reference_intervals,
        )
        collapsed = collapse_random_partition_metrics(representation_results)
        _write_family(
            output / "representation_comparison",
            representation_results,
            representation_diagnostics,
            inferential_results=collapsed,
        )

    if "tradeoff" in args.families:
        tradeoff_config = families["resolution_tradeoff"]
        tradeoff_results, tradeoff_diagnostics = run_tradeoff_experiment(
            features,
            mapping,
            dataset_id=args.dataset_id,
            epsilon_list=epsilon_list,
            mechanisms=mechanisms,
            sample_sizes=["all"] if args.smoke else tradeoff_config["sample_sizes"],
            repeat_runs=1 if args.smoke else int(tradeoff_config["repeat_runs"]),
            seed=seed,
            metadata=metadata,
            reference_intervals=reference_intervals,
        )
        _write_family(output / "resolution_tradeoff", tradeoff_results, tradeoff_diagnostics)

    inputs = [feature_path, external_feature_path, mapping_path, interval_path, config_path]
    inputs.extend(path for path in [metadata_path, external_metadata_path] if path is not None)
    inputs.extend(
        PROJECT_ROOT / config["knowledge"][key]
        for key in ["edge_path", "resource_manifest"]
    )
    write_json(output / "manifest.json", build_run_manifest(config, inputs))


if __name__ == "__main__":
    main()
