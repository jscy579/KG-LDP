"""CLI for the prespecified formal SemMed-LDP ablation family."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semmed_ldp.formal_experiment import (
    build_run_manifest,
    run_formal_ablation_experiment,
    summarize_with_bootstrap_ci,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--entity-id", required=True)
    parser.add_argument("--mapping", default="configs/formal_attribute_mapping.csv")
    parser.add_argument("--reference-intervals", default="configs/reference_intervals.csv")
    parser.add_argument("--config", default="configs/formal_experiment.yaml")
    parser.add_argument("--output-dir", default="results/formal_v2")
    args = parser.parse_args()

    paths = {name: ROOT / value for name, value in {
        "features": args.features,
        "metadata": args.metadata,
        "mapping": args.mapping,
        "intervals": args.reference_intervals,
        "config": args.config,
    }.items()}
    config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    features = pd.read_csv(paths["features"]).set_index(args.entity_id)
    metadata = pd.read_csv(paths["metadata"]).set_index(args.entity_id)
    mapping = pd.read_csv(paths["mapping"])
    intervals = pd.read_csv(paths["intervals"])
    results, diagnostics, support = run_formal_ablation_experiment(
        features,
        mapping,
        dataset_id=args.dataset_id,
        epsilon_list=config["privacy"]["epsilon_list"],
        mechanisms=config["privacy"]["mechanisms"],
        sample_sizes=config["experiments"]["sample_sizes"],
        repeat_runs=int(config["privacy"]["repeat_runs"]),
        seed=int(config["experiments"]["seed"]),
        metadata=metadata,
        reference_intervals=intervals,
    )
    output = ROOT / args.output_dir / args.dataset_id / "ablations"
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "run_metrics.csv", index=False)
    diagnostics.to_csv(output / "run_diagnostics.csv", index=False)
    support.to_csv(output / "semantic_state_support.csv", index=False)
    summarize_with_bootstrap_ci(results).to_csv(output / "summary_bootstrap_ci.csv", index=False)
    manifest_inputs = list(paths.values()) + [
        ROOT / config["knowledge"]["edge_path"],
        ROOT / config["knowledge"]["resource_manifest"],
        ROOT / config["primary_scope"]["scope_path"],
    ]
    write_json(output / "manifest.json", build_run_manifest(config, manifest_inputs))


if __name__ == "__main__":
    main()
