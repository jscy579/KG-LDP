"""Run baseline grouping methods from an existing patient feature matrix."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semmed_ldp.baselines import build_baseline_state_matrices
from semmed_ldp.data_loader import load_yaml_config
from semmed_ldp.frequency_estimation import run_frequency_estimation
from semmed_ldp.umls_mapping import load_attribute_config


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(config_path: str | Path = PROJECT_ROOT / "configs" / "experiment_config.yaml") -> None:
    """Run only baseline LDP estimation after patient matrix construction."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    config = load_yaml_config(config_path)
    paths = config.get("paths", {})
    patient_matrix_path = resolve_project_path(paths.get("processed_dir", "data/processed")) / "patient_feature_matrix.csv"
    attribute_config = load_attribute_config(resolve_project_path(paths.get("attribute_config", "configs/semmed_attribute_config.csv")))
    patient_matrix = pd.read_csv(patient_matrix_path, index_col=0)
    quality = config.get("quality", {})
    mode = config.get("semantic_mode", "binary_abnormal")
    domain_size = 2 if mode == "binary_abnormal" else 4

    state_matrices = build_baseline_state_matrices(
        patient_matrix,
        attribute_config,
        mode=mode,
        seed=int(config.get("seed", 2026)),
        min_group_attributes=int(quality.get("min_group_attributes", 1)),
    )

    rows = []
    for method, matrix in state_matrices.items():
        if method == "UMLS-Grouped-LDP":
            continue
        rows.append(
            run_frequency_estimation(
                matrix,
                epsilon_list=[float(value) for value in config.get("epsilon_list", [0.5, 1.0, 2.0])],
                repeat_runs=int(config.get("repeat_runs", 5)),
                domain_size=domain_size,
                mechanisms=list(config.get("ldp_mechanisms", ["GRR", "OUE"])),
                seed=int(config.get("seed", 2026)),
                method_name=method,
                min_group_samples=int(quality.get("min_group_samples", 1)),
            )
        )
    output = resolve_project_path(paths.get("tables_dir", "results/tables")) / "baseline_frequency_estimation_results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(output, index=False)
    logging.info("Saved baseline results to %s.", output)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "configs" / "experiment_config.yaml")
