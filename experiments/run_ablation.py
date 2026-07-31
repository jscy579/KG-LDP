"""Minimal ablation entrypoint for semantic state modes."""

from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_semmed_ldp import main as run_main
from semmed_ldp.data_loader import dump_yaml_config, load_yaml_config


def main(config_path: str | Path = PROJECT_ROOT / "configs" / "experiment_config.yaml") -> None:
    """Run first-stage ablations for binary and severity semantic states."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    base_config = load_yaml_config(config_path)
    for mode in ["binary_abnormal", "severity_level"]:
        config = copy.deepcopy(base_config)
        config["semantic_mode"] = mode
        config["paths"] = dict(config.get("paths", {}))
        config["paths"]["tables_dir"] = f"results/tables/ablation_{mode}"
        config["paths"]["figures_dir"] = f"results/figures/ablation_{mode}"
        config["paths"]["logs_dir"] = f"results/logs/ablation_{mode}"
        temp_config = PROJECT_ROOT / "data" / "interim" / f"experiment_config_{mode}.yaml"
        dump_yaml_config(config, temp_config)
        run_main(temp_config)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "configs" / "experiment_config.yaml")
