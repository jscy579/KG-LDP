"""Main SemMed-LDP experiment entrypoint."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semmed_ldp.baselines import build_baseline_state_matrices
from semmed_ldp.data_loader import PICDataLoader, dump_yaml_config, load_yaml_config
from semmed_ldp.frequency_estimation import run_frequency_estimation
from semmed_ldp.patient_matrix import build_patient_feature_matrix
from semmed_ldp.semantic_graph import save_semantic_mapping_summary
from semmed_ldp.semantic_reduction import build_semantic_state_matrix
from semmed_ldp.umls_mapping import build_umls_mappings, load_attribute_config
from semmed_ldp.visualization import generate_basic_plots


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def setup_logging(log_dir: Path) -> None:
    """Configure console and file logging for experiments."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run_semmed_ldp.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )


def save_metric_summary(results: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Save a compact method-level metric summary."""
    if results.empty:
        summary = pd.DataFrame()
    else:
        metric_cols = ["MSE", "MAE", "KL", "JS", "TVD", "Wasserstein"]
        group_cols = ["method", "mechanism", "epsilon"]
        if "semantic_mode" in results.columns:
            group_cols = ["semantic_mode", *group_cols]
        unique_cols = [*group_cols, "semantic_group"]
        unique_metric_rows = results.drop_duplicates(unique_cols)
        summary = unique_metric_rows.groupby(group_cols, as_index=False)[metric_cols].agg(["mean", "std"])
        summary.columns = ["_".join([str(part) for part in col if part]) for col in summary.columns.to_flat_index()]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary


def configured_semantic_modes(config: dict) -> list[str]:
    """Return one or more semantic modes configured for the experiment."""
    modes = config.get("semantic_modes")
    if modes:
        return list(modes)
    return [config.get("semantic_mode", "binary_abnormal")]


def configured_mechanisms(config: dict) -> list[str]:
    """Return configured LDP protocol names."""
    return list(config.get("protocols") or config.get("ldp_mechanisms", ["GRR", "OUE"]))


def build_semantic_group_summary(
    semantic_states: pd.DataFrame,
    attribute_config: pd.DataFrame,
    threshold_log: pd.DataFrame,
    missing_report: pd.DataFrame,
    *,
    semantic_mode: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Create sanity-check rows for semantic group coverage and states."""
    rows = []
    groups = sorted({group for value in attribute_config["semantic_group"].fillna("") for group in str(value).replace("|", ";").split(";") if group.strip()})
    fallback_attrs = set()
    if not threshold_log.empty and "fallback_quantile_threshold" in threshold_log.columns:
        fallback_attrs = set(threshold_log.loc[threshold_log["fallback_quantile_threshold"].astype(bool), "attribute_name"].astype(str))
    missing_lookup = {}
    if not missing_report.empty and {"attribute_name", "missing_rate"}.issubset(missing_report.columns):
        missing_lookup = dict(zip(missing_report["attribute_name"].astype(str), pd.to_numeric(missing_report["missing_rate"], errors="coerce")))

    for group in groups:
        group_attrs = attribute_config[attribute_config["semantic_group"].fillna("").str.contains(group, regex=False)]["attribute_name"].astype(str).tolist()
        fallback_count = sum(attr in fallback_attrs for attr in group_attrs)
        high_missing = [attr for attr in group_attrs if missing_lookup.get(attr, 0) > 0.5]
        if group not in semantic_states.columns:
            logger.warning("Semantic group '%s' is empty after reduction.", group)
            distribution = ""
            num_valid = 0
            single_state = False
        else:
            series = pd.to_numeric(semantic_states[group], errors="coerce").dropna().astype(int)
            num_valid = len(series)
            counts = series.value_counts(normalize=True).sort_index()
            distribution = ";".join(f"{int(state)}:{freq:.6f}" for state, freq in counts.items())
            single_state = series.nunique() <= 1
            if single_state:
                logger.warning("Semantic group '%s' has all patients in one state for mode %s.", group, semantic_mode)
        if high_missing:
            logger.warning("Semantic group '%s' has high-missing attributes: %s", group, high_missing)
        fallback_ratio = fallback_count / len(group_attrs) if group_attrs else 0.0
        if fallback_ratio > 0.5:
            logger.warning("Semantic group '%s' has high fallback threshold ratio %.3f.", group, fallback_ratio)
        rows.append(
            {
                "semantic_mode": semantic_mode,
                "semantic_group": group,
                "num_attributes": len(group_attrs),
                "num_patients_valid": num_valid,
                "state_distribution_true": distribution,
                "fallback_attribute_count": fallback_count,
                "fallback_attribute_ratio": fallback_ratio,
                "empty_group": group not in semantic_states.columns,
                "single_state_group": single_state,
                "high_missing_attributes": "|".join(high_missing),
            }
        )
    return pd.DataFrame(rows)


def main(config_path: str | Path = PROJECT_ROOT / "configs" / "experiment_config.yaml") -> None:
    """Run the first-stage SemMed-LDP workflow."""
    config = load_yaml_config(config_path)
    paths = config.get("paths", {})
    processed_dir = resolve_project_path(paths.get("processed_dir", "data/processed"))
    interim_dir = resolve_project_path(paths.get("interim_dir", "data/interim"))
    tables_dir = resolve_project_path(paths.get("tables_dir", "results/tables"))
    figures_dir = resolve_project_path(paths.get("figures_dir", "results/figures"))
    logs_dir = resolve_project_path(paths.get("logs_dir", "results/logs"))
    setup_logging(logs_dir)

    logger = logging.getLogger("run_semmed_ldp")
    logger.info("Starting SemMed-LDP first-stage workflow.")

    attribute_config_path = resolve_project_path(paths.get("attribute_config", "configs/semmed_attribute_config.csv"))
    pic_paths_path = resolve_project_path(paths.get("pic_paths", "configs/pic_paths.yaml"))
    attribute_config = load_attribute_config(attribute_config_path)
    _, group_to_attributes = build_umls_mappings(attribute_config)
    save_semantic_mapping_summary(attribute_config, interim_dir / "semantic_mapping_summary.csv")

    loader = PICDataLoader(pic_paths_path, project_root=PROJECT_ROOT)
    requested_tables = sorted(set(attribute_config["source_table"].dropna().astype(str)))
    tables = loader.read_tables(requested_tables, required=False)
    loader.table_metadata().to_csv(interim_dir / "pic_table_metadata.csv", index=False)

    quality = config.get("quality", {})
    patient_matrix, missing_report = build_patient_feature_matrix(
        tables,
        attribute_config,
        loader.tables_config,
        entity_id=config.get("entity_id", "patient_id"),
        aggregation=config.get("aggregation", "median"),
        max_missing_rate=float(quality.get("max_missing_rate", 1.0)),
        output_path=processed_dir / "patient_feature_matrix.csv",
        missing_report_path=tables_dir / "attribute_missingness_report.csv",
    )
    logger.info("Patient matrix shape: %s; missing report rows: %d.", patient_matrix.shape, len(missing_report))

    all_results = []
    all_summaries = []
    last_semantic_states = pd.DataFrame()
    for semantic_mode in configured_semantic_modes(config):
        semantic_states, threshold_log = build_semantic_state_matrix(
            patient_matrix,
            attribute_config,
            mode=semantic_mode,
            min_group_attributes=int(quality.get("min_group_attributes", 1)),
            output_path=processed_dir / f"semantic_state_matrix_{semantic_mode}.csv",
            log_path=logs_dir / f"semantic_threshold_fallback_log_{semantic_mode}.csv",
        )
        last_semantic_states = semantic_states
        logger.info("Semantic state matrix for %s shape: %s; threshold log rows: %d.", semantic_mode, semantic_states.shape, len(threshold_log))
        all_summaries.append(
            build_semantic_group_summary(
                semantic_states,
                attribute_config,
                threshold_log,
                missing_report,
                semantic_mode=semantic_mode,
                logger=logger,
            )
        )

        state_matrices = {"UMLS-Grouped-LDP": semantic_states}
        if config.get("baselines", {}).get("enabled", True):
            state_matrices = build_baseline_state_matrices(
                patient_matrix,
                attribute_config,
                mode=semantic_mode,
                seed=int(config.get("baselines", {}).get("random_group_seed", config.get("seed", 2026))),
                min_group_attributes=int(quality.get("min_group_attributes", 1)),
            )
            state_matrices["UMLS-Grouped-LDP"] = semantic_states

        domain_size = 2 if semantic_mode == "binary_abnormal" else 4
        for method, matrix in state_matrices.items():
            result = run_frequency_estimation(
                matrix,
                epsilon_list=[float(value) for value in config.get("epsilon_list", [0.5, 1.0, 2.0])],
                repeat_runs=int(config.get("repeat_runs", 5)),
                domain_size=domain_size,
                mechanisms=configured_mechanisms(config),
                seed=int(config.get("seed", 2026)),
                method_name=method,
                min_group_samples=int(quality.get("min_group_samples", 1)),
            )
            if not result.empty:
                result.insert(0, "semantic_mode", semantic_mode)
            all_results.append(result)

    if not last_semantic_states.empty:
        last_semantic_states.to_csv(processed_dir / "semantic_state_matrix.csv")
    semantic_group_summary = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    semantic_group_summary.to_csv(tables_dir / "semantic_group_summary.csv", index=False)
    logger.info("Saved semantic group sanity summary with %d rows.", len(semantic_group_summary))

    results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    results_path = tables_dir / "ldp_frequency_estimation_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    logger.info("Saved combined LDP results to %s with %d rows.", results_path, len(results))

    save_metric_summary(results, tables_dir / "metric_summary_mean_std.csv")
    if not results.empty:
        generate_basic_plots(results, group_to_attributes, figures_dir)
    else:
        logger.warning("No result rows were generated. Check PIC data paths and semmed_attribute_config.csv.")

    run_config_snapshot = logs_dir / "experiment_config_snapshot.yaml"
    dump_yaml_config(config, run_config_snapshot)
    logger.info("SemMed-LDP workflow finished.")


if __name__ == "__main__":
    cli_config = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "configs" / "experiment_config.yaml"
    main(cli_config)
