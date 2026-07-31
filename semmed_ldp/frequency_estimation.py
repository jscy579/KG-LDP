"""LDP frequency estimation over semantic group state variables."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from semmed_ldp.ldp_mechanisms import privatize_and_estimate, validate_categorical_values
from semmed_ldp.metrics import compute_all_metrics

LOGGER = logging.getLogger(__name__)


def true_frequency(values: np.ndarray, domain_size: int) -> np.ndarray:
    """Compute the empirical categorical distribution."""
    values = validate_categorical_values(values, domain_size)
    if values.size == 0:
        return np.ones(domain_size) / domain_size
    return np.bincount(values, minlength=domain_size).astype(float) / values.size


def estimate_series_distribution(
    values: pd.Series,
    *,
    epsilon: float,
    domain_size: int,
    mechanism: str,
    repeat_runs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Estimate one categorical state distribution over repeated LDP runs."""
    clean_values = pd.to_numeric(values, errors="coerce").dropna().astype(int).to_numpy()
    clean_values = clean_values[(clean_values >= 0) & (clean_values < domain_size)]
    true_dist = true_frequency(clean_values, domain_size)
    estimates = []
    for run in range(repeat_runs):
        run_seed = seed + run
        estimates.append(privatize_and_estimate(clean_values, epsilon, domain_size, mechanism, seed=run_seed))
    stacked = np.vstack(estimates) if estimates else np.empty((0, domain_size))
    return true_dist, stacked.mean(axis=0), stacked.std(axis=0), estimates


def run_frequency_estimation(
    state_matrix: pd.DataFrame,
    *,
    epsilon_list: list[float],
    repeat_runs: int = 5,
    domain_size: int = 2,
    mechanisms: list[str] | None = None,
    seed: int = 2026,
    method_name: str = "UMLS-Grouped-LDP",
    min_group_samples: int = 1,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run LDP frequency estimation for each semantic group and epsilon."""
    mechanisms = mechanisms or ["GRR", "OUE"]
    rows = []
    if state_matrix.empty:
        LOGGER.warning("State matrix for %s is empty; no frequency estimates generated.", method_name)
        return pd.DataFrame()

    for group in state_matrix.columns:
        values = pd.to_numeric(state_matrix[group], errors="coerce").dropna().astype(int)
        values = values[(values >= 0) & (values < domain_size)]
        if len(values) < min_group_samples:
            LOGGER.warning("Skipping group '%s' for %s because it has only %d valid samples.", group, method_name, len(values))
            continue
        for mechanism in mechanisms:
            for epsilon in epsilon_list:
                true_dist, mean_est, std_est, _ = estimate_series_distribution(
                    values,
                    epsilon=float(epsilon),
                    domain_size=domain_size,
                    mechanism=mechanism,
                    repeat_runs=repeat_runs,
                    seed=seed + abs(hash((method_name, group, mechanism, float(epsilon)))) % 1_000_000,
                )
                metric_values = compute_all_metrics(true_dist, mean_est)
                for state in range(domain_size):
                    rows.append(
                        {
                            "method": method_name,
                            "semantic_group": group,
                            "mechanism": mechanism.upper(),
                            "epsilon": float(epsilon),
                            "state": state,
                            "n": len(values),
                            "true_frequency": true_dist[state],
                            "estimated_mean": mean_est[state],
                            "estimated_std": std_est[state],
                            "estimated_mean_std": f"{mean_est[state]:.6f} +/- {std_est[state]:.6f}",
                            **metric_values,
                        }
                    )

    result = pd.DataFrame(rows)
    LOGGER.info("Generated %d frequency-estimation rows for %s.", len(result), method_name)
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        LOGGER.info("Saved frequency estimation results to %s.", output)
    return result
