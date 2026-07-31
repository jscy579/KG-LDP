"""Multidimensional categorical frequency protocols under user-level LDP.

The module implements the three reporting strategies used in the revised
SemMed-LDP experiments: privacy-budget splitting (Spl), sample one dimension
(Smp), and random sampling plus fake data (RS+FD).  All strategies estimate the
same per-dimension categorical distributions and use one total user-level
privacy budget ``epsilon``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from semmed_ldp.ldp_mechanisms import grr_estimate, grr_perturb, oue_estimate, oue_perturb


@dataclass(frozen=True)
class MultidimensionalProtocol:
    """Configuration for a complete multidimensional user report."""

    epsilon: float
    mechanism: str
    strategy: str
    domain_size: int = 3

    def validate(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.domain_size < 2:
            raise ValueError("domain_size must be at least 2")
        if self.mechanism.upper() not in {"GRR", "OUE"}:
            raise ValueError("mechanism must be GRR or OUE")
        if self.strategy.upper() not in {"SMP", "SPL", "RS+FD"}:
            raise ValueError("strategy must be Smp, Spl, or RS+FD")


def _state_values(state_matrix: pd.DataFrame, domain_size: int) -> np.ndarray:
    if state_matrix.empty or state_matrix.shape[1] == 0:
        raise ValueError("state_matrix must contain users and dimensions")
    values = state_matrix.to_numpy()
    if pd.isna(values).any():
        raise ValueError("state_matrix contains NA; missingness must be an explicit state")
    values = values.astype(int)
    if values.min() < 0 or values.max() >= domain_size:
        raise ValueError(f"states must lie in [0, {domain_size - 1}]")
    return values


def _simplex_projection(raw: np.ndarray) -> np.ndarray:
    projected = np.clip(np.asarray(raw, dtype=float), 0, None)
    total = projected.sum()
    if total == 0:
        return np.full(len(projected), 1.0 / len(projected))
    return projected / total


def amplified_epsilon(epsilon: float, n_dimensions: int) -> float:
    """Return the internal RS+FD budget after amplification by sampling."""
    if epsilon <= 0 or n_dimensions < 1:
        raise ValueError("epsilon and n_dimensions must be positive")
    return float(np.log(n_dimensions * np.expm1(epsilon) + 1.0))


def _rows_from_matrix(
    estimates: np.ndarray,
    dimensions: Iterable[str],
    n_reports: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for dimension_id, dimension in enumerate(dimensions):
        for state, frequency in enumerate(estimates[dimension_id]):
            rows.append(
                {
                    "semantic_dimension": str(dimension),
                    "state": int(state),
                    "estimated_frequency": float(frequency),
                    "n_reports": int(n_reports),
                }
            )
    return pd.DataFrame(rows)


def _serialized_size(
    strategy: str,
    mechanism: str,
    epsilon: float,
    payloads: list[object],
) -> float:
    prefix = f"semmed-ldp-report-v3|{strategy}|{mechanism}|{epsilon}|"
    sizes = [
        len((prefix + json.dumps(payload, separators=(",", ":"))).encode("utf-8"))
        for payload in payloads
    ]
    return float(np.mean(sizes)) if sizes else 0.0


def run_multidimensional_protocol(
    state_matrix: pd.DataFrame,
    protocol: MultidimensionalProtocol,
    *,
    seed: int,
    measure_communication: bool = True,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Privatize one user matrix and estimate all categorical marginals.

    RS+FD with OUE uses the zero-vector fake-data variant (RS+FD[OUE-z])
    described by Arcolezi et al.  The returned estimates are clipped and
    normalized only after applying the corresponding unbiased estimator.
    """
    protocol.validate()
    values = _state_values(state_matrix, protocol.domain_size)
    n_users, n_dimensions = values.shape
    mechanism = protocol.mechanism.upper()
    strategy = protocol.strategy.upper()
    rng = np.random.default_rng(seed)

    if strategy == "SMP":
        selected = rng.integers(0, n_dimensions, size=n_users)
        selected_values = values[np.arange(n_users), selected]
        if mechanism == "GRR":
            privatized = grr_perturb(
                selected_values, protocol.epsilon, protocol.domain_size, seed=seed + 1
            )
        else:
            privatized = oue_perturb(
                selected_values, protocol.epsilon, protocol.domain_size, seed=seed + 1
            )
        estimates = np.empty((n_dimensions, protocol.domain_size), dtype=float)
        for dimension_id in range(n_dimensions):
            subset = privatized[selected == dimension_id]
            if len(subset) == 0:
                estimates[dimension_id] = np.full(
                    protocol.domain_size, 1.0 / protocol.domain_size
                )
            elif mechanism == "GRR":
                estimates[dimension_id] = grr_estimate(
                    subset, protocol.epsilon, protocol.domain_size
                )
            else:
                estimates[dimension_id] = oue_estimate(
                    subset, protocol.epsilon, protocol.domain_size
                )
        if measure_communication:
            if mechanism == "GRR":
                payloads = [[int(selected[i]), int(privatized[i])] for i in range(n_users)]
            else:
                payloads = [
                    [int(selected[i]), privatized[i].astype(int).tolist()]
                    for i in range(n_users)
                ]
            mean_bytes = _serialized_size("Smp", mechanism, protocol.epsilon, payloads)
        else:
            mean_bytes = float("nan")
        return _rows_from_matrix(estimates, state_matrix.columns, n_users), {
            "strategy": "Smp",
            "internal_epsilon": float(protocol.epsilon),
            "uploads_per_user": 1.0,
            "reported_dimensions_per_user": 1.0,
            "mean_report_bytes": mean_bytes,
        }

    if strategy == "SPL":
        local_epsilon = protocol.epsilon / n_dimensions
        estimates = np.empty((n_dimensions, protocol.domain_size), dtype=float)
        payload_columns: list[np.ndarray] = []
        for dimension_id in range(n_dimensions):
            local_seed = seed + 10_000 + dimension_id
            if mechanism == "GRR":
                privatized = grr_perturb(
                    values[:, dimension_id], local_epsilon, protocol.domain_size, seed=local_seed
                )
                estimates[dimension_id] = grr_estimate(privatized, local_epsilon, protocol.domain_size)
            else:
                privatized = oue_perturb(
                    values[:, dimension_id], local_epsilon, protocol.domain_size, seed=local_seed
                )
                estimates[dimension_id] = oue_estimate(privatized, local_epsilon, protocol.domain_size)
            payload_columns.append(privatized)
        if measure_communication:
            if mechanism == "GRR":
                payloads = np.column_stack(payload_columns).astype(int).tolist()
            else:
                payloads = [
                    [payload_columns[j][i].astype(int).tolist() for j in range(n_dimensions)]
                    for i in range(n_users)
                ]
            mean_bytes = _serialized_size("Spl", mechanism, protocol.epsilon, payloads)
        else:
            mean_bytes = float("nan")
        return _rows_from_matrix(estimates, state_matrix.columns, n_users), {
            "strategy": "Spl",
            "internal_epsilon": float(local_epsilon),
            "uploads_per_user": 1.0,
            "reported_dimensions_per_user": float(n_dimensions),
            "mean_report_bytes": mean_bytes,
        }

    internal_epsilon = amplified_epsilon(protocol.epsilon, n_dimensions)
    selected = rng.integers(0, n_dimensions, size=n_users)
    estimates = np.empty((n_dimensions, protocol.domain_size), dtype=float)
    if mechanism == "GRR":
        privatized = rng.integers(0, protocol.domain_size, size=(n_users, n_dimensions))
        selected_values = values[np.arange(n_users), selected]
        true_reports = grr_perturb(
            selected_values, internal_epsilon, protocol.domain_size, seed=seed + 1
        )
        privatized[np.arange(n_users), selected] = true_reports
        exp_eps = np.exp(internal_epsilon)
        p = exp_eps / (exp_eps + protocol.domain_size - 1)
        q = 1.0 / (exp_eps + protocol.domain_size - 1)
        for dimension_id in range(n_dimensions):
            counts = np.bincount(
                privatized[:, dimension_id], minlength=protocol.domain_size
            ).astype(float)
            raw = (
                counts * n_dimensions * protocol.domain_size
                - n_users * (n_dimensions - 1 + q * protocol.domain_size)
            ) / (n_users * protocol.domain_size * (p - q))
            estimates[dimension_id] = _simplex_projection(raw)
        payloads = privatized.astype(int).tolist() if measure_communication else []
    else:
        p = 0.5
        q = 1.0 / (np.exp(internal_epsilon) + 1.0)
        probabilities = np.full(
            (n_users, n_dimensions, protocol.domain_size), q, dtype=float
        )
        probabilities[np.arange(n_users), selected, selected_values := values[np.arange(n_users), selected]] = p
        privatized = (rng.random(probabilities.shape) < probabilities).astype(int)
        for dimension_id in range(n_dimensions):
            counts = privatized[:, dimension_id, :].sum(axis=0).astype(float)
            raw = n_dimensions * (counts - n_users * q) / (n_users * (p - q))
            estimates[dimension_id] = _simplex_projection(raw)
        payloads = privatized.astype(int).tolist() if measure_communication else []

    mean_bytes = (
        _serialized_size("RS+FD", mechanism, protocol.epsilon, payloads)
        if measure_communication
        else float("nan")
    )

    return _rows_from_matrix(estimates, state_matrix.columns, n_users), {
        "strategy": "RS+FD",
        "internal_epsilon": float(internal_epsilon),
        "uploads_per_user": 1.0,
        "reported_dimensions_per_user": float(n_dimensions),
        "mean_report_bytes": mean_bytes,
    }
