"""User-level sample-one-dimension reporting under local differential privacy.

The dimension choice is sampled independently of the user's values.  Each user
then invokes exactly one epsilon-LDP categorical mechanism.  Consequently the
complete transmitted report is epsilon-LDP; epsilon is not silently multiplied
by the number of semantic dimensions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from semmed_ldp.ldp_mechanisms import grr_estimate, grr_perturb, oue_estimate, oue_perturb


SCHEMA_VERSION = "semmed-ldp-report-v2"


@dataclass(frozen=True)
class ReportingProtocol:
    epsilon: float
    mechanism: str
    domain_size: int = 3
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.domain_size < 2:
            raise ValueError("domain_size must be at least 2")
        if self.mechanism.upper() not in {"GRR", "OUE"}:
            raise ValueError("mechanism must be GRR or OUE")


def _validate_state_matrix(state_matrix: pd.DataFrame, domain_size: int) -> np.ndarray:
    if state_matrix.empty or state_matrix.shape[1] == 0:
        raise ValueError("state_matrix must contain users and dimensions")
    values = state_matrix.to_numpy()
    if pd.isna(values).any():
        raise ValueError("state_matrix contains NA; encode missingness as an explicit state")
    values = values.astype(int)
    if values.min() < 0 or values.max() >= domain_size:
        raise ValueError(f"states must lie in [0, {domain_size - 1}]")
    return values


def sample_one_dimension_reports(
    state_matrix: pd.DataFrame,
    protocol: ReportingProtocol,
    *,
    seed: int,
    dimension_probabilities: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Return one anonymous privatized report per input row.

    ``dimension_probabilities`` must be public and data independent.  Uniform
    sampling is used by default.  No patient identifier or unperturbed state is
    included in the returned table.
    """
    protocol.validate()
    values = _validate_state_matrix(state_matrix, protocol.domain_size)
    n_users, n_dimensions = values.shape
    if dimension_probabilities is None:
        probabilities = np.full(n_dimensions, 1.0 / n_dimensions)
    else:
        probabilities = np.asarray(list(dimension_probabilities), dtype=float)
        if probabilities.shape != (n_dimensions,) or (probabilities <= 0).any():
            raise ValueError("dimension_probabilities must be positive and match the number of dimensions")
        probabilities = probabilities / probabilities.sum()

    rng = np.random.default_rng(seed)
    selected = rng.choice(n_dimensions, size=n_users, p=probabilities)
    selected_states = values[np.arange(n_users), selected]
    mechanism = protocol.mechanism.upper()
    if mechanism == "GRR":
        privatized = grr_perturb(selected_states, protocol.epsilon, protocol.domain_size, seed=seed + 1)
        payloads = [str(int(value)) for value in privatized]
    else:
        privatized = oue_perturb(selected_states, protocol.epsilon, protocol.domain_size, seed=seed + 1)
        payloads = [json.dumps(row.astype(int).tolist(), separators=(",", ":")) for row in privatized]

    return pd.DataFrame(
        {
            "schema_version": protocol.schema_version,
            # The public configuration maps this compact integer identifier to
            # a semantic label. Sending labels would confound communication
            # cost with string length rather than protocol content.
            "sampled_dimension": selected.astype(int),
            "sampling_probability": probabilities[selected],
            "mechanism": mechanism,
            "epsilon": float(protocol.epsilon),
            "privatized_payload": payloads,
        }
    )


def estimate_distributions_from_reports(
    reports: pd.DataFrame,
    dimensions: Iterable[str],
    protocol: ReportingProtocol,
) -> pd.DataFrame:
    """Estimate one categorical distribution for every public dimension."""
    protocol.validate()
    required = {
        "schema_version",
        "sampled_dimension",
        "sampling_probability",
        "mechanism",
        "epsilon",
        "privatized_payload",
    }
    missing = required.difference(reports.columns)
    if missing:
        raise ValueError(f"report table is missing columns: {sorted(missing)}")
    if not reports.empty:
        if set(reports["schema_version"]) != {protocol.schema_version}:
            raise ValueError("report schema version mismatch")
        if set(reports["mechanism"].str.upper()) != {protocol.mechanism.upper()}:
            raise ValueError("report mechanism mismatch")
        if not np.allclose(reports["epsilon"].astype(float), protocol.epsilon):
            raise ValueError("report epsilon mismatch")

    rows: list[dict[str, float | int | str]] = []
    dimension_list = list(dimensions)
    if not reports.empty:
        dimension_ids = pd.to_numeric(reports["sampled_dimension"], errors="coerce")
        if dimension_ids.isna().any() or not dimension_ids.between(0, len(dimension_list) - 1).all():
            raise ValueError("sampled_dimension must be a valid public integer dimension identifier")
    for dimension_id, dimension in enumerate(dimension_list):
        subset = reports.loc[pd.to_numeric(reports["sampled_dimension"]) == dimension_id]
        mechanism = protocol.mechanism.upper()
        if subset.empty:
            estimate = np.full(protocol.domain_size, 1.0 / protocol.domain_size)
        elif mechanism == "GRR":
            payload = subset["privatized_payload"].astype(int).to_numpy()
            estimate = grr_estimate(payload, protocol.epsilon, protocol.domain_size)
        else:
            payload = np.asarray([json.loads(value) for value in subset["privatized_payload"]], dtype=int)
            estimate = oue_estimate(payload, protocol.epsilon, protocol.domain_size)
        for state, frequency in enumerate(estimate):
            rows.append(
                {
                    "semantic_dimension": str(dimension),
                    "state": int(state),
                    "estimated_frequency": float(frequency),
                    "n_reports": int(len(subset)),
                }
            )
    return pd.DataFrame(rows)


def empirical_state_distributions(state_matrix: pd.DataFrame, domain_size: int) -> pd.DataFrame:
    """Return the non-private empirical reference distributions."""
    values = _validate_state_matrix(state_matrix, domain_size)
    rows: list[dict[str, float | int | str]] = []
    for index, dimension in enumerate(state_matrix.columns):
        frequencies = np.bincount(values[:, index], minlength=domain_size).astype(float) / len(values)
        for state, frequency in enumerate(frequencies):
            rows.append(
                {
                    "semantic_dimension": str(dimension),
                    "state": int(state),
                    "true_frequency": float(frequency),
                }
            )
    return pd.DataFrame(rows)


def report_size_bytes(reports: pd.DataFrame) -> float:
    """Estimate mean UTF-8 serialized payload size per anonymous report."""
    if reports.empty:
        return 0.0
    columns = ["schema_version", "sampled_dimension", "mechanism", "epsilon", "privatized_payload"]
    sizes = reports[columns].astype(str).agg("|".join, axis=1).map(lambda value: len(value.encode("utf-8")))
    return float(sizes.mean())
