"""LoPub-derived Bloom-EM baseline for common semantic-query evaluation.

This module implements the Bloom-filter local transformation and EM-based
joint-distribution estimator described by Ren et al. (IEEE TIFS 2018, DOI:
10.1109/TIFS.2018.2812146).  For the paper comparison, these components are
embedded in the same sample-one-group (Smp) protocol as SemMed-LDP: a user
reports raw Bloom encodings only for one data-independently selected group.
This is a component baseline, not an implementation of LoPub's Lasso or
synthetic-data generation stages.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from semmed_ldp.formal_semantic_reduction import (
    ABNORMALITY_PRESENT_STATE,
    MISSING_STATE,
    NO_ABNORMALITY_STATE,
    attribute_abnormal_indicators,
    validate_formal_mapping,
)
from semmed_ldp.metrics import compute_all_metrics


RAW_DOMAIN_SIZE = 3


@dataclass(frozen=True)
class LoPubParameters:
    """Reproducible parameters for the LoPub-derived Bloom-RR and EM stages."""

    bloom_bits: int = 64
    hash_functions: int = 5
    convergence_gap: float = 0.001
    min_iterations: int = 25
    max_iterations: int = 500

    def validate(self) -> None:
        if self.bloom_bits < 2:
            raise ValueError("bloom_bits must be at least 2")
        if self.hash_functions < 1:
            raise ValueError("hash_functions must be positive")
        if not 0 < self.convergence_gap < 1:
            raise ValueError("convergence_gap must lie in (0, 1)")
        if self.min_iterations < 1:
            raise ValueError("min_iterations must be positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.min_iterations > self.max_iterations:
            raise ValueError("min_iterations cannot exceed max_iterations")


def flipping_probability(epsilon: float, *, dimensions: int, hash_functions: int) -> float:
    """Invert epsilon = 2*d*h*log((2-f)/f) from the LoPub privacy analysis."""
    if epsilon <= 0 or dimensions < 1 or hash_functions < 1:
        raise ValueError("epsilon, dimensions, and hash_functions must be positive")
    exponent = epsilon / (2.0 * dimensions * hash_functions)
    return float(2.0 / (1.0 + np.exp(exponent)))


def privacy_budget_from_flipping(
    f: float, *, dimensions: int, hash_functions: int
) -> float:
    """Return the sequentially composed record-level epsilon stated by LoPub."""
    if not 0 < f < 1 or dimensions < 1 or hash_functions < 1:
        raise ValueError("f must lie in (0, 1); dimensions and hashes must be positive")
    return float(2.0 * dimensions * hash_functions * np.log((2.0 - f) / f))


def bloom_codebook(
    attribute_name: str,
    *,
    domain_size: int = RAW_DOMAIN_SIZE,
    bloom_bits: int = 64,
    hash_functions: int = 5,
) -> np.ndarray:
    """Build a deterministic domain-by-bit Bloom-filter codebook."""
    if domain_size < 2 or bloom_bits < 2 or hash_functions < 1:
        raise ValueError("invalid codebook dimensions")
    codebook = np.zeros((domain_size, bloom_bits), dtype=np.uint8)
    for value in range(domain_size):
        for hash_index in range(hash_functions):
            payload = f"SemMed-LDP|LoPub|{attribute_name}|{value}|{hash_index}".encode("utf-8")
            position = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % bloom_bits
            codebook[value, position] = 1
    return codebook


def build_raw_attribute_states(
    features: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    metadata: pd.DataFrame | None = None,
    reference_intervals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Encode each raw attribute as missing, observed-normal, or abnormal."""
    mapping = validate_formal_mapping(mapping)
    indicators = attribute_abnormal_indicators(
        features,
        mapping,
        metadata=metadata,
        reference_intervals=reference_intervals,
    )
    states = pd.DataFrame(index=indicators.index)
    for attribute in mapping["attribute_name"]:
        values = indicators[attribute]
        encoded = pd.Series(MISSING_STATE, index=values.index, dtype="int8")
        encoded.loc[values.notna()] = NO_ABNORMALITY_STATE
        encoded.loc[values.eq(1)] = ABNORMALITY_PRESENT_STATE
        states[attribute] = encoded
    return states


def perturb_raw_states(
    raw_states: pd.DataFrame,
    *,
    epsilon: float,
    parameters: LoPubParameters,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    """Apply the LoPub Bloom-filter bit randomization to a complete record."""
    parameters.validate()
    values = raw_states.to_numpy(dtype=int)
    if values.size == 0 or values.min() < 0 or values.max() >= RAW_DOMAIN_SIZE:
        raise ValueError("raw states must be a nonempty matrix over {0,1,2}")
    dimensions = raw_states.shape[1]
    f = flipping_probability(
        epsilon,
        dimensions=dimensions,
        hash_functions=parameters.hash_functions,
    )
    random_one_probability = f / 2.0
    rng = np.random.default_rng(seed)
    reports: dict[str, np.ndarray] = {}
    codebooks: dict[str, np.ndarray] = {}
    for column_index, attribute in enumerate(raw_states.columns):
        codebook = bloom_codebook(
            str(attribute),
            bloom_bits=parameters.bloom_bits,
            hash_functions=parameters.hash_functions,
        )
        true_bits = codebook[values[:, column_index]]
        probability_one = np.where(
            true_bits == 1,
            1.0 - random_one_probability,
            random_one_probability,
        )
        reports[str(attribute)] = (rng.random(true_bits.shape) < probability_one).astype(
            np.uint8
        )
        codebooks[str(attribute)] = codebook
    return reports, codebooks, f


def _attribute_log_likelihoods(
    reports: dict[str, np.ndarray],
    codebooks: dict[str, np.ndarray],
    *,
    f: float,
) -> dict[str, np.ndarray]:
    q = float(np.clip(f / 2.0, 1e-12, 1.0 - 1e-12))
    log_match = np.log1p(-q)
    log_mismatch = np.log(q)
    likelihoods: dict[str, np.ndarray] = {}
    for attribute, observed in reports.items():
        codebook = codebooks[attribute]
        observed_int = observed.astype(np.int16, copy=False)
        codebook_int = codebook.astype(np.int16, copy=False)
        intersection = observed_int @ codebook_int.T
        mismatches = (
            observed_int.sum(axis=1, keepdims=True)
            + codebook_int.sum(axis=1)[None, :]
            - 2 * intersection
        )
        matches = observed.shape[1] - mismatches
        likelihoods[attribute] = matches * log_match + mismatches * log_mismatch
    return likelihoods


def em_joint_distribution(
    attributes: list[str],
    attribute_log_likelihoods: dict[str, np.ndarray],
    *,
    convergence_gap: float = 0.001,
    min_iterations: int = 1,
    max_iterations: int = 500,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Estimate a joint distribution by the LoPub EM-JD update."""
    if not attributes:
        raise ValueError("at least one attribute is required")
    candidates = np.asarray(
        list(itertools.product(range(RAW_DOMAIN_SIZE), repeat=len(attributes))),
        dtype=np.int8,
    )
    n_users = attribute_log_likelihoods[attributes[0]].shape[0]
    joint_log_likelihood = np.zeros((n_users, len(candidates)), dtype=float)
    for attribute_index, attribute in enumerate(attributes):
        local = attribute_log_likelihoods[attribute]
        if local.shape != (n_users, RAW_DOMAIN_SIZE):
            raise ValueError(f"unexpected likelihood shape for {attribute}")
        joint_log_likelihood += local[:, candidates[:, attribute_index]]

    probability = np.full(len(candidates), 1.0 / len(candidates), dtype=float)
    converged = False
    for iteration in range(1, max_iterations + 1):
        log_weight = joint_log_likelihood + np.log(np.clip(probability, 1e-300, None))
        posterior = np.exp(log_weight - logsumexp(log_weight, axis=1, keepdims=True))
        updated = posterior.mean(axis=0)
        updated /= updated.sum()
        if (
            iteration >= min_iterations
            and np.max(np.abs(updated - probability)) <= convergence_gap
        ):
            probability = updated
            converged = True
            break
        probability = updated
    return candidates, probability, iteration, converged


def sample_group_indices(n_users: int, n_groups: int, *, seed: int) -> np.ndarray:
    """Draw the public, data-independent Smp group identifier for every user."""
    if n_users < 1 or n_groups < 1:
        raise ValueError("n_users and n_groups must be positive")
    return np.random.default_rng(seed).integers(0, n_groups, size=n_users)


def aggregate_joint_to_semantic(
    candidates: np.ndarray, probability: np.ndarray
) -> np.ndarray:
    """Post-process a raw-state joint distribution into the common three-state query."""
    if candidates.ndim != 2 or probability.shape != (len(candidates),):
        raise ValueError("candidate and probability shapes do not align")
    semantic = np.full(len(candidates), NO_ABNORMALITY_STATE, dtype=np.int8)
    semantic[np.all(candidates == MISSING_STATE, axis=1)] = MISSING_STATE
    semantic[np.any(candidates == ABNORMALITY_PRESENT_STATE, axis=1)] = (
        ABNORMALITY_PRESENT_STATE
    )
    result = np.bincount(semantic, weights=probability, minlength=RAW_DOMAIN_SIZE).astype(
        float
    )
    return result / result.sum()


def semantic_truth_from_raw(
    raw_states: pd.DataFrame, mapping: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Return exact common-query distributions for the ontology groups."""
    truth: dict[str, np.ndarray] = {}
    for anchor, rows in mapping.groupby("semantic_anchor", sort=True):
        values = raw_states[rows["attribute_name"].tolist()].to_numpy(dtype=int)
        semantic = np.full(len(values), NO_ABNORMALITY_STATE, dtype=np.int8)
        semantic[np.all(values == MISSING_STATE, axis=1)] = MISSING_STATE
        semantic[np.any(values == ABNORMALITY_PRESENT_STATE, axis=1)] = (
            ABNORMALITY_PRESENT_STATE
        )
        truth[str(anchor)] = np.bincount(
            semantic, minlength=RAW_DOMAIN_SIZE
        ).astype(float) / len(semantic)
    return truth


def evaluate_lopub_em_run(
    raw_states: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    epsilon: float,
    parameters: LoPubParameters,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame, dict[str, float | int | bool]]:
    """Run one LoPub EM-JD replicate and evaluate the common four-group workload."""
    mapping = validate_formal_mapping(mapping)
    reports, codebooks, f = perturb_raw_states(
        raw_states,
        epsilon=epsilon,
        parameters=parameters,
        seed=seed,
    )
    log_likelihoods = _attribute_log_likelihoods(reports, codebooks, f=f)
    truth = semantic_truth_from_raw(raw_states, mapping)
    metric_rows: list[dict[str, float]] = []
    estimate_rows: list[dict[str, float | int | str]] = []
    iterations: list[int] = []
    convergence: list[bool] = []
    for anchor, rows in mapping.groupby("semantic_anchor", sort=True):
        attributes = rows["attribute_name"].astype(str).tolist()
        candidates, probability, n_iterations, converged = em_joint_distribution(
            attributes,
            log_likelihoods,
            convergence_gap=parameters.convergence_gap,
            max_iterations=parameters.max_iterations,
        )
        estimate = aggregate_joint_to_semantic(candidates, probability)
        target = truth[str(anchor)]
        metric_rows.append(compute_all_metrics(target, estimate))
        iterations.append(n_iterations)
        convergence.append(converged)
        for state in range(RAW_DOMAIN_SIZE):
            estimate_rows.append(
                {
                    "semantic_dimension": str(anchor),
                    "state": state,
                    "true_frequency": float(target[state]),
                    "estimated_frequency": float(estimate[state]),
                }
            )
    metrics = {
        key: float(np.mean([row[key] for row in metric_rows]))
        for key in metric_rows[0]
    }
    diagnostics: dict[str, float | int | bool] = {
        "flipping_probability": f,
        "record_epsilon_reconstructed": privacy_budget_from_flipping(
            f,
            dimensions=raw_states.shape[1],
            hash_functions=parameters.hash_functions,
        ),
        "client_payload_bits": raw_states.shape[1] * parameters.bloom_bits,
        "mean_em_iterations": float(np.mean(iterations)),
        "max_em_iterations": int(max(iterations)),
        "all_groups_converged": bool(all(convergence)),
    }
    return metrics, pd.DataFrame(estimate_rows), diagnostics


def evaluate_lopub_smp_em_run(
    raw_states: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    epsilon: float,
    parameters: LoPubParameters,
    seed: int,
    selected_groups: np.ndarray | None = None,
) -> tuple[dict[str, float], pd.DataFrame, dict[str, object]]:
    """Evaluate a fair Smp + group-local Bloom-EM component baseline.

    Group selection is independent of the record.  Conditional on the selected
    group ``j``, all ``k_j`` raw attributes in that group are Bloom encoded and
    jointly randomized using

    ``epsilon = 2 * k_j * h * log((2-f_j)/f_j)``.

    Hence the complete transmitted report ``(j, BloomReport_j)`` is
    ``epsilon``-LDP at the user-record level.  No budget is composed across
    groups because the other groups are not transmitted.
    """
    parameters.validate()
    mapping = validate_formal_mapping(mapping)
    grouped = list(mapping.groupby("semantic_anchor", sort=True))
    n_users = len(raw_states)
    if selected_groups is None:
        selected_groups = sample_group_indices(n_users, len(grouped), seed=seed)
    selected_groups = np.asarray(selected_groups, dtype=int)
    if selected_groups.shape != (n_users,):
        raise ValueError("selected_groups must contain one group identifier per user")
    if selected_groups.min() < 0 or selected_groups.max() >= len(grouped):
        raise ValueError("selected_groups contains an invalid group identifier")

    truth = semantic_truth_from_raw(raw_states, mapping)
    metric_rows: list[dict[str, float]] = []
    estimate_rows: list[dict[str, float | int | str]] = []
    group_diagnostics: list[dict[str, object]] = []
    for group_id, (anchor, rows) in enumerate(grouped):
        mask = selected_groups == group_id
        attributes = rows["attribute_name"].astype(str).tolist()
        local_states = raw_states.loc[mask, attributes]
        if local_states.empty:
            estimate = np.full(RAW_DOMAIN_SIZE, 1.0 / RAW_DOMAIN_SIZE)
            f = flipping_probability(
                epsilon,
                dimensions=len(attributes),
                hash_functions=parameters.hash_functions,
            )
            n_iterations = 0
            converged = False
        else:
            reports, codebooks, f = perturb_raw_states(
                local_states,
                epsilon=epsilon,
                parameters=parameters,
                seed=seed + 10_000 * (group_id + 1),
            )
            likelihoods = _attribute_log_likelihoods(reports, codebooks, f=f)
            candidates, probability, n_iterations, converged = em_joint_distribution(
                attributes,
                likelihoods,
                convergence_gap=parameters.convergence_gap,
                min_iterations=parameters.min_iterations,
                max_iterations=parameters.max_iterations,
            )
            estimate = aggregate_joint_to_semantic(candidates, probability)
        target = truth[str(anchor)]
        metric_rows.append(compute_all_metrics(target, estimate))
        for state in range(RAW_DOMAIN_SIZE):
            estimate_rows.append(
                {
                    "semantic_dimension": str(anchor),
                    "state": state,
                    "true_frequency": float(target[state]),
                    "estimated_frequency": float(estimate[state]),
                    "n_reports": int(mask.sum()),
                }
            )
        reconstructed = privacy_budget_from_flipping(
            f,
            dimensions=len(attributes),
            hash_functions=parameters.hash_functions,
        )
        group_diagnostics.append(
            {
                "semantic_dimension": str(anchor),
                "n_attributes": len(attributes),
                "n_reports": int(mask.sum()),
                "flipping_probability": f,
                "record_epsilon_reconstructed": reconstructed,
                "em_iterations": n_iterations,
                "em_converged": converged,
            }
        )

    metrics = {
        key: float(np.mean([row[key] for row in metric_rows]))
        for key in metric_rows[0]
    }
    diagnostics: dict[str, object] = {
        "strategy": "Smp",
        "n_groups": len(grouped),
        "reports_per_user": 1.0,
        "reported_groups_per_user": 1.0,
        "client_payload_bits_mean": float(
            np.mean(
                [
                    len(grouped[group_id][1]) * parameters.bloom_bits
                    for group_id in selected_groups
                ]
            )
        ),
        "mean_em_iterations": float(
            np.mean([row["em_iterations"] for row in group_diagnostics])
        ),
        "max_em_iterations": int(
            max(row["em_iterations"] for row in group_diagnostics)
        ),
        "all_groups_converged": bool(
            all(row["em_converged"] for row in group_diagnostics)
        ),
        "group_diagnostics": group_diagnostics,
    }
    return metrics, pd.DataFrame(estimate_rows), diagnostics
