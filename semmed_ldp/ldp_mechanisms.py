"""Local differential privacy mechanisms for categorical variables."""

from __future__ import annotations

import numpy as np


def validate_categorical_values(values: np.ndarray, domain_size: int) -> np.ndarray:
    """Return integer categorical values and validate that they fit the domain."""
    values = np.asarray(values, dtype=int)
    if domain_size < 2:
        raise ValueError("domain_size must be at least 2.")
    if values.size and (values.min() < 0 or values.max() >= domain_size):
        raise ValueError(f"Values must be in [0, {domain_size - 1}].")
    return values


def grr_perturb(values: np.ndarray, epsilon: float, domain_size: int, seed: int | None = None) -> np.ndarray:
    """Perturb categorical values with Generalized Randomized Response."""
    values = validate_categorical_values(values, domain_size)
    rng = np.random.default_rng(seed)
    exp_eps = np.exp(epsilon)
    p = exp_eps / (exp_eps + domain_size - 1)
    keep = rng.random(values.size) < p
    reported = values.copy()
    if (~keep).any():
        alternatives = np.empty((~keep).sum(), dtype=int)
        original = values[~keep]
        draws = rng.integers(0, domain_size - 1, size=original.size)
        alternatives = draws + (draws >= original)
        reported[~keep] = alternatives
    return reported


def grr_estimate(reports: np.ndarray, epsilon: float, domain_size: int) -> np.ndarray:
    """Estimate the original categorical distribution from GRR reports."""
    reports = validate_categorical_values(reports, domain_size)
    n = reports.size
    if n == 0:
        return np.ones(domain_size) / domain_size
    exp_eps = np.exp(epsilon)
    p = exp_eps / (exp_eps + domain_size - 1)
    q = 1 / (exp_eps + domain_size - 1)
    counts = np.bincount(reports, minlength=domain_size).astype(float)
    estimated_counts = (counts - n * q) / (p - q)
    estimated_counts = np.clip(estimated_counts, 0, None)
    if estimated_counts.sum() == 0:
        return np.ones(domain_size) / domain_size
    return estimated_counts / estimated_counts.sum()


def oue_perturb(values: np.ndarray, epsilon: float, domain_size: int, seed: int | None = None) -> np.ndarray:
    """Perturb categorical values with Optimized Unary Encoding."""
    values = validate_categorical_values(values, domain_size)
    rng = np.random.default_rng(seed)
    p = 0.5
    q = 1 / (np.exp(epsilon) + 1)
    unary = np.zeros((values.size, domain_size), dtype=int)
    if values.size:
        unary[np.arange(values.size), values] = 1
    probabilities = np.where(unary == 1, p, q)
    return (rng.random(unary.shape) < probabilities).astype(int)


def oue_estimate(reports: np.ndarray, epsilon: float, domain_size: int) -> np.ndarray:
    """Estimate the original categorical distribution from OUE bit reports."""
    reports = np.asarray(reports, dtype=int)
    if reports.ndim != 2 or reports.shape[1] != domain_size:
        raise ValueError("OUE reports must have shape (n, domain_size).")
    n = reports.shape[0]
    if n == 0:
        return np.ones(domain_size) / domain_size
    p = 0.5
    q = 1 / (np.exp(epsilon) + 1)
    counts = reports.sum(axis=0).astype(float)
    estimated_counts = (counts - n * q) / (p - q)
    estimated_counts = np.clip(estimated_counts, 0, None)
    if estimated_counts.sum() == 0:
        return np.ones(domain_size) / domain_size
    return estimated_counts / estimated_counts.sum()


def privatize_and_estimate(
    values: np.ndarray,
    epsilon: float,
    domain_size: int,
    mechanism: str,
    seed: int | None = None,
) -> np.ndarray:
    """Run one categorical LDP perturbation and frequency estimation pass."""
    mechanism_upper = mechanism.upper()
    if mechanism_upper == "GRR":
        reports = grr_perturb(values, epsilon, domain_size, seed=seed)
        return grr_estimate(reports, epsilon, domain_size)
    if mechanism_upper == "OUE":
        reports = oue_perturb(values, epsilon, domain_size, seed=seed)
        return oue_estimate(reports, epsilon, domain_size)
    raise ValueError("mechanism must be 'GRR' or 'OUE'.")
