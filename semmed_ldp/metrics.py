"""Distribution error metrics for SemMed-LDP."""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


def normalize_distribution(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Clip and normalize a vector into a probability distribution."""
    arr = np.asarray(values, dtype=float)
    arr = np.clip(arr, 0, None)
    total = arr.sum()
    if total <= eps:
        return np.ones_like(arr, dtype=float) / len(arr)
    return arr / total


def mse(true_distribution: np.ndarray, estimated_distribution: np.ndarray) -> float:
    """Mean squared error between two distributions."""
    p = normalize_distribution(true_distribution)
    q = normalize_distribution(estimated_distribution)
    return float(np.mean((p - q) ** 2))


def mae(true_distribution: np.ndarray, estimated_distribution: np.ndarray) -> float:
    """Mean absolute error between two distributions."""
    p = normalize_distribution(true_distribution)
    q = normalize_distribution(estimated_distribution)
    return float(np.mean(np.abs(p - q)))


def kl_divergence(true_distribution: np.ndarray, estimated_distribution: np.ndarray, eps: float = 1e-9) -> float:
    """Smoothed KL divergence KL(true || estimated)."""
    p = normalize_distribution(np.asarray(true_distribution, dtype=float) + eps)
    q = normalize_distribution(np.asarray(estimated_distribution, dtype=float) + eps)
    return float(np.sum(p * np.log(p / q)))


def js_divergence(true_distribution: np.ndarray, estimated_distribution: np.ndarray, eps: float = 1e-9) -> float:
    """Smoothed Jensen-Shannon divergence."""
    p = normalize_distribution(np.asarray(true_distribution, dtype=float) + eps)
    q = normalize_distribution(np.asarray(estimated_distribution, dtype=float) + eps)
    midpoint = normalize_distribution((p + q) / 2)
    return float(0.5 * kl_divergence(p, midpoint, eps) + 0.5 * kl_divergence(q, midpoint, eps))


def total_variation_distance(true_distribution: np.ndarray, estimated_distribution: np.ndarray) -> float:
    """Total variation distance between two categorical distributions."""
    p = normalize_distribution(true_distribution)
    q = normalize_distribution(estimated_distribution)
    return float(0.5 * np.sum(np.abs(p - q)))


def wasserstein_distance_metric(true_distribution: np.ndarray, estimated_distribution: np.ndarray) -> float:
    """One-dimensional Wasserstein distance over ordered category labels."""
    p = normalize_distribution(true_distribution)
    q = normalize_distribution(estimated_distribution)
    support = np.arange(len(p))
    return float(wasserstein_distance(support, support, p, q))


def compute_all_metrics(true_distribution: np.ndarray, estimated_distribution: np.ndarray) -> dict[str, float]:
    """Compute all supported distribution error metrics."""
    return {
        "MSE": mse(true_distribution, estimated_distribution),
        "MAE": mae(true_distribution, estimated_distribution),
        "KL": kl_divergence(true_distribution, estimated_distribution),
        "JS": js_divergence(true_distribution, estimated_distribution),
        "TVD": total_variation_distance(true_distribution, estimated_distribution),
        "Wasserstein": wasserstein_distance_metric(true_distribution, estimated_distribution),
    }
