import numpy as np
import pandas as pd

from semmed_ldp.lopub_em import (
    aggregate_joint_to_semantic,
    bloom_codebook,
    em_joint_distribution,
    flipping_probability,
    privacy_budget_from_flipping,
    sample_group_indices,
)


def test_lopub_privacy_parameter_inversion():
    epsilon = 1.7
    f = flipping_probability(epsilon, dimensions=10, hash_functions=5)
    reconstructed = privacy_budget_from_flipping(f, dimensions=10, hash_functions=5)
    assert np.isclose(reconstructed, epsilon)


def test_bloom_codebook_is_deterministic_and_binary():
    first = bloom_codebook("sodium", bloom_bits=64, hash_functions=5)
    second = bloom_codebook("sodium", bloom_bits=64, hash_functions=5)
    assert np.array_equal(first, second)
    assert first.shape == (3, 64)
    assert set(np.unique(first)) <= {0, 1}


def test_em_joint_distribution_recovers_separable_reports():
    log_likelihoods = {
        "a": np.asarray([[0.0, -30.0, -30.0], [-30.0, 0.0, -30.0], [-30.0, 0.0, -30.0]]),
        "b": np.asarray([[0.0, -30.0, -30.0], [-30.0, 0.0, -30.0], [-30.0, -30.0, 0.0]]),
    }
    candidates, probability, _, converged = em_joint_distribution(
        ["a", "b"],
        log_likelihoods,
        convergence_gap=1e-8,
        max_iterations=100,
    )
    expected = {(0, 0): 1 / 3, (1, 1): 1 / 3, (1, 2): 1 / 3}
    for candidate, value in zip(candidates, probability):
        assert np.isclose(value, expected.get(tuple(candidate), 0.0), atol=1e-6)
    assert converged


def test_joint_distribution_aggregates_to_three_semantic_states():
    candidates = np.asarray(
        [(a, b) for a in range(3) for b in range(3)],
        dtype=np.int8,
    )
    probability = np.zeros(9)
    probability[0] = 0.2  # (0,0): all missing
    probability[1] = 0.3  # (0,1): observed, no abnormality
    probability[5] = 0.5  # (1,2): abnormality present
    result = aggregate_joint_to_semantic(candidates, probability)
    assert np.allclose(result, [0.2, 0.3, 0.5])


def test_sample_group_indices_matches_smp_selection():
    selected = sample_group_indices(100, 4, seed=2026)
    expected = np.random.default_rng(2026).integers(0, 4, size=100)
    assert np.array_equal(selected, expected)


def test_em_respects_minimum_iteration_count():
    likelihoods = {
        "a": np.zeros((10, 3)),
        "b": np.zeros((10, 3)),
    }
    _, _, iterations, converged = em_joint_distribution(
        ["a", "b"],
        likelihoods,
        convergence_gap=1e-6,
        min_iterations=7,
        max_iterations=20,
    )
    assert converged
    assert iterations == 7
