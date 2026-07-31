import numpy as np
import pandas as pd

from semmed_ldp.sampled_reporting import (
    ReportingProtocol,
    estimate_distributions_from_reports,
    sample_one_dimension_reports,
)
from semmed_ldp.multidimensional_reporting import (
    MultidimensionalProtocol,
    amplified_epsilon,
    run_multidimensional_protocol,
)


def test_one_anonymous_report_per_user() -> None:
    states = pd.DataFrame({"renal": [0, 1, 2, 1, 0], "hepatic": [2, 1, 2, 1, 0]})
    protocol = ReportingProtocol(epsilon=1.0, mechanism="GRR", domain_size=3)
    reports = sample_one_dimension_reports(states, protocol, seed=7)
    assert len(reports) == len(states)
    assert "patient_id" not in reports.columns
    assert "true_state" not in reports.columns
    assert set(reports["sampled_dimension"]).issubset(range(states.shape[1]))


def test_sampled_grr_estimator_is_approximately_unbiased() -> None:
    rng = np.random.default_rng(11)
    n = 120_000
    states = pd.DataFrame(
        {
            "renal": rng.choice(3, n, p=[0.1, 0.55, 0.35]),
            "hepatic": rng.choice(3, n, p=[0.05, 0.60, 0.35]),
        }
    )
    protocol = ReportingProtocol(epsilon=3.0, mechanism="GRR", domain_size=3)
    reports = sample_one_dimension_reports(states, protocol, seed=13)
    estimates = estimate_distributions_from_reports(reports, states.columns, protocol)
    renal = estimates.loc[estimates["semantic_dimension"] == "renal", "estimated_frequency"].to_numpy()
    truth = np.bincount(states["renal"], minlength=3) / n
    assert np.max(np.abs(renal - truth)) < 0.025


def test_oue_payload_has_domain_size_bits() -> None:
    states = pd.DataFrame({"a": [0, 1, 2] * 20})
    protocol = ReportingProtocol(epsilon=1.0, mechanism="OUE", domain_size=3)
    reports = sample_one_dimension_reports(states, protocol, seed=17)
    assert reports["privatized_payload"].str.count(",").eq(2).all()


def test_multidimensional_strategies_share_the_same_target() -> None:
    rng = np.random.default_rng(31)
    n = 100_000
    states = pd.DataFrame(
        {
            "g1": rng.choice(3, n, p=[0.1, 0.6, 0.3]),
            "g2": rng.choice(3, n, p=[0.2, 0.5, 0.3]),
            "g3": rng.choice(3, n, p=[0.05, 0.7, 0.25]),
            "g4": rng.choice(3, n, p=[0.15, 0.55, 0.3]),
        }
    )
    for strategy in ["Smp", "Spl", "RS+FD"]:
        protocol = MultidimensionalProtocol(
            epsilon=4.0,
            mechanism="GRR",
            strategy=strategy,
            domain_size=3,
        )
        estimate, diagnostics = run_multidimensional_protocol(states, protocol, seed=37)
        for dimension in states.columns:
            observed = estimate.loc[
                estimate["semantic_dimension"] == dimension, "estimated_frequency"
            ].to_numpy()
            truth = np.bincount(states[dimension], minlength=3) / n
            assert np.max(np.abs(observed - truth)) < 0.035
        assert diagnostics["uploads_per_user"] == 1.0


def test_split_and_rsfd_privacy_parameters() -> None:
    states = pd.DataFrame({f"g{i}": [0, 1, 2] * 50 for i in range(4)})
    split = MultidimensionalProtocol(1.0, "OUE", "Spl")
    _, split_diagnostics = run_multidimensional_protocol(states, split, seed=41)
    assert split_diagnostics["internal_epsilon"] == 0.25

    rsfd = MultidimensionalProtocol(1.0, "OUE", "RS+FD")
    _, rsfd_diagnostics = run_multidimensional_protocol(states, rsfd, seed=41)
    assert np.isclose(rsfd_diagnostics["internal_epsilon"], amplified_epsilon(1.0, 4))
    assert rsfd_diagnostics["mean_report_bytes"] > split_diagnostics["mean_report_bytes"] * 0.5
