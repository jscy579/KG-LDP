"""Formal, provenance-aware SemMed-LDP distribution-estimation experiments."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.cluster import AgglomerativeClustering

from semmed_ldp.formal_semantic_reduction import (
    DOMAIN_SIZE,
    attribute_abnormal_indicators,
    build_ontology_group_states,
    validate_formal_mapping,
)
from semmed_ldp.metrics import compute_all_metrics
from semmed_ldp.multidimensional_reporting import (
    MultidimensionalProtocol,
    run_multidimensional_protocol,
)
from semmed_ldp.sampled_reporting import (
    ReportingProtocol,
    empirical_state_distributions,
    estimate_distributions_from_reports,
    report_size_bytes,
    sample_one_dimension_reports,
)


def _with_group_column(mapping: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    frame = mapping.copy()
    frame["semantic_anchor"] = groups.astype(str).to_numpy()
    return frame


def _random_group_labels(mapping: pd.DataFrame, seed: int) -> pd.Series:
    anchors = mapping["semantic_anchor"].astype(str)
    sizes = anchors.value_counts(sort=False).tolist()
    labels = [f"random_{index + 1}" for index, size in enumerate(sizes) for _ in range(size)]
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)
    return pd.Series(labels, index=mapping.index)


def _correlation_group_labels(
    features: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    metadata: pd.DataFrame | None = None,
    reference_intervals: pd.DataFrame | None = None,
) -> pd.Series:
    abnormal = attribute_abnormal_indicators(
        features,
        mapping,
        metadata=metadata,
        reference_intervals=reference_intervals,
    ).astype(float).fillna(0)
    n_groups = mapping["semantic_anchor"].nunique()
    if n_groups <= 1:
        return pd.Series("correlation_1", index=mapping.index)
    corr = abnormal.corr().fillna(0).abs()
    distance = 1 - corr
    np.fill_diagonal(distance.values, 0)
    try:
        model = AgglomerativeClustering(n_clusters=n_groups, metric="precomputed", linkage="average")
    except TypeError:  # scikit-learn < 1.2 compatibility
        model = AgglomerativeClustering(n_clusters=n_groups, affinity="precomputed", linkage="average")
    labels = model.fit_predict(distance)
    by_attribute = {attribute: f"correlation_{label + 1}" for attribute, label in zip(abnormal.columns, labels)}
    return mapping["attribute_name"].map(by_attribute)


def build_formal_method_states(
    features: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    seed: int,
    metadata: pd.DataFrame | None = None,
    reference_intervals: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build proposed and matched baseline state matrices from the same users."""
    mapping = validate_formal_mapping(mapping)
    methods: dict[str, pd.DataFrame] = {}
    reduction_kwargs = {"metadata": metadata, "reference_intervals": reference_intervals}
    methods["SemMed-LDP"], _ = build_ontology_group_states(features, mapping, **reduction_kwargs)

    if "loinc_group_id" in mapping and mapping["loinc_group_id"].astype(str).str.strip().ne("").all():
        loinc_mapping = _with_group_column(mapping, mapping["loinc_group_id"])
        methods["LOINC-Groups-LDP"], _ = build_ontology_group_states(features, loinc_mapping, **reduction_kwargs)

    if "source_table" in mapping and mapping["source_table"].astype(str).str.strip().ne("").all():
        type_groups = mapping["source_table"].astype(str).copy()
        if "source_dictionary_category" in mapping:
            category = mapping["source_dictionary_category"].fillna("").astype(str).str.strip()
            type_groups = category.where(category.ne(""), type_groups)
        type_mapping = _with_group_column(mapping, type_groups)
        methods["Type-Grouped-LDP"], _ = build_ontology_group_states(features, type_mapping, **reduction_kwargs)

    random_mapping = _with_group_column(mapping, _random_group_labels(mapping, seed))
    methods["Random-Grouped-LDP"], _ = build_ontology_group_states(features, random_mapping, **reduction_kwargs)

    correlation_mapping = _with_group_column(
        mapping,
        _correlation_group_labels(
            features,
            mapping,
            metadata=metadata,
            reference_intervals=reference_intervals,
        ),
    )
    methods["Correlation-Grouped-LDP-oracle"], _ = build_ontology_group_states(
        features, correlation_mapping, **reduction_kwargs
    )

    raw_groups = pd.Series([f"raw::{value}" for value in mapping["attribute_name"]], index=mapping.index)
    raw_mapping = _with_group_column(mapping, raw_groups)
    methods["Raw-Attribute-LDP"], _ = build_ontology_group_states(features, raw_mapping, **reduction_kwargs)
    return methods


def build_tradeoff_method_states(
    features: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    metadata: pd.DataFrame | None = None,
    reference_intervals: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build the prespecified resolution--error trade-off representations."""
    mapping = validate_formal_mapping(mapping)
    kwargs = {"metadata": metadata, "reference_intervals": reference_intervals}
    methods: dict[str, pd.DataFrame] = {}
    methods["SemMed-LDP"], _ = build_ontology_group_states(features, mapping, **kwargs)

    loinc_mapping = _with_group_column(mapping, mapping["loinc_group_id"])
    methods["LOINC-Group-LDP"], _ = build_ontology_group_states(features, loinc_mapping, **kwargs)

    source_groups = mapping["source_table"].astype(str).copy()
    if "source_dictionary_category" in mapping:
        category = mapping["source_dictionary_category"].fillna("").astype(str).str.strip()
        source_groups = category.where(category.ne(""), source_groups)
    source_mapping = _with_group_column(mapping, source_groups)
    methods["Source-Schema-LDP"], _ = build_ontology_group_states(features, source_mapping, **kwargs)

    if "loinc_class" not in mapping or mapping["loinc_class"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("trade-off experiment requires a reviewed loinc_class for every attribute")
    class_mapping = _with_group_column(mapping, mapping["loinc_class"])
    methods["LOINC-Class-LDP"], _ = build_ontology_group_states(features, class_mapping, **kwargs)

    raw_groups = pd.Series([f"raw::{value}" for value in mapping["attribute_name"]], index=mapping.index)
    raw_mapping = _with_group_column(mapping, raw_groups)
    methods["Raw-Attribute-LDP"], _ = build_ontology_group_states(features, raw_mapping, **kwargs)
    return methods


def build_representation_method_states(
    target_features: pd.DataFrame,
    target_mapping: pd.DataFrame,
    *,
    external_features: pd.DataFrame,
    external_mapping: pd.DataFrame,
    n_random_partitions: int,
    seed: int,
    target_metadata: pd.DataFrame | None = None,
    external_metadata: pd.DataFrame | None = None,
    reference_intervals: pd.DataFrame | None = None,
    external_dataset_id: str,
) -> dict[str, tuple[pd.DataFrame, str]]:
    """Build dimension-matched semantic, random, and external-correlation states."""
    target_mapping = validate_formal_mapping(target_mapping)
    external_mapping = validate_formal_mapping(external_mapping)
    if set(target_mapping["attribute_name"]) != set(external_mapping["attribute_name"]):
        raise ValueError("target and external mappings must contain the same canonical attributes")
    target_kwargs = {
        "metadata": target_metadata,
        "reference_intervals": reference_intervals,
    }
    methods: dict[str, tuple[pd.DataFrame, str]] = {}
    semantic, _ = build_ontology_group_states(target_features, target_mapping, **target_kwargs)
    methods["SemMed-LDP"] = (semantic, "semantic")

    for partition in range(n_random_partitions):
        labels = _random_group_labels(target_mapping, seed + 1000 + partition)
        random_mapping = _with_group_column(target_mapping, labels)
        states, _ = build_ontology_group_states(target_features, random_mapping, **target_kwargs)
        methods[f"Matched-Random-4D::{partition + 1:03d}"] = (states, f"random_{partition + 1:03d}")

    external_labels = _correlation_group_labels(
        external_features,
        external_mapping,
        metadata=external_metadata,
        reference_intervals=reference_intervals,
    )
    labels_by_attribute = dict(zip(external_mapping["attribute_name"], external_labels))
    target_labels = target_mapping["attribute_name"].map(labels_by_attribute)
    if target_labels.isna().any():
        raise ValueError("external correlation grouping did not cover every target attribute")
    external_group_mapping = _with_group_column(target_mapping, target_labels)
    states, _ = build_ontology_group_states(target_features, external_group_mapping, **target_kwargs)
    methods["External-Correlation-4D"] = (states, f"trained_on_{external_dataset_id}")
    return methods


def semantic_state_diagnostics(states: pd.DataFrame) -> pd.DataFrame:
    """Summarize state support and reject empty or constant semantic modules."""
    if states.empty or states.shape[1] == 0:
        raise ValueError("semantic state matrix must contain users and dimensions")
    rows = []
    for dimension in states.columns:
        counts = states[dimension].value_counts(dropna=False)
        rows.append(
            {
                "semantic_dimension": str(dimension),
                "n_users": int(len(states)),
                "n_unique_states": int(len(counts)),
                "missing_state_rate": float(states[dimension].eq(0).mean()),
                "constant": len(counts) < 2,
            }
        )
    diagnostics = pd.DataFrame(rows)
    if diagnostics["constant"].any():
        bad = diagnostics.loc[diagnostics["constant"], "semantic_dimension"].tolist()
        raise ValueError(f"constant semantic modules are forbidden: {bad}")
    return diagnostics


def build_formal_ablation_states(
    features: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    seed: int,
    metadata: pd.DataFrame | None = None,
    reference_intervals: pd.DataFrame | None = None,
) -> dict[str, tuple[pd.DataFrame, int]]:
    """Build prespecified, data-independent ablation state matrices."""
    mapping = validate_formal_mapping(mapping)
    kwargs = {"metadata": metadata, "reference_intervals": reference_intervals}
    full, _ = build_ontology_group_states(features, mapping, **kwargs)
    ablations: dict[str, tuple[pd.DataFrame, int]] = {"Full-SemMed-LDP": (full, DOMAIN_SIZE)}

    no_hpo = _with_group_column(mapping, mapping["loinc_group_id"])
    states, _ = build_ontology_group_states(features, no_hpo, **kwargs)
    ablations["No-HPO-LOINC-Groups"] = (states, DOMAIN_SIZE)

    shuffled = _with_group_column(mapping, _random_group_labels(mapping, seed))
    states, _ = build_ontology_group_states(features, shuffled, **kwargs)
    ablations["Shuffled-Knowledge-Edges"] = (states, DOMAIN_SIZE)

    three_anchor = mapping["semantic_anchor"].replace(
        {
            "circulating_nitrogen_compounds": "metabolic_and_renal",
            "metabolism_homeostasis": "metabolic_and_renal",
        }
    )
    three_mapping = _with_group_column(mapping, three_anchor)
    states, _ = build_ontology_group_states(features, three_mapping, **kwargs)
    ablations["Anchor-Count-3"] = (states, DOMAIN_SIZE)
    two_anchor = mapping["semantic_anchor"].replace(
        {
            "blood_ion_homeostasis": "chemistry_homeostasis",
            "circulating_nitrogen_compounds": "chemistry_homeostasis",
            "metabolism_homeostasis": "chemistry_homeostasis",
        }
    )
    two_mapping = _with_group_column(mapping, two_anchor)
    states, _ = build_ontology_group_states(features, two_mapping, **kwargs)
    ablations["Anchor-Count-2"] = (states, DOMAIN_SIZE)
    return ablations


def run_formal_ablation_experiment(
    features: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    dataset_id: str,
    epsilon_list: Iterable[float],
    mechanisms: Iterable[str],
    sample_sizes: Iterable[int | str],
    repeat_runs: int,
    seed: int,
    metadata: pd.DataFrame | None = None,
    reference_intervals: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the formal ablation family with one report per user."""
    mapping = validate_formal_mapping(mapping, dataset_id=dataset_id)
    features = features[mapping["attribute_name"].tolist()].copy()
    if metadata is not None:
        metadata = metadata.loc[features.index]
    ablations = build_formal_ablation_states(
        features,
        mapping,
        seed=seed,
        metadata=metadata,
        reference_intervals=reference_intervals,
    )
    support_rows = []
    for method, (states, domain_size) in ablations.items():
        local = semantic_state_diagnostics(states)
        local.insert(0, "method", method)
        local["domain_size"] = domain_size
        support_rows.append(local)

    metric_rows: list[dict[str, float | int | str]] = []
    diagnostic_rows: list[dict[str, float | int | str]] = []
    for sample_size in sample_sizes:
        requested = len(features) if str(sample_size).lower() == "all" else int(sample_size)
        actual = min(requested, len(features))
        for run in range(repeat_runs):
            run_seed = seed + run
            rng = np.random.default_rng(run_seed)
            selected_index = features.index[rng.choice(len(features), size=actual, replace=False)]
            for method, (full_states, domain_size) in ablations.items():
                states = full_states.loc[selected_index]
                truth = empirical_state_distributions(states, domain_size)
                for mechanism in mechanisms:
                    for epsilon in epsilon_list:
                        protocol = ReportingProtocol(
                            epsilon=float(epsilon), mechanism=str(mechanism), domain_size=domain_size
                        )
                        started = time.perf_counter()
                        reports = sample_one_dimension_reports(states, protocol, seed=run_seed)
                        client_seconds = time.perf_counter() - started
                        started = time.perf_counter()
                        estimate = estimate_distributions_from_reports(reports, states.columns, protocol)
                        server_seconds = time.perf_counter() - started
                        metric_rows.append(
                            {
                                "dataset_id": dataset_id,
                                "method": method,
                                "mechanism": str(mechanism).upper(),
                                "epsilon": float(epsilon),
                                "sample_size": actual,
                                "run": run,
                                "seed": run_seed,
                                "n_dimensions": states.shape[1],
                                "domain_size": domain_size,
                                **_metric_row(truth, estimate),
                            }
                        )
                        diagnostic_rows.append(
                            {
                                "dataset_id": dataset_id,
                                "method": method,
                                "mechanism": str(mechanism).upper(),
                                "epsilon": float(epsilon),
                                "sample_size": actual,
                                "run": run,
                                "n_users": len(reports),
                                "n_reports": len(reports),
                                "reports_per_user": len(reports) / len(states),
                                "n_dimensions": states.shape[1],
                                "domain_size": domain_size,
                                "compression_ratio": len(mapping) / states.shape[1],
                                "mean_report_bytes": report_size_bytes(reports),
                                "client_seconds": client_seconds,
                                "server_seconds": server_seconds,
                            }
                        )
    return pd.DataFrame(metric_rows), pd.DataFrame(diagnostic_rows), pd.concat(support_rows, ignore_index=True)


def _metric_row(true_frame: pd.DataFrame, estimate_frame: pd.DataFrame) -> dict[str, float]:
    merged = true_frame.merge(estimate_frame, on=["semantic_dimension", "state"], validate="one_to_one")
    per_dimension = []
    for _, group in merged.groupby("semantic_dimension"):
        per_dimension.append(compute_all_metrics(group["true_frequency"].to_numpy(), group["estimated_frequency"].to_numpy()))
    return {key: float(np.mean([row[key] for row in per_dimension])) for key in per_dimension[0]}


def _selected_user_index(
    index: pd.Index,
    sample_size: int | str,
    *,
    seed: int,
) -> pd.Index:
    requested = len(index) if str(sample_size).lower() == "all" else int(sample_size)
    actual = min(requested, len(index))
    positions = np.random.default_rng(seed).choice(len(index), size=actual, replace=False)
    return index[positions]


def _run_protocol_on_states(
    states: pd.DataFrame,
    *,
    epsilon: float,
    mechanism: str,
    strategy: str,
    seed: int,
    measure_communication: bool,
    truth: pd.DataFrame | None = None,
) -> tuple[dict[str, float], dict[str, float | int | str]]:
    if truth is None:
        truth = empirical_state_distributions(states, DOMAIN_SIZE)
    protocol = MultidimensionalProtocol(
        epsilon=float(epsilon),
        mechanism=str(mechanism),
        strategy=strategy,
        domain_size=DOMAIN_SIZE,
    )
    started = time.perf_counter()
    estimate, diagnostics = run_multidimensional_protocol(
        states,
        protocol,
        seed=seed,
        measure_communication=measure_communication,
    )
    elapsed = time.perf_counter() - started
    diagnostics = {**diagnostics, "total_seconds": elapsed}
    return _metric_row(truth, estimate), diagnostics


def run_protocol_comparison_experiment(
    features: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    dataset_id: str,
    epsilon_list: Iterable[float],
    mechanisms: Iterable[str],
    strategies: Iterable[str],
    sample_sizes: Iterable[int | str],
    repeat_runs: int,
    seed: int,
    metadata: pd.DataFrame | None = None,
    reference_intervals: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare Smp, Spl, and RS+FD on the identical SemMed 4D truth."""
    mapping = validate_formal_mapping(mapping, dataset_id=dataset_id)
    features = features[mapping["attribute_name"].tolist()].copy()
    if metadata is not None:
        metadata = metadata.loc[features.index]
    full_states, _ = build_ontology_group_states(
        features,
        mapping,
        metadata=metadata,
        reference_intervals=reference_intervals,
    )
    if full_states.shape[1] != 4:
        raise ValueError(f"protocol comparison requires SemMed 4D, found {full_states.shape[1]}D")

    metric_rows: list[dict[str, float | int | str]] = []
    diagnostic_rows: list[dict[str, float | int | str]] = []
    for sample_size in sample_sizes:
        for run in range(repeat_runs):
            run_seed = seed + run
            selected = _selected_user_index(features.index, sample_size, seed=run_seed)
            states = full_states.loc[selected]
            truth = empirical_state_distributions(states, DOMAIN_SIZE)
            for mechanism in mechanisms:
                for epsilon in epsilon_list:
                    for strategy in strategies:
                        metrics, diagnostic = _run_protocol_on_states(
                            states,
                            epsilon=float(epsilon),
                            mechanism=str(mechanism),
                            strategy=str(strategy),
                            seed=run_seed,
                            measure_communication=run == 0,
                            truth=truth,
                        )
                        method = f"SemMed+{diagnostic['strategy']}"
                        shared = {
                            "dataset_id": dataset_id,
                            "experiment_family": "protocol_comparison",
                            "method": method,
                            "representation": "SemMed-4D",
                            "strategy": diagnostic["strategy"],
                            "mechanism": str(mechanism).upper(),
                            "epsilon": float(epsilon),
                            "sample_size": len(states),
                            "run": run,
                            "seed": run_seed,
                            "n_dimensions": states.shape[1],
                            "domain_size": DOMAIN_SIZE,
                        }
                        metric_rows.append({**shared, **metrics})
                        diagnostic_rows.append(
                            {
                                **shared,
                                **diagnostic,
                                "n_users": len(states),
                                "compression_ratio": len(mapping) / states.shape[1],
                            }
                        )
    return pd.DataFrame(metric_rows), pd.DataFrame(diagnostic_rows)


def run_representation_comparison_experiment(
    target_features: pd.DataFrame,
    target_mapping: pd.DataFrame,
    *,
    external_features: pd.DataFrame,
    external_mapping: pd.DataFrame,
    dataset_id: str,
    external_dataset_id: str,
    epsilon_list: Iterable[float],
    mechanisms: Iterable[str],
    sample_sizes: Iterable[int | str],
    repeat_runs: int,
    n_random_partitions: int,
    seed: int,
    target_metadata: pd.DataFrame | None = None,
    external_metadata: pd.DataFrame | None = None,
    reference_intervals: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare three 4D representations under fixed Smp and randomizer."""
    target_mapping = validate_formal_mapping(target_mapping, dataset_id=dataset_id)
    external_mapping = validate_formal_mapping(
        external_mapping, dataset_id=external_dataset_id
    )
    target_features = target_features[target_mapping["attribute_name"].tolist()].copy()
    if target_metadata is not None:
        target_metadata = target_metadata.loc[target_features.index]
    methods = build_representation_method_states(
        target_features,
        target_mapping,
        external_features=external_features,
        external_mapping=external_mapping,
        n_random_partitions=n_random_partitions,
        seed=seed,
        target_metadata=target_metadata,
        external_metadata=external_metadata,
        reference_intervals=reference_intervals,
        external_dataset_id=external_dataset_id,
    )
    if any(states.shape[1] != 4 for states, _ in methods.values()):
        raise ValueError("every representation comparator must have exactly four dimensions")

    metric_rows: list[dict[str, float | int | str]] = []
    diagnostic_rows: list[dict[str, float | int | str]] = []
    for sample_size in sample_sizes:
        for run in range(repeat_runs):
            run_seed = seed + run
            selected = _selected_user_index(target_features.index, sample_size, seed=run_seed)
            for method_key, (full_states, grouping_id) in methods.items():
                states = full_states.loc[selected]
                method = "Matched-Random-4D" if method_key.startswith("Matched-Random-4D::") else method_key
                truth = empirical_state_distributions(states, DOMAIN_SIZE)
                for mechanism in mechanisms:
                    for epsilon in epsilon_list:
                        metrics, diagnostic = _run_protocol_on_states(
                            states,
                            epsilon=float(epsilon),
                            mechanism=str(mechanism),
                            strategy="Smp",
                            seed=run_seed,
                            measure_communication=False,
                            truth=truth,
                        )
                        shared = {
                            "dataset_id": dataset_id,
                            "external_dataset_id": external_dataset_id,
                            "experiment_family": "representation_comparison",
                            "method": method,
                            "grouping_id": grouping_id,
                            "strategy": "Smp",
                            "mechanism": str(mechanism).upper(),
                            "epsilon": float(epsilon),
                            "sample_size": len(states),
                            "run": run,
                            "seed": run_seed,
                            "n_dimensions": states.shape[1],
                            "domain_size": DOMAIN_SIZE,
                        }
                        metric_rows.append({**shared, **metrics})
                        diagnostic_rows.append({**shared, **diagnostic, "n_users": len(states)})
    return pd.DataFrame(metric_rows), pd.DataFrame(diagnostic_rows)


def run_tradeoff_experiment(
    features: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    dataset_id: str,
    epsilon_list: Iterable[float],
    mechanisms: Iterable[str],
    sample_sizes: Iterable[int | str],
    repeat_runs: int,
    seed: int,
    metadata: pd.DataFrame | None = None,
    reference_intervals: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate the resolution--privacy trade-off under a fixed Smp protocol."""
    mapping = validate_formal_mapping(mapping, dataset_id=dataset_id)
    features = features[mapping["attribute_name"].tolist()].copy()
    if metadata is not None:
        metadata = metadata.loc[features.index]
    methods = build_tradeoff_method_states(
        features,
        mapping,
        metadata=metadata,
        reference_intervals=reference_intervals,
    )
    expected_dimensions = {
        "Raw-Attribute-LDP": 10,
        "LOINC-Group-LDP": 10,
        "SemMed-LDP": 4,
        "Source-Schema-LDP": None,
        "LOINC-Class-LDP": 2,
    }
    for method, states in methods.items():
        expected = expected_dimensions[method]
        if expected is not None and states.shape[1] != expected:
            raise ValueError(f"{method} must be {expected}D, found {states.shape[1]}D")
        if method == "Source-Schema-LDP" and states.shape[1] not in {2, 3}:
            raise ValueError(f"Source-Schema-LDP must be 2--3D, found {states.shape[1]}D")

    metric_rows: list[dict[str, float | int | str]] = []
    diagnostic_rows: list[dict[str, float | int | str]] = []
    for sample_size in sample_sizes:
        for run in range(repeat_runs):
            run_seed = seed + run
            selected = _selected_user_index(features.index, sample_size, seed=run_seed)
            for method, full_states in methods.items():
                states = full_states.loc[selected]
                truth = empirical_state_distributions(states, DOMAIN_SIZE)
                for mechanism in mechanisms:
                    for epsilon in epsilon_list:
                        metrics, diagnostic = _run_protocol_on_states(
                            states,
                            epsilon=float(epsilon),
                            mechanism=str(mechanism),
                            strategy="Smp",
                            seed=run_seed,
                            measure_communication=run == 0,
                            truth=truth,
                        )
                        shared = {
                            "dataset_id": dataset_id,
                            "experiment_family": "resolution_tradeoff",
                            "method": method,
                            "strategy": "Smp",
                            "mechanism": str(mechanism).upper(),
                            "epsilon": float(epsilon),
                            "sample_size": len(states),
                            "run": run,
                            "seed": run_seed,
                            "n_dimensions": states.shape[1],
                            "domain_size": DOMAIN_SIZE,
                            "estimand_id": method,
                        }
                        metric_rows.append({**shared, **metrics})
                        diagnostic_rows.append(
                            {
                                **shared,
                                **diagnostic,
                                "n_users": len(states),
                                "compression_ratio": len(mapping) / states.shape[1],
                            }
                        )
    return pd.DataFrame(metric_rows), pd.DataFrame(diagnostic_rows)


def run_formal_distribution_experiment(
    features: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    dataset_id: str,
    epsilon_list: Iterable[float],
    mechanisms: Iterable[str],
    sample_sizes: Iterable[int | str],
    repeat_runs: int,
    seed: int,
    metadata: pd.DataFrame | None = None,
    reference_intervals: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run fair user-level LDP comparisons and return metrics and diagnostics."""
    mapping = validate_formal_mapping(mapping, dataset_id=dataset_id)
    features = features[mapping["attribute_name"].tolist()].copy()
    if metadata is not None:
        metadata = metadata.loc[features.index]
    method_states = build_formal_method_states(
        features,
        mapping,
        seed=seed,
        metadata=metadata,
        reference_intervals=reference_intervals,
    )
    metric_rows: list[dict[str, float | int | str]] = []
    diagnostic_rows: list[dict[str, float | int | str]] = []
    for sample_size in sample_sizes:
        requested = len(features) if str(sample_size).lower() == "all" else int(sample_size)
        actual = min(requested, len(features))
        for run in range(repeat_runs):
            run_seed = seed + run
            rng = np.random.default_rng(run_seed)
            selected_positions = rng.choice(len(features), size=actual, replace=False)
            selected_index = features.index[selected_positions]
            for method, full_states in method_states.items():
                states = full_states.loc[selected_index]
                truth = empirical_state_distributions(states, DOMAIN_SIZE)
                for mechanism in mechanisms:
                    for epsilon in epsilon_list:
                        protocol = ReportingProtocol(epsilon=float(epsilon), mechanism=str(mechanism), domain_size=DOMAIN_SIZE)
                        started = time.perf_counter()
                        reports = sample_one_dimension_reports(states, protocol, seed=run_seed)
                        client_seconds = time.perf_counter() - started
                        started = time.perf_counter()
                        estimate = estimate_distributions_from_reports(reports, states.columns, protocol)
                        server_seconds = time.perf_counter() - started
                        metrics = _metric_row(truth, estimate)
                        metric_rows.append(
                            {
                                "dataset_id": dataset_id,
                                "method": method,
                                "mechanism": str(mechanism).upper(),
                                "epsilon": float(epsilon),
                                "sample_size": actual,
                                "run": run,
                                "seed": run_seed,
                                "n_dimensions": states.shape[1],
                                **metrics,
                            }
                        )
                        diagnostic_rows.append(
                            {
                                "dataset_id": dataset_id,
                                "method": method,
                                "mechanism": str(mechanism).upper(),
                                "epsilon": float(epsilon),
                                "sample_size": actual,
                                "run": run,
                                "n_users": len(reports),
                                "n_reports": len(reports),
                                "reports_per_user": len(reports) / len(states),
                                "n_dimensions": states.shape[1],
                                "compression_ratio": len(mapping) / states.shape[1],
                                "mean_report_bytes": report_size_bytes(reports),
                                "client_seconds": client_seconds,
                                "server_seconds": server_seconds,
                            }
                        )
    return pd.DataFrame(metric_rows), pd.DataFrame(diagnostic_rows)


def summarize_with_bootstrap_ci(results: pd.DataFrame, *, seed: int = 2026, n_bootstrap: int = 2000) -> pd.DataFrame:
    metrics = ["MSE", "MAE", "KL", "JS", "TVD", "Wasserstein"]
    group_columns = ["dataset_id", "method", "mechanism", "epsilon", "sample_size", "n_dimensions"]
    rng = np.random.default_rng(seed)
    rows = []
    for keys, group in results.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        for metric in metrics:
            values = group[metric].astype(float).to_numpy()
            boot = np.asarray([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_bootstrap)])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_ci_low"] = float(np.quantile(boot, 0.025))
            row[f"{metric}_ci_high"] = float(np.quantile(boot, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def collapse_random_partition_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Average matched random partitions within each run before inference."""
    random_rows = results[results["method"].eq("Matched-Random-4D")].copy()
    other_rows = results[~results["method"].eq("Matched-Random-4D")].copy()
    if random_rows.empty:
        return results.copy()
    metrics = ["MSE", "MAE", "KL", "JS", "TVD", "Wasserstein"]
    group_columns = [
        column
        for column in random_rows.columns
        if column not in metrics + ["grouping_id"]
    ]
    collapsed = random_rows.groupby(group_columns, as_index=False, dropna=False)[metrics].mean()
    collapsed["grouping_id"] = "mean_of_matched_random_partitions"
    return pd.concat([other_rows, collapsed], ignore_index=True, sort=False)


def paired_wilcoxon_holm(results: pd.DataFrame, *, proposed: str = "SemMed-LDP", metric: str = "MSE") -> pd.DataFrame:
    condition_columns = ["dataset_id", "mechanism", "epsilon", "sample_size"]
    rows = []
    for condition, frame in results.groupby(condition_columns):
        proposed_frame = frame.loc[frame["method"] == proposed, ["run", metric]].rename(columns={metric: "proposed"})
        local_rows = []
        for baseline in sorted(set(frame["method"]) - {proposed}):
            baseline_frame = frame.loc[frame["method"] == baseline, ["run", metric]].rename(columns={metric: "baseline"})
            paired = proposed_frame.merge(baseline_frame, on="run", validate="one_to_one")
            differences = paired["baseline"] - paired["proposed"]
            if len(paired) == 0 or np.allclose(differences, 0):
                statistic, p_value = 0.0, 1.0
                rank_biserial = 0.0
            else:
                statistic, p_value = wilcoxon(paired["proposed"], paired["baseline"], alternative="two-sided")
                ranks = rankdata(np.abs(differences))
                positive = ranks[differences > 0].sum()
                negative = ranks[differences < 0].sum()
                rank_biserial = float((positive - negative) / (positive + negative)) if positive + negative else 0.0
            local_rows.append(
                {
                    **dict(zip(condition_columns, condition)),
                    "baseline": baseline,
                    "metric": metric,
                    "n_pairs": len(paired),
                    "median_baseline_minus_proposed": float(differences.median()) if len(paired) else np.nan,
                    "rank_biserial_effect": rank_biserial,
                    "wilcoxon_statistic": float(statistic),
                    "p_value": float(p_value),
                }
            )
        order = np.argsort([row["p_value"] for row in local_rows])
        adjusted = np.ones(len(local_rows))
        running = 0.0
        for rank, index in enumerate(order):
            candidate = min(1.0, local_rows[index]["p_value"] * (len(local_rows) - rank))
            running = max(running, candidate)
            adjusted[index] = running
        for row, value in zip(local_rows, adjusted):
            row["p_holm"] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_manifest(config: dict, inputs: Iterable[str | Path]) -> dict:
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = "unavailable"
    return {
        "project_id": "SemMed-LDP",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit,
        "config": config,
        "inputs": [
            {"path": str(Path(path)), "sha256": sha256_file(path)}
            for path in inputs
            if Path(path).is_file()
        ],
    }


def write_json(path: str | Path, value: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
