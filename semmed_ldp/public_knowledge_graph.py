"""Versioned LOINC--HPO--UBERON graph construction and graph-derived weights."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import networkx as nx
import numpy as np
import pandas as pd


EDGE_COLUMNS = {
    "source_id",
    "target_id",
    "relation",
    "source_vocabulary",
    "vocabulary_version",
    "provenance",
}


DEFAULT_RELATION_COSTS: dict[str, float] = {
    "mapped_to_loinc": 1.0,
    "member_of_group": 1.0,
    "parent_group": 1.0,
    "has_abnormal_phenotype": 1.5,
    "has_anatomical_system": 1.5,
    "subclass_of": 1.0,
    "anchored_to": 1.0,
}


def load_edge_table(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = EDGE_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"edge table is missing columns: {sorted(missing)}")
    if frame[list(EDGE_COLUMNS)].isna().any().any():
        raise ValueError("edge table contains missing required values")
    return frame


def build_public_medical_graph(
    edge_table: pd.DataFrame,
    *,
    relation_costs: Mapping[str, float] | None = None,
) -> nx.Graph:
    """Build a deterministic weighted undirected graph from a provenance table."""
    missing = EDGE_COLUMNS.difference(edge_table.columns)
    if missing:
        raise ValueError(f"edge table is missing columns: {sorted(missing)}")
    costs = dict(DEFAULT_RELATION_COSTS)
    if relation_costs:
        costs.update({str(key): float(value) for key, value in relation_costs.items()})
    graph = nx.Graph()
    ordered = edge_table.sort_values(["source_id", "target_id", "relation"], kind="stable")
    for row in ordered.itertuples(index=False):
        relation = str(row.relation)
        if relation not in costs or costs[relation] <= 0:
            raise ValueError(f"no positive cost configured for relation {relation!r}")
        graph.add_edge(
            str(row.source_id),
            str(row.target_id),
            relation=relation,
            weight=float(costs[relation]),
            source_vocabulary=str(row.source_vocabulary),
            vocabulary_version=str(row.vocabulary_version),
            provenance=str(row.provenance),
        )
    return graph


def derive_attribute_anchor_weights(
    graph: nx.Graph,
    attribute_nodes: Mapping[str, str],
    anchor_nodes: Mapping[str, str],
    *,
    distance_decay: float = 1.0,
) -> pd.DataFrame:
    """Convert public shortest-path distances to normalized attribute weights."""
    if distance_decay <= 0:
        raise ValueError("distance_decay must be positive")
    rows: list[dict[str, float | str]] = []
    for attribute, attribute_node in sorted(attribute_nodes.items()):
        if attribute_node not in graph:
            raise ValueError(f"attribute node {attribute_node!r} is absent from the graph")
        distances: list[tuple[str, float]] = []
        for anchor, anchor_node in sorted(anchor_nodes.items()):
            if anchor_node not in graph:
                raise ValueError(f"anchor node {anchor_node!r} is absent from the graph")
            try:
                distance = nx.shortest_path_length(graph, attribute_node, anchor_node, weight="weight")
            except nx.NetworkXNoPath:
                continue
            distances.append((anchor, float(distance)))
        if not distances:
            raise ValueError(f"attribute {attribute!r} has no path to a semantic anchor")
        raw = np.asarray([np.exp(-distance_decay * distance) for _, distance in distances])
        normalized = raw / raw.sum()
        for (anchor, distance), weight in zip(distances, normalized):
            rows.append(
                {
                    "attribute_name": attribute,
                    "semantic_anchor": anchor,
                    "graph_distance": distance,
                    "graph_weight": float(weight),
                }
            )
    return pd.DataFrame(rows)


def derive_primary_anchor_assignments(
    graph: nx.Graph,
    attribute_nodes: Mapping[str, str],
    anchor_nodes: Mapping[str, str],
    *,
    distance_decay: float = 1.0,
) -> pd.DataFrame:
    """Assign each attribute to its nearest public semantic anchor."""
    if distance_decay <= 0:
        raise ValueError("distance_decay must be positive")
    rows: list[dict[str, float | str]] = []
    for attribute, attribute_node in sorted(attribute_nodes.items()):
        choices: list[tuple[float, str, str]] = []
        for anchor, anchor_node in sorted(anchor_nodes.items()):
            try:
                distance = float(nx.shortest_path_length(graph, attribute_node, anchor_node, weight="weight"))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            choices.append((distance, anchor, anchor_node))
        if not choices:
            raise ValueError(f"attribute {attribute!r} has no path to a semantic anchor")
        distance, anchor, anchor_node = min(choices, key=lambda item: (item[0], item[1]))
        rows.append(
            {
                "attribute_name": attribute,
                "semantic_anchor": anchor,
                "anchor_node": anchor_node,
                "graph_distance": distance,
                "graph_weight": float(np.exp(-distance_decay * distance)),
            }
        )
    return pd.DataFrame(rows)


def validate_graph_provenance(edge_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize version and provenance coverage for release QA."""
    missing = EDGE_COLUMNS.difference(edge_table.columns)
    if missing:
        raise ValueError(f"edge table is missing columns: {sorted(missing)}")
    return (
        edge_table.groupby(["source_vocabulary", "vocabulary_version", "provenance"], dropna=False)
        .size()
        .rename("n_edges")
        .reset_index()
        .sort_values(["source_vocabulary", "vocabulary_version", "provenance"])
    )
