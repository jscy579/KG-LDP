"""Build a lightweight clinical attribute--medical concept bipartite graph."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from semmed_ldp.umls_mapping import split_multi_value

LOGGER = logging.getLogger(__name__)

try:
    import networkx as nx
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback.
    nx = None


def build_semantic_bipartite_graph(attribute_config: pd.DataFrame):
    """Construct an attribute-concept bipartite graph from config metadata."""
    if nx is None:
        raise ModuleNotFoundError("networkx is required for graph objects. Install requirements.txt or use semantic_mapping_summary().")
    graph = nx.Graph()
    for _, row in attribute_config.iterrows():
        attribute = str(row.get("attribute_name", "")).strip()
        if not attribute:
            continue
        graph.add_node(attribute, node_type="clinical_attribute", source_table=row.get("source_table", ""))

        concepts = split_multi_value(row.get("clinical_concepts", ""))
        if not concepts:
            concepts = split_multi_value(row.get("umls_name", ""))
        if not concepts:
            concepts = split_multi_value(row.get("semantic_group", ""))

        for concept in concepts:
            graph.add_node(concept, node_type="medical_concept", umls_cui=row.get("umls_cui", ""))
            graph.add_edge(attribute, concept, relation="associated_with")

    LOGGER.info(
        "Built semantic bipartite graph with %d nodes and %d edges.",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph


def semantic_mapping_summary(attribute_config: pd.DataFrame) -> pd.DataFrame:
    """Create an edge-list style summary for attribute-concept associations."""
    rows: list[dict[str, Any]] = []
    for _, row in attribute_config.iterrows():
        attribute = str(row.get("attribute_name", "")).strip()
        groups = split_multi_value(row.get("semantic_group", ""))
        concepts = split_multi_value(row.get("clinical_concepts", ""))
        if not concepts:
            concepts = split_multi_value(row.get("umls_name", "")) or groups

        for concept in concepts:
            rows.append(
                {
                    "attribute_name": attribute,
                    "medical_concept": concept,
                    "semantic_groups": "|".join(groups),
                    "umls_cui": row.get("umls_cui", ""),
                    "source_table": row.get("source_table", ""),
                    "relation": "associated_with",
                }
            )
    summary = pd.DataFrame(rows)
    LOGGER.info("Created semantic mapping summary with %d attribute-concept edges.", len(summary))
    return summary


def save_semantic_mapping_summary(attribute_config: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    """Save ``semantic_mapping_summary.csv`` and return the summary frame."""
    summary = semantic_mapping_summary(attribute_config)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    LOGGER.info("Saved semantic mapping summary to %s.", output)
    return summary
