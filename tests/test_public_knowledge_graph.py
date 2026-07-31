import pandas as pd

from semmed_ldp.public_knowledge_graph import (
    build_public_medical_graph,
    derive_attribute_anchor_weights,
    derive_primary_anchor_assignments,
)


def test_graph_weights_are_deterministic_and_normalized() -> None:
    edges = pd.DataFrame(
        [
            ["ATTR:sodium", "LOINC:2951-2", "mapped_to_loinc", "LOINC", "2.82", "test"],
            ["LOINC:2951-2", "ANCHOR:electrolyte", "anchored_to", "SemMed", "1", "test"],
            ["LOINC:2951-2", "ANCHOR:renal", "anchored_to", "SemMed", "1", "test"],
        ],
        columns=["source_id", "target_id", "relation", "source_vocabulary", "vocabulary_version", "provenance"],
    )
    graph = build_public_medical_graph(edges)
    weights = derive_attribute_anchor_weights(
        graph,
        {"sodium": "ATTR:sodium"},
        {"electrolyte": "ANCHOR:electrolyte", "renal": "ANCHOR:renal"},
    )
    assert weights["graph_weight"].sum() == 1.0
    assert weights["graph_weight"].tolist() == [0.5, 0.5]
    primary = derive_primary_anchor_assignments(
        graph,
        {"sodium": "ATTR:sodium"},
        {"electrolyte": "ANCHOR:electrolyte", "renal": "ANCHOR:renal"},
    )
    assert primary.loc[0, "semantic_anchor"] == "electrolyte"
    assert primary.loc[0, "graph_weight"] > 0
