"""Assemble candidate mappings and provenance edges without approving them."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import networkx as nx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANCHORS = {
    "sodium": ("blood_ion_homeostasis", "HP:0003111"),
    "potassium": ("blood_ion_homeostasis", "HP:0003111"),
    "chloride": ("blood_ion_homeostasis", "HP:0003111"),
    "ionized_calcium": ("blood_ion_homeostasis", "HP:0003111"),
    "total_calcium": ("blood_ion_homeostasis", "HP:0003111"),
    "magnesium": ("blood_ion_homeostasis", "HP:0003111"),
    "phosphate": ("blood_ion_homeostasis", "HP:0003111"),
    "creatinine": ("circulating_nitrogen_compounds", "HP:0004364"),
    "urea_nitrogen": ("circulating_nitrogen_compounds", "HP:0004364"),
    "glucose": ("metabolism_homeostasis", "HP:0001939"),
    "albumin": ("metabolism_homeostasis", "HP:0001939"),
    "hemoglobin": ("blood_and_blood_forming_tissues", "HP:0001871"),
    "platelet_count": ("blood_and_blood_forming_tissues", "HP:0001871"),
}

LOINC_OVERRIDES = {
    ("PIC", "glucose"): (
        "2339-0",
        "Glucose [Mass/volume] in Blood",
        "PIC source is a blood glucose electrode result in mmol/L; code represents the standardized post-conversion mg/dL value",
    ),
    ("PIC", "platelet_count"): (
        "777-3",
        "Platelets [#/volume] in Blood by Automated count",
        "PIC D_LABITEMS explicitly records local LOINC_CODE 777-3",
    ),
    ("NHANES_2021_2023", "platelet_count"): (
        "777-3",
        "Platelets [#/volume] in Blood by Automated count",
        "NHANES CBC documentation specifies an automated hematology analyzer",
    ),
}


def parse_hpo(path: Path) -> tuple[nx.DiGraph, dict[str, str], list[tuple[str, str]]]:
    graph = nx.DiGraph()
    names: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    current: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw == "[Term]":
            current = None
        elif raw.startswith("id: HP:"):
            current = raw[4:]
            graph.add_node(current)
        elif current and raw.startswith("name: "):
            names[current] = raw[6:]
        elif current and raw.startswith("is_a: HP:"):
            parent = raw.split()[1]
            graph.add_edge(current, parent)
            edges.append((current, parent))
    return graph, names, edges


def primary_loinc_groups(codes: pd.Series) -> pd.DataFrame:
    group_dir = PROJECT_ROOT / "data/knowledge/loinc/2.82/AccessoryFiles/GroupFile"
    terms = pd.read_csv(group_dir / "GroupLoincTerms.csv", dtype=str)
    groups = pd.read_csv(group_dir / "Group.csv", dtype=str)
    joined = terms.merge(groups[["GroupId", "ParentGroupId", "Group"]], on="GroupId", how="left")
    joined = joined.loc[joined["LoincNumber"].isin(set(codes))].copy()
    joined["priority"] = joined["ParentGroupId"].map({"LG100-4": 0, "LG27-5": 1, "LG55-6": 2}).fillna(3)
    return joined.sort_values(["LoincNumber", "priority", "GroupId"]).drop_duplicates("LoincNumber")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pic-raw-dir",
        default="E:/research/data/pic_raw_data/paediatic-intensive-care-database-1.2.0",
    )
    args = parser.parse_args()
    mapping_path = PROJECT_ROOT / "configs/formal_attribute_mapping_working.csv"
    mapping = pd.read_csv(mapping_path, dtype=str).fillna("")
    evidence_columns = [
        "source_dictionary_label", "source_dictionary_fluid", "source_dictionary_category",
        "source_dictionary_loinc", "source_documentation_url", "mapping_issue", "mapping_confidence_basis",
    ]
    for column in evidence_columns:
        if column not in mapping:
            mapping[column] = ""
    candidate_dir = PROJECT_ROOT / "data/interim/knowledge_candidates"
    loinc = pd.read_csv(candidate_dir / "loinc_candidates.csv", dtype=str).groupby(
        ["dataset_id", "attribute_name"], sort=False
    ).head(1)
    hpo = pd.read_csv(candidate_dir / "hpo_candidates.csv", dtype=str).groupby(
        ["dataset_id", "attribute_name", "query_role"], sort=False
    ).head(1)
    uberon = pd.read_csv(candidate_dir / "uberon_candidates.csv", dtype=str).groupby(
        ["dataset_id", "attribute_name"], sort=False
    ).head(1)

    for (dataset_id, attribute_name), (code, label, issue) in LOINC_OVERRIDES.items():
        match = loinc["dataset_id"].eq(dataset_id) & loinc["attribute_name"].eq(attribute_name)
        loinc.loc[match, "candidate_loinc_code"] = code
        loinc.loc[match, "candidate_name"] = label
        loinc.loc[match, "similarity"] = "1.0"
        mapping_match = mapping["dataset_id"].eq(dataset_id) & mapping["attribute_name"].eq(attribute_name)
        mapping.loc[mapping_match, "mapping_issue"] = issue

    pic_dictionary_path = Path(args.pic_raw_dir) / "D_LABITEMS.csv"
    if pic_dictionary_path.exists():
        dictionary = pd.read_csv(pic_dictionary_path, dtype=str).fillna("").set_index("ITEMID")
        for index, row in mapping.loc[mapping["dataset_id"].eq("PIC")].iterrows():
            source_id = str(row["source_variable_id"])
            if source_id in dictionary.index:
                source = dictionary.loc[source_id]
                mapping.loc[index, "source_dictionary_label"] = source.get("LABEL", "")
                mapping.loc[index, "source_dictionary_fluid"] = source.get("FLUID", "")
                mapping.loc[index, "source_dictionary_category"] = source.get("CATEGORY", "")
                mapping.loc[index, "source_dictionary_loinc"] = source.get("LOINC_CODE", "")

    mapping.loc[mapping["dataset_id"].eq("PIC"), "source_documentation_url"] = "PIC D_LABITEMS.csv; https://physionet.org/content/picdb/1.1.0/"
    mapping.loc[mapping["dataset_id"].eq("NHANES_2021_2023") & mapping["source_table"].eq("CBC_L"), "source_documentation_url"] = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/CBC_L.htm"
    mapping.loc[mapping["dataset_id"].eq("NHANES_2021_2023") & mapping["source_table"].eq("BIOPRO_L"), "source_documentation_url"] = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BIOPRO_L.htm"

    loinc_lookup = loinc.set_index(["dataset_id", "attribute_name"])
    hpo_lookup = hpo.set_index(["dataset_id", "attribute_name", "query_role"])
    uberon_lookup = uberon.set_index(["dataset_id", "attribute_name"])
    group_lookup = primary_loinc_groups(loinc["candidate_loinc_code"]).set_index("LoincNumber")
    hpo_graph, hpo_names, hpo_edges = parse_hpo(PROJECT_ROOT / "data/knowledge/hpo/hp.obo")

    graph_rows: list[dict[str, object]] = []
    used_hpo_nodes: set[str] = set()
    for index, row in mapping.iterrows():
        key = (row["dataset_id"], row["attribute_name"])
        lrow = loinc_lookup.loc[key]
        low = hpo_lookup.loc[(*key, "low")]
        high = hpo_lookup.loc[(*key, "high")]
        anatomy = uberon_lookup.loc[key]
        code = str(lrow["candidate_loinc_code"])
        group = group_lookup.loc[code]
        anchor_label, anchor_id = ANCHORS[row["attribute_name"]]
        distances = []
        for phenotype_id in [str(low["candidate_id"]), str(high["candidate_id"])]:
            used_hpo_nodes.add(phenotype_id)
            used_hpo_nodes.update(nx.descendants(hpo_graph, phenotype_id))
            distances.append(nx.shortest_path_length(hpo_graph, phenotype_id, anchor_id))
        mean_distance = float(sum(distances) / len(distances))
        mapping.loc[index, "loinc_code"] = code
        mapping.loc[index, "loinc_label"] = lrow["candidate_name"].split(" | ")[0]
        mapping.loc[index, "loinc_group_id"] = group["GroupId"]
        mapping.loc[index, "hpo_low_id"] = low["candidate_id"]
        mapping.loc[index, "hpo_high_id"] = high["candidate_id"]
        mapping.loc[index, "uberon_id"] = anatomy["candidate_id"]
        mapping.loc[index, "semantic_anchor"] = anchor_label
        mapping.loc[index, "graph_weight"] = math.exp(-mean_distance)
        confidence = 0.95 if row["dataset_id"] == "NHANES_2021_2023" else 0.80
        confidence_basis = "official variable documentation plus terminology match; manual review required"
        dictionary_code = str(mapping.loc[index, "source_dictionary_loinc"]).strip()
        if dictionary_code and dictionary_code == code:
            confidence = 0.98
            confidence_basis = "exact PIC dictionary LOINC plus terminology match; manual review required"
        if key == ("PIC", "potassium"):
            confidence = 0.65
            confidence_basis = "strong analyte match but PIC fluid field says Other Body Fluid"
        elif key == ("PIC", "ionized_calcium"):
            confidence = 0.65
            confidence_basis = "Chinese label and scale support ionized calcium, but English label says total calcium"
        elif key == ("PIC", "glucose"):
            confidence = 0.75
            confidence_basis = "blood glucose match; selected code describes the standardized post-conversion value"
        mapping.loc[index, "mapping_confidence"] = confidence
        mapping.loc[index, "mapping_confidence_basis"] = confidence_basis
        mapping.loc[index, "provenance"] = "ranked_candidates+LOINC_Groups+HPO_ancestor_path; manual_review_required"

        attr_id = f'ATTR:{row["dataset_id"]}:{row["attribute_name"]}'
        edge_specs = [
            (attr_id, f"LOINC:{code}", "mapped_to_loinc", "LOINC", "2.82", "ranked candidate; manual review required"),
            (f"LOINC:{code}", f'LOINC_GROUP:{group["GroupId"]}', "member_of_group", "LOINC", "2.82", "GroupLoincTerms.csv"),
            (f'LOINC_GROUP:{group["GroupId"]}', f'LOINC_PARENT:{group["ParentGroupId"]}', "parent_group", "LOINC", "2.82", "Group.csv"),
            (f"LOINC:{code}", str(low["candidate_id"]), "has_abnormal_phenotype", "HPO", "2026-02-16", "controlled term candidate; manual review required"),
            (f"LOINC:{code}", str(high["candidate_id"]), "has_abnormal_phenotype", "HPO", "2026-02-16", "controlled term candidate; manual review required"),
            (f"LOINC:{code}", str(anatomy["candidate_id"]), "has_anatomical_system", "UBERON", "2024-11-25", "specimen-system candidate; manual review required"),
            (anchor_id, f"ANCHOR:{anchor_label}", "anchored_to", "SemMed-LDP", "1.0", "prespecified task anchor"),
        ]
        for source, target, relation, vocabulary, version, provenance in edge_specs:
            graph_rows.append(
                {
                    "source_id": source,
                    "target_id": target,
                    "relation": relation,
                    "source_vocabulary": vocabulary,
                    "vocabulary_version": version,
                    "provenance": provenance,
                    "reviewer_status": "needs_review" if "candidate" in provenance else "source_asserted",
                }
            )

    for child, parent in hpo_edges:
        if child in used_hpo_nodes and parent in used_hpo_nodes:
            graph_rows.append(
                {
                    "source_id": child,
                    "target_id": parent,
                    "relation": "subclass_of",
                    "source_vocabulary": "HPO",
                    "vocabulary_version": "2026-02-16",
                    "provenance": "hp.obo",
                    "reviewer_status": "source_asserted",
                }
            )

    mapping.to_csv(mapping_path, index=False)
    edge_frame = pd.DataFrame(graph_rows).drop_duplicates(
        ["source_id", "target_id", "relation", "source_vocabulary", "vocabulary_version"]
    )
    edge_frame.to_csv(PROJECT_ROOT / "configs/formal_graph_edges_working.csv", index=False)
    anchor_rows = [
        {"semantic_anchor": label, "anchor_node": node, "anchor_label": hpo_names.get(node, "")}
        for label, node in sorted(set(ANCHORS.values()))
    ]
    pd.DataFrame(anchor_rows).to_csv(PROJECT_ROOT / "configs/semantic_anchors.csv", index=False)
    print(f"Updated {len(mapping)} candidate mapping rows and wrote {len(edge_frame)} provenance edges")


if __name__ == "__main__":
    main()
