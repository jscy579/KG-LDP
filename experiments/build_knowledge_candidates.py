"""Generate review candidates from pinned LOINC and OBO resources.

The script never auto-approves a medical mapping.  It writes ranked candidates
that must be reviewed before they can enter ``formal_attribute_mapping.csv``.
"""

from __future__ import annotations

import argparse
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def loinc_candidates(attributes: pd.DataFrame, loinc_csv: Path, top_k: int) -> pd.DataFrame:
    loinc = pd.read_csv(loinc_csv, dtype=str, low_memory=False)
    name_columns = [column for column in ["LONG_COMMON_NAME", "COMPONENT", "SHORTNAME"] if column in loinc]
    if "LOINC_NUM" not in loinc or not name_columns:
        raise ValueError("LOINC table lacks LOINC_NUM or searchable name columns")
    if "STATUS" in loinc:
        loinc = loinc.loc[loinc["STATUS"].astype(str).str.upper().eq("ACTIVE")].reset_index(drop=True)
    searchable = loinc[name_columns].fillna("").agg(" | ".join, axis=1)
    normalized_fields = {column: loinc[column].fillna("").map(normalize) for column in name_columns}
    rows = []
    for attribute in attributes.itertuples(index=False):
        query = normalize(
            getattr(attribute, "query_text", "")
            or getattr(attribute, "canonical_name", "")
            or getattr(attribute, "attribute_name")
        )
        canonical = normalize(getattr(attribute, "canonical_name", "") or getattr(attribute, "attribute_name"))
        tokens = [token for token in canonical.split() if len(token) >= 3]
        component = normalized_fields.get("COMPONENT", pd.Series("", index=loinc.index))
        candidate_mask = component.map(lambda value: any(token in value for token in tokens))
        candidate_index = loinc.index[candidate_mask]
        if len(candidate_index) < top_k:
            candidate_index = loinc.index
        scores = pd.Series(0.0, index=candidate_index)
        for column, normalized_values in normalized_fields.items():
            field_scores = normalized_values.loc[candidate_index].map(
                lambda candidate: SequenceMatcher(None, query, candidate).ratio()
            )
            scores = pd.concat([scores, field_scores], axis=1).max(axis=1)
        component_values = normalized_fields.get("COMPONENT", pd.Series("", index=loinc.index)).loc[candidate_index]
        scores += component_values.eq(canonical).astype(float) * 0.35
        system_hint = normalize(getattr(attribute, "system_hint", ""))
        if system_hint and "SYSTEM" in loinc:
            normalized_system = loinc.loc[candidate_index, "SYSTEM"].fillna("").map(normalize)
            if system_hint == "whole blood":
                system_match = normalized_system.isin({"bld", "blood", "bldv", "blda", "bldc"})
            elif system_hint == "serum or plasma":
                system_match = normalized_system.str.contains("serpl|ser plas|serum|plasma", regex=True)
            else:
                system_match = normalized_system.map(lambda value: system_hint in value)
            scores += system_match.astype(float) * 0.25
        requested_unit = normalize(getattr(attribute, "ucum_unit", ""))
        if requested_unit and "EXAMPLE_UCUM_UNITS" in loinc:
            candidate_units = loinc.loc[candidate_index, "EXAMPLE_UCUM_UNITS"].fillna("").map(normalize)
            scores += candidate_units.map(lambda value: requested_unit == value).astype(float) * 0.2
        for index in scores.nlargest(top_k).index:
            rows.append(
                {
                    "dataset_id": getattr(attribute, "dataset_id"),
                    "attribute_name": getattr(attribute, "attribute_name"),
                    "query": query,
                    "candidate_loinc_code": loinc.loc[index, "LOINC_NUM"],
                    "candidate_name": searchable.loc[index],
                    "candidate_class": loinc.loc[index, "CLASS"] if "CLASS" in loinc else "",
                    "candidate_property": loinc.loc[index, "PROPERTY"] if "PROPERTY" in loinc else "",
                    "candidate_system": loinc.loc[index, "SYSTEM"] if "SYSTEM" in loinc else "",
                    "candidate_example_ucum_units": loinc.loc[index, "EXAMPLE_UCUM_UNITS"] if "EXAMPLE_UCUM_UNITS" in loinc else "",
                    "candidate_status": loinc.loc[index, "STATUS"] if "STATUS" in loinc else "",
                    "similarity": float(scores.loc[index]),
                    "reviewer_status": "needs_review",
                }
            )
    return pd.DataFrame(rows)


def parse_obo_terms(path: Path) -> pd.DataFrame:
    rows = []
    current: dict[str, object] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == "[Term]":
            if current.get("id") and current.get("name"):
                rows.append(current)
            current = {"synonyms": []}
        elif line.startswith("id: "):
            current["id"] = line[4:]
        elif line.startswith("name: "):
            current["name"] = line[6:]
        elif line.startswith("synonym: "):
            match = re.match(r'synonym: "([^"]+)"', line)
            if match:
                current.setdefault("synonyms", []).append(match.group(1))
    if current.get("id") and current.get("name"):
        rows.append(current)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"no OBO terms parsed from {path}")
    frame["search_text"] = frame.apply(lambda row: " | ".join([row["name"], *row["synonyms"]]), axis=1)
    return frame


def obo_candidates(queries: pd.DataFrame, ontology: pd.DataFrame, ontology_name: str, top_k: int) -> pd.DataFrame:
    prefix = "HP:" if ontology_name == "HPO" else "UBERON:"
    ontology = ontology.loc[ontology["id"].astype(str).str.startswith(prefix)].reset_index(drop=True)
    normalized = ontology["search_text"].map(normalize)
    normalized_names = ontology["name"].map(normalize)
    rows = []
    for query in queries.itertuples(index=False):
        search = normalize(getattr(query, "query_text"))
        scores = normalized.map(lambda candidate: SequenceMatcher(None, search, candidate).ratio())
        scores += normalized_names.eq(search).astype(float)
        for index in scores.nlargest(top_k).index:
            rows.append(
                {
                    "dataset_id": getattr(query, "dataset_id", ""),
                    "attribute_name": getattr(query, "attribute_name"),
                    "query_role": getattr(query, "query_role", ""),
                    "query": search,
                    "ontology": ontology_name,
                    "candidate_id": ontology.loc[index, "id"],
                    "candidate_name": ontology.loc[index, "name"],
                    "similarity": float(scores.loc[index]),
                    "reviewer_status": "needs_review",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attributes", default="configs/formal_attribute_mapping_template.csv")
    parser.add_argument("--loinc", default="data/knowledge/loinc/2.82/LoincTable/Loinc.csv")
    parser.add_argument("--hpo", default="data/knowledge/hpo/hp.obo")
    parser.add_argument("--uberon", default="data/knowledge/uberon/uberon.obo")
    parser.add_argument("--output-dir", default="data/interim/knowledge_candidates")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--loinc-only", action="store_true", help="Generate LOINC candidates without requiring OBO files")
    args = parser.parse_args()
    attributes = pd.read_csv(args.attributes)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loinc_candidates(attributes, Path(args.loinc), args.top_k).to_csv(output / "loinc_candidates.csv", index=False)
    if args.loinc_only:
        return
    dataset = attributes["dataset_id"].astype(str)
    canonical = attributes["canonical_name"].fillna(attributes["attribute_name"]).astype(str)
    low_queries = attributes.get("hpo_low_query", "Low " + canonical + " level")
    high_queries = attributes.get("hpo_high_query", "High " + canonical + " level")
    hpo_queries = pd.concat(
        [
            pd.DataFrame({"dataset_id": dataset, "attribute_name": attributes["attribute_name"], "query_role": "low", "query_text": low_queries}),
            pd.DataFrame({"dataset_id": dataset, "attribute_name": attributes["attribute_name"], "query_role": "high", "query_text": high_queries}),
        ],
        ignore_index=True,
    )
    anatomy_query = attributes.get("anatomy_query", pd.Series("blood", index=attributes.index)).fillna("blood")
    uberon_queries = pd.DataFrame(
        {
            "dataset_id": dataset,
            "attribute_name": attributes["attribute_name"],
            "query_role": "observation_system",
            "query_text": anatomy_query,
        }
    )
    hpo = parse_obo_terms(Path(args.hpo))
    uberon = parse_obo_terms(Path(args.uberon))
    obo_candidates(hpo_queries, hpo, "HPO", args.top_k).to_csv(
        output / "hpo_candidates.csv", index=False
    )
    obo_candidates(uberon_queries, uberon, "UBERON", args.top_k).to_csv(
        output / "uberon_candidates.csv", index=False
    )


if __name__ == "__main__":
    main()
