"""Write a versioned resource manifest without redistributing source packages."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESOURCES = [
    ("LOINC table", "2.82", "data/knowledge/loinc/2.82/LoincTable/Loinc.csv", "https://loinc.org/downloads/", "https://loinc.org/kb/license/"),
    ("LOINC Groups", "2.82", "data/knowledge/loinc/2.82/AccessoryFiles/GroupFile/Group.csv", "https://loinc.org/groups/", "https://loinc.org/kb/license/"),
    ("LOINC Group terms", "2.82", "data/knowledge/loinc/2.82/AccessoryFiles/GroupFile/GroupLoincTerms.csv", "https://loinc.org/groups/", "https://loinc.org/kb/license/"),
    ("LOINC Parent Groups", "2.82", "data/knowledge/loinc/2.82/AccessoryFiles/GroupFile/ParentGroup.csv", "https://loinc.org/groups/", "https://loinc.org/kb/license/"),
    ("LOINC license", "5.8", "data/knowledge/loinc/2.82/LoincLicense_5.8.txt", "https://loinc.org/kb/license/", "https://loinc.org/kb/license/"),
    ("Human Phenotype Ontology", "2026-02-16", "data/knowledge/hpo/hp.obo", "https://github.com/obophenotype/human-phenotype-ontology/releases/tag/v2026-02-16", "https://hpo.jax.org/app/license"),
    ("UBERON", "2024-11-25", "data/knowledge/uberon/uberon.obo", "https://purl.obolibrary.org/obo/uberon/releases/2024-11-25/uberon.obo", "https://github.com/obophenotype/uberon/blob/master/LICENSE"),
    ("NHANES demographics", "2021-2023", "data/raw/nhanes_2021_2023/DEMO_L.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.htm", "https://www.cdc.gov/other/agencymaterials.html"),
    ("NHANES complete blood count", "2021-2023", "data/raw/nhanes_2021_2023/CBC_L.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/CBC_L.htm", "https://www.cdc.gov/other/agencymaterials.html"),
    ("NHANES biochemistry profile", "2021-2023", "data/raw/nhanes_2021_2023/BIOPRO_L.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BIOPRO_L.htm", "https://www.cdc.gov/other/agencymaterials.html"),
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    rows = []
    for name, version, relative, source, license_url in RESOURCES:
        path = PROJECT_ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"resource is absent or empty: {path}")
        rows.append(
            {
                "name": name,
                "version": version,
                "project_relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": digest(path),
                "source_url": source,
                "license_url": license_url,
                "redistribution_policy": "do_not_redistribute" if name.startswith("LOINC") else "follow_source_license",
            }
        )
    output = PROJECT_ROOT / "data/knowledge/resource_manifest.json"
    output.write_text(
        json.dumps(
            {"project_id": "SemMed-LDP", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "resources": rows},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} verified resources to {output}")


if __name__ == "__main__":
    main()
