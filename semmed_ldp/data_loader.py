"""PIC data loading utilities for SemMed-LDP."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal environments.
    yaml = None

LOGGER = logging.getLogger(__name__)


PATIENT_ID_CANDIDATES = ["subject_id", "patient_id", "pid"]
ADMISSION_ID_CANDIDATES = ["hadm_id", "admission_id", "adm_id"]
ICU_ID_CANDIDATES = ["stay_id", "icustay_id"]
VARIABLE_ID_CANDIDATES = ["itemid", "item_id", "code", "icd_code", "icd10_code_cn", "spec_itemid", "org_itemid", "ab_itemid"]
NUMERIC_VALUE_CANDIDATES = ["valuenum", "value_num", "value", "chartvalue", "lab_value", "amount", "dilution_value"]
VALUE_CANDIDATES = ["value", "chartvalue", "lab_value", "drug_name", "drug", "interpretation"]
LABEL_CANDIDATES = ["label", "item_label", "name", "item_name", "long_title", "drug", "drug_name", "title", "label_cn", "title_cn"]


DEFAULT_TABLE_ALIASES = {
    "admission": "admissions",
    "admissions": "admissions",
    "patient": "patients",
    "patients": "patients",
    "icustay": "icustays",
    "icustays": "icustays",
    "chart": "chartevents",
    "chartevent": "chartevents",
    "chartevents": "chartevents",
    "vital": "chartevents",
    "vitals": "chartevents",
    "vital_signs": "chartevents",
    "lab": "labevents",
    "labs": "labevents",
    "laboratory": "labevents",
    "laboratory_tests": "labevents",
    "labevent": "labevents",
    "labevents": "labevents",
    "diagnosis": "diagnoses_icd",
    "diagnoses": "diagnoses_icd",
    "diagnoses_icd": "diagnoses_icd",
    "diagnosis_icd": "diagnoses_icd",
    "medication": "prescriptions",
    "medications": "prescriptions",
    "drug": "prescriptions",
    "drugs": "prescriptions",
    "prescription": "prescriptions",
    "prescriptions": "prescriptions",
    "inputevents": "inputevents",
    "input": "inputevents",
    "outputevents": "outputevents",
    "output": "outputevents",
    "microbiologyevents": "microbiologyevents",
    "microbiology": "microbiologyevents",
    "d_items": "d_items",
    "d_labitems": "d_labitems",
    "d_icd_diagnoses": "d_icd_diagnoses",
}


def infer_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first existing column that matches a candidate name.

    Matching is case-insensitive and ignores surrounding whitespace.
    """
    normalized = {str(column).strip().lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def _strip_yaml_comment(line: str) -> str:
    """Strip simple YAML comments outside quoted strings."""
    in_single = False
    in_double = False
    for idx, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def _parse_simple_yaml_scalar(value: str) -> Any:
    """Parse the small YAML scalar subset used by project configs."""
    value = value.strip()
    if value == "":
        return ""
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_simple_yaml_scalar(part.strip()) for part in inner.split(",")]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Load a minimal indentation-based YAML mapping.

    This fallback intentionally supports only the config subset used by this
    project: nested dictionaries, scalar values, and inline lists.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = _strip_yaml_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        if ":" not in content:
            continue
        key, value = content.split(":", 1)
        key = key.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_simple_yaml_scalar(value)
    return root


def dump_yaml_config(config: dict[str, Any], output_path: str | Path) -> None:
    """Write a readable YAML config, using PyYAML when available."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
        return

    def render_mapping(mapping: dict[str, Any], indent: int = 0) -> list[str]:
        lines: list[str] = []
        prefix = " " * indent
        for key, value in mapping.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.extend(render_mapping(value, indent + 2))
            elif isinstance(value, list):
                rendered = ", ".join(str(item) for item in value)
                lines.append(f"{prefix}{key}: [{rendered}]")
            else:
                lines.append(f"{prefix}{key}: {value}")
        return lines

    path.write_text("\n".join(render_mapping(config)) + "\n", encoding="utf-8")


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return an empty dict for blank files."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        text = handle.read()
    if yaml is not None:
        return yaml.safe_load(text) or {}
    LOGGER.warning("PyYAML is not installed; using a minimal YAML parser for %s.", path)
    return _load_simple_yaml(text)


class PICDataLoader:
    """Load PIC source tables from a path configuration file.

    The loader expects a YAML file with a top-level ``tables`` mapping. Each
    table entry should include a ``path`` and may include column-name metadata
    such as ``patient_id_col``, ``admission_id_col``, ``itemid_col``,
    ``value_col``, and ``value_numeric_col``.
    """

    def __init__(self, config_path: str | Path, project_root: str | Path | None = None) -> None:
        self.config_path = Path(config_path)
        self.project_root = Path(project_root) if project_root else self.config_path.parent.parent
        self.config = load_yaml_config(self.config_path)
        raw_data_dir = self.config.get("raw_data_dir", "")
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else self.project_root / "data" / "raw"
        self.tables_config = self.config.get("tables", {})
        LOGGER.info("Loaded PIC path config from %s with %d table entries.", self.config_path, len(self.tables_config))

    def normalize_table_name(self, table_name: str) -> str:
        """Normalize common table aliases to configured table names."""
        key = str(table_name).strip()
        lowered = key.lower()
        return DEFAULT_TABLE_ALIASES.get(lowered, key)

    def get_table_config(self, table_name: str) -> dict[str, Any]:
        """Return configuration for one source table."""
        normalized = self.normalize_table_name(table_name)
        if normalized not in self.tables_config:
            raise KeyError(f"Table '{table_name}' is not configured in {self.config_path}.")
        table_config = self.tables_config[normalized]
        if isinstance(table_config, str):
            return {"file": table_config}
        if isinstance(table_config, dict):
            return dict(table_config)
        raise TypeError(f"Table '{table_name}' config must be a file name string or a mapping.")

    def resolve_table_path(self, table_name: str) -> Path:
        """Resolve a configured table path under ``raw_data_dir`` when needed."""
        table_config = self.get_table_config(table_name)
        raw_path = table_config.get("path") or table_config.get("file") or table_config.get("filename")
        if not raw_path:
            raise ValueError(f"Table '{table_name}' has no path configured.")
        path = Path(raw_path)
        if path.is_absolute():
            return path
        raw_data_dir = self.raw_data_dir if self.raw_data_dir.is_absolute() else self.project_root / self.raw_data_dir
        return raw_data_dir / path

    def check_table_paths(self) -> pd.DataFrame:
        """Check configured CSV availability without loading table contents."""
        rows = []
        for table_name in self.tables_config:
            try:
                path = self.resolve_table_path(table_name)
                rows.append({"table_name": table_name, "path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0})
            except Exception as exc:  # pragma: no cover - defensive metadata path.
                rows.append({"table_name": table_name, "path": "", "exists": False, "size_bytes": 0, "warning": str(exc)})
        return pd.DataFrame(rows)

    def read_table(
        self,
        table_name: str,
        *,
        required: bool = False,
        usecols: list[str] | None = None,
        nrows: int | None = None,
        **read_csv_kwargs: Any,
    ) -> pd.DataFrame:
        """Read one configured CSV table.

        Missing optional files return an empty ``DataFrame`` and a warning.
        Set ``required=True`` to raise ``FileNotFoundError`` instead.
        """
        normalized = self.normalize_table_name(table_name)
        path = self.resolve_table_path(normalized)
        if not path.exists():
            message = f"Configured PIC table '{normalized}' does not exist: {path}"
            if required:
                raise FileNotFoundError(message)
            LOGGER.warning("%s. Returning an empty table.", message)
            return pd.DataFrame()

        kwargs = {"low_memory": False}
        kwargs.update(read_csv_kwargs)
        if usecols is not None:
            kwargs["usecols"] = usecols
        if nrows is not None:
            kwargs["nrows"] = nrows

        frame = pd.read_csv(path, **kwargs)
        LOGGER.info("Read PIC table '%s' from %s with %d rows and %d columns.", normalized, path, *frame.shape)
        return frame

    def read_tables(self, table_names: list[str] | None = None, *, required: bool = False) -> dict[str, pd.DataFrame]:
        """Read multiple configured PIC CSV tables into a dictionary."""
        names = table_names or list(self.tables_config)
        tables: dict[str, pd.DataFrame] = {}
        for name in names:
            normalized = self.normalize_table_name(name)
            tables[normalized] = self.read_table(normalized, required=required)
        LOGGER.info("Loaded %d PIC source tables.", len(tables))
        return tables

    def read_available_tables(self) -> dict[str, pd.DataFrame]:
        """Read only configured CSV tables whose paths currently exist."""
        available = []
        for table_name in self.tables_config:
            try:
                if self.resolve_table_path(table_name).exists():
                    available.append(table_name)
            except (KeyError, ValueError):
                LOGGER.warning("Skipping malformed table config for '%s'.", table_name)
        return self.read_tables(available, required=False)

    def table_metadata(self) -> pd.DataFrame:
        """Return a compact summary of configured table paths and availability."""
        rows = []
        for name in self.tables_config:
            cfg = self.get_table_config(name)
            path = self.resolve_table_path(name)
            rows.append(
                {
                    "table_name": name,
                    "path": str(path),
                    "exists": path.exists(),
                    "size_bytes": path.stat().st_size if path.exists() else 0,
                    "patient_id_col": cfg.get("patient_id_col", ""),
                    "admission_id_col": cfg.get("admission_id_col", ""),
                    "itemid_col": cfg.get("itemid_col", ""),
                    "value_col": cfg.get("value_col", ""),
                    "value_numeric_col": cfg.get("value_numeric_col", ""),
                }
            )
        return pd.DataFrame(rows)
