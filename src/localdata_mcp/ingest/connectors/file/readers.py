"""localdata_mcp/ingest/connectors/file/readers.py — the 14-format table (E8.2).

I-2's reader registry over the §7.2 core formats: CSV, TSV, JSON,
YAML, TOML, INI, XML, Excel (.xlsx/.xls), ODS, Numbers, Parquet,
Feather, Arrow, HDF5 — one reader per format, dispatched by declared
name or suffix. Tabular formats return a DataFrame; document formats
(YAML/TOML/INI/JSON-object, XML) return their natural nested-mapping
shape (I-2 allows tree shapes). The NFR-107 hardening is IN the
readers, not around them:

- **YAML** parses with `SafeLoader` exclusively AND the loaded tree is
  scanned whole for any non-plain primitive (belt and braces — a
  loader regression cannot silently admit objects).
- **XML** parses through `defusedxml` on every path: DOCTYPE, entity
  resolution, and external references are structurally impossible
  (XXE/SSRF closure) — never a "hardened flag on lxml" that a
  refactor could drop.
- **HDF5** refuses any file carrying an external link or a virtual
  dataset — the file's own binary content can never direct `h5py` to
  open a second path outside `path_contain`'s verdict (I-2's channel
  closure; refusal is the observable the battery asserts).

Containment (NFR-108) is the TOOL's step before any reader runs
(tools.py crosses the guard's contain_path) — readers assume a
contained real path and do no security beyond their own format's
hardening. Neighbors: tools.py dispatches here; the inventory registry
records each format's streaming classification (E8.4).
"""

from __future__ import annotations

import configparser
import datetime
import io
import json
import tomllib
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd
import yaml
from defusedxml import ElementTree as DefusedElementTree


class FileIngestError(ValueError):
    """A reader refused its file — unknown format, malformed content,
    or a hardening violation; shaped through NX-3 by the wrapper."""


# -- format table -----------------------------------------------------

_SUFFIX_TO_FORMAT: Mapping[str, str] = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".xml": "xml",
    ".xlsx": "xlsx",
    ".xls": "xls",
    ".ods": "ods",
    ".numbers": "numbers",
    ".parquet": "parquet",
    ".feather": "feather",
    ".arrow": "arrow",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
}


def resolve_format(path: Path, declared: str) -> str:
    """The effective format: the declared name, or the suffix when
    `auto` — an unknown answer is a refusal, never a guess."""
    if declared != "auto":
        if declared not in _READERS:
            raise FileIngestError(
                f"unknown format {declared!r} — supported: {sorted(_READERS)}"
            )
        return declared
    suffix_format = _SUFFIX_TO_FORMAT.get(path.suffix.lower())
    if suffix_format is None:
        raise FileIngestError(
            f"cannot infer a format from suffix {path.suffix!r} — pass "
            f"format explicitly (supported: {sorted(_READERS)})"
        )
    return suffix_format


def read_path(path: Path, format_name: str) -> Any:
    """Dispatch to the one reader for `format_name` (already resolved)."""
    return _READERS[format_name](path)


# -- tabular readers --------------------------------------------------


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def _read_xlsx(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl")


def _read_xls(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, engine="xlrd")


def _read_ods(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, engine="odf")


def _read_numbers(path: Path) -> pd.DataFrame:
    from numbers_parser import Document

    table = Document(str(path)).sheets[0].tables[0]
    rows = table.rows(values_only=True)
    if not rows:
        return pd.DataFrame()
    # Numbers tables are padded to a fixed grid, so trailing columns
    # with an empty header carry no data — trim to the named columns.
    header = list(rows[0])
    width = len(header)
    while width > 0 and (header[width - 1] is None or header[width - 1] == ""):
        width -= 1
    columns = [str(cell) for cell in header[:width]]
    body = [row[:width] for row in rows[1:]]
    return pd.DataFrame(body, columns=columns)


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


def _read_feather(path: Path) -> pd.DataFrame:
    return pd.read_feather(path)


def _read_arrow(path: Path) -> pd.DataFrame:
    import pyarrow.ipc

    with pyarrow.ipc.open_file(str(path)) as reader:
        return reader.read_all().to_pandas()


# -- document readers -------------------------------------------------


def _read_json(path: Path) -> Any:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return _tabular_if_records(loaded)


def _read_yaml(path: Path) -> Any:
    """SafeLoader exclusively, then the whole-tree primitive scan
    (NFR-107): nothing outside the plain-data vocabulary survives."""
    loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.SafeLoader)
    _refuse_non_plain(loaded, "yaml document")
    return _tabular_if_records(loaded)


def _read_toml(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _read_ini(path: Path) -> Mapping[str, Any]:
    parser = configparser.ConfigParser()
    parser.read_string(path.read_text(encoding="utf-8"))
    return {section: dict(parser.items(section)) for section in parser.sections()}


def _read_xml(path: Path) -> Mapping[str, Any]:
    """defusedxml on every path: a DOCTYPE or entity refuses at parse
    (XXE/SSRF closed structurally); the tree comes back as a nested
    mapping (I-2's tree shape)."""
    try:
        root = DefusedElementTree.parse(str(path)).getroot()
    except Exception as failure:
        raise FileIngestError(
            f"XML refused or malformed (hardened parse): {failure}"
        ) from failure
    return {_local_tag(root.tag): _element_to_mapping(root)}


def _read_hdf5(path: Path) -> Any:
    """h5py with the file-content-directed channel CLOSED: an external
    link or virtual dataset anywhere refuses the whole file (I-2)."""
    import h5py

    with h5py.File(path, "r") as handle:
        _refuse_hdf5_indirection(handle, handle.name)
        datasets: dict[str, Any] = {}

        def collect(name: str, item: Any) -> None:
            if isinstance(item, h5py.Dataset):
                datasets[name] = item[()].tolist()

        handle.visititems(collect)
    if len(datasets) == 1:
        only = next(iter(datasets.values()))
        frame = pd.DataFrame(only)
        return frame
    return datasets


def _refuse_hdf5_indirection(group: Any, base: str) -> None:
    import h5py

    for key in group.keys():
        link = group.get(key, getlink=True)
        if isinstance(link, h5py.ExternalLink):
            raise FileIngestError(
                f"HDF5 external link at {base}/{key} — refused (NFR-107: "
                "file content cannot direct reads to a second path)"
            )
        item = group.get(key)
        if isinstance(item, h5py.Dataset) and item.is_virtual:
            raise FileIngestError(
                f"HDF5 virtual dataset at {base}/{key} — refused (NFR-107)"
            )
        if isinstance(item, h5py.Group):
            _refuse_hdf5_indirection(item, f"{base}/{key}")


# -- shared helpers ---------------------------------------------------

_PLAIN_TYPES = (
    type(None),
    bool,
    int,
    float,
    str,
    datetime.date,
    datetime.datetime,
)


def _refuse_non_plain(value: Any, where: str) -> None:
    """The NFR-107 whole-tree scan: only plain data survives."""
    if isinstance(value, dict):
        for key, entry in value.items():
            _refuse_non_plain(key, where)
            _refuse_non_plain(entry, where)
    elif isinstance(value, (list, tuple)):
        for entry in value:
            _refuse_non_plain(entry, where)
    elif not isinstance(value, _PLAIN_TYPES):
        raise FileIngestError(
            f"forbidden primitive {type(value).__name__} inside {where} — "
            "refused (NFR-107)"
        )


def _tabular_if_records(loaded: Any) -> Any:
    """A top-level list of flat mappings is tabular data; anything
    else keeps its document shape."""
    if (
        isinstance(loaded, list)
        and loaded
        and all(isinstance(entry, Mapping) for entry in loaded)
    ):
        return pd.DataFrame(loaded)
    return loaded


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _element_to_mapping(element: Any) -> Any:
    """One XML element as plain data: attributes under `@name`, text
    under `#text`, children nested by tag (repeated tags → list)."""
    node: dict[str, Any] = {f"@{k}": v for k, v in element.attrib.items()}
    text = (element.text or "").strip()
    children = list(element)
    if not children:
        return text if not node else {**node, "#text": text} if text else node
    for child in children:
        key = _local_tag(child.tag)
        rendered = _element_to_mapping(child)
        if key in node:
            existing = node[key]
            node[key] = (
                existing + [rendered]
                if isinstance(existing, list)
                else [
                    existing,
                    rendered,
                ]
            )
        else:
            node[key] = rendered
    if text:
        node["#text"] = text
    return node


_READERS: Mapping[str, Callable[[Path], Any]] = {
    "csv": _read_csv,
    "tsv": _read_tsv,
    "json": _read_json,
    "yaml": _read_yaml,
    "toml": _read_toml,
    "ini": _read_ini,
    "xml": _read_xml,
    "xlsx": _read_xlsx,
    "xls": _read_xls,
    "ods": _read_ods,
    "numbers": _read_numbers,
    "parquet": _read_parquet,
    "feather": _read_feather,
    "arrow": _read_arrow,
    "hdf5": _read_hdf5,
}
