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
(tools.py crosses the guard's contain_path). The NFR-105 MEMORY gate is
this module's step: `read_path` crosses the chokepoint's admission seam
(`admit`, injected) BEFORE any whole-file materialization (CR-005) so a
decompression bomb inside allowed_paths cannot OOM the server. Three
admission regimes, by how the format's materialized size can be bounded
ahead of the read:

- CSV/TSV charge their growing residency chunk by chunk (a high-ratio
  file refused mid-read), behind a coarse `st_size` pre-gate that bounds
  a wide/newline-free first chunk (CR-031).
- Uncompressed whole-file formats (the text tree, legacy .xls) cross one
  upfront `st_size * factor` estimate — sound because the bytes are
  present on disk.
- Compressed containers (parquet/feather/arrow/HDF5, zip-packed
  xlsx/ods/numbers) cross an estimate read from the container's own
  declared LOGICAL shape (rows x column width, dataset shapes, archive
  uncompressed total), NEVER `st_size` — an on-disk-size estimate is
  defeatable by a dictionary/RLE or deflate bomb and would fail open
  (CR-029, `_logical_materialization`).

Otherwise readers assume a contained real path and do no security beyond
their own format's hardening. Neighbors: tools.py dispatches here; the
inventory registry records each format's streaming classification (E8.4).
"""

from __future__ import annotations

import configparser
import datetime
import io
import json
import os
import tomllib
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd
import yaml
from defusedxml import ElementTree as DefusedElementTree


class FileIngestError(ValueError):
    """A reader refused its file — unknown format, malformed content,
    or a hardening violation; shaped through NX-3 by the wrapper."""


AdmitLoad = Callable[[int], None]
"""The chokepoint's upfront memory-admission seam (Chokepoint.admit_load):
given an estimated resident-byte count it returns on headroom and raises
ResourceRefusedError otherwise. Injected by tools.py — readers.py never
imports the chokepoint (the import-graph gate)."""

# Rows per chunk on the running-charge streaming path: small enough that
# one chunk's residency is a fraction of any realistic ceiling, so a
# high-ratio file is refused after the first over-headroom chunk rather
# than after the whole file has materialized (CR-005).
_READ_CHUNK_ROWS = 50_000

# Upfront in-memory estimate for the non-streaming formats. Two regimes,
# split by whether the on-disk bytes bound the materialized size:
#
# - UNCOMPRESSED on disk (the text tree + legacy .xls, an OLE2/BIFF
#   stream): the file's own bytes are present, so `st_size * factor` is a
#   sound upper bound on the parsed object graph — text parsed to a Python
#   object graph runs several times its source; .xls a small multiple.
#   Over-estimation is the fail-safe direction.
#
# - COMPRESSED containers (columnar parquet/feather/arrow, HDF5, and the
#   zip-packed spreadsheets xlsx/ods/numbers): on-disk size says NOTHING
#   about materialized size — a dictionary/RLE or deflate bomb expands
#   hundreds-to-thousands to one (a 92 KB dict/RLE parquet materializes
#   160 MB), so `st_size * factor` is trivially defeatable and FAILS OPEN
#   (CR-029). These are estimated from the container's own declared
#   LOGICAL shape instead (rows x materialized column width, dataset
#   shapes x itemsize, or the archive's declared-uncompressed total) —
#   never st_size. See `_logical_materialization`.
_EXPANSION_FACTOR: Mapping[str, int] = {
    "json": 10,
    "yaml": 10,
    "toml": 10,
    "ini": 10,
    "xml": 10,
    "xls": 20,
}

# The compressed containers: each estimated from declared logical size,
# never from st_size (CR-029). Dispatched in `_logical_materialization`.
_LOGICAL_SIZE_FORMATS: frozenset[str] = frozenset(
    {"parquet", "feather", "arrow", "hdf5", "xlsx", "ods", "numbers"}
)

# Per-cell allowance for a variable-width (object/string) column: a Python
# str carries ~49 bytes of object header on top of the 8-byte array
# pointer, so a column of many short distinct strings materializes far
# past its declared byte total. 60 bytes/cell bounds that (and hugely
# over-estimates a dictionary-shared column, which is the fail-safe
# direction).
_OBJECT_CELL_BYTES = 60

# Multiplier applied to every logical-size estimate to cover the pandas
# frame bookkeeping the per-column arithmetic omits (the index, per-block
# overhead, object slack) — the raw column sum undershoots the measured
# `memory_usage(deep=True)` by a small constant, so a 10% margin keeps the
# estimate a true upper bound.
_MATERIALIZATION_MARGIN = 1.1

# Coarse upfront pre-gate for the streaming (CSV/TSV) path: the running
# per-chunk charge fires only AFTER a chunk materializes, so a wide or
# newline-free file whose whole content lands in the first chunk could
# OOM before the first charge (CR-031). CSV/TSV are uncompressed, so
# `st_size` is a floor on the file's own bytes; charging `st_size * 2`
# before the read bounds that first-chunk materialization (the parsed
# frame runs a small multiple of the delimited text).
_STREAM_COARSE_FACTOR = 2

# Formats whose reader library validates the PATH itself (by package name
# or suffix) and so cannot consume a `/dev/fd/<fd>` proxy: they read the
# O_NOFOLLOW-validated real path (CR-024 residual documented in read_path).
_PATH_ONLY_FORMATS = frozenset({"numbers"})


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
    supported = _supported_formats()
    if declared != "auto":
        if declared not in supported:
            raise FileIngestError(
                f"unknown format {declared!r} — supported: {sorted(supported)}"
            )
        return declared
    suffix_format = _SUFFIX_TO_FORMAT.get(path.suffix.lower())
    if suffix_format is None:
        raise FileIngestError(
            f"cannot infer a format from suffix {path.suffix!r} — pass "
            f"format explicitly (supported: {sorted(supported)})"
        )
    return suffix_format


def read_path(real: Path, format_name: str, admit: AdmitLoad) -> Any:
    """Dispatch to the one reader for `format_name` over an ATOMICALLY
    contained descriptor (CR-024), crossing the chokepoint's
    memory-admission seam BEFORE the file materializes (CR-005/NFR-105,
    GP3 fail-safe).

    The contained path is re-opened with O_NOFOLLOW and its descriptor
    identity re-validated (`_contained_open`), then every reader consumes
    the file through `/dev/fd/<fd>` — never by re-resolving the path
    string — so a symlink swapped into the final component after
    containment cannot redirect the read (the resolve-then-reopen TOCTOU
    is closed). A streamable format (CSV/TSV) is read chunk by chunk with
    its growing residency charged after each (a high-ratio file refused
    mid-read); every other format crosses one upfront estimate keyed off
    the descriptor's size. No reader materializes a whole file without
    first passing the memory gate.
    """
    fd = _contained_open(real)
    try:
        if format_name in _STREAMING_READERS:
            # CR-031: bound a wide/newline-free first chunk BEFORE the
            # running per-chunk charge (which only fires after a chunk
            # materializes) can be reached — CSV/TSV are uncompressed so
            # st_size is a sound floor.
            admit(os.fstat(fd).st_size * _STREAM_COARSE_FACTOR)
            return _STREAMING_READERS[format_name](fd, admit)
        admit(_upfront_estimate(fd, real, format_name))
        os.lseek(fd, 0, os.SEEK_SET)
        # Most readers consume the descriptor via /dev/fd; a format whose
        # library validates the path itself (numbers_parser checks the
        # `.numbers` package name) reads the O_NOFOLLOW-validated real
        # path instead — the symlinked-final-component vector is already
        # closed by _contained_open; the residual is the narrow reopen
        # window, bounded in the single-user deployment.
        source = real if format_name in _PATH_ONLY_FORMATS else Path(_fd_path(fd))
        return _READERS[format_name](source)
    finally:
        os.close(fd)


def _fd_path(fd: int) -> str:
    """The `/dev/fd/<fd>` path that refers to THIS open descriptor, not
    the original path string — the read cannot be redirected by a later
    symlink swap (CR-024)."""
    return f"/dev/fd/{fd}"


def _contained_open(real: Path) -> int:
    """Atomic contain-and-open (CR-024): re-open the already-contained
    path with O_NOFOLLOW so a symlink swapped into the final component
    after NX-6 resolved it cannot redirect the read, then confirm the
    descriptor still names the same file (device+inode) containment
    validated. Callers read through `/dev/fd/<fd>`, never by re-resolving
    the path string. Residual (bounded in the single-user deployment): a
    parent directory swapped in the microwindow between the identity stat
    and the open."""
    fd = os.open(real, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        current = os.stat(real)
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise FileIngestError(
                "contain-and-open identity mismatch: the file changed "
                "between containment and open (CR-024 TOCTOU) — refused"
            )
    except BaseException:
        os.close(fd)
        raise
    return fd


def _supported_formats() -> set[str]:
    """Every format read_file accepts — streaming and whole-file alike."""
    return set(_READERS) | set(_STREAMING_READERS)


def _upfront_estimate(fd: int, real: Path, format_name: str) -> int:
    """The pre-read materialized-memory estimate for a non-streaming
    format. A compressed container is estimated from its own declared
    LOGICAL shape (CR-029); an uncompressed format from `st_size` times
    its conservative expansion factor."""
    if format_name in _LOGICAL_SIZE_FORMATS:
        # The metadata read goes through the SAME contained descriptor the
        # reader will use (`/dev/fd/<fd>`), never a re-resolved path string
        # — except the path-only formats, which read the O_NOFOLLOW-
        # validated real path (CR-024).
        source = real if format_name in _PATH_ONLY_FORMATS else Path(_fd_path(fd))
        return _logical_materialization(source, format_name)
    return os.fstat(fd).st_size * _EXPANSION_FACTOR.get(format_name, 1)


def _logical_materialization(source: Path, format_name: str) -> int:
    """Declared materialized size of a compressed container, read from
    metadata WITHOUT materializing the data (CR-029). A metadata read that
    cannot bound the file raises — GP3 fail-safe never falls back to the
    defeatable `st_size` estimate."""
    if format_name in {"parquet", "feather", "arrow"}:
        return _columnar_materialization(source, format_name)
    if format_name == "hdf5":
        return _hdf5_materialization(source)
    # zip-packed spreadsheets: xlsx / ods / numbers
    return _zip_materialization(source)


def _with_margin(raw: int) -> int:
    """A logical column/dataset sum times the frame-overhead margin —
    a true upper bound on the measured resident size."""
    return int(raw * _MATERIALIZATION_MARGIN)


def _arrow_cell_bytes(arrow_type: Any, rows: int) -> int:
    """Materialized bytes for one arrow-typed column of `rows` rows: a
    fixed-width numeric/temporal/bool column is `rows * itemsize`; a
    variable-width (string/binary/nested) column materializes to a pandas
    object array charged at `_OBJECT_CELL_BYTES` per cell (the declared
    byte total is added by the caller for parquet)."""
    import pyarrow as pa

    if pa.types.is_boolean(arrow_type):
        return rows * 1
    if (
        pa.types.is_integer(arrow_type)
        or pa.types.is_floating(arrow_type)
        or pa.types.is_temporal(arrow_type)
    ):
        width = (
            (arrow_type.bit_width // 8) if getattr(arrow_type, "bit_width", 0) else 8
        )
        return rows * width
    return rows * _OBJECT_CELL_BYTES


def _columnar_materialization(source: Path, format_name: str) -> int:
    """parquet: rows x per-column width from the footer (+ declared
    uncompressed bytes for variable-width columns), metadata-only.
    feather/arrow (arrow IPC): schema width x row count, read through a
    memory map so per-batch counting stays bounded (one batch at a time,
    released between reads) rather than materializing the whole file."""
    if format_name == "parquet":
        import pyarrow.parquet as pq

        metadata = pq.ParquetFile(str(source)).metadata
        rows = metadata.num_rows
        schema = metadata.schema.to_arrow_schema()
        estimate = 0
        for index, field in enumerate(schema):
            estimate += _arrow_cell_bytes(field.type, rows)
            if _is_variable(field.type):
                # Add the declared uncompressed character data on top of
                # the per-cell object allowance already charged above.
                estimate += sum(
                    metadata.row_group(rg).column(index).total_uncompressed_size
                    for rg in range(metadata.num_row_groups)
                )
        return _with_margin(estimate)
    import pyarrow as pa
    import pyarrow.ipc as ipc

    with pa.memory_map(str(source), "r") as handle:
        reader = ipc.open_file(handle)
        schema = reader.schema
        rows = sum(
            reader.get_batch(batch).num_rows
            for batch in range(reader.num_record_batches)
        )
    return _with_margin(sum(_arrow_cell_bytes(field.type, rows) for field in schema))


def _is_variable(arrow_type: Any) -> bool:
    import pyarrow as pa

    return not (
        pa.types.is_boolean(arrow_type)
        or pa.types.is_integer(arrow_type)
        or pa.types.is_floating(arrow_type)
        or pa.types.is_temporal(arrow_type)
    )


def _hdf5_materialization(source: Path) -> int:
    """Sum every dataset's `shape.prod x dtype.itemsize` (h5py metadata,
    no data read) — an upper bound on what any single-key read
    materializes."""
    import h5py
    import numpy as np

    total = 0

    def visit(_name: str, item: Any) -> None:
        nonlocal total
        if isinstance(item, h5py.Dataset):
            total += int(np.prod(item.shape)) * item.dtype.itemsize

    with h5py.File(source, "r") as handle:
        handle.visititems(visit)
    return _with_margin(total)


def _zip_materialization(source: Path) -> int:
    """Declared-uncompressed total of a zip-packed spreadsheet
    (xlsx/ods/numbers): the sum of the archive's central-directory
    uncompressed sizes. This is the decompressed archive (its XML/IWA
    parts), a conservative upper bound on the materialized cell frame
    (the markup is at least as large as the values it carries)."""
    import zipfile

    with zipfile.ZipFile(source) as archive:
        return _with_margin(sum(entry.file_size for entry in archive.infolist()))


# -- tabular readers --------------------------------------------------


def _read_csv_streaming(fd: int, admit: AdmitLoad) -> pd.DataFrame:
    return _stream_delimited(fd, admit, sep=",")


def _read_tsv_streaming(fd: int, admit: AdmitLoad) -> pd.DataFrame:
    return _stream_delimited(fd, admit, sep="\t")


def _stream_delimited(fd: int, admit: AdmitLoad, *, sep: str) -> pd.DataFrame:
    """Running-charge chunked read (CR-005) over the contained descriptor
    (CR-024): pull the file in fixed-row chunks through `/dev/fd/<fd>` and
    charge the growing resident size through the chokepoint after each —
    the whole-file frame never materializes past the memory ceiling,
    mirroring how bounded SQL fetch admits per batch. A ledger refusal on
    any chunk propagates as the structured over-budget refusal (tools.py)
    before the rest of the file is read."""
    source = _fd_path(fd)
    frames: list[pd.DataFrame] = []
    resident = 0
    os.lseek(fd, 0, os.SEEK_SET)
    with pd.read_csv(source, sep=sep, chunksize=_READ_CHUNK_ROWS) as reader:
        for chunk in reader:
            resident += int(chunk.memory_usage(deep=True).sum())
            admit(resident)
            frames.append(chunk)
    if frames:
        # CR-032: pd.concat allocates a new full-size frame while `frames`
        # is still held — a transient ~2x the admitted resident — so the
        # peak is admitted before it is reached.
        admit(resident * 2)
        return pd.concat(frames, ignore_index=True)
    os.lseek(fd, 0, os.SEEK_SET)  # header-only file: re-read just the header
    return pd.read_csv(source, sep=sep, nrows=0)


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


_STREAMING_READERS: Mapping[str, Callable[[int, AdmitLoad], Any]] = {
    "csv": _read_csv_streaming,
    "tsv": _read_tsv_streaming,
}

_READERS: Mapping[str, Callable[[Path], Any]] = {
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
