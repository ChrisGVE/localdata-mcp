"""Documents that parse to nested data: JSON, JSON Lines and YAML.

These formats are tables only sometimes, so most of this module decides where
the tables are in a parsed document: the document itself when it is an array
of objects, or every array of objects found under its keys, each named from its
key path. It also stores a nested value as JSON text, a rule the XML reader
uses too. JSON and YAML share the module because they parse to the same
structures and follow the same rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..errors import LoadError
from ..names import table_names
from .result import NamedFrame, ReadResult, require_library, single_table


#: How the JSON readers decode a file: UTF-8, with a byte-order mark at the very
#: start skipped rather than refused (#154). ``utf-8-sig`` drops the mark only
#: when it is the first thing in the stream and reads the rest as plain UTF-8.
_UTF8_TEXT = "utf-8-sig"


def read_json(path: Path) -> ReadResult:
    """Read a JSON document: the table in it, or every table in it.

    A JSON file is only *sometimes* tabular, so this reader says which shapes it
    takes rather than reshaping whatever it finds:

    * **An array of objects** is the table, and nothing is assumed.
    * **An object with exactly one non-empty array of objects in it**, at any
      depth — the shape an API dump takes, ``{"count": 2, "employees": [...]}``
      — is that array, and the note says which key path it came from. There is
      no choice to make when there is one candidate, and refusing would be a
      dead end: an agent holding this file has no way to lift the array out of
      it, so a refusal it cannot act on is worse than a load it is told about.
    * **An object with several** is several tables, and all of them land, the
      way a workbook's sheets do: a datasource is a database, and leaving one
      of them unreachable would be the silent loss the workbook rule exists to
      prevent. Each is named from its key path (see :func:`_tables_within`).
    * **Anything else is refused**, naming what was found.

    A UTF-8 byte-order mark at the start is skipped, as RFC 8259 §8.1 allows
    and as the other text formats already do: Windows editors and Excel write
    one, and nothing about the data changes by dropping it.
    """
    try:
        document = json.loads(path.read_text(encoding=_UTF8_TEXT))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LoadError(f"Could not read {path.name}: {exc}") from exc

    return _document_tables(document, path)


def read_jsonl(path: Path) -> ReadResult:
    """Read JSON Lines: one object per line, blank lines ignored.

    No shape ambiguity exists here — the format *is* a sequence of records — so
    unlike ``.json`` this reader never has anything to report. A byte-order
    mark is skipped at the start of the file only, as for ``.json``; one at the
    start of a later line is a stray character in the data, and refused.
    """
    records = []
    with path.open(encoding=_UTF8_TEXT) as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise LoadError(
                    f"Could not read {path.name}: line {number} is not JSON ({exc})"
                ) from exc
            if not isinstance(value, dict):
                raise LoadError(
                    f"Could not read {path.name}: line {number} is "
                    f"{_json_kind(value)}, and JSON Lines is one object per line."
                )
            records.append(value)
    return _frame_of_records(records, ())


def read_yaml(path: Path) -> ReadResult:
    """YAML parses to the same structures JSON does, so it gets the same rules.

    ``safe_load``, never ``load``: the full loader constructs arbitrary Python
    objects from a document, and every document here arrives from outside.
    """
    yaml = require_library("yaml", "yaml", "Reading YAML")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise LoadError(f"Could not read {path.name}: {exc}") from exc

    return _document_tables(document, path)


#: A key path, from the document's root down to one value: ``("meta", "orders")``.
KeyPath = tuple[str, ...]


def _shown(path: KeyPath) -> str:
    """A key path as a caller reads it, and as its table is named: ``meta.orders``."""
    return ".".join(path)


def _is_table(value: object) -> bool:
    """A non-empty array of objects: each one a row, its keys the columns.

    Empty arrays are not tables, which is what makes the common
    ``{"data": [...], "errors": []}`` one table rather than two.
    """
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, dict) for item in value)
    )


def _walk(document: dict) -> tuple[list[tuple[KeyPath, list[dict]]], list[KeyPath]]:
    """Every table under an object, in document order, and every key left out.

    Only objects are walked into. An array of objects is a table and is not
    looked inside — a list of objects within one of its rows is that row's
    value, stored as JSON text like any other nested value — and anything else
    is a value rather than a place a table can be. Which keys count as left out
    is set out at :func:`_left_out_keys`.

    Iterative rather than recursive, so a document nested as deep as the parser
    accepted is never deeper than this walk survives.
    """
    tables: list[tuple[KeyPath, list[dict]]] = []
    visited: list[tuple[KeyPath, object]] = []
    stack: list[tuple[KeyPath, object]] = [((), document)]
    while stack:
        path, value = stack.pop()
        if path:
            visited.append((path, value))
        if _is_table(value):
            tables.append((path, value))
        elif isinstance(value, dict):
            # Reversed onto the stack, so the walk comes off it in file order.
            stack.extend(
                ((*path, str(key)), child) for key, child in reversed(value.items())
            )

    return tables, _left_out_keys(tables, visited)


def _left_out_keys(
    tables: list[tuple[KeyPath, list[dict]]], visited: list[tuple[KeyPath, object]]
) -> list[KeyPath]:
    """The visited keys that lead to no table.

    A key is *left out* when it is neither a table nor an object holding one,
    and it is reported at the level where it stops leading anywhere: a sibling
    ``{"page": {"n": 1}}`` is reported as ``page``, not as ``page.n``. Keys
    inside an object that holds no table are not walked for the report, which
    keeps the one-table note exactly what it was before nesting was walked.
    """
    leading = {path[:depth] for path, _ in tables for depth in range(len(path) + 1)}
    return [path for path, _ in visited if path not in leading and path[:-1] in leading]


def _tables_within(
    document: object, name: str
) -> tuple[list[tuple[KeyPath | None, list[dict]]], tuple[str, ...]]:
    """Find the tables in a parsed JSON or YAML document, or refuse and say why.

    Returns each table's key path — ``None`` for the one table of a document
    whose answer is a single table, which is then named after the file as it
    always was — with its records, and the note saying what was assumed.
    """
    if isinstance(document, list):
        _refuse_array_of_non_objects(document, name)
        return [(None, document)], ()

    if isinstance(document, dict):
        tables, left_out = _walk(document)
        others = ", ".join(_shown(path) for path in left_out)
        if len(tables) == 1:
            path, records = tables[0]
            return [(None, records)], (_one_table_note(name, path, others),)
        if tables:
            note = _several_tables_note(name, tables, left_out, others)
            return list(tables), (note,)

    raise LoadError(
        f"Could not read {name}: it holds no table. This reader takes an array "
        f"of objects, or an object with arrays of objects in it — at any depth, "
        f"each becoming a table of its own."
    )


def _refuse_array_of_non_objects(array: list, name: str) -> None:
    """Refuse an array holding a non-object, naming what the first one is."""
    non_objects = [value for value in array if not isinstance(value, dict)]
    if non_objects:
        raise LoadError(
            f"Could not read {name}: it is an array of "
            f"{_json_kind(non_objects[0])}, and a table needs an array of objects "
            f"— each one a row, its keys the columns."
        )


def _one_table_note(name: str, path: KeyPath, others: str) -> str:
    """The note for an object holding exactly one table, at key path ``path``."""
    return (
        f"{name} is an object rather than an array, and the array under "
        f"{_shown(path)!r} was the only table in it — that is what was "
        f"loaded. The other keys ({others}) are not part of this table."
    )


def _several_tables_note(
    name: str,
    tables: list[tuple[KeyPath, list[dict]]],
    left_out: list[KeyPath],
    others: str,
) -> str:
    """The note for an object holding several tables: what each landed as."""
    shown = [_shown(path) for path, _ in tables]
    _, landing = table_names(shown, Path(name).stem)
    where = ", ".join(f"{given!r} as {table}" for given, table in zip(shown, landing))
    rest = f" The other keys ({others}) are not part of any table." if left_out else ""
    return (
        f"{name} holds {len(tables)} tables, each an array of objects "
        f"under its own key, and every one of them landed as a table of "
        f"its own, named from its key path — the keys from the top of "
        f"the document down, joined by '.', then made safe for SQL the "
        f"way a sheet name is: {where}. Each loaded entry names its key "
        f"path as its sheet.{rest}"
    )


def _document_tables(document: object, path: Path) -> ReadResult:
    """Every table in a parsed JSON or YAML document, as frames with their notes."""
    found, notes = _tables_within(document, path.name)
    tables = []
    for key_path, records in found:
        name = None if key_path is None else _shown(key_path)
        frame, nested = frame_and_nested_note(
            records, None if name is None else repr(name)
        )
        tables.append(NamedFrame(frame, name))
        notes = notes + nested
    return ReadResult(tuple(tables), notes)


def _frame_of_records(records: list[dict], notes: tuple[str, ...]) -> ReadResult:
    """One unnamed table from JSON records, with the reader's notes and its own."""
    frame, nested = frame_and_nested_note(records, None)
    return single_table(frame, notes + nested)


def frame_and_nested_note(
    records: list[dict], table: str | None
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Build a frame from JSON records, encoding anything SQL cannot hold.

    A nested value has no SQL type, so it is written as its JSON text. That is
    lossless and reversible, and unlike dropping or flattening it invents
    nothing — but the resulting column looks like ordinary text, so the note
    names the columns and the function that reads back into them. ``table`` is
    how the note names the table, for a file holding more than one: a reader's
    notes reach every table the file holds, so each has to say which it is about.
    """
    frame = pd.DataFrame(records)
    encoded = _encode_nested_columns(frame)

    if not encoded:
        return frame, ()
    where = ", ".join(encoded) if table is None else f"{', '.join(encoded)} of {table}"
    return frame, (
        f"Nested values in {where} were stored as JSON text, "
        f"because SQL has no nested type. Read into them with "
        f"json_extract(column, '$.key'); the text is exactly what was in the "
        f"file, so nothing was lost.",
    )


def _encode_nested_columns(frame: pd.DataFrame) -> list[str]:
    """Replace nested values in ``frame`` with their JSON text; name those columns."""
    encoded = []
    for name in frame.columns:
        series = frame[name]
        if not series.map(lambda value: isinstance(value, (dict, list))).any():
            continue
        frame[name] = series.map(
            lambda value: (
                json.dumps(value) if isinstance(value, (dict, list)) else value
            )
        )
        encoded.append(str(name))
    return encoded


def _json_kind(value: object) -> str:
    """What a JSON value is, in JSON's own words rather than Python's."""
    if value is None:
        return "null"
    return {
        bool: "a boolean",
        int: "a number",
        float: "a number",
        str: "a string",
        list: "an array",
        dict: "an object",
    }.get(type(value), f"a {type(value).__name__}")
