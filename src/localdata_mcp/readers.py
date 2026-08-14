"""The readers: one function per input format, and the shapes they return.

A reader is handed a path that has already been judged safe and returns the
tables it found in it, as pandas frames. That is the whole contract. Nothing
here opens a database, holds a slot, or knows a nickname: a file becomes frames
here, and everything downstream works from the frames.

They live apart from ``loader`` so that ``formats`` — the one table naming what
reads and what writes each suffix — can reach them without importing the module
that *calls* them. The same split put the writers in ``writers``.

**A reader may refuse.** Reading at the wrong separator, or from a file whose
shape is not a table, does not produce bad rows to be caught later — it produces
a ``LoadError`` naming what was wrong with the file, because a wrong answer that
looks like data is the failure nothing downstream can report.

**A reader may also warn.** Several formats have a shape a table does not: a
JSON object with one array under it, a workbook with several sheets. Those
come back as notes on the result rather than as refusals, because the data is
readable and the caller is the one who can say whether it was read the way they
meant.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
from xml.etree import ElementTree

import pandas as pd

from .errors import LoadError

__all__ = ["NamedFrame", "ReadResult", "Reader"]


@dataclass(frozen=True)
class NamedFrame:
    """One table out of a file, with the name the file gave it if it had one.

    ``name`` is ``None`` for a format that holds a single unnamed table — a CSV
    is just rows, and what to call them is the caller's or the filename's. A
    spreadsheet sheet names itself, and that name is the one to use.
    """

    frame: pd.DataFrame
    name: str | None = None


@dataclass(frozen=True)
class ReadResult:
    """The tables in a file, and anything about the reading the data cannot show.

    **A file may hold more than one table**, and a workbook is the obvious case:
    three sheets are three tables. Reading only the first and ignoring the rest
    would leave data that is present in the file unreachable through the server,
    which is the same silent loss as dropping a nested value — so the reader
    returns all of them and the datasource, being a database, holds all of them.

    Most formats need no second field either: a CSV's rows are the whole story.
    Some cannot say everything in the data. A JSON file whose tables hang under
    a key was read from *one* of those keys. That is a fact about the source the
    caller can act on and would otherwise have to infer from a shape that looks
    perfectly ordinary — so it is carried out rather than dropped here.

    A note is a sentence, and the contract is the same as the warnings the load
    path already produces: state what happened and what to do about it. It is
    not a place to guess.
    """

    tables: tuple[NamedFrame, ...]
    notes: tuple[str, ...] = ()


def _one(frame: pd.DataFrame, notes: tuple[str, ...] = ()) -> ReadResult:
    """A result for the common case: a format holding a single unnamed table."""
    return ReadResult((NamedFrame(frame),), notes)


#: A reader turns a path into a frame plus whatever it had to assume or choose.
Reader = Callable[[Path], ReadResult]

#: Delimiters worth naming in a warning. The character that separated the file
#: is never among the candidates, because a column name cannot contain it.
_COMMON_DELIMITERS = {";": "';'", "\t": "a tab", "|": "'|'", ",": "','"}


def delimited(separator: str) -> Reader:
    """A reader for character-separated text, at a given separator.

    The default separator comes from the extension — comma for ``.csv`` and
    ``.txt``, tab for ``.tsv`` — and an explicit ``delimiter`` replaces the
    reader rather than being threaded through every other format's signature.
    """

    def read(path: Path) -> ReadResult:
        # `keep_default_na` is left on: pandas' blank/NA handling is what turns
        # an empty cell into NULL rather than the string "".
        frame = pd.read_csv(path, sep=separator)
        return _one(frame, fat_column_note(frame, separator, path.name))

    return read


def fat_column_note(frame: pd.DataFrame, separator: str, name: str) -> tuple[str, ...]:
    """Say when a file has plainly been read at the wrong separator.

    The parameter alone does not fix the silent failure: a caller who does not
    know the file is semicolon-separated gets one column holding every field and
    no signal at all — the whole header becomes the column's name. One column
    whose *name* still contains a common delimiter is that, and nothing else, so
    it is worth saying and worth naming the parameter that fixes it.

    This states what it found; it does not re-read the file at the guessed
    separator. Sniffing is the fail-open shape this project keeps being bitten
    by, and a guess that is usually right is the worst kind.
    """
    if len(frame.columns) != 1:
        return ()

    column = str(frame.columns[0])
    found = [
        spelling
        for character, spelling in _COMMON_DELIMITERS.items()
        if character != separator and character in column
    ]
    if not found:
        return ()

    return (
        f"{name} loaded as a single column whose name contains "
        f"{' and '.join(found)}, which is what a file separated by something "
        f"other than {_COMMON_DELIMITERS[separator]} looks like when read at "
        f"{_COMMON_DELIMITERS[separator]}. If that is the case, attach it again "
        f"with delimiter set to the right character. Nothing here guesses it.",
    )


def read_json(path: Path) -> ReadResult:
    """Read a JSON document that holds one table.

    A JSON file is only *sometimes* tabular, so this reader says which shapes it
    takes rather than reshaping whatever it finds:

    * **An array of objects** is the table, and nothing is assumed.
    * **An object with exactly one non-empty array of objects under it** — the
      shape an API dump takes, ``{"count": 2, "employees": [...]}`` — is that
      array, and the note says which key it came from. There is no choice to
      make when there is one candidate, and refusing would be a dead end: an
      agent holding this file has no way to lift the array out of it, so a
      refusal it cannot act on is worse than a load it is told about.
    * **Anything else is refused**, naming what was found. Two candidate arrays
      is a genuine choice between tables, and choosing is the caller's.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LoadError(f"Could not read {path.name}: {exc}") from exc

    records, notes = _table_within(document, path.name)
    return _frame_of_records(records, notes)


def read_jsonl(path: Path) -> ReadResult:
    """Read JSON Lines: one object per line, blank lines ignored.

    No shape ambiguity exists here — the format *is* a sequence of records — so
    unlike ``.json`` this reader never has anything to report.
    """
    records = []
    with path.open(encoding="utf-8") as handle:
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


def _require(module: str, extra: str, doing: str):
    """Import an optional format library, or say how to install it.

    Every format is *known* to this server whether or not its library is here,
    so a caller asking for one that is not installed gets an instruction rather
    than a mystery. Listing only the installed formats would make the tool's own
    description vary by environment, which is worse: the agent could not learn
    what this server does without discovering what it happens to have.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise LoadError(
            f"{doing} needs {module}, which is not installed. Install it with: "
            f"pip install 'localdata-mcp[{extra}]' (or [all] for every format)."
        ) from exc


def read_yaml(path: Path) -> ReadResult:
    """YAML parses to the same structures JSON does, so it gets the same rules.

    ``safe_load``, never ``load``: the full loader constructs arbitrary Python
    objects from a document, and every document here arrives from outside.
    """
    yaml = _require("yaml", "yaml", "Reading YAML")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise LoadError(f"Could not read {path.name}: {exc}") from exc

    records, notes = _table_within(document, path.name)
    return _frame_of_records(records, notes)


def read_columnar(path: Path) -> ReadResult:
    """Parquet, Feather/Arrow and ORC — typed formats, so nothing is inferred.

    Feather goes through ``pyarrow.ipc`` rather than ``pandas.read_feather``,
    which routes to a ``pyarrow.feather`` entry point deprecated as of pyarrow
    24 and warns on every call. Same file format either way.
    """
    suffix = path.suffix.lower()
    _require("pyarrow", "parquet", f"Reading {suffix}")
    try:
        if suffix == ".parquet":
            frame = pd.read_parquet(path)
        elif suffix == ".orc":
            frame = pd.read_orc(path)
        else:
            import pyarrow as pa

            with pa.memory_map(str(path), "rb") as source:
                frame = pa.ipc.open_file(source).read_all().to_pandas()
    except LoadError:
        raise
    except Exception as exc:
        raise LoadError(f"Could not read {path.name}: {exc}") from exc
    return _one(frame)


def workbook(module: str, extra: str) -> Reader:
    """A reader for one workbook format, bound to the library that reads it.

    A factory rather than a lookup, for the same reason ``delimited`` is one:
    which library handles a format is a fact about the format, so it is declared
    once in ``formats`` and reaches both this reader and the matching writer
    from there. Held here as a table instead it was written down a second time
    on the write side, where the two could disagree without anything noticing
    (#99).

    ``module`` is what pandas dispatches on and what is imported to check the
    library is here; ``extra`` is this project's optional-dependency group, and
    the two differ, so both have to be carried.
    """

    def read_workbook(path: Path) -> ReadResult:
        """Every sheet of a workbook, each as a table under its own sheet name.

        ``sheet_name=None`` rather than the default ``0``: the default reads the
        first sheet and says nothing about the others, which leaves data that is
        present in the file unreachable through the server. A workbook is a database
        and its sheets are its tables, so all of them land.
        """
        suffix = path.suffix.lower()
        _require(module, extra, f"Reading {suffix}")

        try:
            sheets = pd.read_excel(path, sheet_name=None)
        except LoadError:
            raise
        except Exception as exc:
            raise LoadError(f"Could not read {path.name}: {exc}") from exc

        # A workbook may carry a sheet that is entirely empty; it is a sheet with no
        # table in it, and dropping it is not loss. Refusing the whole file over one
        # would be, so only a workbook with nothing in any sheet is refused.
        tables = tuple(
            NamedFrame(frame, name)
            for name, frame in sheets.items()
            if not frame.empty or len(frame.columns)
        )
        if not tables:
            raise LoadError(f"Could not read {path.name}: every sheet in it is empty.")
        return ReadResult(tables)

    return read_workbook


def read_numbers(path: Path) -> ReadResult:
    """Apple Numbers, whose sheets each hold their own named tables.

    Two levels rather than one: a Numbers sheet is a canvas that may carry
    several tables, so the name here is the table's, qualified by its sheet only
    when the same table name appears on more than one.
    """
    parser = _require("numbers_parser", "numbers", "Reading Apple Numbers")
    try:
        document = parser.Document(str(path))
        found = [
            (sheet.name, table.name, table.rows(values_only=True))
            for sheet in document.sheets
            for table in sheet.tables
        ]
    except LoadError:
        raise
    except Exception as exc:
        raise LoadError(f"Could not read {path.name}: {exc}") from exc

    counts: dict[str, int] = {}
    for _, table_name, _ in found:
        counts[table_name] = counts.get(table_name, 0) + 1

    tables = []
    for sheet_name, table_name, rows in found:
        trimmed = _without_grid_padding(rows)
        if trimmed is None:
            continue
        header, body = trimmed
        frame = pd.DataFrame(body, columns=header)
        name = table_name if counts[table_name] == 1 else f"{sheet_name}_{table_name}"
        tables.append(NamedFrame(_inferred_types(frame), name))

    if not tables:
        raise LoadError(f"Could not read {path.name}: it holds no table with rows.")
    return ReadResult(tuple(tables))


def _without_grid_padding(rows: Sequence[Sequence[Any]]):
    """Strip a Numbers table's empty grid, and nothing that holds a value.

    A Numbers table is a fixed canvas — a new one is 8 columns by 12 rows —
    so the cells beyond the data come back as ``None`` and would otherwise
    become columns named ``None`` and a tail of all-null rows.

    Emptiness is tested on the *whole* column, not on its header: a column with
    values but no header is real data that happens to be unlabelled, and
    dropping it on the strength of a blank header would be exactly the silent
    loss this reader is meant to avoid. Returns ``None`` for a table that is
    entirely empty.
    """
    if not rows:
        return None

    header, *body = rows
    keep = [
        index
        for index in range(len(header))
        if header[index] is not None
        or any(index < len(row) and row[index] is not None for row in body)
    ]
    if not keep:
        return None

    kept_body = [[row[index] for index in keep] for row in body]
    kept_body = [row for row in kept_body if any(cell is not None for cell in row)]
    if not kept_body:
        return None

    return [header[index] for index in keep], kept_body


def read_xml(path: Path) -> ReadResult:
    """Read an XML document whose root holds one repeated element per row.

    Written rather than delegated to ``pandas.read_xml``, which is fail-open on
    two shapes this corpus contains. Measured against the stdlib parser: a row
    holding a nested element comes back with that column ``NaN`` — the subtree
    silently dropped — and a row with a tag repeated twice keeps only the last
    one. Both are data loss with nothing reported, which is the class this
    server exists to refuse.

    Columns are the row's attributes and the tags of its direct children. A
    child with children of its own is kept as its XML text (as a nested JSON
    value is kept as JSON text); a tag appearing twice in one row is a list
    rather than a column, and is refused by name.
    """
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError as exc:
        raise LoadError(f"Could not read {path.name}: {exc}") from exc

    rows, notes = _rows_within(root, path.name)

    records = []
    nested: dict[str, None] = {}  # insertion-ordered, and deduplicating
    for element in rows:
        record, held = _record_of(element, path.name)
        records.append(record)
        nested.update(dict.fromkeys(held))

    if nested:
        notes = notes + (
            f"Nested elements in {', '.join(nested)} were kept as XML text, "
            f"because SQL has no nested type. The text is the element as it "
            f"stood in the file, so nothing was lost.",
        )

    result = _frame_of_records(records, notes)
    frame = result.tables[0].frame
    return _one(_inferred_types(frame), result.notes)


def _inferred_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Read numbers out of text, since every value in XML arrives as text.

    The same inference ``read_csv`` performs, applied here because this reader
    builds its frame by hand. A column is converted only when **every** non-null
    value in it parses, so a column mixing numbers and text stays text and the
    mixed-column signal downstream still has something to find.

    The skip test asks what a column *is* rather than comparing its dtype to
    ``object``: pandas 3 infers a dedicated ``str`` dtype for text, so an
    ``!= object`` guard here skipped every column it was meant to convert.
    """
    for name in frame.columns:
        series = frame[name]
        if pd.api.types.is_numeric_dtype(
            series
        ) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        present = series.notna().sum()
        if not present:
            continue
        candidate = pd.to_numeric(series, errors="coerce")
        if candidate.notna().sum() == present:
            frame[name] = candidate
    return frame


def _rows_within(
    root: ElementTree.Element, name: str
) -> tuple[list[ElementTree.Element], tuple[str, ...]]:
    """Pick the repeated element that is the table, on JSON's rules.

    One kind of child is the table. Several kinds, one of which repeats, is the
    wrapped shape — a ``<generated>`` beside the rows — and the note names what
    was left out. Two kinds that both repeat are two tables, which is a choice,
    so it is refused.
    """
    groups: dict[str, list[ElementTree.Element]] = {}
    for child in root:
        groups.setdefault(child.tag, []).append(child)

    if not groups:
        raise LoadError(f"Could not read {name}: <{root.tag}> holds no elements.")
    if len(groups) == 1:
        return next(iter(groups.values())), ()

    repeated = [tag for tag, members in groups.items() if len(members) > 1]
    if len(repeated) == 1:
        tag = repeated[0]
        others = ", ".join(f"<{other}>" for other in groups if other != tag)
        return groups[tag], (
            f"{name} holds <{tag}> repeated among other elements, and <{tag}> "
            f"was loaded as the table. The rest ({others}) are not part of it.",
        )
    if repeated:
        listed = ", ".join(f"<{tag}>" for tag in repeated)
        raise LoadError(
            f"Could not read {name}: it holds more than one table ({listed}), "
            f"and which one you want is not something this server should "
            f"decide. Split the file, or attach it as one table per file."
        )
    raise LoadError(
        f"Could not read {name}: <{root.tag}> holds one each of "
        f"{', '.join(f'<{tag}>' for tag in groups)}, so nothing in it repeats "
        f"as rows do."
    )


def _record_of(element: ElementTree.Element, name: str) -> tuple[dict, list[str]]:
    """One row — attributes then children by tag — and which parts were nested."""
    record: dict[str, object] = dict(element.attrib)
    nested = []

    for child in element:
        if child.tag in record:
            raise LoadError(
                f"Could not read {name}: <{child.tag}> appears more than once "
                f"in a single <{element.tag}>, which makes it a list rather "
                f"than a column. A table cannot hold it without choosing which "
                f"one to keep, and choosing would lose the rest."
            )
        if len(child):
            # Kept whole. Dropping it is what pandas does, and it reports nothing.
            record[child.tag] = ElementTree.tostring(child, encoding="unicode").strip()
            nested.append(child.tag)
        else:
            record[child.tag] = child.text

    if not record:
        raise LoadError(
            f"Could not read {name}: <{element.tag}> has no attributes and no "
            f"child elements, so it names no columns. A row needs named parts."
        )
    return record, nested


def _table_within(document: object, name: str) -> tuple[list[dict], tuple[str, ...]]:
    """Find the one table in a parsed JSON document, or refuse and say why."""
    if isinstance(document, list):
        offender = next(
            (v for v in document if not isinstance(v, dict)),
            None,
        )
        if offender is not None:
            raise LoadError(
                f"Could not read {name}: it is an array of "
                f"{_json_kind(offender)}, and a table needs an array of objects "
                f"— each one a row, its keys the columns."
            )
        return document, ()

    if isinstance(document, dict):
        # Empty arrays are not candidates, which is what makes the common
        # `{"data": [...], "errors": []}` unambiguous rather than a refusal.
        candidates = [
            key
            for key, value in document.items()
            if isinstance(value, list)
            and value
            and all(isinstance(item, dict) for item in value)
        ]
        if len(candidates) == 1:
            key = candidates[0]
            return document[key], (
                f"{name} is an object rather than an array, and the array under "
                f"{key!r} was the only table in it — that is what was loaded. "
                f"The other keys ({', '.join(k for k in document if k != key)}) "
                f"are not part of this table.",
            )
        if candidates:
            raise LoadError(
                f"Could not read {name}: it holds more than one table "
                f"({', '.join(repr(k) for k in candidates)}), and which one you "
                f"want is not something this server should decide. Split the "
                f"file, or attach it as one table per file."
            )

    raise LoadError(
        f"Could not read {name}: it holds no table. This reader takes an array "
        f"of objects, or an object with exactly one array of objects under it."
    )


def _frame_of_records(records: list[dict], notes: tuple[str, ...]) -> ReadResult:
    """Build a frame from JSON records, encoding anything SQL cannot hold.

    A nested value has no SQL type, so it is written as its JSON text. That is
    lossless and reversible, and unlike dropping or flattening it invents
    nothing — but the resulting column looks like ordinary text, so the note
    names the columns and the function that reads back into them.
    """
    frame = pd.DataFrame(records)

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

    if encoded:
        notes = notes + (
            f"Nested values in {', '.join(encoded)} were stored as JSON text, "
            f"because SQL has no nested type. Read into them with "
            f"json_extract(column, '$.key'); the text is exactly what was in the "
            f"file, so nothing was lost.",
        )
    return _one(frame, notes)


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
