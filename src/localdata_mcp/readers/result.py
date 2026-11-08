"""The contract every reader shares: what a reader is, and what it returns.

A reader takes a path and returns a ``ReadResult``: the tables found in the
file, each a ``NamedFrame``, and any notes about the reading. The helpers here
are the ones more than one format needs — wrapping a single unnamed table,
importing an optional format library or saying how to install it, and reading
numbers out of columns that arrived as text. They sit in their own module so
that every format module can depend on them without depending on each other.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from ..errors import LoadError


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
    keys was read from those keys, and each table is named from its key path.
    That is a fact about the source the caller can act on and would otherwise
    have to infer from a shape that looks perfectly ordinary — so it is carried
    out rather than dropped here.

    A note is a sentence, and the contract is the same as the warnings the load
    path already produces: state what happened and what to do about it. It is
    not a place to guess.
    """

    tables: tuple[NamedFrame, ...]
    notes: tuple[str, ...] = ()


def single_table(frame: pd.DataFrame, notes: tuple[str, ...] = ()) -> ReadResult:
    """A result for the common case: a format holding a single unnamed table."""
    return ReadResult((NamedFrame(frame),), notes)


#: A reader turns a path into a frame plus whatever it had to assume or choose.
Reader = Callable[[Path], ReadResult]


def require_library(module: str, extra: str, doing: str):
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


def inferred_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Read numbers out of text, since every value in XML arrives as text.

    The same inference ``read_csv`` performs, applied here because the XML and
    Apple Numbers readers build their frames by hand. A column is converted only when **every** non-null
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
