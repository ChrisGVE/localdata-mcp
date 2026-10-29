"""How a name a file gives becomes a name SQL can use.

A CSV header, a workbook's sheet, a Numbers table: each arrives named however
its author liked, with spaces, punctuation and capitals, and leaves here as an
identifier a statement can carry unquoted. Cleaning a name up can make two of
them the same — ``Q2 Prices`` and ``Q2-Prices`` are both ``q2_prices`` — and a
table or a column landing on a name another already holds overwrites it or is
refused by the database, so what is cleaned up together is also made distinct
together.

One module, so that columns and tables follow one rule. They used to follow two
— columns were de-duplicated and tables were not — and a workbook whose sheet
names cleaned up alike lost a sheet without a word (#91).

It sits apart from ``loader`` because a reader needs it too: the Numbers reader
decides whether to qualify a table by its sheet from the name SQL will see, and
``readers`` cannot import the module that calls it.
"""

from __future__ import annotations

import re
from typing import Sequence

__all__ = ["clashes", "sanitize", "unique"]

_UNSAFE = re.compile(r"\W+")


def sanitize(name: object, fallback: str) -> str:
    """One name made safe for SQL: letters, digits and underscores, lower case.

    A name that cleans up to nothing, or to something starting with a digit, is
    prefixed with ``fallback`` — ``2024`` becomes ``table_2024`` — since neither
    is an identifier a statement can use unquoted.
    """
    cleaned = _UNSAFE.sub("_", str(name).strip()).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"{fallback}_{cleaned}" if cleaned else fallback
    return cleaned.lower()


def unique(names: Sequence[str]) -> list[str]:
    """Make already-cleaned names distinct, keeping their order.

    The first holder of a name keeps it bare. Each later one gets the smallest
    numeric suffix, from ``_1`` up, that nothing before it already holds — so
    ``a, a, a`` becomes ``a, a_1, a_2``, and ``a, a, a_1`` becomes ``a, a_1,
    a_1_1`` rather than ``a_1`` twice, because the third name is itself the
    suffix the second was given.

    Order decides who keeps the bare name, and order is the file's: the first
    column, or the first sheet. That makes the answer the same on every read.
    """
    taken: set[str] = set()
    #: Where the search for each name's next free suffix starts, so a file of
    #: ten thousand identical headers is not searched from ``_1`` every time.
    next_suffix: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        candidate = name
        if candidate in taken:
            suffix = next_suffix.get(name, 1)
            while f"{name}_{suffix}" in taken:
                suffix += 1
            next_suffix[name] = suffix + 1
            candidate = f"{name}_{suffix}"
        taken.add(candidate)
        out.append(candidate)
    return out


def clashes(wanted: Sequence[str], given: Sequence[str]) -> list[tuple[str, list[int]]]:
    """Which entries ran into one another, as ``(name, positions)`` pairs.

    ``wanted`` is each entry's name once cleaned up and ``given`` the name
    :func:`unique` settled on. For every name somebody wanted and did not get,
    the positions are the entry that holds it and each entry moved off it, in
    file order. With ``a, a!, a_1`` that is two clashes: ``a`` held by the
    first and wanted by the second, then ``a_1`` held by the second and wanted
    by the third.
    """
    holder = {name: position for position, name in enumerate(given)}
    found: dict[str, list[int]] = {}
    for position, (want, got) in enumerate(zip(wanted, given)):
        if want != got:
            found.setdefault(want, [holder[want]]).append(position)
    return [(name, sorted(positions)) for name, positions in found.items()]
