"""Character-separated text: CSV, TSV, and any file split at one character.

Reading such a file is one call to pandas. What needs care is the warning when
the file was read at the wrong separator, which is most of this module: it
recognises the two shapes a wrong separator leaves in the header, and names the
separator it used and the one the header holds. It is its own module because
that warning is also used outside the readers, by the load path.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .result import ReadResult, Reader, single_table


#: Delimiters worth naming in a warning. The character that separated the file
#: is never among the candidates, because a column name cannot contain it.
_COMMON_DELIMITERS = {";": "';'", "\t": "a tab", "|": "'|'", ",": "','"}


def _spelled(separator: str) -> str:
    """How a warning names the separator a file was read at.

    A common delimiter has its spelling above, which says "a tab" rather than
    quoting an invisible character. Any other one the caller declared, ``:``
    say, is quoted as its ``repr``, so it reads ``':'``. The map holds only the
    candidates a note may name, so indexing it with the separator used would
    fail on every other character (#156).
    """
    return _COMMON_DELIMITERS.get(separator, repr(separator))


def delimited(separator: str) -> Reader:
    """A reader for character-separated text, at the separator it is given.

    Which separator that is is decided before this is called, by
    ``formats.resolve_separator``: the one the caller declared, else the
    format's published standard (a comma for ``.csv``, a tab for ``.tsv``). This
    reader only applies it. A factory rather than a parameter so that the
    separator does not have to be threaded through the signature of every other
    format's reader, none of which has one.
    """

    def read(path: Path) -> ReadResult:
        # `keep_default_na` is left on: pandas' blank/NA handling is what turns
        # an empty cell into NULL rather than the string "".
        frame = pd.read_csv(path, sep=separator)
        return single_table(frame, fat_column_note(frame, separator, path.name))

    return read


def fat_column_note(frame: pd.DataFrame, separator: str, name: str) -> tuple[str, ...]:
    """Say when a file was plainly read at the wrong separator.

    This is the net under the standard default. A file named ``.csv`` is read at
    a comma unless the caller says otherwise, and one that is really separated
    by ``;`` — common from European tools — then loads wrongly. The same holds
    for a separator the caller declared wrongly. Either way the character used
    is known, so the note can name it and name the one the header holds. Two
    shapes are the signature of the mistake:

    - **One column whose header still contains a common delimiter.** The whole
      header line became the column.
    - **Two or more columns, every one of whose names contains the same other
      common delimiter (#152).** The header happened to hold the separator
      tried, so the file split, at the wrong places: ``region;price, eur;units``
      read at ``,`` gives ``region;price`` and `` eur;units``. *Every* name, not
      some or most: a correctly read ``;`` file with one column called
      ``price, eur`` must stay quiet, and a threshold would be a guess.

    The edge this leaves open, deliberately: a header where some field lacks
    the other separator (``x,y;z`` read at ``,`` gives ``x`` and ``y;z``) is not
    caught. Catching it means warning on names that merely hold such a
    character, as ``price, eur`` does.

    This states what it found; it does not re-read the file at the character it
    spotted. Sniffing is the fail-open shape this project keeps being bitten by,
    and a guess that is usually right is the worst kind.

    **What it quotes is the header line, not the column names the caller ends
    up with.** Identifier sanitisation runs downstream of every reader, and it
    replaces exactly the characters this note is about, so a stored name never
    contains the one being reported. Citing it named evidence the answer no
    longer held (#101); the header is the same evidence and the caller can check
    it against the file. With several columns it is the names the reader split,
    joined again at the separator used.
    """
    names = [str(column) for column in frame.columns]
    if len(names) == 1:
        return _single_column_note(names[0], separator, name)
    return _split_header_note(names, separator, name)


def _single_column_note(header: str, separator: str, name: str) -> tuple[str, ...]:
    """The note for a header line that became one column."""
    found = [
        spelling
        for character, spelling in _COMMON_DELIMITERS.items()
        if character != separator and character in header
    ]
    if not found:
        return ()

    return (
        f"{name} was read at {_spelled(separator)} and loaded as a "
        f"single column, from a header line reading {_shortened(header)} — "
        f"which contains {' and '.join(found)}, what a file separated by one of "
        f"those looks like when it is read at {_spelled(separator)}. "
        f"The stored column name is that header sanitised, so it is not the "
        f"string quoted here. If that is the case, attach it again with "
        f"delimiter set to the right character. Nothing here guesses it.",
    )


def _split_header_note(names: list[str], separator: str, name: str) -> tuple[str, ...]:
    """The note for a header that split, every piece holding the same character."""
    shared = [
        spelling
        for character, spelling in _COMMON_DELIMITERS.items()
        if character != separator and all(character in each for each in names)
    ]
    if not shared:
        return ()

    used = _spelled(separator)
    which = "that character" if len(shared) == 1 else "one of those characters"
    return (
        f"{name} was read at {used} and loaded as {len(names)} columns, from a "
        f"header line reading {_shortened(separator.join(names))} — every one of "
        f"those column names contains {' and '.join(shared)}, what a file "
        f"separated by {which} looks like when it is read at {used}. The "
        f"stored column names are those pieces sanitised, so they are not the "
        f"strings quoted here. If that is the case, attach it again with "
        f"delimiter set to the right character. Nothing here guesses it.",
    )


#: How much of a header line a warning quotes. A file read at the wrong
#: separator turns its whole first line into one name, and a wide file's is
#: hundreds of characters — long enough to bury the sentence explaining it.
_QUOTED_HEADER_LIMIT = 60


def _shortened(text: str) -> str:
    """The header as a quoted string, cut to something a sentence can carry."""
    if len(text) <= _QUOTED_HEADER_LIMIT:
        return repr(text)
    return repr(text[:_QUOTED_HEADER_LIMIT] + "…")
