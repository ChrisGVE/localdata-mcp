"""The two failures the data layer raises, in the one place both sides can see.

They live here rather than in the modules that raise them because the format
layer sits *below* both. A reader has to be able to say a file could not be
read, and a writer that a destination will not be written, without either of
them importing the other's module — which is what an error class defined beside
its raiser would have forced.

The distinction between them is not tidiness. A ``LoadError`` is about a source
the caller pointed at: something that is already there, and the answer is what
is wrong with it. An ``ExportError`` is about a destination the caller named:
nothing exists yet, and the answer is why this server will not produce it. They
are handled differently upstream for that reason — an export that fails cleans
up the partial file it was in the middle of, and a load has nothing to clean.

Both carry messages written **for the caller**, not for a log. Upstream code
puts them into a refusal payload close to verbatim, so the text has to stand on
its own without the exception type or a stack trace beside it.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["ExportError", "LoadError", "undeclared_separator"]


class LoadError(RuntimeError):
    """A source that could not be loaded."""


class ExportError(ValueError):
    """A result this server will not write in the form it was asked for."""


def undeclared_separator(name: str, *, reading: bool) -> str:
    """The one sentence both sides say when a separator has to be declared.

    A delimiter is only ever missing for a suffix with no published standard
    separator. ``.csv`` has one (a comma, RFC 4180) and ``.tsv`` has one (a tab,
    the IANA ``text/tab-separated-values`` registration), and each is the
    default when nothing is declared (project first principle 7). ``.txt`` has
    none, so for it nothing is assumed and the caller is asked instead.

    It lives here, beside the two errors that carry it, because the read side
    and the write side each raise their own error type over the *same* rule.
    Said twice it would drift, and the halves that drift are exactly the halves
    a caller compares when a round trip stops working.

    ``reading`` picks the next step, which is the only part that legitimately
    differs. Reading, the caller may not be able to see the file, so the step
    that works without a shell is given too: try a separator and read the
    warnings, which name the one the header holds when the guess was wrong.
    Writing, the caller is choosing, so it is told how to choose.
    """
    suffix = Path(name).suffix.lower() or "this"
    gap = (
        f"{name} is character-separated text, and a {suffix} file has no "
        f"published standard separator — unlike .csv (a comma) or .tsv (a "
        f"tab) — so nothing is assumed for it."
    )
    if reading:
        return (
            f"{gap} Pass delimiter. If you can see the file's first line, pass "
            f"the character between its fields. If you cannot, attach at the "
            f"likeliest one (',' ';' a tab or '|') and read warnings: read at "
            f"the wrong one, the file loads as a single column, or as columns "
            f"whose every name holds the same other separator, and a warning "
            f"names that separator — detach and attach again at that one."
        )
    return (
        f"{gap} Pass delimiter to say what should separate it, or name the "
        f"file .csv or .tsv to write at that format's standard."
    )
