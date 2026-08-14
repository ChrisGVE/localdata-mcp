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

__all__ = ["ExportError", "LoadError", "undeclared_separator"]


class LoadError(RuntimeError):
    """A source that could not be loaded."""


class ExportError(ValueError):
    """A result this server will not write in the form it was asked for."""


def undeclared_separator(name: str, *, reading: bool) -> str:
    """The one sentence both sides say when no separator was declared.

    It lives here, beside the two errors that carry it, because the read side
    and the write side each raise their own error type over the *same* rule —
    character-separated text has no separator until the caller says what it is.
    Said twice it would drift, and the halves that drift are exactly the halves
    a caller compares when a round trip stops working.

    ``reading`` picks the verb and the consequence, which are the only parts
    that legitimately differ: reading at the wrong separator misreads data that
    already exists, and writing at an assumed one produces a file whose name
    disagrees with its contents.
    """
    if reading:
        return (
            f"{name} is character-separated text, and this server does not "
            f"guess what separates it. Pass delimiter — a file called .csv is "
            f"as likely to be separated by ';' or '|' or a tab as by ',', and "
            f"reading it at the wrong one changes what the data is while the "
            f"result looks perfectly ordinary. The extension does not decide "
            f"this and neither does anything else here."
        )
    return (
        f"{name} is character-separated text, so say what should separate it: "
        f"pass delimiter. The suffix does not decide it, here or anywhere else "
        f"— a file whose name and separator disagree is something only whatever "
        f"reads it next finds out about, and by then the file is what exists."
    )
