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

__all__ = ["ExportError", "LoadError"]


class LoadError(RuntimeError):
    """A source that could not be loaded."""


class ExportError(ValueError):
    """A result this server will not write in the form it was asked for."""
