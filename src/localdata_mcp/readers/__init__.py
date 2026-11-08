"""The readers: one function per input format, and the shapes they return.

A reader is handed a path that has already been judged safe and returns the
tables it found in it, as pandas frames. That is the whole contract. Nothing
here opens a database, holds a slot, or knows a nickname: a file becomes frames
here, and everything downstream works from the frames.

They live apart from ``loader`` so that ``formats`` — the one table naming what
reads and what writes each suffix — can reach them without importing the module
that *calls* them. The same split put the writers in ``writers``.

**A reader may refuse.** Reading from a file whose shape is not a table does
not produce bad rows to be caught later — it produces a ``LoadError`` naming
what was wrong with the file, because a wrong answer that looks like data is the
failure nothing downstream can report.

**A reader may also warn.** Several formats have a shape a table does not: a
JSON object with arrays under its keys, a workbook with several sheets. Those
come back as notes on the result rather than as refusals, because the data is
readable and the caller is the one who can say whether it was read the way they
meant. Text read at the wrong separator is the same case: it is readable, only
wrongly split, and ``fat_column_note`` says so when the header shows it.

**One module per kind of file**, and this package re-exports what the rest of
the server uses, so callers import from ``readers`` alone. ``result`` holds the
contract every reader shares; ``separator`` reads character-separated text;
``documents`` reads JSON, JSON Lines and YAML; ``markup`` reads XML;
``workbooks`` reads Excel, OpenDocument and Apple Numbers; ``columnar`` reads
Parquet, Feather/Arrow and ORC.
"""

from __future__ import annotations

# Re-exported for the rest of the server. Each is written ``name as name``,
# which is how a package says an import is there to be re-exported: ``__all__``
# lists only the shapes, as it did when this was a single module.
from .columnar import read_columnar as read_columnar
from .documents import read_json as read_json
from .documents import read_jsonl as read_jsonl
from .documents import read_yaml as read_yaml
from .markup import read_xml as read_xml
from .result import NamedFrame, ReadResult, Reader
from .separator import delimited as delimited
from .separator import fat_column_note as fat_column_note
from .workbooks import read_numbers as read_numbers
from .workbooks import workbook as workbook

# tests/test_uncommon_separator_note.py reaches these as readers._single_column_note
# and readers._split_header_note, and the tests are frozen.
from .separator import _single_column_note as _single_column_note
from .separator import _split_header_note as _split_header_note

__all__ = ["NamedFrame", "ReadResult", "Reader"]
