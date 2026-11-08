"""Columnar files: Parquet, Feather/Arrow and ORC.

These formats store a type with every column, so the reader infers nothing and
reports nothing: it hands the file to pyarrow and returns the one table. It is
its own module because it shares nothing else with the text formats.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..errors import LoadError
from .result import ReadResult, require_library, single_table


def read_columnar(path: Path) -> ReadResult:
    """Parquet, Feather/Arrow and ORC — typed formats, so nothing is inferred.

    Feather goes through ``pyarrow.ipc`` rather than ``pandas.read_feather``,
    which routes to a ``pyarrow.feather`` entry point deprecated as of pyarrow
    24 and warns on every call. Same file format either way.
    """
    suffix = path.suffix.lower()
    require_library("pyarrow", "parquet", f"Reading {suffix}")
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
    return single_table(frame)
