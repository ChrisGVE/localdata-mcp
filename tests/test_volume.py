"""Volume behaviour: the memory invariant, asserted as a growth property.

Two claims are defended here, and both are only meaningful as a *shape*:

* **the row sequence handed to executemany is never materialised** — the load
  side;
* **a query written to a file is never materialised** — the export side, where
  the rows go from the database cursor to the writer without a list of them
  existing anywhere in between.

Asserting "the peak stays under N megabytes" would pass on a machine with more
headroom and hide a per-row accumulation entirely. Asserting the peak barely
moves while the row count grows four-fold catches it.

Measured reference: the pandas writer peaks at ~35x the frame's own size and
scales with total rows even when chunked, because it materialises before it
chunks. Feeding a lazy iterator holds a flat peak from 100,000 rows through
1,600,000.

Marked slow; run with ``-m slow`` or as part of the full suite.
"""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path

import pandas as pd
import pytest

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.formats import DELIMITED
from localdata_mcp.config import Config
from localdata_mcp.loader import Workspace

pytestmark = pytest.mark.slow

SMALL_ROWS = 50_000
LARGE_ROWS = 200_000


@pytest.fixture()
def root(monkeypatch, tmp_path):
    config_module.use(Config(roots=(tmp_path,)))
    return tmp_path


def _write_csv(path: Path, rows: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("id,name,score,flag\n")
        for index in range(rows):
            handle.write(f"{index},name_{index},{index * 1.5},{index % 2}\n")


def _insert_peak(path: Path, table: str) -> tuple[float, int, float]:
    """Return (peak MB, rows landed, seconds) for the *insert* alone.

    The frame is read before tracing starts. That boundary is the whole point:
    a DataFrame's own size scales with row count no matter how it is written, so
    tracing the read as well measures pandas rather than our insert path and
    reports linear growth whatever we do.

    **The returned duration is inflated ~5x and must not be quoted as
    throughput.** ``tracemalloc`` costs about that much wall clock, measured
    here at 5.32x and 5.11x. Untraced, the same inserts run 200,000 rows in
    1.84 s and 800,000 in 8.09 s. Nothing asserts on the duration for exactly
    this reason.
    """
    workspace = Workspace.in_memory()
    workspace.attach_memory("bulk")
    try:
        frame = pd.read_csv(path)

        tracemalloc.start()
        started = time.perf_counter()
        info = workspace.insert_frame(frame, table, source=str(path), tag="bulk")
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # A write that does nothing is very fast and allocates nothing, which is
        # the same shape as the result we want. Prove the rows are really there.
        _, rows = workspace.query("bulk", f"SELECT count(*) FROM {table}")
        assert rows[0][0] == info.row_count
        return peak / 1_048_576, info.row_count, elapsed
    finally:
        workspace.close()


def test_insert_peak_does_not_scale_with_row_count(root):
    small_path = root / "small.csv"
    large_path = root / "large.csv"
    _write_csv(small_path, SMALL_ROWS)
    _write_csv(large_path, LARGE_ROWS)

    small_peak, small_rows, _ = _insert_peak(small_path, "small")
    large_peak, large_rows, _ = _insert_peak(large_path, "large")

    assert small_rows == SMALL_ROWS
    assert large_rows == LARGE_ROWS

    # Four times the rows must not cost four times the peak. The generous bound
    # is deliberate: this test is here to catch a per-row accumulation, which
    # would show up as a ratio near 4, not to police the constant.
    growth = large_peak / max(small_peak, 0.001)
    assert growth < 2.0, (
        f"insert peak grew {growth:.2f}x for a 4x row increase "
        f"({small_peak:.2f} MB -> {large_peak:.2f} MB). The row iterator is "
        f"probably being materialised somewhere."
    )


def test_large_file_answers_correctly(root):
    """Volume must not cost correctness."""
    path = root / "big.csv"
    _write_csv(path, LARGE_ROWS)

    workspace = Workspace.in_memory()
    workspace.attach_memory("bulk")
    try:
        (info,) = workspace.load_file(str(path), "bulk", delimiter=",")
        assert info.row_count == LARGE_ROWS

        _, rows = workspace.query("bulk", "SELECT sum(id) FROM big")
        assert rows[0][0] == LARGE_ROWS * (LARGE_ROWS - 1) // 2

        _, rows = workspace.query("bulk", "SELECT count(*) FROM big WHERE flag = 1")
        assert rows[0][0] == LARGE_ROWS // 2

        _, rows = workspace.query("bulk", "SELECT max(score) FROM big")
        assert rows[0][0] == pytest.approx((LARGE_ROWS - 1) * 1.5)
    finally:
        workspace.close()


def _export_peak(
    root: Path, rows: int, name: str, suffix: str = ".csv"
) -> tuple[float, int]:
    """Return (peak MB, rows written) for ``query`` writing straight to a file.

    Driven through ``server.query`` rather than through ``Workspace`` and
    ``export_rows`` separately, because the claim is about the *path* between
    them: either of those two halves can be perfectly lazy while the code that
    joins them builds a list. Testing the seam underneath would validate the
    seam and leave the join untested.

    The source file is written and attached before tracing starts, so what is
    measured is the cursor-to-disk path and nothing else.
    """
    source = root / f"{name}.csv"
    _write_csv(source, rows)

    server_module._reset()
    try:
        attached = server_module.attach(str(source), delimiter=",")
        assert attached["ok"], attached
        nickname = attached["nickname"]
        target = root / f"{name}-out{suffix}"

        tracemalloc.start()
        answer = server_module.query(
            nickname,
            f"SELECT id, name, score, flag FROM {name}",
            path=str(target),
            **({"delimiter": ","} if suffix in DELIMITED else {}),
        )
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert answer["ok"], answer
        # Checked against the file rather than the report: a writer that
        # streamed nothing and said it wrote everything is the shape this whole
        # test exists to catch. Line-counting is exact for the delimited case
        # and meaningless for the rest, which get the weaker check.
        if suffix == ".csv":
            with target.open() as handle:
                assert sum(1 for _ in handle) == rows + 1
        else:
            assert target.stat().st_size > 0
        return peak / 1_048_576, answer["rows_written"]
    finally:
        server_module._reset()


def test_export_peak_does_not_scale_with_row_count(root):
    """A result written to a file never exists as a list of rows.

    ``Workspace.query`` streams from the driver but returns a list, so a query
    exported to a file used to hold every row in memory before the writer saw
    the first one — on top of whatever the writer itself builds. For a format
    that writes row by row, the peak must be flat.
    """
    small_peak, small_rows = _export_peak(root, SMALL_ROWS, "small")
    large_peak, large_rows = _export_peak(root, LARGE_ROWS, "large")

    assert small_rows == SMALL_ROWS
    assert large_rows == LARGE_ROWS

    # Same generous bound and same reasoning as the insert test above: this
    # catches an accumulation, which shows as a ratio near 4, not a constant.
    growth = large_peak / max(small_peak, 0.001)
    assert growth < 2.0, (
        f"export peak grew {growth:.2f}x for a 4x row increase "
        f"({small_peak:.2f} MB -> {large_peak:.2f} MB). The result is probably "
        f"being materialised between the cursor and the writer."
    )


def test_yaml_export_peak_does_not_scale_with_row_count(root):
    """YAML is written a chunk at a time, so it inherits the flat peak.

    It is the format that made this worth doing. The writer used to build a
    dict per row into one list and hand the whole list to ``safe_dump``, which
    then built its own representation of all of it — two full copies of the
    result on top of the list of rows that reached it. A million rows of eleven
    columns crossed 16 GB on the way back in (CONSTRAINTS §10.7); the write side
    is now bounded whatever the read side does.
    """
    pytest.importorskip("yaml")

    small_peak, small_rows = _export_peak(root, SMALL_ROWS, "ysmall", ".yaml")
    large_peak, large_rows = _export_peak(root, LARGE_ROWS, "ylarge", ".yaml")

    assert small_rows == SMALL_ROWS
    assert large_rows == LARGE_ROWS

    growth = large_peak / max(small_peak, 0.001)
    assert growth < 2.0, (
        f"yaml export peak grew {growth:.2f}x for a 4x row increase "
        f"({small_peak:.2f} MB -> {large_peak:.2f} MB). The writer is probably "
        f"collecting the rows before dumping them."
    )


def test_export_of_a_large_result_is_complete(root):
    """Volume must not cost completeness on the way out."""
    from localdata_mcp.export import export_rows

    path = root / "big.csv"
    _write_csv(path, SMALL_ROWS)

    workspace = Workspace.in_memory()
    workspace.attach_memory("bulk")
    try:
        workspace.load_file(str(path), "bulk", delimiter=",")
        columns, rows = workspace.query("bulk", "SELECT id, score FROM big")
        target = root / "exported.csv"
        result = export_rows(columns, rows, str(target), delimiter=",")

        assert result.row_count == SMALL_ROWS
        # Header plus every row, counted from the file rather than the report.
        with target.open() as handle:
            assert sum(1 for _ in handle) == SMALL_ROWS + 1
    finally:
        workspace.close()
