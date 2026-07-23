"""tests/v3/test_ingest_streaming.py — E8.4: the I-4 streaming tool surface.

FR-404's exit-gate leg over the real stack: `query` cuts over to a
genuinely-streaming stream past the inline budget (the peeked frames
re-chained, nothing re-executed), every advertised chunk is
retrievable in cursor order through `fetch_chunk`, exhaustion reports
the final total as metadata and closes the stream, expiry and the
row-24 admission cap are structured refusals with the declared
recovery paths, and `read_file`/`query_file` serve oversized results
from their admitted load-then-serve buffers (I-2's honest
classification, recorded in the inventory registry). Envelope shapes
for `StreamOpened`/`ServedChunk` are asserted through the shaper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

import localdata_mcp.ingest.runtime as runtime
from localdata_mcp.ingest.connectors.file.tools import query_file, read_file
from localdata_mcp.ingest.connectors.sql.tools import query
from localdata_mcp.ingest.streams import close_stream, fetch_chunk
from localdata_mcp.nexus.chokepoint.guard import (
    Chokepoint,
    GuardedExecutionError,
    Result,
    ServedChunk,
    StreamOpened,
)
from localdata_mcp.nexus.config.models import (
    ConfigModel,
    QueryConfig,
    ResponseConfig,
    SecurityConfig,
)
from localdata_mcp.nexus.config.endpoints import EndpointDeclaration
from localdata_mcp.nexus.contract.inventory import (
    Kind,
    StreamingClass,
    entries,
)

# Tiny, provably-non-S8 budgets: results beyond 3 rows (or 3 rows per
# chunk) stream; the byte side is left roomy so the row side decides.
_TINY_ROWS = 3
_ROOMY_BYTES = 65_536


def _config(tmp_path: Path) -> ConfigModel:
    return ConfigModel(
        query=QueryConfig(default_chunk_size=2),
        response=ResponseConfig(
            inline_max_rows=_TINY_ROWS, inline_max_bytes=_ROOMY_BYTES
        ),
        security=SecurityConfig(allowed_paths=(str(tmp_path),)),
        endpoints={
            "warehouse": EndpointDeclaration(
                name="warehouse",
                dsn=f"sqlite:///{tmp_path / 'w.db'}",
                posture="read_write",
            ),
        },
    )


def _seed_rows(guard: Chokepoint, count: int) -> None:
    with guard._persistence.connection("warehouse") as connection:
        from sqlalchemy import text

        connection.execute(text("CREATE TABLE t (id INTEGER, label TEXT)"))
        for index in range(count):
            connection.execute(
                text("INSERT INTO t VALUES (:i, :l)"),
                {"i": index, "l": f"row-{index}"},
            )
        connection.commit()


@pytest.fixture()
def booted(tmp_path: Path) -> Iterator[Chokepoint]:
    guard = Chokepoint.boot(_config(tmp_path), environ={})
    _seed_rows(guard, 9)
    runtime.configure_ingest(guard)
    yield guard
    runtime._CHOKEPOINT = None
    guard.shutdown()


class TestQueryCutover:
    def test_small_result_stays_inline(self, booted: Chokepoint) -> None:
        result = query("warehouse", "SELECT id FROM t WHERE id < 2 ORDER BY id")
        assert isinstance(result, Result)
        assert result.rows == ((0,), (1,))

    def test_large_result_opens_a_stream(self, booted: Chokepoint) -> None:
        opened = query("warehouse", "SELECT id, label FROM t ORDER BY id")
        assert isinstance(opened, StreamOpened)
        assert opened.columns == ("id", "label")
        assert opened.advertised_chunks > 0

    def test_every_advertised_chunk_is_retrievable_in_order(
        self, booted: Chokepoint
    ) -> None:
        """FR-404: drain the stream; rows arrive complete and ordered."""
        opened = query("warehouse", "SELECT id FROM t ORDER BY id")
        assert isinstance(opened, StreamOpened)
        collected: list[int] = []
        while True:
            served = fetch_chunk(opened.stream_id)
            assert isinstance(served, ServedChunk)
            if served.chunk_id is None:
                assert served.closed is True
                assert served.total_chunks == 5  # 9 rows / chunk_size 2
                break
            collected.extend(row[0] for row in served.rows)
        assert collected == list(range(9))

    def test_fetch_after_exhaustion_is_the_expired_refusal(
        self, booted: Chokepoint
    ) -> None:
        opened = query("warehouse", "SELECT id FROM t ORDER BY id")
        assert isinstance(opened, StreamOpened)
        while fetch_chunk(opened.stream_id).chunk_id is not None:
            pass
        with pytest.raises(GuardedExecutionError) as refusal:
            fetch_chunk(opened.stream_id)
        assert "re-issue" in refusal.value.structured.suggestion.lower()

    def test_close_stream_releases_and_is_idempotent(self, booted: Chokepoint) -> None:
        opened = query("warehouse", "SELECT id FROM t ORDER BY id")
        assert isinstance(opened, StreamOpened)
        outcome = close_stream(opened.stream_id)
        assert outcome == {"stream_id": opened.stream_id, "closed": True}
        again = close_stream(opened.stream_id)
        assert again["closed"] is True
        with pytest.raises(GuardedExecutionError):
            fetch_chunk(opened.stream_id)

    def test_stream_cap_refusal_names_close_stream(self, tmp_path: Path) -> None:
        """The row-24 admission row: the cap holds and the refusal
        carries the declared recovery."""
        config = _config(tmp_path)
        capped = ConfigModel(
            query=QueryConfig(
                default_chunk_size=2, max_concurrent_streams_per_endpoint=1
            ),
            response=config.response,
            security=config.security,
            endpoints=config.endpoints,
        )
        guard = Chokepoint.boot(capped, environ={})
        _seed_rows(guard, 9)
        runtime.configure_ingest(guard)
        try:
            first = query("warehouse", "SELECT id FROM t ORDER BY id")
            assert isinstance(first, StreamOpened)
            with pytest.raises(GuardedExecutionError) as refusal:
                query("warehouse", "SELECT label FROM t ORDER BY id")
            assert "close_stream" in refusal.value.structured.suggestion
        finally:
            runtime._CHOKEPOINT = None
            guard.shutdown()


class TestLoadThenServeCutover:
    def test_read_file_streams_past_the_budget(
        self, booted: Chokepoint, tmp_path: Path
    ) -> None:
        big = tmp_path / "big.csv"
        big.write_text("a,b\n" + "\n".join(f"{i},{i * 2}" for i in range(8)))
        opened = read_file(str(big))
        assert isinstance(opened, StreamOpened)
        collected: list[int] = []
        while True:
            served = fetch_chunk(opened.stream_id)
            if served.chunk_id is None:
                break
            collected.extend(int(row[0]) for row in served.rows)
        assert collected == list(range(8))

    def test_read_file_small_stays_inline(
        self, booted: Chokepoint, tmp_path: Path
    ) -> None:
        small = tmp_path / "small.csv"
        small.write_text("a,b\n1,2\n3,4\n")
        result = read_file(str(small))
        assert isinstance(result, Result)
        assert result.row_count == 2

    def test_query_file_streams_from_the_admitted_buffer(
        self, booted: Chokepoint, tmp_path: Path
    ) -> None:
        import sqlite3

        db = tmp_path / "local.db"
        with sqlite3.connect(db) as connection:
            connection.execute("CREATE TABLE t (id INTEGER)")
            connection.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(7)])
        opened = query_file(str(db), "SELECT id FROM t ORDER BY id")
        assert isinstance(opened, StreamOpened)
        served = fetch_chunk(opened.stream_id)
        assert [row[0] for row in served.rows] == [0, 1]


class TestInventoryClassification:
    def test_sql_engines_are_genuinely_streaming(self) -> None:
        for entry in entries(kind=Kind.SQL_ENGINE):
            assert entry.streaming is StreamingClass.GENUINELY_STREAMING

    def test_file_formats_and_stores_are_load_then_serve(self) -> None:
        """I-2's honest-capability cell: every non-SQL entry —
        including the `query_file` EphemeralFileConnection path's file
        formats — is declared load-then-serve."""
        for entry in entries():
            if entry.kind is not Kind.SQL_ENGINE:
                assert entry.streaming is StreamingClass.LOAD_THEN_SERVE


class TestEnvelopeShapes:
    def test_stream_reference_and_chunk_envelopes(self, booted: Chokepoint) -> None:
        """The wrapper-level truth: shaped tool calls render the
        stream reference and chunk shapes with the stream's state."""
        from localdata_mcp.nexus.contract.registry import default_registry
        from localdata_mcp.nexus.contract.spec_modules import load_spec_modules
        from localdata_mcp.nexus.response.envelope import ResponseShaper

        load_spec_modules()
        shaper = ResponseShaper(booted._config, default_registry())
        opened = query("warehouse", "SELECT id FROM t ORDER BY id")
        assert isinstance(opened, StreamOpened)
        spec = default_registry().lookup("query")
        envelope = shaper.shape_envelope(opened, spec)
        assert opened.stream_id in envelope.inline
        assert "fetch_chunk" in envelope.inline
        served = fetch_chunk(opened.stream_id)
        chunk_spec = default_registry().lookup("fetch_chunk")
        chunk_envelope = shaper.shape_envelope(served, chunk_spec)
        assert "Chunk 0" in chunk_envelope.inline
        assert "| id |" in chunk_envelope.inline
