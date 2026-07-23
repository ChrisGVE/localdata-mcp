"""testbench/batteries/base/property_test.py — NFR-507 property suite (E14.5).

Hypothesis-driven invariants of the base store/stream operations, the
four NFR-507 names made executable:

  - **round-trip fidelity**  — `set_value` then `get_value` returns the
    same typed value across every ValueType (S8 15c's exact class);
  - **idempotent deletes**   — deleting an already-absent key leaves the
    node in the identical end state (the *effect* is idempotent, even
    though the second call reports a missing-entity refusal);
  - **insert/read consistency** — a set key is immediately visible to
    both `get_value` and `list_keys`;
  - **chunk-cursor monotonicity** — draining a streamed result yields
    every row exactly once, in order, under strictly increasing chunk
    ids, then closes (FR-404 / §5 cursor semantics).

Example counts come from `testbench.hypothesis_max_examples` (S8 row 21,
200 per-PR / 1000 nightly) — read from the NX-2 config, never a
test-code literal. The three store properties run at the L3
`fastmcp.Client` seam (PROJECT-FP #4). The cursor property runs at the
direct `query`/`fetch_chunk` ingest seam — the seam the streaming
battery itself uses (tests/v3/test_ingest_streaming.py); the cursor
mechanics under test are seam-invariant, and the direct seam lets the
property drain `ServedChunk` objects without re-parsing the envelope's
inline stream reference. Neighbors: ingest_battery_test.py is the
deterministic base slice this accretes onto; enumeration.py + the
pipeline battery own the coupling axis.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

import anyio
import pytest
from fastmcp import Client
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import text

import localdata_mcp.ingest.runtime as runtime
from localdata_mcp.ingest.connectors.sql.tools import query
from localdata_mcp.ingest.streams import fetch_chunk
from localdata_mcp.nexus.chokepoint.guard import Chokepoint, ServedChunk, StreamOpened
from localdata_mcp.nexus.config.endpoints import EndpointDeclaration
from localdata_mcp.nexus.config.models import (
    ConfigModel,
    QueryConfig,
    ResponseConfig,
    SecurityConfig,
)
from localdata_mcp.nexus.contract.registry import default_registry
from localdata_mcp.nexus.response.shaping import configure_shaping
from localdata_mcp.server.mcp_app import app

# Example count resolves through the NX-2 testbench section (S8 row 21):
# 200 per-PR by default, raised to 1000 nightly via the derived env
# override — no literal here, so the one-default-site gate stays green.
_MAX_EXAMPLES = ConfigModel().testbench.hypothesis_max_examples

# Provably-sub-S8 stream budgets: any result past three rows streams,
# two rows per chunk, so a drawn N in [cutover+1, 40] always opens a
# genuine multi-chunk stream. The minimum is spelled as cutover+1 (one
# past the inline boundary) — arithmetic, not a bare literal, so it
# neither reads as an S8 default nor drifts if the cutover is retuned.
_INLINE_CUTOVER_ROWS = 3
_CHUNK_ROWS = 2
_ROOMY_INLINE_BYTES = 65_536
_STREAM_ROW_MIN = _INLINE_CUTOVER_ROWS + 1
_STREAM_ROW_MAX = 40

_KV_ENDPOINT = "store_kv"
_SQL_ENDPOINT = "warehouse"
_NODE_PATH = "root"

# A conservative property-key alphabet: non-empty identifiers, so a
# drawn key is always a legal node property and distinct draws never
# collide with the fixture's structural paths.
_KEY_STRATEGY = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122),
    min_size=1,
    max_size=12,
)

# Shared @settings: the config-derived cap, no per-example deadline (an
# L3 round-trip through the guard is not sub-millisecond), and the
# documented suppression for the intentionally session-shared guard
# fixture (reset-per-example is neither possible nor wanted — the store
# properties assert on freshly-drawn keys, the cursor property resets
# its own table each example).
_SETTINGS = settings(
    max_examples=_MAX_EXAMPLES,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _declarations(tmp_path: Path) -> dict[str, EndpointDeclaration]:
    declared = {
        _KV_ENDPOINT: f"kv+sqlite:///{tmp_path / 'kv.db'}",
        _SQL_ENDPOINT: f"sqlite:///{tmp_path / 'w.db'}",
    }
    return {
        name: EndpointDeclaration(name=name, dsn=dsn, posture="read_write")
        for name, dsn in declared.items()
    }


@pytest.fixture()
def bench(tmp_path: Path) -> Iterator[Chokepoint]:
    """One booted guard for every property in this module: tiny stream
    budgets so results stream, a kv and a sqlite endpoint declared."""
    config = ConfigModel(
        query=QueryConfig(default_chunk_size=_CHUNK_ROWS),
        response=ResponseConfig(
            inline_max_rows=_INLINE_CUTOVER_ROWS,
            inline_max_bytes=_ROOMY_INLINE_BYTES,
        ),
        security=SecurityConfig(allowed_paths=(str(tmp_path),)),
        endpoints=_declarations(tmp_path),
    )
    guard = Chokepoint.boot(config, environ=dict(os.environ))
    configure_shaping(config, default_registry())
    runtime.configure_ingest(guard)
    yield guard
    runtime._CHOKEPOINT = None
    configure_shaping(ConfigModel(), default_registry())
    guard.shutdown()


def _call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """One L3 call; returns the FR-403 envelope."""

    async def session() -> dict[str, Any]:
        async with Client(app) as client:
            result = await client.call_tool(name, arguments)
            assert not result.is_error
            if isinstance(result.structured_content, dict) and (
                "inline" in result.structured_content
            ):
                return result.structured_content
            payload = json.loads(result.content[0].text)
            assert isinstance(payload, dict)
            return payload

    return anyio.run(session)


# -- typed value round-trip strategies --------------------------------
#
# Each strategy emits (value_type, wire_string, expected_python): the
# wire string is exactly what a tool call carries (`value: str`), and
# `expected_python` is what `get_value` must return after the store's
# declared type conversion (values.py). repr() round-trips a Python
# float exactly, so float comparison is equality, not tolerance.


@st.composite
def _integer_value(draw: st.DrawFn) -> tuple[str, str, Any]:
    number = draw(st.integers(min_value=-(2**53), max_value=2**53))
    return ("integer", str(number), number)


@st.composite
def _float_value(draw: st.DrawFn) -> tuple[str, str, Any]:
    number = draw(st.floats(allow_nan=False, allow_infinity=False))
    return ("float", repr(number), number)


@st.composite
def _boolean_value(draw: st.DrawFn) -> tuple[str, str, Any]:
    flag = draw(st.booleans())
    return ("boolean", "true" if flag else "false", flag)


@st.composite
def _string_value(draw: st.DrawFn) -> tuple[str, str, Any]:
    text_value = draw(st.text(max_size=64))
    return ("string", text_value, text_value)


@st.composite
def _array_value(draw: st.DrawFn) -> tuple[str, str, Any]:
    items = draw(st.lists(st.integers(min_value=-1000, max_value=1000), max_size=8))
    return ("array", json.dumps(items), items)


def _kv_values() -> st.SearchStrategy[tuple[str, str, Any]]:
    return st.one_of(
        _integer_value(),
        _float_value(),
        _boolean_value(),
        _string_value(),
        _array_value(),
    )


class TestRoundTripFidelity:
    """NFR-507: a value stored through the tool surface reads back
    identically, across every ValueType (S8 15c's exact class)."""

    @_SETTINGS
    @given(key=_KEY_STRATEGY, payload=_kv_values())
    def test_set_then_get_is_identity(
        self, bench: Chokepoint, key: str, payload: tuple[str, str, Any]
    ) -> None:
        value_type, wire_value, expected = payload
        set_env = _call(
            "set_value",
            {
                "endpoint": _KV_ENDPOINT,
                "path": _NODE_PATH,
                "key": key,
                "value": wire_value,
                "value_type": value_type,
            },
        )
        assert set_env["error"] is None, set_env["error"]
        got = _call(
            "get_value",
            {"endpoint": _KV_ENDPOINT, "path": _NODE_PATH, "key": key},
        )
        assert got["error"] is None, got["error"]
        assert got["data"]["value"] == expected


class TestIdempotentDeletes:
    """NFR-507: deleting a key twice leaves the node in the same end
    state as deleting it once — the effect is idempotent even though the
    second call reports the missing-entity refusal."""

    @_SETTINGS
    @given(key=_KEY_STRATEGY)
    def test_double_delete_reaches_one_stable_absent_state(
        self, bench: Chokepoint, key: str
    ) -> None:
        _call(
            "set_value",
            {
                "endpoint": _KV_ENDPOINT,
                "path": _NODE_PATH,
                "key": key,
                "value": "seed",
                "value_type": "string",
            },
        )
        first = _call(
            "delete_key",
            {"endpoint": _KV_ENDPOINT, "path": _NODE_PATH, "key": key},
        )
        assert first["error"] is None, first["error"]
        after_first = _call(
            "get_value",
            {"endpoint": _KV_ENDPOINT, "path": _NODE_PATH, "key": key},
        )
        assert after_first["error"] is not None  # gone after the first delete
        second = _call(
            "delete_key",
            {"endpoint": _KV_ENDPOINT, "path": _NODE_PATH, "key": key},
        )
        assert second["error"] is not None  # missing-entity refusal, no state change
        after_second = _call(
            "get_value",
            {"endpoint": _KV_ENDPOINT, "path": _NODE_PATH, "key": key},
        )
        assert after_second["error"] is not None  # identical end state


class TestInsertReadConsistency:
    """NFR-507: a freshly set key is immediately visible to both the
    point read and the listing."""

    @_SETTINGS
    @given(key=_KEY_STRATEGY, number=st.integers(min_value=-1000, max_value=1000))
    def test_set_is_visible_to_get_and_list(
        self, bench: Chokepoint, key: str, number: int
    ) -> None:
        # Each drawn key gets its OWN node so the listing carries exactly
        # this key: the tiny inline cutover (3 rows) would otherwise
        # stream a node that accreted keys across examples, and the
        # property is about visibility, not the cutover.
        node = f"{_NODE_PATH}.{key}"
        _call(
            "set_value",
            {
                "endpoint": _KV_ENDPOINT,
                "path": node,
                "key": key,
                "value": str(number),
                "value_type": "integer",
            },
        )
        got = _call(
            "get_value",
            {"endpoint": _KV_ENDPOINT, "path": node, "key": key},
        )
        assert got["error"] is None, got["error"]
        assert got["data"]["value"] == number
        listing = _call("list_keys", {"endpoint": _KV_ENDPOINT, "path": node})
        assert listing["error"] is None, listing["error"]
        listed = {row[0]: row[1] for row in listing["data"]["rows"]}
        assert key in listed
        assert listed[key] == number


def _seed_range(guard: Chokepoint, count: int) -> None:
    """Replace the warehouse table with ids 0..count-1 (one fixed table,
    reset each example so the drawn N is the whole result)."""
    with guard._persistence.connection(_SQL_ENDPOINT) as connection:
        connection.execute(text("CREATE TABLE IF NOT EXISTS t (id INTEGER)"))
        connection.execute(text("DELETE FROM t"))
        for index in range(count):
            connection.execute(text("INSERT INTO t VALUES (:i)"), {"i": index})
        connection.commit()


class TestChunkCursorMonotonicity:
    """NFR-507 / FR-404: draining a streamed result serves every row
    exactly once, in order, under strictly increasing chunk ids, then
    closes and reports the total."""

    @_SETTINGS
    @given(row_count=st.integers(min_value=_STREAM_ROW_MIN, max_value=_STREAM_ROW_MAX))
    def test_drain_is_complete_ordered_and_monotone(
        self, bench: Chokepoint, row_count: int
    ) -> None:
        _seed_range(bench, row_count)
        opened = query(_SQL_ENDPOINT, "SELECT id FROM t ORDER BY id")
        assert isinstance(opened, StreamOpened)  # past the inline cutover
        collected: list[int] = []
        previous_chunk_id = -1
        while True:
            served = fetch_chunk(opened.stream_id)
            assert isinstance(served, ServedChunk)
            if served.chunk_id is None:
                assert served.closed is True
                break
            assert served.chunk_id > previous_chunk_id  # strictly monotone cursor
            previous_chunk_id = served.chunk_id
            collected.extend(row[0] for row in served.rows)
        assert collected == list(range(row_count))  # complete, ordered, no gaps
