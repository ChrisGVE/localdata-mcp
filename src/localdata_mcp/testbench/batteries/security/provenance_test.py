"""testbench/batteries/security/provenance_test.py — NFR-114 at the L3 seam.

Endpoint provenance: networked connections are established by an
operator-declared endpoint name resolved through the Config nexus, never
by a caller-supplied DSN at tool-call time; local file-engine sources may
be opened ad hoc by path, but only inside `allowed_paths` and with a
default read-only posture that is not fail-open. Proven through the wire
the agent uses, across every branch of NFR-114's acceptance:

- **no caller DSN (structural, i-a)** — the whole tool surface is
  enumerated and asserted to expose *no* connection-string/DSN parameter,
  so a raw-DSN connection to an undeclared networked backend is not
  expressible in the first place.
- **undeclared endpoint name (ii)** — a read or a mutation naming an
  endpoint absent from operator config is refused, not silently created as
  a fresh unconfigured connection (the NFR-113 bypass this closes).
- **ad-hoc local file (i-b)** — a `query_file` open inside `allowed_paths`
  succeeds at the default read-only posture; the same open *outside* is
  refused (tying to NFR-108); a mutation through that ad-hoc read-only
  source is refused (the posture is not fail-open); and an
  operator-declared read-write endpoint over the same file is the positive
  control that an explicit operator grant — never a caller DSN — is what
  admits the write.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import anyio
import pytest
from fastmcp import Client

from localdata_mcp.server.mcp_app import app

from . import _seam

# Parameter names that would let a caller hand the server a raw connection
# target at call time — exactly what NFR-114 forbids for networked stores.
_DSN_PARAM_MARKERS = (
    "dsn",
    "conn_string",
    "connection_string",
    "conn_str",
    "connection",
)


@pytest.fixture()
def seeded(tmp_path: Path):
    """A contained root holding one seeded SQLite file, booted with NO
    declared endpoints — the substrate for the undeclared-name and ad-hoc
    file-open branches."""
    db = tmp_path / "local.db"
    connection = sqlite3.connect(db)
    connection.executescript("CREATE TABLE t (id INTEGER); INSERT INTO t VALUES (1);")
    connection.close()
    with _seam.booted(allowed_paths=(str(tmp_path),)):
        yield tmp_path, db


# -- (i-a) the no-caller-DSN structural property ------------------------


def test_no_tool_exposes_a_caller_supplied_dsn_parameter() -> None:
    async def enumerate_params() -> list[tuple[str, str]]:
        async with Client(app) as client:
            offenders: list[tuple[str, str]] = []
            for tool in await client.list_tools():
                schema: dict[str, Any] = tool.inputSchema or {}
                for name in schema.get("properties") or {}:
                    if any(marker in name.lower() for marker in _DSN_PARAM_MARKERS):
                        offenders.append((tool.name, name))
            return offenders

    offenders = anyio.run(enumerate_params)
    assert offenders == [], f"caller-supplied DSN parameters present: {offenders}"


# -- (ii) an undeclared endpoint name is refused, never auto-created -----


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("query", {"endpoint": "undeclared_x", "sql": "SELECT 1"}),
        (
            "write_query",
            {"endpoint": "undeclared_x", "sql": "INSERT INTO t VALUES (9)"},
        ),
    ],
    ids=["read", "mutation"],
)
def test_undeclared_endpoint_is_refused(seeded, tool: str, arguments: dict) -> None:
    error = _seam.expect_refused(_seam.call_envelope(tool, arguments))
    assert "NFR-114" in error["message"], error["message"]


# -- (i-b) ad-hoc local file: read-only-by-default, contained -----------


def test_adhoc_file_read_inside_root_succeeds(seeded) -> None:
    _root, db = seeded
    data = _seam.expect_ok(
        _seam.call_envelope("query_file", {"path": str(db), "sql": "SELECT id FROM t"})
    )
    assert data["rows"] == [[1]]


def test_adhoc_file_read_outside_root_is_refused(seeded, tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.db"
    connection = sqlite3.connect(outside)
    connection.executescript("CREATE TABLE t (id INTEGER); INSERT INTO t VALUES (7);")
    connection.close()
    try:
        _seam.expect_refused(
            _seam.call_envelope(
                "query_file", {"path": str(outside), "sql": "SELECT id FROM t"}
            )
        )
    finally:
        outside.unlink(missing_ok=True)


def test_adhoc_file_mutation_is_refused(seeded) -> None:
    _root, db = seeded
    error = _seam.expect_refused(
        _seam.call_envelope(
            "query_file", {"path": str(db), "sql": "INSERT INTO t VALUES (2)"}
        )
    )
    # The default posture is read-only and NOT fail-open.
    assert "read-only" in error["message"], error["message"]
    assert "NFR-114" in error["message"], error["message"]


# -- positive control: an operator read-write endpoint admits the write --


def test_operator_declared_readwrite_endpoint_allows_mutation(tmp_path: Path) -> None:
    db = tmp_path / "declared.db"
    connection = sqlite3.connect(db)
    connection.executescript("CREATE TABLE t (id INTEGER); INSERT INTO t VALUES (1);")
    connection.close()
    declarations = _seam.declare(local=(f"sqlite:///{db}", "read_write"))
    with _seam.booted(allowed_paths=(str(tmp_path),), declarations=declarations):
        _seam.expect_ok(
            _seam.call_envelope(
                "write_query", {"endpoint": "local", "sql": "INSERT INTO t VALUES (2)"}
            )
        )
        data = _seam.expect_ok(
            _seam.call_envelope(
                "query", {"endpoint": "local", "sql": "SELECT COUNT(*) FROM t"}
            )
        )
    assert data["rows"][0][0] == 2
