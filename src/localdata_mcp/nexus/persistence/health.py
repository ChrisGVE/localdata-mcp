"""localdata_mcp/nexus/persistence/health.py — the FR-803 health probe.

The harvested successor of `connection_manager/health.py`, shrunk to
v3's shape: an on-demand probe (one round-trip through the endpoint's
own engine handle) instead of a background daemon thread — a stdio MCP
server warms up at startup (§4e) and re-checks on demand, and FR-803's
acceptance is a call, not a monitor. The probe's failure text passes
the NFR-110 credential redaction AT CONSTRUCTION — a driver error
happily echoes the DSN it failed to reach — so no `HealthCheckResult`
ever holds an unredacted secret, upholding §5's invariant before the
NX-3/NX-4 edge backstops apply. Neighbors: manager.py probes during
warm-up and on request; record.py stores the latest result;
config/dsn_patterns.py owns the credential shapes applied here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from localdata_mcp.nexus.config.dsn_patterns import redact_credentials_text
from localdata_mcp.nexus.persistence.engines import EngineHandle

# The one-round-trip liveness statement; Oracle needs its FROM clause,
# and an rdf store speaks SPARQL (the cheapest complete query is ASK).
_PROBE_SQL = "SELECT 1"
_PROBE_SQL_ORACLE = "SELECT 1 FROM DUAL"
_PROBE_SPARQL = "ASK {}"


@dataclass(frozen=True)
class HealthCheckResult:
    """One probe's outcome; `detail` is always redacted text."""

    healthy: bool
    response_time_ms: float
    detail: str

    def __post_init__(self) -> None:
        """No result carries a credential, whatever the caller passed."""
        object.__setattr__(self, "detail", redact_credentials_text(self.detail))


def probe(handle: EngineHandle, backend_kind: str) -> HealthCheckResult:
    """One liveness round-trip through the endpoint's own engine.

    Never raises: an unreachable backend IS the down half of FR-803's
    up/downed acceptance, reported as a result with redacted detail.
    """
    if backend_kind == "oracle":
        statement = _PROBE_SQL_ORACLE
    elif backend_kind == "rdf":
        statement = _PROBE_SPARQL
    else:
        statement = _PROBE_SQL
    started = time.monotonic()
    try:
        with handle.connect() as connection:
            _execute_probe(connection, backend_kind, statement)
    except Exception as failure:
        return HealthCheckResult(
            healthy=False,
            response_time_ms=(time.monotonic() - started) * 1000,
            detail=f"{type(failure).__name__}: {failure}",
        )
    return HealthCheckResult(
        healthy=True,
        response_time_ms=(time.monotonic() - started) * 1000,
        detail="probe ok",
    )


def _execute_probe(connection: object, backend_kind: str, statement: str) -> None:
    """Issue the statement through the connection's own dialect: DuckDB's
    native connection and the rdf cursor connection take the string
    directly; SQLAlchemy connections take a `text()` construct."""
    if backend_kind in ("duckdb", "rdf"):
        connection.execute(statement)  # type: ignore[attr-defined]
        return
    from sqlalchemy import text

    connection.execute(text(statement))  # type: ignore[attr-defined]
