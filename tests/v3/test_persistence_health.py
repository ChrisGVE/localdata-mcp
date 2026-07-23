"""tests/v3/test_persistence_health.py — FR-803 up/downed + redaction.

The acceptance pair verbatim: a live fixture backend probes healthy, a
deliberately-downed one probes unhealthy — same call, correct status
each way. Plus the E5.5 redaction clause: a failure detail carrying a
credential-shaped span never survives into the result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localdata_mcp.nexus.config.endpoints import EndpointDeclaration
from localdata_mcp.nexus.persistence.engines import create_handle
from localdata_mcp.nexus.persistence.health import HealthCheckResult, probe
from localdata_mcp.nexus.persistence.limits import ResourceLimits

LIMITS = ResourceLimits(
    max_connections=2, statement_timeout_seconds=30, max_concurrent_streams=1
)


def handle_for(dsn: str):
    return create_handle(EndpointDeclaration(name="ep", dsn=dsn), LIMITS, {})


class TestUpDownedFixture:
    def test_live_sqlite_fixture_probes_healthy(self, tmp_path: Path) -> None:
        result = probe(handle_for(f"sqlite:///{tmp_path / 'live.db'}"), "sqlite")
        assert result.healthy
        assert result.response_time_ms >= 0

    def test_live_duckdb_fixture_probes_healthy(self, tmp_path: Path) -> None:
        import duckdb

        path = tmp_path / "live.duckdb"
        duckdb.connect(str(path)).close()
        declaration = EndpointDeclaration(name="ep", dsn=f"duckdb:///{path}")
        result = probe(create_handle(declaration, LIMITS, {}), "duckdb")
        assert result.healthy

    def test_downed_fixture_probes_unhealthy_without_raising(
        self, tmp_path: Path
    ) -> None:
        # A read-only open of a nonexistent file cannot succeed — the
        # "deliberately-downed" fixture backend.
        missing = tmp_path / "no-such-dir" / "gone.duckdb"
        declaration = EndpointDeclaration(name="ep", dsn=f"duckdb:///{missing}")
        result = probe(create_handle(declaration, LIMITS, {}), "duckdb")
        assert not result.healthy
        assert result.detail  # names the failure, redacted


class TestHealthTextRedaction:
    def test_credential_shaped_detail_is_redacted_at_construction(self) -> None:
        result = HealthCheckResult(
            healthy=False,
            response_time_ms=1.0,
            detail=(
                "connect failed for postgresql://svc:hunter2@db.example/wh "
                "(options password=hunter2)"
            ),
        )
        assert "hunter2" not in result.detail
        assert "[REDACTED]" in result.detail

    def test_probe_failure_detail_carries_no_probe_target_secret(
        self, tmp_path: Path
    ) -> None:
        class ExplodingHandle:
            def connect(self):
                raise RuntimeError("no route to postgresql://svc:hunter2@db.example/wh")

            def dispose(self) -> None: ...

        result = probe(ExplodingHandle(), "postgresql")
        assert not result.healthy
        assert "hunter2" not in result.detail
