"""tests/v3/test_startup_report.py — E2.6 startup pinned/shadowed report.

Forward-ports d5fb7280's intent (ARCHITECTURE.md section 8.1): at
startup the process logs, through NX-4 and therefore redacted and on
stderr, which config fields were set by which source, every shadowed
contribution with its disposition, and every trust refusal — derived
entirely from NX-2's Provenance, never from a second bookkeeping
path. One subprocess case proves the report fires on the real boot
path. Neighbors: nexus/observability/report.py under test;
test_config_provenance.py covers the Provenance data it reads.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest

from localdata_mcp.nexus.config import ConfigurationError, merge_sources
from localdata_mcp.nexus.config.merge import ConfigLoadResult
from localdata_mcp.nexus.config.provenance import Layer, LayerSource, Provenance
from localdata_mcp.nexus.observability import bootstrap
from localdata_mcp.nexus.observability.report import log_startup_report
from localdata_mcp.testbench.purity_runner import (
    initialize_request,
    initialized_notification,
    run_session,
)


@pytest.fixture(autouse=True)
def fresh_logging_state() -> Iterator[None]:
    bootstrap(force=True)
    yield
    bootstrap(force=True)


def layered_result() -> ConfigLoadResult:
    """System pins a security field, the project layer tries to shadow
    it and also sets an ordinary field — one winner, one refusal."""
    system = LayerSource(
        "system-file",
        Layer.SYSTEM,
        0,
        {"resources": {"memory_ceiling_bytes": 2 * 2**30}},
    )
    project = LayerSource(
        "project-file",
        Layer.PROJECT,
        0,
        {
            "resources": {"memory_ceiling_bytes": 2**30},
            "query": {"default_chunk_size": 50},
        },
    )
    return merge_sources([system, project])


class TestStartupReport:
    def test_reports_winner_with_source(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log_startup_report(layered_result())
        err = capsys.readouterr().err
        assert "resources.memory_ceiling_bytes" in err
        assert "system-file" in err

    def test_reports_shadowed_contribution_with_disposition(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log_startup_report(layered_result())
        err = capsys.readouterr().err
        assert "pin_refused" in err

    def test_reports_refusals_as_warnings(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log_startup_report(layered_result())
        err = capsys.readouterr().err
        assert '"level": "warning"' in err

    def test_default_only_fields_are_not_itemized(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log_startup_report(layered_result())
        err = capsys.readouterr().err
        # An untouched default (process.* section) stays out of the
        # per-field listing — the report covers configured fields only.
        assert "bootstrap_default_resamples" not in err

    def test_summary_counts_are_logged(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log_startup_report(layered_result())
        err = capsys.readouterr().err
        assert '"configured_fields": 2' in err
        assert '"refusals": 1' in err

    def test_refusal_text_is_redacted_on_the_way_out(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = ConfigLoadResult(
            model=layered_result().model,
            provenance=Provenance({}),
            refusals=(
                ConfigurationError(
                    "refused endpoint mysql://svc:sekrit123@db/x",
                    source="project-file",
                ),
            ),
        )
        log_startup_report(result)
        err = capsys.readouterr().err
        assert "sekrit123" not in err
        assert "[REDACTED]" in err


class TestRealBootPath:
    def test_report_fires_on_startup(self, tmp_path: Path) -> None:
        (tmp_path / ".localdata.toml").write_text(
            "[query]\ndefault_chunk_size = 50\n", encoding="utf-8"
        )
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("LOCALDATA_")
        }
        env["HOME"] = str(tmp_path)
        session = run_session(
            [initialize_request(request_id=1), initialized_notification()],
            cwd=tmp_path,
            env=env,
        )
        assert session.returncode == 0
        assert b"query.default_chunk_size" in session.stderr
        assert b"project-file" in session.stderr
