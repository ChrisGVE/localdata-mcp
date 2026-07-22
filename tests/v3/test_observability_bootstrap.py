"""tests/v3/test_observability_bootstrap.py — E2.1 two-phase bring-up.

NX-4's bring-up contract (ARCHITECTURE.md section 4e): a stderr-only
bootstrap mode is active from the first import of the package, before
any config is read; once NX-2's merged model is available the process
reconfigures (level) with stderr-only as the invariant floor NX-2 can
refine but never move off. Assertions target the nexus-owned
StderrHandler (pytest injects its own capture handlers on the root
logger); the OS-level fd-1 truth is asserted by the E2.5 subprocess
battery, not here. Neighbors: test_redaction.py covers the processor
chain's redaction stage; test_purity_static_check.py covers the
no-print/no-second-handler teeth.
"""

from __future__ import annotations

import logging
import sys
from typing import Iterator

import pytest

from localdata_mcp.nexus import observability
from localdata_mcp.nexus.observability import (
    bootstrap,
    get_logger,
    logging_phase,
    reconfigure,
)
from localdata_mcp.nexus.observability.config import StderrHandler


@pytest.fixture(autouse=True)
def fresh_logging_state() -> Iterator[None]:
    """Re-enter bootstrap phase before each test and restore it after,
    so reconfigure() calls cannot leak between tests."""
    bootstrap(force=True)
    yield
    bootstrap(force=True)


def nexus_handlers() -> list[StderrHandler]:
    """The root handlers owned by NX-4 (ignoring pytest's capture
    handlers, which the nexus deliberately leaves alone)."""
    return [
        handler
        for handler in logging.getLogger().handlers
        if isinstance(handler, StderrHandler)
    ]


class TestBootstrapPhase:
    def test_import_alone_activates_bootstrap_logging(self) -> None:
        # The package import (module scope above) must have configured
        # logging: a phase is set and the nexus stderr handler exists.
        assert observability.logging_phase() in ("bootstrap", "configured")
        assert nexus_handlers()

    def test_bootstrap_reports_its_phase(self) -> None:
        assert logging_phase() == "bootstrap"

    def test_bootstrap_is_idempotent(self) -> None:
        bootstrap()
        bootstrap()
        assert len(nexus_handlers()) == 1

    def test_bootstrap_level_is_info(self) -> None:
        assert logging.getLogger().level == logging.INFO

    def test_nexus_handler_is_late_bound_to_stderr(self) -> None:
        (handler,) = nexus_handlers()
        assert handler.stream is sys.stderr

    def test_log_event_lands_on_stderr_not_stdout(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        get_logger("bootstrap-probe").info("bootstrap probe event")
        out, err = capsys.readouterr()
        assert out == ""
        assert "bootstrap probe event" in err

    def test_json_rendering_outside_debug(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        get_logger("json-probe").info("json probe", answer=42)
        err = capsys.readouterr().err
        assert '"event": "json probe"' in err
        assert '"answer": 42' in err
        assert '"logger": "json-probe"' in err


class TestReconfigurePhase:
    def test_reconfigure_switches_phase(self) -> None:
        reconfigure(level="INFO")
        assert logging_phase() == "configured"

    def test_reconfigure_applies_level(self) -> None:
        reconfigure(level="WARNING")
        assert logging.getLogger().level == logging.WARNING

    def test_reconfigure_replaces_never_stacks_handlers(self) -> None:
        reconfigure(level="DEBUG")
        reconfigure(level="INFO")
        assert len(nexus_handlers()) == 1

    def test_reconfigure_keeps_stderr_binding(self) -> None:
        reconfigure(level="DEBUG")
        (handler,) = nexus_handlers()
        assert handler.stream is sys.stderr

    def test_reconfigure_rejects_unknown_level(self) -> None:
        with pytest.raises(ValueError):
            reconfigure(level="LOUD")

    def test_debug_level_events_flow_after_reconfigure(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        reconfigure(level="DEBUG")
        get_logger("debug-probe").debug("debug probe event")
        assert "debug probe event" in capsys.readouterr().err

    def test_info_events_filtered_at_warning(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        reconfigure(level="WARNING")
        get_logger("filter-probe").info("should not appear")
        assert "should not appear" not in capsys.readouterr().err
