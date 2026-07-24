"""testbench/batteries/conftest.py — results-store persistence plugin.

The one integration point that turns every battery test's live outcome
into an NFR-508 results-store record. It is ENV-GATED: with no
`LOCALDATA_RESULTS_STORE_DIR` set (ordinary local runs) it does nothing
at all, so the batteries stay pure pytest modules. Under CI the
coordinator sets that directory plus the minted run id, and the
registered plugin collects one outcome per executed test — battery
derived from the path segment under `batteries/`, run mode from the
module name — then writes a single per-worker results file at session
end (emit.RunCollector). CI's post-job step merges those worker files
into the canonical store (results_store/merge.py). A test attaches a
numeric output for longitudinal comparison by recording a
`numeric_output` property.

The env boundary lives here by design: results_store/emit.py reads no
environment and no configuration, so it stays unit-testable in isolation.
"""

from __future__ import annotations

import os
from typing import Optional

import pytest

from localdata_mcp.testbench.results_store import emit

# Results-store write tunables for this trusted, non-LLM-facing CI path.
# Not S8 ConfigModel fields (the store's DDL owns them, not the operator
# surface); named here so the magic numbers read as intent.
_WORKER_BUSY_TIMEOUT_MS = 250
_WORKER_MAX_WRITE_ATTEMPTS = 3

# The env contract the CI coordinator fills; store dir absent => inert.
_STORE_DIR_ENV = "LOCALDATA_RESULTS_STORE_DIR"
_RUN_ID_ENV = "LOCALDATA_RESULTS_RUN_ID"
_DATASET_HASH_ENV = "LOCALDATA_DATASET_HASH"
_GIT_SHA_ENV = "GITHUB_SHA"
_XDIST_WORKER_ENV = "PYTEST_XDIST_WORKER"

_BATTERIES_SEGMENT = "batteries/"
_RANDOMIZED_MARKER = "randomized"
_NUMERIC_OUTPUT_PROPERTY = "numeric_output"


def _battery_name(nodeid: str) -> str:
    """The path segment under `batteries/`: base, domain, security, ..."""
    _, _, tail = nodeid.partition(_BATTERIES_SEGMENT)
    head = tail if tail else nodeid
    return head.split("/", 1)[0] if "/" in head else "unknown"


def _test_id(nodeid: str) -> str:
    """The battery-relative node id — stable across runs, unique per test."""
    _, _, tail = nodeid.partition(_BATTERIES_SEGMENT)
    return tail if tail else nodeid


def _run_mode(nodeid: str) -> str:
    """Randomized when the module name marks it so, else deterministic."""
    return "randomized" if _RANDOMIZED_MARKER in nodeid else "deterministic"


def _numeric_output(report: pytest.TestReport) -> Optional[object]:
    """The numeric_output a test recorded, if any (last one wins)."""
    value: Optional[object] = None
    for key, recorded in report.user_properties:
        if key == _NUMERIC_OUTPUT_PROPERTY:
            value = recorded
    return value


class ResultsStorePlugin:
    """Buffers battery outcomes and writes one per-worker file at the end."""

    def __init__(self, collector: emit.RunCollector, store_dir: str) -> None:
        self._collector = collector
        self._store_dir = store_dir

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Record each executed test's outcome on its call phase."""
        if report.when != "call":
            return
        self._collector.record(
            battery_name=_battery_name(report.nodeid),
            run_mode=_run_mode(report.nodeid),
            test_id=_test_id(report.nodeid),
            passed=report.passed,
            numeric_output=_numeric_output(report),
            duration_ms=report.duration * 1000.0,
        )

    def pytest_sessionfinish(self, exitstatus: int) -> None:
        """Persist the per-worker results file (nothing collected => skip)."""
        if self._collector.is_empty():
            return
        os.makedirs(self._store_dir, exist_ok=True)
        worker_id = os.environ.get(_XDIST_WORKER_ENV, "w0")
        self._collector.persist(
            os.path.join(self._store_dir, f"{worker_id}.db"),
            dataset_hash=os.environ.get(_DATASET_HASH_ENV, "unpinned"),
            git_sha=os.environ.get(_GIT_SHA_ENV),
            busy_timeout_ms=_WORKER_BUSY_TIMEOUT_MS,
            max_write_attempts=_WORKER_MAX_WRITE_ATTEMPTS,
        )


def pytest_configure(config: pytest.Config) -> None:
    """Register the persistence plugin iff the store dir is configured."""
    store_dir = os.environ.get(_STORE_DIR_ENV)
    if not store_dir:
        return
    collector = emit.RunCollector(run_id_base=os.environ.get(_RUN_ID_ENV, "local"))
    config.pluginmanager.register(
        ResultsStorePlugin(collector, store_dir), name="localdata-results-store"
    )
