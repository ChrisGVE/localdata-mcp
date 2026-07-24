"""localdata_mcp/testbench/results_store/emit.py — per-worker persistence.

The write side of NFR-508 for a live battery run: a `RunCollector`
accumulates one outcome per battery test as pytest reports it, then
`persist` folds every (battery_name, run_mode) group into a single
per-worker results file (ARCHITECTURE.md section 5) — one `battery_runs`
row per group, all outcomes under it. CI's post-job step merges those
worker files into the canonical store (merge.py). Every tunable (busy
timeout, retry attempts) is a required parameter; this module reads no
configuration and no environment of its own — the pytest plugin that
drives it (batteries/conftest.py) owns that boundary. Neighbors:
schema.py owns the DDL, store.py the write transaction this builds on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import schema, store

# The provenance libraries whose installed versions are recorded per run
# (NFR-508): the numeric-result-bearing stack whose upgrades most often
# move an oracle output. Absent packages are simply omitted.
_PROVENANCE_PACKAGES: Tuple[str, ...] = (
    "localdata-mcp",
    "pandas",
    "numpy",
    "scipy",
    "statsmodels",
    "scikit-learn",
    "sqlglot",
    "sqlalchemy",
    "matplotlib",
)


def software_versions() -> Dict[str, str]:
    """Installed versions of the provenance stack, plus the interpreter."""
    import platform

    versions: Dict[str, str] = {"python": platform.python_version()}
    for name in _PROVENANCE_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _now_iso() -> str:
    """ISO-8601 UTC, second precision — the store's timestamp shape."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class _Outcome:
    """One battery test's reported result, before it is grouped into a run."""

    battery_name: str
    run_mode: str
    test_id: str
    passed: bool
    numeric_output: Optional[Any]
    duration_ms: Optional[float]


@dataclass
class RunCollector:
    """Accumulates per-test outcomes across a live battery session.

    A run's identity is (battery_name, run_mode); `run_id_base` is the
    coordinator-minted prefix (one CI battery invocation), made unique
    per group so a mixed session that touches several batteries yields
    one clean `battery_runs` row apiece.
    """

    run_id_base: str
    seed_for_randomized: Optional[int] = 42
    _outcomes: List[_Outcome] = field(default_factory=list)
    _started_at: str = field(default_factory=_now_iso)

    def record(
        self,
        *,
        battery_name: str,
        run_mode: str,
        test_id: str,
        passed: bool,
        numeric_output: Optional[Any] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Append one test outcome (idempotence is the merge's job, not ours)."""
        self._outcomes.append(
            _Outcome(
                battery_name=battery_name,
                run_mode=run_mode,
                test_id=test_id,
                passed=passed,
                numeric_output=numeric_output,
                duration_ms=duration_ms,
            )
        )

    def is_empty(self) -> bool:
        return not self._outcomes

    def run_id_for(self, battery_name: str, run_mode: str) -> str:
        """The coordinator-minted run id for one (battery, mode) group."""
        return f"{self.run_id_base}-{battery_name}-{run_mode}"

    def persist(
        self,
        worker_path: Path | str,
        *,
        dataset_hash: str,
        git_sha: Optional[str],
        busy_timeout_ms: int,
        max_write_attempts: int,
    ) -> Path:
        """Write every collected group into one per-worker results file."""
        finished_at = _now_iso()
        versions = software_versions()
        connection = schema.connect(worker_path, busy_timeout_ms=busy_timeout_ms)
        schema.ensure_schema(connection)
        try:
            with store.write_transaction(connection, max_attempts=max_write_attempts):
                for (battery_name, run_mode), outcomes in self._grouped().items():
                    run_id = self.run_id_for(battery_name, run_mode)
                    store.write_run(
                        connection,
                        store.BatteryRun(
                            run_id=run_id,
                            battery_name=battery_name,
                            run_mode=run_mode,
                            seed=(
                                self.seed_for_randomized
                                if run_mode == "randomized"
                                else None
                            ),
                            dataset_hash=dataset_hash,
                            software_versions=versions,
                            started_at=self._started_at,
                            finished_at=finished_at,
                            git_sha=git_sha,
                        ),
                    )
                    store.write_results(
                        connection,
                        (
                            store.BatteryResult(
                                run_id=run_id,
                                test_id=outcome.test_id,
                                passed=outcome.passed,
                                numeric_output=outcome.numeric_output,
                                duration_ms=outcome.duration_ms,
                            )
                            for outcome in outcomes
                        ),
                    )
        finally:
            connection.close()
        return Path(worker_path)

    def _grouped(self) -> Dict[Tuple[str, str], List[_Outcome]]:
        """Outcomes bucketed by (battery_name, run_mode), insertion-ordered."""
        groups: Dict[Tuple[str, str], List[_Outcome]] = {}
        for outcome in self._outcomes:
            groups.setdefault((outcome.battery_name, outcome.run_mode), []).append(
                outcome
            )
        return groups
