"""tests/v3/test_merge_battery_results.py — E14.4 post-job merge CLI.

The merge script folds per-worker files into a canonical store, reads
the retention bound from the config nexus (never restated), reports a
JSON summary, and exits non-zero when any worker file is refused. Run
as a subprocess — the established script-test pattern here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from localdata_mcp.nexus.config import load_config
from localdata_mcp.testbench.results_store import schema, store

from .testbench.conftest import (
    MAX_WRITE_ATTEMPTS,
    make_results,
    make_run,
    make_worker_file,
)

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "merge_battery_results.py"


def _run(canonical: Path, *workers: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--canonical", str(canonical), *workers],
        capture_output=True,
        text=True,
    )


def test_merges_workers_and_reports_config_retention(tmp_path: Path) -> None:
    run_a = make_run(index=0)
    run_b = make_run(index=1, battery_name="security")
    make_worker_file(tmp_path / "w0.db", run_a, make_results(run_a.run_id, count=2))
    make_worker_file(tmp_path / "w1.db", run_b, make_results(run_b.run_id, count=3))

    result = _run(
        tmp_path / "canonical.db", str(tmp_path / "w0.db"), str(tmp_path / "w1.db")
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert (
        report["retention_runs"] == load_config().model.testbench.results_retention_runs
    )
    assert report["canonical_run_count"] == 2
    assert len(report["merged"]) == 2
    assert report["refused"] == []

    connection = schema.connect(tmp_path / "canonical.db", busy_timeout_ms=250)
    assert len(store.read_results(connection, run_a.run_id)) == 2
    assert len(store.read_results(connection, run_b.run_id)) == 3
    connection.close()


def test_glob_argument_is_expanded(tmp_path: Path) -> None:
    for index in range(2):
        run = make_run(index=index)
        make_worker_file(tmp_path / f"w{index}.db", run, make_results(run.run_id, 1))
    # One run id survives (same battery/mode group, retention keeps recent).
    result = _run(tmp_path / "canonical.db", str(tmp_path / "w*.db"))
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert len(report["merged"]) == 2


def test_refused_worker_exits_nonzero(tmp_path: Path) -> None:
    broken = schema.connect(tmp_path / "broken.db", busy_timeout_ms=250)
    schema.ensure_schema(broken)
    broken.execute("DROP TABLE battery_results")
    broken.commit()
    broken.close()

    result = _run(tmp_path / "canonical.db", str(tmp_path / "broken.db"))
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["merged"] == []
    assert report["refused"][0]["file"].endswith("broken.db")


def test_missing_worker_is_reported(tmp_path: Path) -> None:
    result = _run(tmp_path / "canonical.db", str(tmp_path / "absent.db"))
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["refused"][0]["reason"] == "missing"
