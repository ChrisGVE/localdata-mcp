"""tests/v3/test_check_battery_run_trailer.py — NFR-506 merge gate CLI.

Builds throwaway git repos with controlled commit dates and runs the
gate against a seeded results store: a fix touching bench-covered code
must carry a valid Battery-Run trailer; a clean two-condition case
passes; non-fix commits and untouched-code commits are exempt.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from sqlite3 import Connection
from typing import List, Tuple

from localdata_mcp.testbench.results_store import store

from .testbench.conftest import MAX_WRITE_ATTEMPTS, open_store

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "check_battery_run_trailer.py"
)

_BASE_DATE = "2026-07-22 09:00:00 +0000"
_COMMIT_DATE = "2026-07-22 12:00:00 +0000"
_BEFORE = "2026-07-22T10:00:00Z"
_AFTER = "2026-07-22T14:00:00Z"


def _git(repo: Path, *args: str, date: str | None = None) -> str:
    env = dict(os.environ, TZ="UTC")
    if date is not None:
        env["GIT_AUTHOR_DATE"] = date
        env["GIT_COMMITTER_DATE"] = date
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout.strip()


def _build_repo(
    repo: Path, *, subject: str, trailer: str = "", touch_covered: bool = True
) -> Tuple[str, str]:
    """Init a repo, one base commit then one subject commit; return (base, head)."""
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "Tester")
    covered = repo / "src" / "localdata_mcp" / "nexus" / "mod.py"
    covered.parent.mkdir(parents=True)
    covered.write_text("x = 1\n")
    (repo / "README.md").write_text("base\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "chore: init", date=_BASE_DATE)
    base = _git(repo, "rev-parse", "HEAD")

    target = covered if touch_covered else (repo / "README.md")
    target.write_text("x = 2\n" if touch_covered else "changed\n")
    _git(repo, "add", "-A")
    message = subject + (f"\n\nBattery-Run: {trailer}\n" if trailer else "\n")
    _git(repo, "commit", "-qm", message, date=_COMMIT_DATE)
    return base, _git(repo, "rev-parse", "HEAD")


def _seed(
    connection: Connection,
    run_id: str,
    *,
    started_at: str,
    battery: str = "base",
    failures: int = 1,
) -> None:
    run = store.BatteryRun(
        run_id=run_id,
        battery_name=battery,
        run_mode="deterministic",
        seed=None,
        dataset_hash="sha256:fx",
        software_versions={"python": "3.12"},
        started_at=started_at,
        finished_at=started_at,
        git_sha="abc",
    )
    results: List[store.BatteryResult] = [
        store.BatteryResult(run_id, f"c_fail_{i}", passed=False)
        for i in range(failures)
    ] + [store.BatteryResult(run_id, "c_pass", passed=True)]
    with store.write_transaction(connection, max_attempts=MAX_WRITE_ATTEMPTS):
        store.write_run(connection, run)
        store.write_results(connection, results)


def _run(
    store_path: Path, repo: Path, base: str, head: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--store",
            str(store_path),
            "--base",
            base,
            "--head",
            head,
        ],
        capture_output=True,
        text=True,
        cwd=str(repo),
    )


def test_fix_without_trailer_is_blocked(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    base, head = _build_repo(repo, subject="fix: broken anova")
    store_path = tmp_path / "store.db"
    open_store(store_path).close()

    result = _run(store_path, repo, base, head)
    assert result.returncode == 1
    violations = json.loads(result.stdout)["violations"]
    assert any("no Battery-Run: trailer" in v for v in violations)


def test_clean_two_condition_case_passes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    base, head = _build_repo(repo, subject="fix: broken anova", trailer="antecedent")
    store_path = tmp_path / "store.db"
    connection = open_store(store_path)
    _seed(connection, "antecedent", started_at=_BEFORE, failures=1)
    _seed(connection, "post-fix", started_at=_AFTER, failures=0)
    connection.close()

    result = _run(store_path, repo, base, head)
    assert result.returncode == 0, result.stdout
    assert json.loads(result.stdout)["violations"] == []


def test_non_fix_commit_is_exempt(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    base, head = _build_repo(repo, subject="feat: new tool")
    store_path = tmp_path / "store.db"
    open_store(store_path).close()

    result = _run(store_path, repo, base, head)
    assert result.returncode == 0
    assert json.loads(result.stdout)["violations"] == []


def test_fix_not_touching_covered_code_is_exempt(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    base, head = _build_repo(repo, subject="fix: docs typo", touch_covered=False)
    store_path = tmp_path / "store.db"
    open_store(store_path).close()

    result = _run(store_path, repo, base, head)
    assert result.returncode == 0
    assert json.loads(result.stdout)["violations"] == []
