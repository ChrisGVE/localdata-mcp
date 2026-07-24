"""tests/v3/testbench/test_trailer.py — NFR-506 batch-then-diagnose gate.

Covers the mechanical trailer check: parsing the Battery-Run trailer
(last one wins, absent -> None), and the two-condition verify — the
cited run must exist, predate the commit, and have recorded a failure,
and a newer run of the same battery must exist after the commit.
"""

from sqlite3 import Connection
from typing import List

from localdata_mcp.testbench.results_store import store, trailer

from .conftest import MAX_WRITE_ATTEMPTS

# ISO timestamps around a fixed commit instant (store's fixed-width shape).
_BEFORE = "2026-07-22T10:00:00Z"
_COMMIT = "2026-07-22T12:00:00Z"
_AFTER = "2026-07-22T14:00:00Z"


def _write_run(
    connection: Connection,
    run_id: str,
    *,
    started_at: str,
    battery_name: str = "base",
    failures: int = 1,
    passes: int = 2,
) -> None:
    run = store.BatteryRun(
        run_id=run_id,
        battery_name=battery_name,
        run_mode="deterministic",
        seed=None,
        dataset_hash="sha256:fixtures",
        software_versions={"python": "3.12.9"},
        started_at=started_at,
        finished_at=started_at,
        git_sha="abc123",
    )
    results: List[store.BatteryResult] = [
        store.BatteryResult(run_id, f"base.case_fail_{i}", passed=False)
        for i in range(failures)
    ] + [
        store.BatteryResult(run_id, f"base.case_pass_{i}", passed=True)
        for i in range(passes)
    ]
    with store.write_transaction(connection, max_attempts=MAX_WRITE_ATTEMPTS):
        store.write_run(connection, run)
        store.write_results(connection, results)


class TestParseTrailer:
    def test_extracts_run_id(self) -> None:
        message = "fix: repair the anova path\n\nBattery-Run: run-base-det-0007\n"
        assert trailer.parse_trailer(message) == "run-base-det-0007"

    def test_last_trailer_wins(self) -> None:
        message = "fix\n\nBattery-Run: old-run\nBattery-Run: new-run\n"
        assert trailer.parse_trailer(message) == "new-run"

    def test_absent_trailer_is_none(self) -> None:
        assert trailer.parse_trailer("fix: no trailer here\n") is None


class TestVerify:
    def test_clean_case_passes(self, canonical: Connection) -> None:
        _write_run(canonical, "antecedent", started_at=_BEFORE, failures=1)
        _write_run(canonical, "after-fix", started_at=_AFTER, failures=0, passes=3)
        assert (
            trailer.verify(canonical, cited_run_id="antecedent", commit_iso=_COMMIT)
            == []
        )

    def test_unknown_run_is_flagged(self, canonical: Connection) -> None:
        violations = trailer.verify(canonical, cited_run_id="ghost", commit_iso=_COMMIT)
        assert len(violations) == 1
        assert "not in the results store" in violations[0]

    def test_run_not_predating_commit_is_flagged(self, canonical: Connection) -> None:
        _write_run(canonical, "too-late", started_at=_AFTER, failures=1)
        violations = trailer.verify(
            canonical, cited_run_id="too-late", commit_iso=_COMMIT
        )
        assert any("must predate the fix" in v for v in violations)

    def test_run_without_failures_is_flagged(self, canonical: Connection) -> None:
        _write_run(canonical, "all-green", started_at=_BEFORE, failures=0, passes=3)
        _write_run(canonical, "after-fix", started_at=_AFTER, failures=0, passes=1)
        violations = trailer.verify(
            canonical, cited_run_id="all-green", commit_iso=_COMMIT
        )
        assert any("recorded no failures" in v for v in violations)

    def test_missing_post_fix_run_is_flagged(self, canonical: Connection) -> None:
        _write_run(canonical, "antecedent", started_at=_BEFORE, failures=1)
        violations = trailer.verify(
            canonical, cited_run_id="antecedent", commit_iso=_COMMIT
        )
        assert any("must be re-run after the fix" in v for v in violations)

    def test_post_fix_run_scoped_to_same_battery(self, canonical: Connection) -> None:
        """A later run of a DIFFERENT battery does not satisfy condition (b)."""
        _write_run(canonical, "antecedent", started_at=_BEFORE, failures=1)
        _write_run(
            canonical,
            "other-battery",
            started_at=_AFTER,
            battery_name="security",
            failures=0,
            passes=1,
        )
        violations = trailer.verify(
            canonical, cited_run_id="antecedent", commit_iso=_COMMIT
        )
        assert any("must be re-run after the fix" in v for v in violations)
