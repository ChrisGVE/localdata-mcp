"""tests/v3/testbench/test_battery_persistence.py — the CI persistence plugin.

Covers the batteries/conftest.py plugin: nodeid -> (battery, mode,
test_id) derivation, numeric-output extraction from user_properties, and
the record -> persist -> merge path a per-worker file takes. Reports are
duck-typed (SimpleNamespace) so the plugin logic is exercised without a
full pytest run.
"""

from pathlib import Path
from types import SimpleNamespace

from localdata_mcp.testbench.batteries import conftest as plugin
from localdata_mcp.testbench.results_store import emit, merge, store

from .conftest import MAX_WRITE_ATTEMPTS, open_store

_BASE = "src/localdata_mcp/testbench/batteries/base/ingest_battery_test.py"
_RAND = "src/localdata_mcp/testbench/batteries/base/ingest_randomized_test.py"


def _report(nodeid, *, passed=True, when="call", duration=0.02, properties=()):
    return SimpleNamespace(
        nodeid=nodeid,
        passed=passed,
        when=when,
        duration=duration,
        user_properties=list(properties),
    )


class TestNodeidDerivation:
    def test_battery_name_from_segment(self) -> None:
        assert plugin._battery_name(f"{_BASE}::TestKv::test_roundtrip") == "base"
        seg = "src/localdata_mcp/testbench/batteries/security/path_containment_test.py"
        assert plugin._battery_name(f"{seg}::x") == "security"

    def test_test_id_is_battery_relative(self) -> None:
        assert (
            plugin._test_id(f"{_BASE}::TestKv::test_roundtrip")
            == "base/ingest_battery_test.py::TestKv::test_roundtrip"
        )

    def test_run_mode_reads_randomized_module(self) -> None:
        assert plugin._run_mode(f"{_RAND}::x") == "randomized"
        assert plugin._run_mode(f"{_BASE}::x") == "deterministic"

    def test_numeric_output_last_wins(self) -> None:
        report = _report(
            _BASE,
            properties=[("numeric_output", {"a": 1}), ("numeric_output", {"a": 2})],
        )
        assert plugin._numeric_output(report) == {"a": 2}

    def test_numeric_output_absent_is_none(self) -> None:
        assert plugin._numeric_output(_report(_BASE)) is None


class TestPluginRoundTrip:
    def test_record_persist_merge(self, canonical, tmp_path: Path) -> None:
        collector = emit.RunCollector(run_id_base="ci-7")
        store_plugin = plugin.ResultsStorePlugin(collector, str(tmp_path / "store"))

        store_plugin.pytest_runtest_logreport(
            _report(
                f"{_BASE}::TestKv::test_roundtrip",
                passed=True,
                properties=[("numeric_output", {"rows": 6})],
            )
        )
        store_plugin.pytest_runtest_logreport(
            _report(f"{_RAND}::TestKv::test_shuffle", passed=False)
        )
        # A non-call phase report must not be recorded.
        store_plugin.pytest_runtest_logreport(
            _report(f"{_BASE}::TestKv::test_setup", when="setup")
        )
        store_plugin.pytest_sessionfinish(exitstatus=0)

        worker = tmp_path / "store" / "w0.db"
        assert worker.exists()
        merge.merge_worker_file(
            canonical,
            worker,
            retention_runs=2,
            max_write_attempts=MAX_WRITE_ATTEMPTS,
        )
        det = store.read_results(canonical, "ci-7-base-deterministic")
        rand = store.read_results(canonical, "ci-7-base-randomized")
        assert [r.test_id for r in det] == [
            "base/ingest_battery_test.py::TestKv::test_roundtrip"
        ]
        assert det[0].numeric_output == {"rows": 6}
        assert rand[0].passed is False

    def test_empty_session_writes_nothing(self, tmp_path: Path) -> None:
        collector = emit.RunCollector(run_id_base="ci-0")
        store_plugin = plugin.ResultsStorePlugin(collector, str(tmp_path / "store"))
        store_plugin.pytest_sessionfinish(exitstatus=0)
        assert not (tmp_path / "store").exists()
