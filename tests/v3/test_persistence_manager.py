"""tests/v3/test_persistence_manager.py — NX-5's owner + the E5 exit gate.

Real records over real tmp_path engines: §4e warm-up, NX-6's
connection seam, the E4.0 protocol satisfied by the REAL
implementation (the same assertions the conformant double passes —
E6's gate re-runs the contract here), and NFR-112's fault-injection
acceptance verbatim: an exception mid-operation, the record disposes
and reissues, and the follow-up query sees no partial mutation.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from sqlalchemy import text

from localdata_mcp.nexus.config.endpoints import EndpointDeclaration
from localdata_mcp.nexus.config.models import ConfigModel
from localdata_mcp.nexus.error.fault_signal import (
    DisposeAndReissueEntry,
    FaultSignal,
)
from localdata_mcp.nexus.error.wire import wrap
from localdata_mcp.nexus.persistence.lifecycle import LifecycleState
from localdata_mcp.nexus.persistence.manager import (
    EndpointUnavailableError,
    PersistenceNexus,
    UnknownEndpointError,
)


def config_for(tmp_path: Path, *names: str) -> ConfigModel:
    """One read_write sqlite endpoint per name, files under tmp_path."""
    endpoints = {
        name: EndpointDeclaration(
            name=name,
            dsn=f"sqlite:///{tmp_path / name}.db",
            posture="read_write",
        )
        for name in names
    }
    return ConfigModel(endpoints=endpoints)


def warmed(tmp_path: Path, *names: str) -> PersistenceNexus:
    nexus = PersistenceNexus(config_for(tmp_path, *names), environ={})
    nexus.warm_up()
    return nexus


class TestWarmUp:
    def test_every_declared_endpoint_gets_a_healthy_record(
        self, tmp_path: Path
    ) -> None:
        nexus = PersistenceNexus(config_for(tmp_path, "alpha", "beta"), environ={})
        results = nexus.warm_up()
        assert set(results) == {"alpha", "beta"}
        assert all(result.healthy for result in results.values())
        assert nexus.record("alpha").state is LifecycleState.HEALTHY

    def test_limits_come_from_the_config_nexus(self, tmp_path: Path) -> None:
        config = config_for(tmp_path, "alpha")
        nexus = PersistenceNexus(config, environ={})
        nexus.warm_up()
        record = nexus.record("alpha")
        assert (
            record.limits.max_connections
            == config.resources.max_connections_per_endpoint
        )


class TestConnectionSeam:
    def test_connection_issues_from_the_named_record(self, tmp_path: Path) -> None:
        nexus = warmed(tmp_path, "alpha")
        with nexus.connection("alpha") as connection:
            assert connection.execute(text("SELECT 1")).scalar() == 1

    def test_unknown_endpoint_is_refused_by_name(self, tmp_path: Path) -> None:
        nexus = warmed(tmp_path, "alpha")
        with pytest.raises(UnknownEndpointError):
            with nexus.connection("nowhere"):
                pass

    def test_closed_record_does_not_issue(self, tmp_path: Path) -> None:
        nexus = warmed(tmp_path, "alpha")
        nexus.close_all()
        with pytest.raises(EndpointUnavailableError):
            with nexus.connection("alpha"):
                pass

    def test_issuable_and_pool_are_read_under_the_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CR-016: the issuability gate and the pool capture must run
        inside the record-collection lock so a concurrent fault walk
        cannot dispose-and-reissue the pool between the check and the
        connect. We prove it by observing that the nexus RLock is owned
        by this thread at the instant `issuable` is read — under the
        pre-fix code (read outside the lock) it is not."""
        nexus = warmed(tmp_path, "alpha")
        record = nexus.record("alpha")
        observed: dict[str, bool] = {}

        class LockObservingIssuable:
            def __get__(self, obj: object, objtype: object = None) -> bool:
                observed["locked"] = nexus._lock._is_owned()  # type: ignore[attr-defined]
                return record.state is LifecycleState.HEALTHY

        monkeypatch.setattr(type(record), "issuable", LockObservingIssuable())
        with nexus.connection("alpha") as connection:
            assert connection.execute(text("SELECT 1")).scalar() == 1
        assert observed["locked"] is True


class TestFaultSignalRealImplementation:
    """The E4.0 contract against the REAL NX-5 (E6's gate, prepared)."""

    def test_the_nexus_satisfies_the_protocol(self, tmp_path: Path) -> None:
        assert isinstance(warmed(tmp_path, "alpha"), FaultSignal)

    def test_mark_faulted_is_synchronous_and_reports_disposal(
        self, tmp_path: Path
    ) -> None:
        nexus = warmed(tmp_path, "alpha")
        entry = nexus.mark_faulted("alpha")
        assert entry == DisposeAndReissueEntry(
            record_id="alpha", disposed=True, reissued=True
        )
        # Synchronous: by return time the walk is complete.
        assert nexus.record("alpha").state is LifecycleState.HEALTHY

    def test_failed_reissue_is_reported_not_raised(self, tmp_path: Path) -> None:
        nested = tmp_path / "nested"
        nested.mkdir()
        nexus = warmed(nested, "alpha")
        shutil.rmtree(nested)  # the backend is now genuinely gone
        entry = nexus.mark_faulted("alpha")
        assert entry.disposed and not entry.reissued
        assert nexus.record("alpha").state is LifecycleState.CLOSED

    def test_unknown_or_closed_record_is_bookkept_never_raised(
        self, tmp_path: Path
    ) -> None:
        nexus = warmed(tmp_path, "alpha")
        assert nexus.mark_faulted("nowhere") == DisposeAndReissueEntry(
            record_id="nowhere", disposed=False, reissued=False
        )
        nexus.close_all()
        entry = nexus.mark_faulted("alpha")
        assert not entry.disposed and not entry.reissued


class TestNfr112FaultInjection:
    """The acceptance text verbatim: fault-injection during an
    operation, then the follow-up query sees no partial mutation."""

    def test_partial_mutation_does_not_survive_dispose_and_reissue(
        self, tmp_path: Path
    ) -> None:
        nexus = warmed(tmp_path, "alpha")
        with nexus.connection("alpha") as connection:
            connection.execute(text("CREATE TABLE t (x INTEGER)"))
            connection.commit()

        # The faulting operation: an uncommitted write, then the wire
        # path fires the E4.0 signal through NX-3 (§4b's ordered walk).
        with pytest.raises(TimeoutError):
            with nexus.connection("alpha") as connection:
                connection.execute(text("INSERT INTO t VALUES (1)"))
                raise TimeoutError("connection timed out mid-operation")

        structured = wrap(
            TimeoutError("connection timed out mid-operation"),
            "generic",
            fault_sink=nexus,
            record_id="alpha",
        )
        assert structured.error_type.signals_connection_fault

        with nexus.connection("alpha") as connection:
            count = connection.execute(text("SELECT COUNT(*) FROM t")).scalar()
        assert count == 0  # no partial mutation survived the fault

    def test_the_record_walked_the_full_cycle_and_serves_again(
        self, tmp_path: Path
    ) -> None:
        nexus = warmed(tmp_path, "alpha")
        wrap(
            TimeoutError("connection dropped"),
            "generic",
            fault_sink=nexus,
            record_id="alpha",
        )
        record = nexus.record("alpha")
        assert record.state is LifecycleState.HEALTHY
        assert record.health is not None and record.health.healthy
        with nexus.connection("alpha") as connection:
            assert connection.execute(text("SELECT 1")).scalar() == 1


class TestEphemeralThroughTheOwner:
    def test_open_ephemeral_reads_the_live_operator_grants(
        self, tmp_path: Path
    ) -> None:
        import sqlite3

        db = tmp_path / "file.db"
        sqlite3.connect(db).close()
        from localdata_mcp.nexus.config.models import SecurityConfig

        config = ConfigModel(
            security=SecurityConfig(ephemeral_write_paths=(str(tmp_path),))
        )
        nexus = PersistenceNexus(config, environ={})
        assert nexus.open_ephemeral(str(db), "sqlite").posture == "read_write"
        assert (
            PersistenceNexus(ConfigModel(), environ={})
            .open_ephemeral(str(db), "sqlite")
            .posture
            == "read_only"
        )


class TestShutdown:
    def test_close_all_closes_every_record(self, tmp_path: Path) -> None:
        nexus = warmed(tmp_path, "alpha", "beta")
        nexus.close_all()
        assert all(
            nexus.record(name).state is LifecycleState.CLOSED
            for name in ("alpha", "beta")
        )
        nexus.close_all()  # idempotent — already-closed records are skipped

class TestOnDemandHealth:
    def test_check_health_probes_and_lands_on_the_record(
        self, tmp_path: Path
    ) -> None:
        nexus = warmed(tmp_path, "alpha")
        result = nexus.check_health("alpha")
        assert result.healthy
        assert nexus.record("alpha").health is result


class TestReissueCreationFailure:
    def test_handle_creation_failure_closes_the_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reissue's other failure shape: rebuilding the handle
        itself raises (driver gone, DSN newly invalid) — same outcome
        as a failed probe: resetting -> closed, reported not raised."""
        import localdata_mcp.nexus.persistence.manager as manager_module

        nexus = warmed(tmp_path, "alpha")

        def exploding_create_handle(*args: object, **kwargs: object) -> object:
            raise RuntimeError("driver unloadable")

        monkeypatch.setattr(
            manager_module, "create_handle", exploding_create_handle
        )
        entry = nexus.mark_faulted("alpha")
        assert entry.disposed and not entry.reissued
        record = nexus.record("alpha")
        assert record.state is LifecycleState.CLOSED
        assert record.health is not None and not record.health.healthy
        assert "reissue failed" in record.health.detail

