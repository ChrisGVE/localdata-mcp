"""localdata_mcp/nexus/persistence/manager.py — NX-5, the sole owner.

The one collection of live connection state in v3 (FR-801 — the
god-class's bare `self.connections` dict MUST NOT recur): a
`ConnectionRecord` per declared endpoint, built at §4e warm-up,
issued to NX-6 exclusively (§6.2 — `connection()` is not in the
tool-importable protocol set), and walked through the §5 lifecycle by
`mark_faulted`, the REAL implementation of E4.0's `FaultSignal`
protocol — NX-3's wire path calls it synchronously on a
connection-class error, and the record disposes-and-reissues rather
than rolling back (a rollback can fail on a broken connection;
disposal cannot). Ephemeral file opens route through the same owner so
the operator rw grant is read from the one live config. Neighbors:
record.py/lifecycle.py hold the state machine; engines.py builds and
rebuilds handles; health.py probes them; nexus/error/wire.py fires the
fault signal.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

from localdata_mcp.nexus.config.models import ConfigModel
from localdata_mcp.nexus.error.fault_signal import DisposeAndReissueEntry
from localdata_mcp.nexus.observability.manager import get_logger
from localdata_mcp.nexus.persistence.engines import backend_kind_of, create_handle
from localdata_mcp.nexus.persistence.ephemeral import (
    EphemeralEngineKind,
    EphemeralFileConnection,
    ephemeral_for,
)
from localdata_mcp.nexus.persistence.health import HealthCheckResult, probe
from localdata_mcp.nexus.persistence.lifecycle import LifecycleState
from localdata_mcp.nexus.persistence.limits import limits_from_config
from localdata_mcp.nexus.persistence.record import ConnectionRecord

logger = get_logger(__name__)


class UnknownEndpointError(LookupError):
    """The caller named an endpoint no operator declared (NFR-114)."""


class EndpointUnavailableError(RuntimeError):
    """The record exists but is not issuable — mid-fault-walk or
    terminally closed; the caller sees a structured NX-3 error."""


class PersistenceNexus:
    """The record collection and its lifecycle driver (§5, §8 NX-5)."""

    def __init__(self, config: ConfigModel, environ: Mapping[str, str]) -> None:
        self._config = config
        self._environ = environ
        self._records: dict[str, ConnectionRecord] = {}
        self._lock = threading.RLock()

    def warm_up(self) -> Mapping[str, HealthCheckResult]:
        """§4e: build and health-check a record per declared endpoint.

        An unhealthy endpoint still gets its record (the operator may
        bring the backend up later; on-demand probes re-check) — warm-up
        reports, it does not refuse startup.
        """
        results: dict[str, HealthCheckResult] = {}
        with self._lock:
            for name, declaration in self._config.endpoints.items():
                limits = limits_from_config(self._config)
                handle = create_handle(declaration, limits, self._environ)
                record = ConnectionRecord(
                    name=name,
                    backend_kind=backend_kind_of(declaration.dsn),
                    posture=declaration.posture,
                    credentials_ref=declaration.credentials_ref,
                    pool=handle,
                    limits=limits,
                )
                record.health = probe(handle, record.backend_kind)
                self._records[name] = record
                results[name] = record.health
                logger.info(
                    "endpoint warmed",
                    endpoint=name,
                    backend_kind=record.backend_kind,
                    healthy=record.health.healthy,
                )
        return results

    def endpoint_names(self) -> tuple[str, ...]:
        """Every declared endpoint's name, in declaration order — the
        enumeration NX-6's summary seam reads (I-1's list_endpoints)."""
        with self._lock:
            return tuple(self._records)

    def record(self, name: str) -> ConnectionRecord:
        """The named record, for NX-6's resolution and for probes."""
        with self._lock:
            try:
                return self._records[name]
            except KeyError:
                raise UnknownEndpointError(
                    f"no declared endpoint named {name!r}"
                ) from None

    @contextmanager
    def connection(self, name: str) -> Iterator[Any]:
        """A live connection from the named record's pool — NX-6's seam
        (§6.2), never importable by tool modules (FR-105/802)."""
        record = self.record(name)
        if not record.issuable:
            raise EndpointUnavailableError(
                f"endpoint {name!r} is {record.state.value}, not issuable"
            )
        with record.pool.connect() as live:
            yield live

    def check_health(self, name: str) -> HealthCheckResult:
        """On-demand FR-803 probe; the result lands on the record."""
        record = self.record(name)
        record.health = probe(record.pool, record.backend_kind)
        return record.health

    def open_ephemeral(
        self, path: str, engine_kind: EphemeralEngineKind
    ) -> EphemeralFileConnection:
        """An ad-hoc file source (E5.3) under the live operator grants —
        containment (NFR-108) is NX-6's check before this call."""
        return ephemeral_for(
            path, engine_kind, self._config.security.ephemeral_write_paths
        )

    def mark_faulted(self, record_id: str) -> DisposeAndReissueEntry:
        """The E4.0 fault signal, real half: walk §5 and dispose-and-reissue.

        Synchronous by contract — returns only after the record has left
        `healthy`. Never raises into NX-3's error path: an unknown or
        already-closed record is bookkept as (disposed=False,
        reissued=False), the caller's wire shape unaffected either way.
        """
        with self._lock:
            record = self._records.get(record_id)
            if record is None or record.state is LifecycleState.CLOSED:
                return DisposeAndReissueEntry(
                    record_id=record_id, disposed=False, reissued=False
                )
            record.transition(LifecycleState.FAULTED)
            record.transition(LifecycleState.RESETTING)
            record.pool.dispose()
            return self._reissue(record)

    def _reissue(self, record: ConnectionRecord) -> DisposeAndReissueEntry:
        """The reissue half: a fresh handle, proven by probe. Success
        walks `resetting → healthy`; failure walks `resetting → closed`
        (§5 — the endpoint is unusable until the operator intervenes)."""
        declaration = self._config.endpoints[record.name]
        try:
            fresh = create_handle(declaration, record.limits, self._environ)
            health = probe(fresh, record.backend_kind)
        except Exception as failure:  # creation itself failed — same outcome
            health = HealthCheckResult(
                healthy=False,
                response_time_ms=0.0,
                detail=f"reissue failed: {type(failure).__name__}: {failure}",
            )
            fresh = None  # type: ignore[assignment]
        record.health = health
        if health.healthy and fresh is not None:
            record.pool = fresh
            record.transition(LifecycleState.HEALTHY)
            logger.info("connection reissued", endpoint=record.name)
            return DisposeAndReissueEntry(
                record_id=record.name, disposed=True, reissued=True
            )
        record.transition(LifecycleState.CLOSED)
        logger.warning(
            "reissue failed; record closed",
            endpoint=record.name,
            detail=health.detail,
        )
        return DisposeAndReissueEntry(
            record_id=record.name, disposed=True, reissued=False
        )

    def close_all(self) -> None:
        """§4e shutdown: every record → closed, every pool disposed."""
        with self._lock:
            for record in self._records.values():
                if record.state is not LifecycleState.CLOSED:
                    record.transition(LifecycleState.CLOSED)
                    record.pool.dispose()
