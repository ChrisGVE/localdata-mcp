"""localdata_mcp/nexus/chokepoint/surfaces_schema.py — the endpoint
enumeration (I-1) and schema-discovery (X-1, E9.1) seams.

`_SchemaSurface` carries the read-only discovery surfaces: every
declared endpoint's capability-narrow summary (the one surface NFR-114
refusals point callers to), and the by-DECLARED-kind schema views —
SQL kinds answer with the table catalog, store kinds are refused a
table catalog (their discovery surfaces are the store tool families,
not a column list). `read_table` is the Explore family's whole-table
read: membership verified against the catalog FIRST, the identifier
quoted by the dialect's own preparer, the fetch admission-gated like
every read. The inspector never crosses the seam (FR-802) — plain data
only. Chokepoint-internal by §6.2: composed into `Chokepoint`
(guard.py).
"""

from __future__ import annotations

from typing import Any

from . import introspection
from .core import _GuardCore
from .execution import fetch_bounded
from .types import EndpointSummary, GuardRefusedError, Result


class _SchemaSurface(_GuardCore):
    """The endpoint-enumeration and schema-discovery read surfaces."""

    # -- the endpoint-enumeration seam (I-1) --------------------------

    def endpoint_summaries(self) -> tuple[EndpointSummary, ...]:
        """Every declared endpoint's summary, backend-kind-agnostic —
        the one discovery surface NFR-114 refusals point callers to
        (list_endpoints reaches NX-5's state through here, §6.2)."""
        return tuple(
            self.endpoint_summary(name) for name in self._persistence.endpoint_names()
        )

    def endpoint_summary(self, name: str) -> EndpointSummary:
        """One named endpoint's summary (E8.3) — the capability-narrow
        kind/posture view store-family tools dispatch on (a kv call
        against a graph endpoint speaks graph tables); raises the same
        `UnknownEndpointError` every guard resolution raises."""
        record = self._persistence.record(name)
        health = record.health
        return EndpointSummary(
            name=record.name,
            backend_kind=record.backend_kind,
            posture=record.posture,
            healthy=None if health is None else health.healthy,
            health_detail="" if health is None else health.detail,
        )

    # -- the schema-discovery seam (X-1, E9.1) ------------------------

    def describe_endpoint(self, endpoint_name: str) -> dict[str, Any]:
        """The endpoint's schema summary, by DECLARED kind: SQL kinds
        answer with the table catalog (columns, keys, row counts),
        store kinds with their semantic shape (key space, graph shape,
        triple counts) — plain data only, the inspector never crosses
        the seam (FR-802)."""
        record = self._persistence.record(endpoint_name)
        with self._wired(endpoint_name, record.backend_kind):
            return introspection.endpoint_schema(record)

    def describe_endpoint_table(
        self, endpoint_name: str, table: str
    ) -> dict[str, Any] | None:
        """One table's schema on a SQL-kind endpoint, or None when the
        table does not exist; store kinds are refused (their discovery
        surfaces are the store tools, not a table catalog)."""
        record = self._persistence.record(endpoint_name)
        self._refuse_store_catalog(endpoint_name, record.backend_kind)
        with self._wired(endpoint_name, record.backend_kind):
            return introspection.table_schema(record, table)

    def endpoint_table_names(self, endpoint_name: str) -> tuple[str, ...]:
        """The SQL-kind endpoint's table catalog (find_table's search
        space); store kinds refused as above."""
        record = self._persistence.record(endpoint_name)
        self._refuse_store_catalog(endpoint_name, record.backend_kind)
        with self._wired(endpoint_name, record.backend_kind):
            return introspection.table_names(record)

    def read_table(self, endpoint_name: str, table: str) -> "Result | None":
        """A whole-table read for the Explore family (X-2/X-4's
        `table=` slot): membership verified against the catalog FIRST,
        the identifier quoted by the dialect's own preparer, the fetch
        admission-gated like every read. None when the table does not
        exist; store kinds refused as catalog reads."""
        record = self._persistence.record(endpoint_name)
        self._refuse_store_catalog(endpoint_name, record.backend_kind)
        with self._wired(endpoint_name, record.backend_kind):
            if table not in introspection.table_names(record):
                return None
            statement = introspection.quoted_select(record, table)
            with self._persistence.connection(endpoint_name) as connection:
                columns, rows = fetch_bounded(
                    connection,
                    statement,
                    None,
                    self._bounds,
                    self._config.query.default_chunk_size,
                )
        return Result(columns=columns, rows=rows, category="query")

    def _refuse_store_catalog(self, endpoint_name: str, backend_kind: str) -> None:
        if backend_kind in ("kv", "tree", "graph", "rdf"):
            raise GuardRefusedError(
                f"endpoint {endpoint_name!r} is a {backend_kind} store — "
                "its schema is semantic, not a table catalog; use "
                "describe_database for the store summary and the store "
                "tool family to browse"
            )
