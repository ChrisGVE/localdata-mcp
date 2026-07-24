"""localdata_mcp/nexus/chokepoint/surfaces_query.py — the guarded
read/write entrypoints (E6.1) and the ephemeral local-file seam.

`_QueryMutationSurface` carries `guarded_query` / `guarded_mutation` —
the only two names §6.2 lets a tool module use to touch a declared
backend — and `guarded_file_query`, NX-6's ad-hoc SQL-over-a-local-file
seam (I-2, NFR-114 i-b). Each resolves the operator-declared endpoint
through NX-5 (tools never see a connection — FR-802), screens the
statement through the inherited `_GuardCore` (the E6.2 allow-list / the
E6.2b SPARQL screen), enforces posture (NFR-113), contains every
extracted path literal (NFR-108), executes over the NX-5 connection,
and hands back the capability-narrow frozen `Result` (GP3's corollary).
A backend failure crosses NX-3's wire IN through the inherited `_wired`.
Chokepoint-internal by §6.2: composed into `Chokepoint` (guard.py).
"""

from __future__ import annotations

from pathlib import Path

from localdata_mcp.nexus.persistence.ephemeral import EphemeralEngineKind

from .core import _GuardCore
from .execution import execute_mutation, fetch_bounded
from .types import GuardRefusedError, QueryRequest, Result


class _QueryMutationSurface(_GuardCore):
    """The two guarded entrypoints and the ephemeral local-file seam."""

    # -- the two entrypoints (§6.2) -----------------------------------

    def guarded_query(self, endpoint_name: str, request: QueryRequest) -> Result:
        """The read path — permitted on ANY posture (NFR-113)."""
        record = self._persistence.record(endpoint_name)
        category = self._screen_read_side(request, record.backend_kind)
        with self._wired(endpoint_name, record.backend_kind):
            with self._persistence.connection(endpoint_name) as connection:
                columns, rows = fetch_bounded(
                    connection,
                    request.text,
                    request.parameters,
                    self._bounds,
                    self._config.query.default_chunk_size,
                )
        return Result(columns=columns, rows=rows, category=category)

    def guarded_mutation(self, endpoint_name: str, request: QueryRequest) -> Result:
        """The write path — `read_write` posture only (NFR-113), for
        the enumerated mutation category, write-side local-file
        constructs, and SPARQL update forms."""
        record = self._persistence.record(endpoint_name)
        if record.posture != "read_write":
            raise GuardRefusedError(
                f"mutation refused: endpoint {endpoint_name!r} is declared "
                "read_only (NFR-113) — the operator must declare "
                "read_write posture for writes"
            )
        category = self._screen_write_side(request, record.backend_kind)
        with self._wired(endpoint_name, record.backend_kind):
            with self._persistence.connection(endpoint_name) as connection:
                affected = execute_mutation(
                    connection, request.text, request.parameters
                )
        return Result(columns=(), rows=(), category=category, affected_rows=affected)

    # -- the ephemeral local-file seam (I-2, NFR-114 i-b) -------------

    def guarded_file_query(
        self,
        path: str | Path,
        request: QueryRequest,
        engine_kind: EphemeralEngineKind,
    ) -> Result:
        """Ad-hoc SQL over a local SQLite/DuckDB file: contained FIRST
        (NFR-108 — an out-of-tree path never reaches NX-5), screened by
        the same allow-list, and READ-ONLY by default — a mutation
        needs an operator `ephemeral_write_paths` grant (§5's ephemeral
        posture rule; NFR-114's local-file branch). The connection
        never outlives this call (no NX-5 pool is pinned, so the row-24
        stream cap does not apply — I-2)."""
        real = self.contain_path(path, mode="read")
        classification = self._classify(request, engine_kind)
        wants_write = (
            classification.category in ("mutation", "local_file_write")
            or classification.contains_mutation_nodes
        )
        connection_spec = self._persistence.open_ephemeral(str(real), engine_kind)
        if wants_write and connection_spec.posture != "read_write":
            raise GuardRefusedError(
                f"mutation refused: ad-hoc file source {str(real)!r} is "
                "read-only by default (NFR-114) — only an operator "
                "security.ephemeral_write_paths grant makes it writable"
            )
        self._contain_all(classification, mode="write" if wants_write else "read")
        with self._wired(f"file:{real}", engine_kind):
            with connection_spec.open() as live:
                if wants_write:
                    affected = execute_mutation(live, request.text, request.parameters)
                    return Result(
                        columns=(),
                        rows=(),
                        category=classification.category,
                        affected_rows=affected,
                    )
                columns, rows = fetch_bounded(
                    live,
                    request.text,
                    request.parameters,
                    self._bounds,
                    self._config.query.default_chunk_size,
                )
        return Result(columns=columns, rows=rows, category=classification.category)
