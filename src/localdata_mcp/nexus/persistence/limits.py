"""localdata_mcp/nexus/persistence/limits.py — per-endpoint limits from NX-2.

E5.4's second half: the resource limits a `ConnectionRecord` carries are
READ from the config nexus (S8 rows 2/3/24 — operator-tunable, never
hardcoded here), applied per endpoint. This is the harvested successor
of `connection_manager/resources.py`'s limit setup, shrunk to what §5
actually stores on the record: connection-count and statement-timeout
bounds (the memory ceiling is process-wide and belongs to the budget
gate, not to any one endpoint). Neighbors: record.py embeds
`ResourceLimits`; engines.py sizes pools from it; NX-6 enforces the
same numbers at the chokepoint for ephemeral connections (§5 — they
inherit these global defaults, having no record of their own).
"""

from __future__ import annotations

from dataclasses import dataclass

from localdata_mcp.nexus.config.models import ConfigModel


@dataclass(frozen=True)
class ResourceLimits:
    """The per-endpoint bounds a record enforces (§5, §6(g))."""

    max_connections: int
    statement_timeout_seconds: int
    max_concurrent_streams: int


def limits_from_config(config: ConfigModel) -> ResourceLimits:
    """One endpoint's limits, read from NX-2's declared fields.

    Every declared endpoint currently shares the operator-set global
    defaults (S8 rows 2/3/24 are per-endpoint numbers with one
    operator-tunable value); a future per-endpoint override would be a
    new NX-2 declaration, never a literal here.
    """
    return ResourceLimits(
        max_connections=config.resources.max_connections_per_endpoint,
        statement_timeout_seconds=config.resources.query_timeout_seconds,
        max_concurrent_streams=config.query.max_concurrent_streams_per_endpoint,
    )
