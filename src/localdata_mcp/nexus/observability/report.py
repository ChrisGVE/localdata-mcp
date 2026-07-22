"""localdata_mcp/nexus/observability/report.py — startup config report.

Forward-ports `d5fb7280`'s startup pinned/shadowed report as designed
in ARCHITECTURE.md section 5: after NX-2's merge, the process logs —
through NX-4, hence redacted and on stderr, never printed — every
field a config source set (winner + each shadowed contribution with
its disposition) and every trust refusal, all derived from the
Provenance the merge already produced; there is no second bookkeeping
path. Neighbors: server/mcp_app.py calls this right after
reconfigure(); nexus/config/provenance.py defines the data read here.
"""

from __future__ import annotations

from typing import Any

import structlog

from ..config.merge import ConfigLoadResult
from ..config.provenance import FieldProvenance
from .manager import get_logger


def log_startup_report(
    result: ConfigLoadResult,
    *,
    logger: "structlog.stdlib.BoundLogger | None" = None,
) -> None:
    """Log the pinned/shadowed/refused story of one config load."""
    log = logger if logger is not None else get_logger(__name__)
    configured = [entry for entry in result.provenance.values() if entry.winning_source]
    log.info(
        "configuration report",
        configured_fields=len(configured),
        refusals=len(result.refusals),
    )
    for entry in configured:
        log.info(
            "config field set",
            field_path=entry.field_path,
            value=repr(entry.value),
            source=entry.winning_source,
            shadowed=_shadowed_contributions(entry),
        )
    for refusal in result.refusals:
        log.warning(
            "config contribution refused",
            reason=str(refusal),
            field_path=refusal.field_path,
            source=refusal.source,
        )


def _shadowed_contributions(entry: FieldProvenance) -> list[dict[str, Any]]:
    """Every losing attempt at the field, with how it lost. Values go
    out as repr strings so any shape survives the JSON renderer."""
    return [
        {
            "source": contribution.source,
            "layer": contribution.layer.name,
            "value": repr(contribution.value),
            "disposition": contribution.disposition,
        }
        for contribution in entry.contributions
        if contribution.disposition != "won"
    ]
