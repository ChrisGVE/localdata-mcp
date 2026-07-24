"""localdata_mcp/nexus/chokepoint/resource_bounds.py — NFR-105 bounds (E6.5).

The fail-SAFE revival of the kept memory-budget machinery: `main`'s
gate returned safe defaults when its own bound-checking failed
(`streaming/memory.py:50-60` — the NFR-105 fail-open defect); here an
internal error inside ANY admission check is itself a structured
refusal (§4c "reject, fail-safe"), never a pass. One instance owns the
process-wide aggregate accounting (§5 bound 2): every live registry's
residency is charged against the ONE `resources.memory_ceiling_bytes`,
so concurrent streams and analyses cannot individually pass while
jointly exceeding the ceiling. Analytical admission is DYNAMIC (S8
row 13): the row count is capped by `query.max_analysis_rows` AND the
estimated working set (rows × per-row estimate) is checked against
`ceiling − live residency`, the refusal naming current residency. There
is NO disk-spill/staging branch here: v3 streaming bounds memory by
look-ahead backpressure (chunk_registry.py pauses the pull at the K/B
bound), not by spilling to disk, so no spill/staging WRITE path exists
in the tree for a disk gate to sit in front of — the earlier
`admit_spill`/`release_spill`/`live_spill` machinery (and its `_spill`
ledger) was fully implemented but had zero production callers, so it was
removed rather than left as a gate whose comment falsely claimed it was
wired (CR-006/GP5). The `resources.max_spill_bytes` and
`min_free_disk_bytes` NX-2 fields were removed with it (they had no
remaining consumer); the only disk WRITE the tree performs is NX-8's
`export_to_file`, which
enforces its own containment and atomic-write and is out of NX-6's
memory accounting. Every memory bound is read from NX-2 — no literal in
this file may restate an S8 default (NFR-403). Neighbors:
chunk_registry.py charges stream residency here; guard.py admits every
analytical and load path through here and shapes refusals via NX-3.
"""

from __future__ import annotations

import functools
import threading
from typing import Any, Callable, Literal, TypeVar, cast

from localdata_mcp.nexus.config.models import ConfigModel

ResourceClass = Literal["memory", "internal"]

_F = TypeVar("_F", bound=Callable[..., Any])


class ResourceRefusedError(RuntimeError):
    """An NFR-105 structured refusal naming its resource class;
    guard.py shapes it through NX-3."""

    def __init__(self, message: str, *, resource_class: ResourceClass) -> None:
        super().__init__(message)
        self.resource_class: ResourceClass = resource_class


def _fail_safe(check: _F) -> _F:
    """The fail-open fix, structural: an internal error inside a bound
    check REFUSES (§4c) — the defect class where `main` answered its
    own failure with safe defaults cannot recur on any wrapped path."""

    @functools.wraps(check)
    def guarded(self: "ResourceBounds", *args: Any, **kwargs: Any) -> Any:
        try:
            return check(self, *args, **kwargs)
        except ResourceRefusedError:
            raise
        except Exception as failure:
            raise ResourceRefusedError(
                f"internal error while checking {check.__name__} — refused "
                f"fail-safe (NFR-105): {type(failure).__name__}: {failure}",
                resource_class="internal",
            ) from failure

    return cast(_F, guarded)


class ResourceBounds:
    """Aggregate accounting and admissions against the one ceiling.

    Registries (streams, in-flight analyses) charge their live resident
    bytes under a `registry_id`; every admission decision reads the sum
    — never a per-retrieval view — so the ceiling holds jointly (§5
    bound 2). Thread-safe: one lock over the residency ledger.
    """

    def __init__(self, config: ConfigModel) -> None:
        self._resources = config.resources
        self._query = config.query
        self._residency: dict[str, int] = {}
        self._lock = threading.Lock()

    # -- aggregate memory ledger (§5 bound 2) -------------------------

    def live_residency(self) -> int:
        """The sum of every live registry's resident bytes."""
        with self._lock:
            return sum(self._residency.values())

    @_fail_safe
    def charge(self, registry_id: str, resident_bytes: int) -> None:
        """Set `registry_id`'s current residency, refusing an aggregate
        over the ceiling (the charge is rolled back, so a refused pull
        leaves the ledger exactly as it was — the caller may serve
        buffered data and retry once residency drops)."""
        if resident_bytes < 0:
            raise ResourceRefusedError(
                f"negative residency {resident_bytes} for {registry_id!r} — "
                "refused fail-safe (NFR-105)",
                resource_class="internal",
            )
        ceiling = self._resources.memory_ceiling_bytes
        with self._lock:
            previous = self._residency.get(registry_id)
            self._residency[registry_id] = resident_bytes
            aggregate = sum(self._residency.values())
            if aggregate > ceiling:
                if previous is None:
                    del self._residency[registry_id]
                else:
                    self._residency[registry_id] = previous
                raise ResourceRefusedError(
                    f"memory charge refused: {registry_id!r} at "
                    f"{resident_bytes} bytes would put aggregate residency "
                    f"at {aggregate} bytes, over the "
                    f"{ceiling}-byte ceiling (NFR-105)",
                    resource_class="memory",
                )

    def release(self, registry_id: str) -> None:
        """Drop a registry from the ledger (idempotent — releasing an
        unknown id is a no-op, never an error on the teardown path)."""
        with self._lock:
            self._residency.pop(registry_id, None)

    # -- admissions (all fail-safe) -----------------------------------

    @_fail_safe
    def admit_analysis(self, rows: int, per_row_bytes: float) -> None:
        """S8 row 13's DYNAMIC admission: cap by `max_analysis_rows`,
        then check the estimated working set against live headroom.

        A non-positive per-row estimate cannot come from a real
        measurement, and taking it at face value would admit anything —
        refused fail-safe rather than trusted.
        """
        if rows < 0 or per_row_bytes <= 0:
            raise ResourceRefusedError(
                f"analytical admission refused: rows={rows}, "
                f"per_row_bytes={per_row_bytes} is not a real measurement "
                "— refused fail-safe (NFR-105)",
                resource_class="internal",
            )
        max_rows = self._query.max_analysis_rows
        if rows > max_rows:
            raise ResourceRefusedError(
                f"analytical admission refused: {rows} rows exceeds the "
                f"max_analysis_rows cap of {max_rows} (S8 row 13) — "
                "stream the data instead",
                resource_class="memory",
            )
        estimated = int(rows * per_row_bytes)
        self._require_memory_headroom(estimated, what="analytical working set")

    @_fail_safe
    def admit_load(self, estimated_bytes: int) -> None:
        """The load-then-serve upfront gate (§5): a whole-dataset read
        is admitted only if its estimated full size fits current
        headroom — the refusal lands BEFORE the load, which is the only
        point a decompression bomb can be stopped ahead of degradation."""
        if estimated_bytes < 0:
            raise ResourceRefusedError(
                f"load admission refused: negative size estimate "
                f"{estimated_bytes} — refused fail-safe (NFR-105)",
                resource_class="internal",
            )
        self._require_memory_headroom(estimated_bytes, what="full load")

    def _require_memory_headroom(self, estimated: int, *, what: str) -> None:
        """Refuse when `estimated` exceeds `ceiling − live residency`,
        naming current residency (S8 row 13's refusal contract)."""
        ceiling = self._resources.memory_ceiling_bytes
        residency = self.live_residency()
        headroom = ceiling - residency
        if estimated > headroom:
            raise ResourceRefusedError(
                f"memory admission refused: estimated {what} of "
                f"{estimated} bytes exceeds headroom of {headroom} bytes "
                f"(ceiling {ceiling}, live aggregate residency "
                f"{residency}) — NFR-105",
                resource_class="memory",
            )
