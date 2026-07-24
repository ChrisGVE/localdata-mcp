"""testbench/batteries/security/resource_failsafe_test.py — NFR-105 at the bound seam.

The fail-open→fail-closed fix, exercised by fault injection. `main`'s
memory gate answered its *own* internal failure with safe defaults
(streaming/memory.py:50-60), so a bug in the bound checker silently
admitted anything; v3 wraps every admission in a fail-safe guard, so an
internal error inside any bound check is itself a structured refusal
(§4c "reject, fail-safe"), never a pass.

This control cannot be provoked from the client wire — a caller has no way
to make the checker's own internals bug out — so the battery reaches the
one place the fault lives: the live resource-bounds owned by the booted
chokepoint (obtained through the guard the seam yields, never imported —
NFR-103's own internals-stay-internal rule holds for this module too).
Into each memory admission method (memory charge, analytical admission,
load admission) an internal fault is injected, and each is asserted to
REFUSE with the `internal` resource class rather than return — and rather
than leak the raw exception. A genuine over-budget refusal is
proven to pass through the guard unchanged (not re-wrapped), and an
explicit-guard refusal stays intact, so the fail-safe wrapper never masks a
real bound decision.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

from . import _seam


class _InjectedBug(RuntimeError):
    """A stand-in for an undiscovered bug firing inside a bound check."""


def _raise_bug(*_args: object, **_kwargs: object) -> None:
    raise _InjectedBug("injected checker fault")


def _expect_refusal(action: Callable[[], Any]) -> Any:
    """Run `action`; assert it raised a ResourceRefusedError (resolved by
    type name so no chokepoint internal is imported) and return it. A
    normal return is the fail-open regression this whole file guards
    against — it fails loudly."""
    try:
        action()
    except Exception as failure:  # noqa: BLE001 — concrete type asserted below
        assert type(failure).__name__ == "ResourceRefusedError", type(failure).__name__
        return failure
    raise AssertionError("expected a ResourceRefusedError refusal, got a normal return")


@pytest.fixture()
def bounds(tmp_path: Path) -> Iterator[Any]:
    """The live `ResourceBounds` the booted chokepoint owns — real config
    ceilings, reached through the guard rather than an import."""
    with _seam.booted(allowed_paths=(str(tmp_path),)) as guard:
        yield guard._bounds


# -- an internal fault inside each admission method refuses, never passes --


def test_charge_internal_fault_refuses(bounds) -> None:
    # Reading the ceiling blows up mid-check (a corrupted config handle).
    bounds._resources = object()
    error = _expect_refusal(lambda: bounds.charge("registry", 1))
    assert error.resource_class == "internal"


def test_admit_analysis_internal_fault_refuses(bounds) -> None:
    bounds._require_memory_headroom = _raise_bug
    error = _expect_refusal(lambda: bounds.admit_analysis(rows=1, per_row_bytes=1.0))
    assert error.resource_class == "internal"


def test_admit_load_internal_fault_refuses(bounds) -> None:
    bounds._require_memory_headroom = _raise_bug
    error = _expect_refusal(lambda: bounds.admit_load(estimated_bytes=1))
    assert error.resource_class == "internal"


# -- the guard never masks a genuine bound decision ---------------------


def test_genuine_refusal_passes_through_unchanged(bounds) -> None:
    module = sys.modules[type(bounds).__module__]
    refused = module.ResourceRefusedError

    def refuse_for_real(*_args: object, **_kwargs: object) -> None:
        raise refused("real headroom refusal", resource_class="memory")

    bounds._require_memory_headroom = refuse_for_real
    error = _expect_refusal(lambda: bounds.admit_analysis(rows=1, per_row_bytes=1.0))
    # Re-raised as-is: still the memory class, not re-wrapped to internal.
    assert error.resource_class == "memory"


def test_explicit_guard_still_refuses_bad_input(bounds) -> None:
    # A non-positive per-row estimate can never be a real measurement.
    error = _expect_refusal(lambda: bounds.admit_analysis(rows=1, per_row_bytes=0.0))
    assert error.resource_class == "internal"
