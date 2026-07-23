"""tests/v3/test_process_inventory.py — FR-703 orphan-inventory gate (E10.x9).

FR-703's acceptance made mechanical: the process-domain tool surface
is enumerated and each function is either tool-registered or listed in
the deferred-items registry with a rationale. Two legs:

1. the LIVE registry's process surface equals `LAUNCH_PROCESS_TOOLS`
   exactly (a new or dropped tool forces a reviewed edit to the
   inventory — no silent orphan, no silent surface change);
2. every deferred main capability carries a non-empty rationale (the
   "explicitly listed with rationale" half of the acceptance).
"""

from __future__ import annotations

from localdata_mcp.nexus.contract.registry import default_registry
from localdata_mcp.nexus.contract.spec_modules import load_spec_modules
from localdata_mcp.process.inventory import (
    DEFERRED_DOMAIN_CAPABILITIES,
    LAUNCH_PROCESS_TOOLS,
    deferred_without_rationale,
)


def test_live_process_surface_equals_the_declared_inventory() -> None:
    load_spec_modules()
    registered = {spec.name for spec in default_registry() if spec.domain == "process"}
    assert registered == set(LAUNCH_PROCESS_TOOLS), (
        "the live process tool surface diverged from the FR-703 inventory; "
        "register the new tool and add it to LAUNCH_PROCESS_TOOLS, or "
        f"defer it with rationale. Delta: "
        f"{registered.symmetric_difference(LAUNCH_PROCESS_TOOLS)}"
    )


def test_every_deferred_capability_carries_a_rationale() -> None:
    assert deferred_without_rationale() == []
    assert DEFERRED_DOMAIN_CAPABILITIES, "the deferred-items list must not be empty"
