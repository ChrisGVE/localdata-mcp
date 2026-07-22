"""localdata_mcp/nexus/config/fields.py — field-declaration helpers for NX-2.

Sits under nexus/config/ beside models.py, which uses `cfg_field` to
declare every S8 default with its metadata (pin-eligibility, derivation,
rationale) on the dataclass field itself — ARCHITECTURE.md section 5's
"pin-eligibility has exactly one home" rule. merge.py and env_derive.py
read the metadata keys defined here; they never keep a parallel list.
"""

from __future__ import annotations

from dataclasses import field
from typing import Any, Callable, Final

# Sentinel default for a field whose value is derived from other fields
# at model construction (S8 row 13's "derivation declared on the field").
# Any explicit operator value replaces it; validation refuses negatives,
# so the sentinel can never leak into a resolved model.
DERIVED: Final[int] = -1

# Metadata keys consumed by models.py, merge.py, and env_derive.py.
META_PIN: Final[str] = "pin"  # bool | None — None inherits the section class
META_DERIVE: Final[str] = "derive"  # Callable[[ConfigModel], value] | None
META_DOC: Final[str] = "doc"  # str — the S8 row reference + rationale
META_INTRODUCTION_GATED: Final[str] = "introduction_gated"  # bool


def cfg_field(
    default: Any,
    *,
    doc: str,
    pin: bool | None = None,
    derive: Callable[[Any], Any] | None = None,
    introduction_gated: bool = False,
) -> Any:
    """Declare one config truth: default + metadata, in one place.

    `pin=None` defers to the owning section's security classification
    (fail-closed: a security-classed section pins every field unless the
    field explicitly opts out with `pin=False`).
    """
    return field(
        default=default,
        metadata={
            META_PIN: pin,
            META_DERIVE: derive,
            META_DOC: doc,
            META_INTRODUCTION_GATED: introduction_gated,
        },
    )
