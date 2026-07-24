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
# When True, the one-default-site scanner (default_site_check.py) skips
# this field's value: a pervasive magic number (e.g. 0.05) whose one-home
# discipline rests on review because it cannot be mechanically told apart
# from unrelated uses of the same literal — mirroring the ints-0-3 exempt
# rule. The field is still a real operator knob (env mapping, one home);
# only the AST gate stands aside.
META_UNSCANNED: Final[str] = "unscanned"  # bool


def cfg_field(
    default: Any,
    *,
    doc: str,
    pin: bool | None = None,
    derive: Callable[[Any], Any] | None = None,
    introduction_gated: bool = False,
    unscanned: bool = False,
) -> Any:
    """Declare one config truth: default + metadata, in one place.

    `pin=None` defers to the owning section's security classification
    (fail-closed: a security-classed section pins every field unless the
    field explicitly opts out with `pin=False`). `unscanned=True` exempts
    the value from the one-default-site AST gate (review-only, for a
    pervasive magic literal — see META_UNSCANNED).
    """
    return field(
        default=default,
        metadata={
            META_PIN: pin,
            META_DERIVE: derive,
            META_DOC: doc,
            META_INTRODUCTION_GATED: introduction_gated,
            META_UNSCANNED: unscanned,
        },
    )
