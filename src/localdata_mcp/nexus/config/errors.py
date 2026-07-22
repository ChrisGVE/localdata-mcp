"""localdata_mcp/nexus/config/errors.py — NX-2's typed refusal taxonomy.

One class per refusal path (ARCHITECTURE.md section 5: a typed
ConfigurationError on validation failure, never a silent print).
Neighbors: loaders.py and env_derive.py raise the fatal kinds
(unknown field, type mismatch, invalid value); merge.py records the
trust-refusal kinds (introduction, pin shadowing) on the load result
so untrusted layers cannot break startup — see merge.py's rationale.
NX-3 (E4) will absorb these into the one error model; until then NX-2
owns them, matching the E1-before-E4 build order.
"""

from __future__ import annotations

from typing import Any


class ConfigurationError(Exception):
    """Base of every NX-2 refusal; carries reportable context."""

    def __init__(
        self,
        message: str,
        *,
        field_path: str | None = None,
        source: str | None = None,
        attempted_value: Any = None,
    ) -> None:
        super().__init__(message)
        self.field_path = field_path
        self.source = source
        self.attempted_value = attempted_value


class UnknownFieldError(ConfigurationError):
    """A source names a section or field the model does not declare."""


class TypeMismatchError(ConfigurationError):
    """A source's value cannot be read as the field's declared type."""


class InvalidValueError(ConfigurationError):
    """A well-typed value fails the field's validation rules."""


class IntroductionRefusedError(ConfigurationError):
    """A non-operator layer tried to introduce a security-relevant
    declaration (endpoint, credential ref, allowed_paths entry)."""


class PinShadowingError(ConfigurationError):
    """A lower-trust layer tried to shadow a pinned, operator-set value."""


class InlineCredentialError(ConfigurationError):
    """A config source carries literal credential material instead of a
    credentials_ref environment-variable indirection (NFR-110)."""
