"""localdata_mcp/nexus/config/endpoints.py — endpoint declarations.

The config entity behind NFR-114's endpoint-declaration model: the
operator declares endpoints (name, DSN, posture, credentials_ref) in
operator-trust config layers; callers only ever name them. Credential
values are never inline (NFR-110): `credentials_ref` names an
environment variable read at connection-issue time, and a DSN carrying
literal credential material is refused. Neighbors: models.py composes
the declarations into ConfigModel; merge.py applies the introduction
and pinning rules; NX-5 consumes the declarations to build pools.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from .errors import (
    InlineCredentialError,
    InvalidValueError,
    UnknownFieldError,
)

Posture = Literal["read_only", "read_write"]

POSTURES: tuple[Posture, ...] = ("read_only", "read_write")

# The exact, mechanical shape of an environment-variable name (PRD S4.1:
# "every endpoint's credentials_ref field is an environment-variable
# name" — a shape check here; the battery scans for known literals).
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")

# Keyword credential markers in non-URL DSNs (ODBC-style strings).
_KEYWORD_CREDENTIAL = re.compile(r"(?i)\b(password|pwd)\s*=")

# URL userinfo carrying a password component: scheme://user:secret@host
_URL_PASSWORD = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://[^/@]*:[^/@]+@")

_KNOWN_KEYS = frozenset({"dsn", "posture", "credentials_ref"})


@dataclass(frozen=True)
class EndpointDeclaration:
    """One declared endpoint; read-only unless the operator says more."""

    name: str
    dsn: str
    posture: Posture = "read_only"
    credentials_ref: str | None = None


def endpoint_from_raw(
    name: str, raw: Mapping[str, Any], *, source: str
) -> EndpointDeclaration:
    """Validate one raw config table into a declaration, typed errors on
    every refusal path."""
    for key in raw:
        if key not in _KNOWN_KEYS:
            raise UnknownFieldError(
                f"endpoint {name!r} declares unknown key {key!r}",
                field_path=f"endpoints.{name}.{key}",
                source=source,
            )
    dsn = raw.get("dsn")
    if not isinstance(dsn, str) or not dsn:
        raise InvalidValueError(
            f"endpoint {name!r} needs a non-empty string dsn",
            field_path=f"endpoints.{name}.dsn",
            source=source,
        )
    _refuse_inline_credentials(name, dsn, source)
    posture = _validated_posture(name, raw.get("posture", "read_only"), source)
    ref = _validated_ref(name, raw.get("credentials_ref"), source)
    return EndpointDeclaration(name=name, dsn=dsn, posture=posture, credentials_ref=ref)


def _refuse_inline_credentials(name: str, dsn: str, source: str) -> None:
    """NFR-110: no config source may carry a literal secret."""
    if _URL_PASSWORD.search(dsn) or _KEYWORD_CREDENTIAL.search(dsn):
        raise InlineCredentialError(
            f"endpoint {name!r} carries credential material in its DSN; "
            "declare a credentials_ref environment variable instead",
            field_path=f"endpoints.{name}.dsn",
            source=source,
        )


def _validated_posture(name: str, value: Any, source: str) -> Posture:
    if value not in POSTURES:
        raise InvalidValueError(
            f"endpoint {name!r} posture must be one of {POSTURES}, got {value!r}",
            field_path=f"endpoints.{name}.posture",
            source=source,
            attempted_value=value,
        )
    return value  # type: ignore[no-any-return]


def _validated_ref(name: str, value: Any, source: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _ENV_NAME.match(value):
        raise InvalidValueError(
            f"endpoint {name!r} credentials_ref must be an environment-"
            f"variable name, got {value!r}",
            field_path=f"endpoints.{name}.credentials_ref",
            source=source,
            attempted_value=value,
        )
    return value


def resolve_credential(
    declaration: EndpointDeclaration, environ: Mapping[str, str]
) -> str | None:
    """Read the declared credential at connection-issue time (NFR-110).

    None ref means the endpoint authenticates without a secret; a ref
    naming an unset variable is refused rather than passed through empty.
    """
    if declaration.credentials_ref is None:
        return None
    try:
        return environ[declaration.credentials_ref]
    except KeyError:
        raise InvalidValueError(
            f"endpoint {declaration.name!r} credentials_ref names unset "
            f"environment variable {declaration.credentials_ref}",
            field_path=f"endpoints.{declaration.name}.credentials_ref",
        ) from None
