"""tests/v3/test_config_endpoints.py — E1.4 endpoint declarations.

Endpoints are config entities (name, DSN, posture, credentials_ref) —
never caller-supplied (NFR-114). Credential values are never inline:
`credentials_ref` names an environment variable NX-2 reads at
connection-issue time (NFR-110, PRD S4.1), and a DSN carrying literal
credential material is refused with a typed error.
"""

from __future__ import annotations

import dataclasses

import pytest

from localdata_mcp.nexus.config.endpoints import (
    EndpointDeclaration,
    endpoint_from_raw,
    resolve_credential,
)
from localdata_mcp.nexus.config.errors import (
    InlineCredentialError,
    InvalidValueError,
    UnknownFieldError,
)
from localdata_mcp.nexus.config.models import ConfigModel


class TestDeclaration:
    def test_defaults_are_fail_safe(self) -> None:
        declaration = EndpointDeclaration(
            name="analytics", dsn="postgresql://db.internal/analytics"
        )
        assert declaration.posture == "read_only"
        assert declaration.credentials_ref is None

    def test_declarations_are_frozen(self) -> None:
        declaration = EndpointDeclaration(name="a", dsn="sqlite:///x.db")
        with pytest.raises(dataclasses.FrozenInstanceError):
            declaration.posture = "read_write"  # type: ignore[misc]

    def test_model_defaults_to_no_endpoints(self) -> None:
        assert ConfigModel().endpoints == {}


class TestParsing:
    def test_parses_a_full_declaration(self) -> None:
        declaration = endpoint_from_raw(
            "warehouse",
            {
                "dsn": "postgresql://db.internal/warehouse",
                "posture": "read_write",
                "credentials_ref": "WAREHOUSE_DB_PASSWORD",
            },
            source="user-file",
        )
        assert declaration == EndpointDeclaration(
            name="warehouse",
            dsn="postgresql://db.internal/warehouse",
            posture="read_write",
            credentials_ref="WAREHOUSE_DB_PASSWORD",
        )

    def test_missing_dsn_is_refused(self) -> None:
        with pytest.raises(InvalidValueError):
            endpoint_from_raw("x", {"posture": "read_only"}, source="s")

    def test_unknown_key_is_refused(self) -> None:
        with pytest.raises(UnknownFieldError) as excinfo:
            endpoint_from_raw(
                "x",
                {"dsn": "sqlite:///d.db", "password": "hunter2"},
                source="user-file",
            )
        assert excinfo.value.field_path == "endpoints.x.password"

    def test_bad_posture_is_refused(self) -> None:
        with pytest.raises(InvalidValueError):
            endpoint_from_raw(
                "x", {"dsn": "sqlite:///d.db", "posture": "rw"}, source="s"
            )

    def test_credentials_ref_must_be_an_env_var_name(self) -> None:
        with pytest.raises(InvalidValueError):
            endpoint_from_raw(
                "x",
                {"dsn": "sqlite:///d.db", "credentials_ref": "not a name"},
                source="s",
            )


class TestInlineCredentialRefusal:
    def test_url_password_is_refused(self) -> None:
        with pytest.raises(InlineCredentialError):
            endpoint_from_raw(
                "x",
                {"dsn": "postgresql://user:hunter2@db.internal/d"},
                source="s",
            )

    def test_keyword_password_is_refused(self) -> None:
        with pytest.raises(InlineCredentialError):
            endpoint_from_raw(
                "x",
                {"dsn": "Driver={ODBC};Server=s;PWD=hunter2"},
                source="s",
            )

    def test_username_without_password_is_allowed(self) -> None:
        declaration = endpoint_from_raw(
            "x", {"dsn": "postgresql://reader@db.internal/d"}, source="s"
        )
        assert declaration.dsn == "postgresql://reader@db.internal/d"


class TestCredentialResolution:
    def test_reads_the_named_environment_variable(self) -> None:
        declaration = EndpointDeclaration(
            name="x", dsn="postgresql://db/d", credentials_ref="X_SECRET"
        )
        assert resolve_credential(declaration, {"X_SECRET": "s3cret"}) == "s3cret"

    def test_no_ref_resolves_to_none(self) -> None:
        declaration = EndpointDeclaration(name="x", dsn="sqlite:///d.db")
        assert resolve_credential(declaration, {}) is None

    def test_unset_variable_is_refused(self) -> None:
        declaration = EndpointDeclaration(
            name="x", dsn="postgresql://db/d", credentials_ref="X_SECRET"
        )
        with pytest.raises(InvalidValueError):
            resolve_credential(declaration, {})
