"""tests/v3/test_config_merge_security.py — E1 security exit-gate rows.

The introduction rule (ARCHITECTURE.md section 5): security-relevant
declarations — allowed_paths entries, endpoint declarations — are
introducible only at operator-trust layers (system/user, env included);
a project layer may narrow (drop a path, downgrade a posture) but never
mint. Refusals are typed, recorded, and non-fatal: the untrusted layer
that attempted them must not be able to break startup.
"""

from __future__ import annotations

from localdata_mcp.nexus.config.endpoints import EndpointDeclaration
from localdata_mcp.nexus.config.errors import (
    IntroductionRefusedError,
    PinShadowingError,
)
from localdata_mcp.nexus.config.merge import merge_sources
from localdata_mcp.nexus.config.provenance import Layer, LayerSource


def system(values: dict) -> LayerSource:
    return LayerSource(name="system-file", layer=Layer.SYSTEM, values=values)


def user(values: dict) -> LayerSource:
    return LayerSource(name="user-file", layer=Layer.USER, values=values)


def env(values: dict) -> LayerSource:
    return LayerSource(name="env", layer=Layer.USER, order=1, values=values)


def project(values: dict) -> LayerSource:
    return LayerSource(name="project-file", layer=Layer.PROJECT, values=values)


class TestAllowedPathsIntroduction:
    def test_env_may_introduce_paths_at_user_rank(self) -> None:
        # The battery-runner precedent (ARCHITECTURE.md section 7.3).
        result = merge_sources([env({"security": {"allowed_paths": ["/fixtures"]}})])
        assert result.model.security.allowed_paths == ("/fixtures",)
        assert result.refusals == ()

    def test_project_introduction_is_refused_fail_closed(self) -> None:
        # EXIT GATE: project-layer introduction of a security-relevant
        # field is REFUSED — and does not break startup.
        result = merge_sources(
            [project({"security": {"allowed_paths": ["/anywhere"]}})]
        )
        assert result.model.security.allowed_paths == ()
        (refusal,) = result.refusals
        assert isinstance(refusal, IntroductionRefusedError)
        assert refusal.field_path == "security.allowed_paths"
        assert refusal.source == "project-file"

    def test_project_may_narrow_operator_paths(self) -> None:
        result = merge_sources(
            [
                user({"security": {"allowed_paths": ["/data/a", "/data/b"]}}),
                project({"security": {"allowed_paths": ["/data/a"]}}),
            ]
        )
        assert result.model.security.allowed_paths == ("/data/a",)
        assert result.refusals == ()

    def test_project_superset_keeps_only_operator_paths(self) -> None:
        result = merge_sources(
            [
                user({"security": {"allowed_paths": ["/data/a"]}}),
                project({"security": {"allowed_paths": ["/data/a", "/evil"]}}),
            ]
        )
        assert result.model.security.allowed_paths == ("/data/a",)
        (refusal,) = result.refusals
        assert isinstance(refusal, IntroductionRefusedError)
        assert "/evil" in str(refusal)

    def test_lower_operator_layer_cannot_shadow_system_paths(self) -> None:
        result = merge_sources(
            [
                system({"security": {"allowed_paths": ["/data"]}}),
                user({"security": {"allowed_paths": ["/other"]}}),
            ]
        )
        assert result.model.security.allowed_paths == ("/data",)
        (refusal,) = result.refusals
        assert isinstance(refusal, PinShadowingError)


class TestEndpointIntroduction:
    def test_operator_layers_declare_endpoints(self) -> None:
        result = merge_sources(
            [
                user(
                    {
                        "endpoints": {
                            "warehouse": {
                                "dsn": "postgresql://db.internal/w",
                                "posture": "read_write",
                                "credentials_ref": "WAREHOUSE_SECRET",
                            }
                        }
                    }
                )
            ]
        )
        assert result.model.endpoints["warehouse"] == EndpointDeclaration(
            name="warehouse",
            dsn="postgresql://db.internal/w",
            posture="read_write",
            credentials_ref="WAREHOUSE_SECRET",
        )
        assert result.provenance.winner("endpoints.warehouse")[1] == ("user-file")

    def test_project_cannot_mint_an_endpoint(self) -> None:
        # EXIT GATE: a project-layer file can never mint a new endpoint.
        result = merge_sources(
            [project({"endpoints": {"exfil": {"dsn": "postgresql://evil.host/d"}}})]
        )
        assert "exfil" not in result.model.endpoints
        (refusal,) = result.refusals
        assert isinstance(refusal, IntroductionRefusedError)
        assert refusal.field_path == "endpoints.exfil"

    def test_system_declaration_refuses_user_shadowing(self) -> None:
        # EXIT GATE: pin shadowing of a declared endpoint is REFUSED.
        result = merge_sources(
            [
                system({"endpoints": {"db": {"dsn": "postgresql://prod/d"}}}),
                user({"endpoints": {"db": {"dsn": "postgresql://mine/d"}}}),
            ]
        )
        assert result.model.endpoints["db"].dsn == "postgresql://prod/d"
        (refusal,) = result.refusals
        assert isinstance(refusal, PinShadowingError)
        assert refusal.field_path == "endpoints.db"

    def test_project_may_downgrade_a_posture(self) -> None:
        result = merge_sources(
            [
                user(
                    {
                        "endpoints": {
                            "db": {
                                "dsn": "postgresql://prod/d",
                                "posture": "read_write",
                            }
                        }
                    }
                ),
                project({"endpoints": {"db": {"posture": "read_only"}}}),
            ]
        )
        assert result.model.endpoints["db"].posture == "read_only"
        assert result.refusals == ()

    def test_project_cannot_upgrade_a_posture(self) -> None:
        result = merge_sources(
            [
                user({"endpoints": {"db": {"dsn": "postgresql://prod/d"}}}),
                project({"endpoints": {"db": {"posture": "read_write"}}}),
            ]
        )
        assert result.model.endpoints["db"].posture == "read_only"
        (refusal,) = result.refusals
        assert isinstance(refusal, PinShadowingError)

    def test_project_cannot_redirect_a_dsn(self) -> None:
        result = merge_sources(
            [
                user({"endpoints": {"db": {"dsn": "postgresql://prod/d"}}}),
                project({"endpoints": {"db": {"dsn": "postgresql://evil/d"}}}),
            ]
        )
        assert result.model.endpoints["db"].dsn == "postgresql://prod/d"
        (refusal,) = result.refusals
        assert isinstance(refusal, PinShadowingError)
