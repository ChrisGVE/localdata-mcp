"""testbench/batteries/security/_seam.py — the NFR-502b battery's shared L3 seam.

Not a test module (underscore-prefixed, so pytest never collects it): the
security battery's hostile-payload modules import this one harness so the
boot/teardown ritual and the FR-403 refusal assertions live in a single
place (NFR-402 — imported, never copied across modules).

Every security row drives its payload through the real `fastmcp.Client`
wire (PROJECT-FP #4 — the surface the agent actually uses, through the
generated wrapper, response shaping, and the chokepoint guard) and asserts
the structured-refusal shape the whole document turns on: the transport
succeeds, the refusal rides the FR-403 envelope's `error` field (NFR-301's
one error nexus, never a bare exception on the wire), and any hostile
side-effect the payload attempted never happened.
"""

from __future__ import annotations

import contextlib
import json
import os
from typing import Any, Iterator

import anyio
from fastmcp import Client

import localdata_mcp.ingest.runtime as runtime
from localdata_mcp.nexus.chokepoint.guard import Chokepoint
from localdata_mcp.nexus.config.endpoints import EndpointDeclaration, Posture
from localdata_mcp.nexus.config.models import ConfigModel, SecurityConfig
from localdata_mcp.nexus.contract.registry import default_registry
from localdata_mcp.nexus.response.shaping import configure_shaping
from localdata_mcp.server.mcp_app import app


def declare(**dsn_posture: tuple[str, Posture]) -> dict[str, EndpointDeclaration]:
    """Build the endpoints mapping from name -> (dsn, posture) pairs."""
    return {
        name: EndpointDeclaration(name=name, dsn=dsn, posture=posture)
        for name, (dsn, posture) in dsn_posture.items()
    }


@contextlib.contextmanager
def booted(
    *,
    allowed_paths: tuple[str, ...],
    declarations: dict[str, EndpointDeclaration] | None = None,
) -> Iterator[Chokepoint]:
    """Boot the chokepoint + response shaping + ingest runtime for one L3
    session, tearing every process-global back down on exit (the base
    battery's teardown contract, factored here)."""
    config = ConfigModel(
        security=SecurityConfig(allowed_paths=allowed_paths),
        endpoints=declarations or {},
    )
    guard = Chokepoint.boot(config, environ=dict(os.environ))
    configure_shaping(config, default_registry())
    runtime.configure_ingest(guard)
    try:
        yield guard
    finally:
        runtime._CHOKEPOINT = None
        configure_shaping(ConfigModel(), default_registry())
        guard.shutdown()


def call_envelope(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """One L3 tool call; returns the FR-403 envelope.

    The transport itself must succeed — a policy refusal is delivered
    *inside* the envelope (NFR-301), so `is_error` staying false is part of
    the contract under test, not an accident of the happy path.
    """

    async def session() -> dict[str, Any]:
        async with Client(app) as client:
            result = await client.call_tool(name, arguments)
            assert not result.is_error
            if isinstance(result.structured_content, dict) and (
                "inline" in result.structured_content
            ):
                return result.structured_content
            payload = json.loads(result.content[0].text)
            assert isinstance(payload, dict)
            return payload

    return anyio.run(session)


def expect_refused(envelope: dict[str, Any]) -> dict[str, Any]:
    """Assert the envelope carries a structured refusal; return the error
    object (a dict with at least a string `message`)."""
    error = envelope["error"]
    assert error is not None, f"expected a refusal, got data: {envelope.get('data')!r}"
    assert isinstance(error.get("message"), str), error
    return error


def expect_ok(envelope: dict[str, Any]) -> Any:
    """Assert the envelope succeeded; return its data. Every security module
    carries at least one positive control so a blanket-refuse regression
    (which would pass every negative row vacuously) fails loudly."""
    assert envelope["error"] is None, envelope["error"]
    return envelope["data"]
