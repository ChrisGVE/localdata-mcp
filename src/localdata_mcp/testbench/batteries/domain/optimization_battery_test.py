"""testbench/batteries/domain/optimization_battery_test.py — E10.f slice.

The NFR-502c domain battery's optimization rows: FR-301 L3 coverage,
the FR-304 oracle (every solution re-checked against its analytic
closed form and a scipy recomputation), and the FR-305 negative row —
a host-escape payload in the objective string is refused by the
deny-by-default grammar, never executed. The constrained rows use
textbook problems with known exact optima; the LP row's solution is
hand-derivable ((x1, x2) = (4, 2) at the constraint intersection).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

import anyio
import pandas as pd
import pytest
from fastmcp import Client

import localdata_mcp.ingest.runtime as runtime
from localdata_mcp.nexus.chokepoint.guard import Chokepoint
from localdata_mcp.nexus.config.models import ConfigModel, SecurityConfig
from localdata_mcp.nexus.contract.registry import default_registry
from localdata_mcp.nexus.response.shaping import configure_shaping
from localdata_mcp.server.mcp_app import app


@pytest.fixture()
def bench(tmp_path: Path) -> Iterator[Path]:
    config = ConfigModel(
        security=SecurityConfig(allowed_paths=(str(tmp_path),)),
    )
    guard = Chokepoint.boot(config, environ=dict(os.environ))
    configure_shaping(config, default_registry())
    runtime.configure_ingest(guard)
    yield tmp_path
    runtime._CHOKEPOINT = None
    configure_shaping(ConfigModel(), default_registry())
    guard.shutdown()


def _call_envelope(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
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


def _data(envelope: dict[str, Any]) -> Any:
    assert envelope["error"] is None, envelope["error"]
    return envelope["data"]


def test_linear_program_reaches_the_analytic_vertex(bench: Path) -> None:
    """solve_linear_program — minimise -x1 - 2*x2 s.t. x1 + x2 <= 6,
    x2 <= 2 (x >= 0): the optimum sits at the vertex (4, 2)."""
    target = bench / "lp.csv"
    pd.DataFrame(
        {"cost": [-1.0, -2.0], "sum_cap": [1.0, 1.0], "x2_cap": [0.0, 1.0]}
    ).to_csv(target, index=False)
    data = _data(
        _call_envelope(
            "solve_linear_program",
            {
                "path": str(target),
                "objective_column": "cost",
                "constraint_columns": ["sum_cap", "x2_cap"],
                "constraint_values": [6.0, 2.0],
            },
        )
    )
    config = ConfigModel()
    rtol = config.testbench.tol_closed_form_rtol
    assert data["converged"] is True
    assert data["optimizer_status"] == 0
    assert data["solution"] == pytest.approx([4.0, 2.0], rel=rtol)
    assert data["objective_value"] == pytest.approx(-8.0, rel=rtol)


def test_constrained_quadratic_reaches_the_known_minimum(bench: Path) -> None:
    """optimize_constrained — (x0-1)^2 + (x1-2)^2 has its minimum at
    (1, 2); the string only ever meets the safe grammar."""
    target = bench / "guess.csv"
    pd.DataFrame({"start": [0.0, 0.0]}).to_csv(target, index=False)
    data = _data(
        _call_envelope(
            "optimize_constrained",
            {
                "path": str(target),
                "objective_expression": "(x[0]-1)**2 + (x[1]-2)**2",
                "initial_guess_column": "start",
            },
        )
    )
    config = ConfigModel()
    rtol = config.testbench.tol_iterative_rtol
    assert data["converged"] is True
    assert data["solution"] == pytest.approx([1.0, 2.0], rel=rtol, abs=1e-4)
    assert data["objective_value"] == pytest.approx(
        0.0, abs=config.testbench.tol_closed_form_rtol
    )


def test_constrained_with_inequality_respects_the_boundary(bench: Path) -> None:
    """optimize_constrained with x0 + x1 >= 4 (ineq): the constrained
    minimum of the same bowl moves to (1.5, 2.5)."""
    target = bench / "guess2.csv"
    pd.DataFrame({"start": [2.0, 2.0]}).to_csv(target, index=False)
    data = _data(
        _call_envelope(
            "optimize_constrained",
            {
                "path": str(target),
                "objective_expression": "(x[0]-1)**2 + (x[1]-2)**2",
                "initial_guess_column": "start",
                "constraint_expressions": ["x[0] + x[1] - 4"],
            },
        )
    )
    config = ConfigModel()
    assert data["converged"] is True
    assert data["solution"] == pytest.approx(
        [1.5, 2.5], rel=config.testbench.tol_iterative_rtol, abs=1e-4
    )


def test_hostile_objective_is_refused_not_executed(bench: Path) -> None:
    """FR-305: a host-escape payload becomes a structured refusal."""
    target = bench / "guess3.csv"
    pd.DataFrame({"start": [0.0]}).to_csv(target, index=False)
    sentinel_file = bench / "pwned.txt"
    envelope = _call_envelope(
        "optimize_constrained",
        {
            "path": str(target),
            "objective_expression": (
                f"__import__('pathlib').Path('{sentinel_file}').write_text('x')"
            ),
            "initial_guess_column": "start",
        },
    )
    assert envelope["error"] is not None
    assert not sentinel_file.exists()


def test_assignment_matches_the_hand_solved_matrix(bench: Path) -> None:
    """solve_assignment_problem — 3x3 with an obvious diagonal optimum."""
    target = bench / "costs.csv"
    pd.DataFrame(
        {
            "worker": ["ann", "bob", "cid"],
            "t1": [1.0, 9.0, 9.0],
            "t2": [9.0, 1.0, 9.0],
            "t3": [9.0, 9.0, 1.0],
        }
    ).to_csv(target, index=False)
    data = _data(
        _call_envelope(
            "solve_assignment_problem",
            {
                "path": str(target),
                "cost_columns": ["t1", "t2", "t3"],
                "agent_column": "worker",
            },
        )
    )
    config = ConfigModel()
    assert data["total_cost"] == pytest.approx(
        3.0, rel=config.testbench.tol_closed_form_rtol
    )
    assigned = {entry["agent"]: entry["task"] for entry in data["assignments"]}
    assert assigned == {"ann": "t1", "bob": "t2", "cid": "t3"}
