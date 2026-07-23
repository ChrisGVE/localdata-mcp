"""testbench/batteries/base/ingest_randomized_test.py — NFR-509 randomized mode (E14.3).

The `--randomize --seed 42` full-assertion half of the base-capability
battery: the SAME connect/retrieve and per-format scenarios the
deterministic slice (ingest_battery_test.py) runs, but executed in a
seed-shuffled ORDER against ONE shared guard — so any order-dependent
state leakage (a store write in one scenario perturbing a later read,
a fixture-file collision, a stateful connection) surfaces as a
randomized-fail where the declaration-order run passed (NFR-509's
"no deterministic-pass/randomized-fail regressions"). The scenario
tables, the L3 call helper, and every assertion are imported from the
deterministic slice — the randomized mode adds order, never a second
copy of the battery (NFR-402's one-declaration discipline).

Seed 42 is the fixed reproducibility seed the E14.3 story names (the
same literal the domain batteries pin); a failure is reproduced by
re-running this module. Neighbors: ingest_battery_test.py is the
deterministic half re-driven here; the pipeline battery carries its own
randomized coupling axis.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, Iterator, List, Tuple

import pytest

import localdata_mcp.ingest.runtime as runtime
from localdata_mcp.nexus.chokepoint.guard import Chokepoint
from localdata_mcp.nexus.config.models import ConfigModel, SecurityConfig
from localdata_mcp.nexus.contract.registry import default_registry
from localdata_mcp.nexus.response.shaping import configure_shaping

from .ingest_battery_test import (
    _BACKEND_ROWS,
    _FORMAT_ROWS,
    _MYSQL_DSN,
    _PG_DSN,
    _call,
    _data,
    _declarations,
    _seed_sql,
)

# The fixed reproducibility seed the E14.3 story names — the same
# literal the domain batteries pin; not an S8 config value.
_SEED = 42

# One scenario: a stable id and a zero-context action that runs the
# scenario against the shared guard, asserting internally.
Scenario = Tuple[str, Callable[[Chokepoint, Path], None]]


@pytest.fixture()
def bench(tmp_path: Path) -> Iterator[Chokepoint]:
    """The deterministic slice's guard, booted once for the whole
    shuffled run: same declarations, same SQL seeding."""
    config = ConfigModel(
        security=SecurityConfig(allowed_paths=(str(tmp_path),)),
        endpoints=_declarations(tmp_path),
    )
    guard = Chokepoint.boot(config, environ=dict(os.environ))
    for name in config.endpoints:
        if name.startswith("sql_"):
            _seed_sql(guard, name)
    configure_shaping(config, default_registry())
    runtime.configure_ingest(guard)
    yield guard
    runtime._CHOKEPOINT = None
    configure_shaping(ConfigModel(), default_registry())
    guard.shutdown()


def _backend_action(
    endpoint: str, retrieve: Callable[[str], None]
) -> Callable[[Chokepoint, Path], None]:
    def run(guard: Chokepoint, tmp_path: Path) -> None:
        retrieve(endpoint)

    return run


def _format_action(
    write_fixture: Callable[[Path], Path], expect: Callable[[object], None]
) -> Callable[[Chokepoint, Path], None]:
    def run(guard: Chokepoint, tmp_path: Path) -> None:
        fixture = write_fixture(tmp_path)
        expect(_data(_call("read_file", {"path": str(fixture)})))

    return run


def _scenarios() -> List[Scenario]:
    """Every base scenario present in this environment (dockerized SQL
    rows drop out when their DSN is absent, as in the deterministic
    slice), unshuffled — the caller applies the seeded order."""
    scenarios: List[Scenario] = []
    for endpoint, retrieve in _BACKEND_ROWS:
        if endpoint == "sql_postgresql" and not _PG_DSN:
            continue
        if endpoint == "sql_mysql" and not _MYSQL_DSN:
            continue
        scenarios.append((f"backend:{endpoint}", _backend_action(endpoint, retrieve)))
    for name, write_fixture, expect in _FORMAT_ROWS:
        scenarios.append((f"format:{name}", _format_action(write_fixture, expect)))
    return scenarios


def test_base_scenarios_pass_in_seed_shuffled_order(
    bench: Chokepoint, tmp_path: Path
) -> None:
    """NFR-509: the full base battery, run in seed-42 order against one
    shared guard, passes with the identical assertions — proving the
    deterministic pass is order-independent, not a declaration-order
    artifact."""
    scenarios = _scenarios()
    random.Random(_SEED).shuffle(scenarios)
    executed: list[str] = []
    for scenario_id, action in scenarios:
        try:
            action(bench, tmp_path)
        except AssertionError as failure:
            raise AssertionError(
                f"randomized base scenario {scenario_id!r} failed"
                f" (order so far: {executed})"
            ) from failure
        executed.append(scenario_id)
    # Every non-dockerized scenario ran: 6 embedded backends + 14 formats.
    assert len(executed) == len(scenarios)
    assert len(executed) >= len(_FORMAT_ROWS)
