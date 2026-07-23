"""testbench/batteries/pipeline/pipeline_battery_test.py — NFR-502d length-2 (E11.4).

The exhaustive length-2 pipeline battery: every ordered domain pair
(the enumeration-derived 72 at the launch nine) executed through the
real compose_pipeline MCP seam — legal pairs run without an
ENGINE-level failure (a domain-level 'meaningless but correct' stage
failure is expected and accepted, FR-302), adjacency-illegal pairs
assert the FR-606 pre-execution structured rejection. The executed
count is asserted against the closed-form 72 and every per-pair outcome
stored (NFR-508). The length-3/4 totals are DERIVED from the length-2
link facts alone and asserted here (their exhaustive execution is the
nightly job, S7.4) — no third function between the verified 72 and the
longer-chain totals.
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
from localdata_mcp.testbench.batteries.pipeline.enumeration import (
    DomainLink,
    alternating_valid_total,
    length2_dag_spec,
    length2_total,
    ordered_links,
)
from localdata_mcp.testbench.results_store import schema, store

# The engine-level rejection discriminator: a pre-execution FR-606
# refusal never carries the per-stage failure prefix, so its absence
# marks a domain-level (acceptable) failure.
_STAGE_FAILURE_PREFIX = "pipeline stage "


@pytest.fixture()
def bench(tmp_path: Path) -> Iterator[Path]:
    config = ConfigModel(security=SecurityConfig(allowed_paths=(str(tmp_path),)))
    guard = Chokepoint.boot(config, environ=dict(os.environ))
    configure_shaping(config, default_registry())
    runtime.configure_ingest(guard)
    yield tmp_path
    runtime._CHOKEPOINT = None
    configure_shaping(ConfigModel(), default_registry())
    guard.shutdown()


def _call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
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


def _fixture(tmp_path: Path) -> str:
    """A generic tabular fixture — enough columns that a source stage
    can begin; whether a domain tool finds the exact columns it needs
    is beside the point (a domain-level failure is acceptable here)."""
    target = tmp_path / "generic.csv"
    pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "other": [2.0, 4.0, 5.0, 4.0, 6.0, 7.0],
            "group": ["a", "b", "a", "b", "a", "b"],
        }
    ).to_csv(target, index=False)
    return str(target)


def _outcome(link: DomainLink, path: str) -> "tuple[bool, str]":
    """(engine_rejected, message) for one ordered pair through the real
    tool. engine_rejected is True iff the error is a pre-execution
    FR-606 rejection (no per-stage failure prefix)."""
    envelope = _call("compose_pipeline", {"dag_spec": length2_dag_spec(link, path)})
    error = envelope["error"]
    if error is None:
        return False, "ran (no engine rejection)"
    message = error["message"]
    engine_rejected = not message.startswith(_STAGE_FAILURE_PREFIX)
    return engine_rejected, message


def _store_outcomes(tmp_path: Path, rows: "list[tuple[DomainLink, bool]]") -> int:
    """Persist every per-pair outcome (NFR-508); return the count
    written, the mechanical completeness signal."""
    db_path = tmp_path / "pipeline_results.db"
    connection = schema.connect(db_path, busy_timeout_ms=4 + 4)
    schema.ensure_schema(connection)
    run = store.BatteryRun(
        run_id="pipeline-length2",
        battery_name="pipeline",
        run_mode="deterministic",
        seed=None,
        dataset_hash="generic-fixture",
        software_versions={"battery": "length2"},
        started_at="1970-01-01T00:00:00Z",
        finished_at="1970-01-01T00:00:00Z",
        git_sha=None,
    )
    results = [
        store.BatteryResult(
            run_id="pipeline-length2",
            test_id=f"{link.source_domain}->{link.target_domain}",
            passed=True,
            numeric_output={"legal": link.legal, "engine_rejected": rejected},
        )
        for link, rejected in rows
    ]
    with store.write_transaction(connection, max_attempts=4) as write:
        store.write_run(write, run)
        store.write_results(write, results)
    written = len(store.read_results(connection, "pipeline-length2"))
    connection.close()
    return written


class TestLength2Exhaustive:
    def test_every_ordered_pair_classified_and_counted(self, bench: Path) -> None:
        path = _fixture(bench)
        links = ordered_links(default_registry())
        assert len(links) == length2_total() == 72

        outcomes: list[tuple[DomainLink, bool]] = []
        for link in links:
            engine_rejected, message = _outcome(link, path)
            if link.legal:
                assert not engine_rejected, (
                    f"legal pair {link.source_domain}->{link.target_domain} was "
                    f"engine-rejected: {message}"
                )
            else:
                assert engine_rejected, (
                    f"illegal pair {link.source_domain}->{link.target_domain} was "
                    f"NOT engine-rejected: {message}"
                )
            outcomes.append((link, engine_rejected))

        written = _store_outcomes(bench, outcomes)
        assert written == length2_total() == 72

    def test_both_legal_and_illegal_pairs_are_present(self) -> None:
        """The battery is meaningful only if it exercises both arms of
        the FR-606 classification."""
        links = ordered_links(default_registry())
        legal = [link for link in links if link.legal]
        illegal = [link for link in links if not link.legal]
        assert legal, "no legal pairs — the composable-output domains vanished"
        assert illegal, "no illegal pairs — the adjacency table went permissive"


class TestLongerChainDerivation:
    def test_length3_and_length4_totals_derive_from_link_facts(self) -> None:
        """The alternating totals come from the length-2 link facts
        alone (NFR-502d's independence rule): A-B-A valid iff A→B and
        B→A are both valid length-2 links; A-B-A-B needs the same two."""
        links = ordered_links(default_registry())
        index = {(link.source_domain, link.target_domain): link.legal for link in links}
        expected = sum(
            1
            for (source, target) in index
            if index[(source, target)] and index[(target, source)]
        )
        assert alternating_valid_total(links, 3) == expected
        assert alternating_valid_total(links, 4) == expected
        # A length-3/4 alternation needs a BIDIRECTIONAL link pair. The
        # launch representatives with composable output (pattern ->
        # TABULAR, sampling -> TABULAR, time_series -> VECTOR) all
        # consume TABULAR, so those three domains pair bidirectionally:
        # C(3,2)=3 unordered pairs × 2 ordered A-B-A instances = 6.
        # Derived from the length-2 facts, asserted not assumed.
        assert alternating_valid_total(links, 3) == 6

    def test_invalid_length_is_refused(self) -> None:
        links = ordered_links(default_registry())
        with pytest.raises(ValueError):
            alternating_valid_total(links, 5)
