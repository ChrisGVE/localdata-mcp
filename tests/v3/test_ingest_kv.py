"""tests/v3/test_ingest_kv.py — E8.3: the kv tool family.

FR-103's L3 set/get leg over the real stack (Chokepoint.boot on
family-prefixed store endpoints, tools crossing the installed
runtime): property CRUD against a kv store AND a graph store through
the one dispatch, `main`'s string-type inference intact, every
mutation refused below read-write posture (NFR-113 — enforced by the
guard the tools cross, not by tool code), kind mismatches and misses
as structured refusals carrying discovery guidance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

import localdata_mcp.ingest.runtime as runtime
from localdata_mcp.ingest.connectors.kv.tools import (
    delete_key,
    get_value,
    list_keys,
    set_value,
)
from localdata_mcp.nexus.chokepoint.guard import (
    Chokepoint,
    GuardedExecutionError,
    GuardRefusedError,
)
from localdata_mcp.nexus.config.endpoints import EndpointDeclaration
from localdata_mcp.nexus.config.models import ConfigModel


def _config(tmp_path: Path) -> ConfigModel:
    kv_file = tmp_path / "kv.db"
    return ConfigModel(
        endpoints={
            "notes": EndpointDeclaration(
                name="notes",
                dsn=f"kv+sqlite:///{kv_file}",
                posture="read_write",
            ),
            "notes_ro": EndpointDeclaration(
                name="notes_ro",
                dsn=f"kv+sqlite:///{kv_file}",
                posture="read_only",
            ),
            "social": EndpointDeclaration(
                name="social",
                dsn=f"graph+sqlite:///{tmp_path / 'g.db'}",
                posture="read_write",
            ),
            "warehouse": EndpointDeclaration(
                name="warehouse",
                dsn=f"sqlite:///{tmp_path / 'w.db'}",
                posture="read_write",
            ),
        }
    )


@pytest.fixture()
def booted(tmp_path: Path) -> Iterator[Chokepoint]:
    guard = Chokepoint.boot(_config(tmp_path), environ={})
    runtime.configure_ingest(guard)
    yield guard
    runtime._CHOKEPOINT = None
    guard.shutdown()


class TestKvOverTreeStore:
    def test_set_then_get_round_trips(self, booted: Chokepoint) -> None:
        set_value("notes", "config.server", "host", "db.internal")
        found = get_value("notes", "config.server", "host")
        assert found["value"] == "db.internal"
        assert found["value_type"] == "string"

    def test_string_type_inference_carries_main_semantics(
        self, booted: Chokepoint
    ) -> None:
        set_value("notes", "config", "port", "5432")
        assert get_value("notes", "config", "port")["value"] == 5432
        set_value("notes", "config", "debug", "true")
        assert get_value("notes", "config", "debug")["value"] is True

    def test_explicit_value_type_overrides_inference(self, booted: Chokepoint) -> None:
        set_value("notes", "config", "version", "42", value_type="string")
        assert get_value("notes", "config", "version")["value"] == "42"

    def test_set_value_auto_creates_ancestors(self, booted: Chokepoint) -> None:
        set_value("notes", "a.b.c", "k", "v")
        listed = list_keys("notes", "a.b.c")
        assert listed.rows == (("k", "v", "string"),)

    def test_upsert_overwrites(self, booted: Chokepoint) -> None:
        set_value("notes", "n", "k", "first")
        set_value("notes", "n", "k", "second")
        assert get_value("notes", "n", "k")["value"] == "second"

    def test_delete_key_removes_and_refuses_a_second_time(
        self, booted: Chokepoint
    ) -> None:
        set_value("notes", "n2", "gone", "x")
        outcome = delete_key("notes", "n2", "gone")
        assert outcome["deleted"] is True
        with pytest.raises(GuardedExecutionError) as refusal:
            delete_key("notes", "n2", "gone")
        assert "not found" in refusal.value.structured.message

    def test_list_keys_paginates_key_ordered(self, booted: Chokepoint) -> None:
        for key in ("beta", "alpha", "gamma"):
            set_value("notes", "page", key, key)
        first_two = list_keys("notes", "page", offset=0, limit=2)
        assert [row[0] for row in first_two.rows] == ["alpha", "beta"]
        rest = list_keys("notes", "page", offset=2, limit=2)
        assert [row[0] for row in rest.rows] == ["gamma"]

    def test_get_value_miss_names_the_discovery_path(self, booted: Chokepoint) -> None:
        set_value("notes", "exists", "k", "v")
        with pytest.raises(GuardedExecutionError) as refusal:
            get_value("notes", "exists", "absent")
        assert "list_keys" in refusal.value.structured.suggestion


class TestKvOverGraphStore:
    def test_property_crud_speaks_graph_tables(self, booted: Chokepoint) -> None:
        set_value("social", "alice", "age", "30")
        assert get_value("social", "alice", "age")["value"] == 30
        listed = list_keys("social", "alice")
        assert listed.rows == (("age", 30, "integer"),)
        assert delete_key("social", "alice", "age")["deleted"] is True

    def test_list_keys_on_missing_graph_node_refused(self, booted: Chokepoint) -> None:
        with pytest.raises(GuardedExecutionError) as refusal:
            list_keys("social", "nobody")
        assert "Node not found" in refusal.value.structured.message


class TestKvRefusals:
    def test_mutation_refused_on_read_only_posture(self, booted: Chokepoint) -> None:
        with pytest.raises(GuardRefusedError, match="read_only"):
            set_value("notes_ro", "n", "k", "v")

    def test_sql_endpoint_kind_mismatch_refused_with_guidance(
        self, booted: Chokepoint
    ) -> None:
        with pytest.raises(GuardedExecutionError) as refusal:
            get_value("warehouse", "n", "k")
        structured = refusal.value.structured
        assert "backend_kind='sqlite'" in structured.message
        assert "list_endpoints()" in structured.suggestion

    def test_undeclared_endpoint_carries_nfr114_guidance(
        self, booted: Chokepoint
    ) -> None:
        with pytest.raises(GuardedExecutionError) as refusal:
            get_value("never-declared", "n", "k")
        assert "list_endpoints()" in refusal.value.structured.suggestion
        assert "NFR-114" in refusal.value.structured.message

    def test_read_only_endpoint_still_reads(self, booted: Chokepoint) -> None:
        set_value("notes", "shared", "k", "v")
        assert get_value("notes_ro", "shared", "k")["value"] == "v"
