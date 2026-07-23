"""testbench/batteries/base/ingest_battery_test.py — E8.5's base slice.

The NFR-502a base-capability battery's E8 accretion: connect/retrieve
per backend and per file format, DETERMINISTIC order (declaration
order of the scenario tables below; fixed fixture content), at the L3
`fastmcp.Client` seam (PROJECT-FP #4 — the wire the agent actually
uses, through the generated wrapper, shaping, and the guard).

Coverage in this slice — and what is deliberately NOT here (no silent
caps): the four file-based/embedded engines' file halves (SQLite,
DuckDB) plus the E8.3 store kinds (kv, tree, graph, rdf) run
unconditionally; PostgreSQL and MySQL rows run whenever the E14.2
fixture script's DSN environment variables are present and SKIP with
a named reason otherwise (the dockerized halves assemble in E14.2 —
FR-102's dockerized acceptance lands there). The format table covers
the 14 core formats through writable fixtures; legacy `.xls` shares
the Excel family and needs an authored binary fixture (no maintained
writer exists) — it lands with E14.1's collect-and-build script.
E14.3 adds the `--randomize --seed 42` full-assertion mode; this
module IS the deterministic half it re-runs.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Callable, Iterator

import anyio
import pandas as pd
import pytest
from fastmcp import Client

import localdata_mcp.ingest.runtime as runtime
from localdata_mcp.nexus.chokepoint.guard import Chokepoint
from localdata_mcp.nexus.config.endpoints import EndpointDeclaration
from localdata_mcp.nexus.config.models import ConfigModel, SecurityConfig
from localdata_mcp.nexus.contract.registry import default_registry
from localdata_mcp.nexus.response.shaping import configure_shaping
from localdata_mcp.server.mcp_app import app

_FRAME = pd.DataFrame({"id": [1, 2], "label": ["a", "b"]})

# Optional dockerized rows (E14.2 provisions these; absent → skip).
_PG_DSN = os.environ.get("LOCALDATA_TEST_POSTGRES_DSN")
_MYSQL_DSN = os.environ.get("LOCALDATA_TEST_MYSQL_DSN")


def _declarations(tmp_path: Path) -> dict[str, EndpointDeclaration]:
    ttl = tmp_path / "kb.ttl"
    ttl.write_text(
        "@prefix ex: <http://example.org/> .\n"
        'ex:alpha ex:label "a" .\n'
        'ex:beta ex:label "b" .\n'
    )
    declared = {
        "sql_sqlite": f"sqlite:///{tmp_path / 'base.db'}",
        "sql_duckdb": f"duckdb:///{tmp_path / 'base.duckdb'}",
        "store_kv": f"kv+sqlite:///{tmp_path / 'kv.db'}",
        "store_tree": f"tree+sqlite:///{tmp_path / 'tree.db'}",
        "store_graph": f"graph+sqlite:///{tmp_path / 'graph.db'}",
        "store_rdf": f"rdf+turtle:///{ttl}",
    }
    if _PG_DSN:
        declared["sql_postgresql"] = _PG_DSN
    if _MYSQL_DSN:
        declared["sql_mysql"] = _MYSQL_DSN
    return {
        name: EndpointDeclaration(name=name, dsn=dsn, posture="read_write")
        for name, dsn in declared.items()
    }


def _seed_sql(guard: Chokepoint, endpoint: str) -> None:
    """Two fixed rows through the NX-5 seam (DDL is outside every
    guarded category by design — seeding is bench scaffolding)."""
    with guard._persistence.connection(endpoint) as connection:
        if hasattr(connection, "exec_driver_sql"):
            connection.exec_driver_sql("CREATE TABLE base_t (id INTEGER, label TEXT)")
            connection.exec_driver_sql("INSERT INTO base_t VALUES (1, 'a'), (2, 'b')")
            connection.commit()
        else:  # native DuckDB connection
            connection.execute("CREATE TABLE base_t (id INTEGER, label TEXT)")
            connection.execute("INSERT INTO base_t VALUES (1, 'a'), (2, 'b')")


@pytest.fixture()
def bench(tmp_path: Path) -> Iterator[Chokepoint]:
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


def _call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """One L3 call; returns the FR-403 envelope."""

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


def _retrieve_sql(endpoint: str) -> None:
    data = _data(
        _call(
            "query",
            {"endpoint": endpoint, "sql": "SELECT id, label FROM base_t ORDER BY id"},
        )
    )
    assert data["rows"] == [[1, "a"], [2, "b"]]


def _retrieve_kv(endpoint: str) -> None:
    _data(
        _call(
            "set_value",
            {"endpoint": endpoint, "path": "root.node", "key": "k", "value": "42"},
        )
    )
    data = _data(
        _call("get_value", {"endpoint": endpoint, "path": "root.node", "key": "k"})
    )
    assert data["value"] == 42


def _retrieve_graph(endpoint: str) -> None:
    _data(
        _call(
            "add_edge",
            {"endpoint": endpoint, "source": "a", "target": "b", "label": "knows"},
        )
    )
    data = _data(_call("get_graph_stats", {"endpoint": endpoint}))
    assert data["node_count"] == 2
    assert data["edge_count"] == 1


def _retrieve_rdf(endpoint: str) -> None:
    data = _data(
        _call(
            "query",
            {
                "endpoint": endpoint,
                "sql": (
                    "SELECT ?s ?o WHERE { ?s <http://example.org/label> ?o } "
                    "ORDER BY ?o"
                ),
            },
        )
    )
    assert [row[1] for row in data["rows"]] == ["a", "b"]


# Deterministic backend order: embedded SQL engines, store kinds, then
# the env-gated dockerized engines — declaration order IS the run order.
_BACKEND_ROWS: tuple[tuple[str, Callable[[str], None]], ...] = (
    ("sql_sqlite", _retrieve_sql),
    ("sql_duckdb", _retrieve_sql),
    ("store_kv", _retrieve_kv),
    ("store_tree", _retrieve_kv),
    ("store_graph", _retrieve_graph),
    ("store_rdf", _retrieve_rdf),
    ("sql_postgresql", _retrieve_sql),
    ("sql_mysql", _retrieve_sql),
)


@pytest.mark.parametrize(
    ("endpoint", "retrieve"),
    _BACKEND_ROWS,
    ids=[name for name, _ in _BACKEND_ROWS],
)
def test_connect_and_retrieve_per_backend(
    bench: Chokepoint, endpoint: str, retrieve: Callable[[str], None]
) -> None:
    if endpoint == "sql_postgresql" and not _PG_DSN:
        pytest.skip("dockerized PostgreSQL fixture absent — provisioned by E14.2")
    if endpoint == "sql_mysql" and not _MYSQL_DSN:
        pytest.skip("dockerized MySQL fixture absent — provisioned by E14.2")
    summaries = {s.name for s in bench.endpoint_summaries()}
    assert endpoint in summaries  # connected: declared, warmed, enumerable
    retrieve(endpoint)


# -- formats ----------------------------------------------------------


def _write_csv(path: Path) -> Path:
    target = path / "d.csv"
    _FRAME.to_csv(target, index=False)
    return target


def _write_tsv(path: Path) -> Path:
    target = path / "d.tsv"
    _FRAME.to_csv(target, sep="\t", index=False)
    return target


def _write_json(path: Path) -> Path:
    target = path / "d.json"
    target.write_text('[{"id": 1, "label": "a"}, {"id": 2, "label": "b"}]')
    return target


def _write_yaml(path: Path) -> Path:
    target = path / "d.yaml"
    target.write_text("- id: 1\n  label: a\n- id: 2\n  label: b\n")
    return target


def _write_toml(path: Path) -> Path:
    target = path / "d.toml"
    target.write_text('[meta]\nname = "base"\n')
    return target


def _write_ini(path: Path) -> Path:
    target = path / "d.ini"
    target.write_text("[meta]\nname = base\n")
    return target


def _write_xml(path: Path) -> Path:
    target = path / "d.xml"
    target.write_text('<root><item id="1">a</item><item id="2">b</item></root>')
    return target


def _write_xlsx(path: Path) -> Path:
    target = path / "d.xlsx"
    _FRAME.to_excel(target, index=False, engine="openpyxl")
    return target


def _write_ods(path: Path) -> Path:
    target = path / "d.ods"
    with pd.ExcelWriter(target, engine="odf") as writer:
        _FRAME.to_excel(writer, index=False)
    return target


def _write_numbers(path: Path) -> Path:
    from numbers_parser import Document

    target = path / "d.numbers"
    document = Document()
    table = document.sheets[0].tables[0]
    table.write(0, 0, "id")
    table.write(0, 1, "label")
    table.write(1, 0, 1)
    table.write(1, 1, "a")
    document.save(str(target))
    return target


def _write_parquet(path: Path) -> Path:
    target = path / "d.parquet"
    _FRAME.to_parquet(target, index=False)
    return target


def _write_feather(path: Path) -> Path:
    import pyarrow
    import pyarrow.feather

    target = path / "d.feather"
    pyarrow.feather.write_feather(pyarrow.Table.from_pandas(_FRAME), target)
    return target


def _write_arrow(path: Path) -> Path:
    import pyarrow
    import pyarrow.ipc

    target = path / "d.arrow"
    table = pyarrow.Table.from_pandas(_FRAME)
    with pyarrow.ipc.new_file(str(target), table.schema) as writer:
        writer.write_table(table)
    return target


def _write_hdf5(path: Path) -> Path:
    import h5py

    target = path / "d.h5"
    with h5py.File(target, "w") as handle:
        handle.create_dataset("values", data=[[1, 2], [3, 4]])
    return target


def _expect_tabular_rows(data: Any) -> None:
    assert data["rows"][0][0] == 1


def _expect_document(data: Any) -> None:
    assert data["meta"]["name"] == "base"


def _expect_xml_tree(data: Any) -> None:
    assert data["root"]["item"][0]["@id"] == "1"


def _expect_hdf5_matrix(data: Any) -> None:
    assert data["rows"][0] == [1, 2]


# The 14 core formats in the section 7.2 declaration order (Excel's
# xlsx variant carries the family here; the authored .xls binary
# fixture lands with E14.1 — stated in the module docstring).
_FORMAT_ROWS: tuple[tuple[str, Callable[[Path], Path], Callable[[Any], None]], ...] = (
    ("csv", _write_csv, _expect_tabular_rows),
    ("tsv", _write_tsv, _expect_tabular_rows),
    ("json", _write_json, _expect_tabular_rows),
    ("yaml", _write_yaml, _expect_tabular_rows),
    ("toml", _write_toml, _expect_document),
    ("ini", _write_ini, _expect_document),
    ("xml", _write_xml, _expect_xml_tree),
    ("excel_xlsx", _write_xlsx, _expect_tabular_rows),
    ("ods", _write_ods, _expect_tabular_rows),
    ("numbers", _write_numbers, _expect_tabular_rows),
    ("parquet", _write_parquet, _expect_tabular_rows),
    ("feather", _write_feather, _expect_tabular_rows),
    ("arrow", _write_arrow, _expect_tabular_rows),
    ("hdf5", _write_hdf5, _expect_hdf5_matrix),
)


@pytest.mark.parametrize(
    ("format_name", "write_fixture", "expect"),
    _FORMAT_ROWS,
    ids=[name for name, _, _ in _FORMAT_ROWS],
)
def test_read_file_per_format(
    bench: Chokepoint,
    tmp_path: Path,
    format_name: str,
    write_fixture: Callable[[Path], Path],
    expect: Callable[[Any], None],
) -> None:
    fixture = write_fixture(tmp_path)
    envelope = _call("read_file", {"path": str(fixture)})
    expect(_data(envelope))


def test_query_file_retrieves_at_the_l3_seam(bench: Chokepoint, tmp_path: Path) -> None:
    """The `query_file` cell of the retrieval matrix, same seam."""
    db = tmp_path / "local.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE t (id INTEGER)")
        connection.executemany("INSERT INTO t VALUES (?)", [(1,), (2,)])
    data = _data(
        _call("query_file", {"path": str(db), "sql": "SELECT id FROM t ORDER BY id"})
    )
    assert data["rows"] == [[1], [2]]
