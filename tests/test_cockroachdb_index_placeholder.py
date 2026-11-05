"""CockroachDB's placeholder for an index it is still building is not an index (#148).

While CockroachDB builds an index, its declarative schema changer keeps a
temporary index of its own beside it, over the same columns. Measured on
v26.3.4 with a build held up by another session's open transaction
(docs/CONSTRAINTS.md §15, amendment of 2026-11-05), SQLAlchemy's reflection
reported both:

    {'name': 'crdb_internal_index_3_name_placeholder', 'column_names': ['salary'], ...}
    {'name': 'ix_<table>_salary', 'column_names': ['salary'], ...}

So ``directory`` listed an index nobody made and nobody can drop, and
``create(type='index')`` sent again refused with "already indexed on salary, by
crdb_internal_index_3_name_placeholder", naming the placeholder instead of the
index the caller asked for. The placeholder goes when the build completes.

Offline, the backend's answer and the workspace leaving out what the backend
names as its own. Live, another session's build behind an open reader, and
``directory`` read while it runs. The live half needs the CockroachDB
container and skips without it.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from endpoints import TARGETS, Unavailable, applied, reach
from fastmcp import Client
from sqlalchemy import create_engine, text

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config
from localdata_mcp.dialects import Backend, CockroachDBBackend, PostgreSQLBackend
from localdata_mcp.loader import Workspace

# ---------------------------------------------------------------------------
# Offline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "crdb_internal_index_3_name_placeholder",
        "crdb_internal_index_12_name_placeholder",
    ],
)
def test_cockroachdb_names_its_build_placeholder_as_its_own(name):
    assert CockroachDBBackend().internal_index(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "ix_people_salary",
        "people_pkey",
        # Only the exact shape CockroachDB gives its placeholder: an index a
        # caller named this way is theirs, and is listed.
        "crdb_internal_index_x_name_placeholder",
        "my_crdb_internal_index_3_name_placeholder",
    ],
)
def test_cockroachdb_lists_every_other_index(name):
    assert CockroachDBBackend().internal_index(name) is False


@pytest.mark.parametrize(
    "backend", [Backend(), PostgreSQLBackend()], ids=lambda b: b.name
)
def test_no_other_backend_hides_an_index(backend):
    assert backend.internal_index("crdb_internal_index_3_name_placeholder") is False


@pytest.fixture()
def workspace(tmp_path):
    config_module.use(Config(roots=(tmp_path,)))
    ws = Workspace.in_memory()
    ws.attach_memory("main")
    with ws.entry("main").engines.write.begin() as conn:
        conn.execute(text("CREATE TABLE people (name TEXT, salary INTEGER)"))
        conn.execute(text("CREATE INDEX ix_people_name ON people (name)"))
        conn.execute(text("CREATE INDEX ix_people_salary ON people (salary)"))
    yield ws
    ws.close()


def test_the_workspace_leaves_out_an_index_its_backend_calls_internal(
    workspace, monkeypatch
):
    backend = type(workspace.entry("main").backend)
    monkeypatch.setattr(
        backend, "internal_index", lambda _self, name: name == "ix_people_name"
    )

    names = {index.name for index in workspace.indexes("main")}
    on_table = {index.name for index in workspace.indexes("main", "people")}

    assert names == {"ix_people_salary"}
    assert on_table == {"ix_people_salary"}


# ---------------------------------------------------------------------------
# Live: another session builds an index while a third holds a read open
# ---------------------------------------------------------------------------


_COCKROACH = [target for target in TARGETS if target.name == "cockroachdb"]


def _call(tool: str, **arguments):
    async def _run():
        async with Client(server_module.mcp) as client:
            result = await client.call_tool(tool, arguments)
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(result.content[0].text)

    return asyncio.run(_run())


@contextmanager
def _server(target, tmp_path) -> Iterator[tuple[str, object]]:
    try:
        reached = reach(target)
    except Unavailable as exc:
        pytest.skip(str(exc))
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    config_module.use(Config(roots=(root,), network_enabled=True))
    server_module._reset()
    admin = create_engine(reached.url)
    try:
        with applied(reached.environ):
            attached = _call(
                "attach", database=reached.url, nickname="endpoint", writable=True
            )
            assert attached["ok"] is True, attached
            yield reached.url, admin
    finally:
        server_module._reset()
        admin.dispose()


@pytest.mark.endpoint
@pytest.mark.parametrize("target", _COCKROACH, ids=lambda t: t.name)
def test_directory_lists_an_index_under_construction_without_its_placeholder(
    target, tmp_path
):
    with _server(target, tmp_path) as (url, admin):
        table = f"t{uuid.uuid4().hex[:8]}"
        index = f"ix_{table}_salary"
        with admin.begin() as conn:
            conn.execute(text(f"CREATE TABLE {table} (id INT PRIMARY KEY, salary INT)"))
            conn.execute(
                text(
                    f"INSERT INTO {table} "
                    f"SELECT g, g % 97 FROM generate_series(1, 1000) g"
                )
            )
        reader = create_engine(url)
        held = reader.connect()
        builder = create_engine(url, isolation_level="AUTOCOMMIT")
        try:
            held.execute(text(f"SELECT count(*) FROM {table}"))
            # The build waits on the open read; its own timeout ends the
            # statement and leaves the build running, as measured.
            with builder.connect() as conn:
                conn.execute(text("SET statement_timeout = '3s'"))
                with pytest.raises(Exception, match="statement timeout"):
                    conn.execute(text(f"CREATE INDEX {index} ON {table} (salary)"))

            listed = _call("directory", nickname="endpoint", table=table)
            assert listed["ok"] is True, listed
            text_of = json.dumps(listed)
            assert index in text_of, listed
            assert "crdb_internal_index" not in text_of, listed

            again = _call(
                "create",
                nickname="endpoint",
                type="index",
                table=table,
                columns=["salary"],
            )
            assert again["ok"] is False, again
            assert f"by {index}." in again["error"], again
        finally:
            held.rollback()
            held.close()
            reader.dispose()
            builder.dispose()
            with admin.begin() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
