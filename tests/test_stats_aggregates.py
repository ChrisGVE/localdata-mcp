"""Which backends report a median and a standard deviation, and that they are right (#95).

``stats`` reports ``median`` and ``stddev`` only where
:meth:`localdata_mcp.dialects.Backend.optional_aggregates` declares them, and the
declaration defaults to empty. An entry is added only where the engine was
driven and answered **correctly** — not merely without error. "Correct" is
checked against values worked out by hand from the files below, never read off
a server:

* ``small`` holds ``n`` = 1, 2, 4, 7, 10, 12 and one gap, and ``x`` = ``n`` + 0.5.
  Six values, an **even** count, so a median has to average the two middle ones:
  (4 + 7) / 2 = 5.5, and 6.0 for ``x``. An engine whose median returns one of the
  two middle values (4 or 7) is answering a different question. The mean is 6,
  the squared deviations sum to 25 + 16 + 4 + 1 + 16 + 36 = 98, so the **sample**
  standard deviation is sqrt(98 / 5) = 4.42718872... and the **population** one,
  which is not what this server reports, is sqrt(98 / 6) = 4.04145188...
  The gap is there so that a median or deviation counting a null as a value
  would show.
* ``big`` holds ``v`` = i * i for i in 0..20000: 20001 rows, an odd count, so the
  exact median is the middle value, 10000 * 10000 = 100000000. An engine whose
  median is an estimate — a reservoir sample, a t-digest — lands near it rather
  than on it, which a three-row table can never show. Its sample standard
  deviation is computed here with :func:`statistics.stdev`, which works in exact
  fractions.
* ``typed`` is built below the verbs, through SQLAlchemy's generic types, because
  a loaded file only ever produces ``INTEGER``, ``REAL`` and ``TEXT`` and an
  attached table can hold anything. ``d`` is a ``Numeric(12, 4)`` holding
  ``x``'s six values, so it must answer exactly as ``x`` does; ``k`` is an
  integer key, 1..6, median 3.5 and sample deviation sqrt(17.5 / 5). No gap
  here: ClickHouse's driver would not write a null into the decimal column this
  builds (``decimal.ConversionSyntax``, measured), and nulls are already
  ``small``'s to test. A median that works on a float column and fails on a decimal
  one would fail ``stats`` mid-profile on a real table, which is the failure an
  undeclared statistic exists to avoid.

:data:`MEASURED` is the record of what was driven, one line per backend, and both
tests below hold the code to it: the offline one checks the declarations
without any database, and the live one drives each engine that has a container
answering.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import uuid
from decimal import Decimal

import pytest
from endpoints import ENDPOINTS, Target, Unavailable, applied, reach
from fastmcp import Client
from sqlalchemy import Column, Integer, MetaData, Numeric, Table, create_engine
from test_endpoints import _drop_everything_named

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config
from localdata_mcp.dialects import BACKENDS, backend_for_url

#: What each backend declares, by the name it calls itself. Every entry was
#: driven against a live engine on 2026-11-03 (docs/CONSTRAINTS.md §37), and the
#: reason a statistic is absent is recorded there.
MEASURED: dict[str, frozenset[str]] = {
    "sqlite": frozenset(),
    "duckdb": frozenset({"median", "stddev"}),
    "postgresql": frozenset({"median", "stddev"}),
    "mysql": frozenset({"stddev"}),
    "mariadb": frozenset({"stddev"}),
    "firebird": frozenset({"stddev"}),
    "cockroachdb": frozenset({"median", "stddev"}),
    "monetdb": frozenset({"median", "stddev"}),
    "yugabytedb": frozenset({"median", "stddev"}),
    "opengauss": frozenset({"median", "stddev"}),
    "cratedb": frozenset({"stddev"}),
    "trino": frozenset({"stddev"}),
    "clickhouse": frozenset({"median", "stddev"}),
    "mssql": frozenset({"stddev"}),
    "ydb": frozenset({"stddev"}),
    "oracle": frozenset({"median", "stddev"}),
    "exasol": frozenset({"median", "stddev"}),
}

SMALL = "n,x\n1,1.5\n2,2.5\n4,4.5\n7,7.5\n10,10.5\n12,12.5\n,\n"
SMALL_SAMPLE_STDDEV = (98 / 5) ** 0.5
BIG_VALUES = [i * i for i in range(20001)]
BIG = "v\n" + "".join(f"{value}\n" for value in BIG_VALUES)
TYPED_ROWS = [
    {"k": k, "d": Decimal(d)}
    for k, d in enumerate(["1.5", "2.5", "4.5", "7.5", "10.5", "12.5"], start=1)
]
TYPED_K_SAMPLE_STDDEV = (17.5 / 5) ** 0.5


def build_typed_table(url: str, name: str) -> None:
    """The ``typed`` table, made the way somebody else's would be.

    Every dialect fact is asked of the backend, as ``_build_typed_table`` in
    :mod:`test_endpoints` does: whether a key is required (it goes on ``k``,
    which is unique), what ``CREATE TABLE`` needs, whether the rows may travel
    with the schema, and how a written row becomes readable.
    """
    backend = backend_for_url(url)
    metadata = MetaData()
    key = Column("k", Integer, primary_key=backend.requires_primary_key())
    defined = Table(
        name, metadata, key, Column("d", Numeric(12, 4)), **backend.table_options()
    )

    def fill(conn) -> None:
        conn.execute(defined.insert(), TYPED_ROWS)
        backend.settle(conn, name)

    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            metadata.create_all(conn)
            if backend.sees_new_tables_in_transaction():
                fill(conn)
        if not backend.sees_new_tables_in_transaction():
            with engine.begin() as conn:
                fill(conn)
    finally:
        engine.dispose()


def test_every_backend_declares_exactly_what_was_measured():
    """The declaration matches the record, for every registered backend.

    A backend missing from :data:`MEASURED` fails here rather than passing on the
    empty default, so a new backend cannot arrive with its coverage unstated;
    and the record names no backend that is not registered, so the loop below
    cannot pass by covering fewer backends than the record claims.
    """
    assert {backend.name for backend in BACKENDS.values()} == set(MEASURED)
    for backend in BACKENDS.values():
        assert backend.name in MEASURED, f"{backend.name} has no measured entry"
        declared = frozenset(backend.optional_aggregates())
        assert declared == MEASURED[backend.name], backend.name


def call(tool: str, **arguments):
    """One tool call over the protocol, decoded. As in :mod:`test_endpoints`."""

    async def _run():
        async with Client(server_module.mcp) as client:
            result = await client.call_tool(tool, arguments)
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(result.content[0].text)

    return asyncio.run(_run())


@pytest.fixture(
    params=[Target(endpoint) for endpoint in ENDPOINTS],
    ids=lambda target: target.name,
)
def landed(request, tmp_path):
    """The three tables in one live endpoint; dropped after.

    ``small`` and ``big`` arrive through ``create``, as a user's file would;
    ``typed`` is built directly, as a table the server merely reaches would be.

    The credentialed target of each endpoint only: an authentication mode
    reaches the same engine, and which aggregates it has is a fact about the
    engine.
    """
    try:
        reached = reach(request.param)
    except Unavailable as exc:
        pytest.skip(str(exc))

    root = (tmp_path / "root").resolve()
    root.mkdir()
    (root / "small.csv").write_text(SMALL)
    (root / "big.csv").write_text(BIG)
    config_module.use(Config(roots=(root,), network_enabled=True))
    server_module._reset()
    prefix = f"t{uuid.uuid4().hex[:8]}"
    with applied(reached.environ):
        try:
            attached = call("attach", database=reached.url, nickname="e", writable=True)
            assert attached["ok"] is True, attached
            engine = backend_for_url(reached.url).name
            # `big` exists to catch an estimated median, so it is landed only
            # where a median is expected: Trino had still not taken in the
            # 20001 rows after twenty minutes, measured, and they buy nothing
            # where there is no median to check.
            stems = ("small", "big") if "median" in MEASURED[engine] else ("small",)
            tables = {}
            for stem in stems:
                made = call(
                    "create",
                    nickname="e",
                    type="table",
                    source=str(root / f"{stem}.csv"),
                    table=f"{prefix}_{stem}",
                    delimiter=",",
                )
                assert made["ok"] is True, made
                tables[stem] = f"{prefix}_{stem}"
            tables["typed"] = f"{prefix}_typed"
            build_typed_table(reached.url, tables["typed"])
            yield engine, tables
        finally:
            server_module._reset()
            _drop_everything_named(reached.url, prefix)


def _profiled(table: str) -> dict[str, dict]:
    answer = call("stats", nickname="e", table=table)
    assert answer["ok"] is True, answer
    return {column["name"].lower(): column for column in answer["columns"]}


@pytest.mark.endpoint
def test_the_engine_reports_exactly_the_measured_statistics_and_they_are_right(
    landed,
):
    """``stats`` against a live engine: the declared statistics, and their values.

    The set of statistics reported is asserted as a whole first, so a backend
    that reports nothing cannot pass the value checks by having none to check.
    """
    engine, tables = landed
    expected = MEASURED[engine]
    small = _profiled(tables["small"])
    typed = _profiled(tables["typed"])
    big = _profiled(tables["big"]) if "big" in tables else None

    floor = {"name", "type", "nulls", "non_nulls", "min", "max", "avg"}
    profiled = {"n": small["n"], "x": small["x"], "k": typed["k"], "d": typed["d"]}
    if big is not None:
        profiled["v"] = big["v"]
    for name, column in profiled.items():
        assert set(column) - floor == expected, (engine, name, column)

    assert small["n"]["nulls"] == 1 and small["n"]["non_nulls"] == 6
    if "median" in expected:
        assert small["n"]["median"] == pytest.approx(5.5)
        assert small["x"]["median"] == pytest.approx(6.0)
        assert big["v"]["median"] == pytest.approx(100000000)
        assert typed["k"]["median"] == pytest.approx(3.5)
        assert typed["d"]["median"] == pytest.approx(6.0)
    if "stddev" in expected:
        assert small["n"]["stddev"] == pytest.approx(SMALL_SAMPLE_STDDEV)
        assert small["x"]["stddev"] == pytest.approx(SMALL_SAMPLE_STDDEV)
        if big is not None:
            assert big["v"]["stddev"] == pytest.approx(statistics.stdev(BIG_VALUES))
        assert typed["k"]["stddev"] == pytest.approx(TYPED_K_SAMPLE_STDDEV)
        assert typed["d"]["stddev"] == pytest.approx(SMALL_SAMPLE_STDDEV)
