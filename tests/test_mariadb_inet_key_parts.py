"""MariaDB's INET4 and INET6 are named by reflection and sized in a key (localdata#161).

tests/test_mariadb_inet_key_parts.py — beside tests/test_mysql_key_budget.py,
which holds the other key-part costs.

SQLAlchemy 2.1.6 ships both types but leaves them out of ``ischema_names``, so
reflection read them as ``NullType`` and an index with a ``TEXT`` beside them was
refused as unsizable. The names now go on a copy of that mapping belonging to the
one engine; the dialect class keeps its own mapping, shared with every other
engine in the process.

No container is needed. The ``SHOW CREATE TABLE`` text below was read off MariaDB
11.8.9, and SQLAlchemy's own parser reads it here as it does on a live attach. The
expected numbers come from that server, by the largest BLOB prefix InnoDB accepted
(a BLOB prefix is counted in bytes, so 3072 minus it is what the column cost):
INET4 took 3068 and answered 1071 at 3069, INET6 took 3056 and answered 1071 at
3057, NULL and NOT NULL alike. Beside a ``VARCHAR(600)`` a TEXT prefix of 167
characters fitted an INET4 and 168 did not; 164 fitted an INET6 and 165 did not.
"""

from __future__ import annotations

import warnings

import pytest
from sqlalchemy import Column, MetaData, String, Table, Text, create_engine
from sqlalchemy.dialects.mysql.base import MySQLDialect
from sqlalchemy.dialects.mysql.mariadb import INET4, INET6
from sqlalchemy.exc import SAWarning
from sqlalchemy.types import NullType

from localdata_mcp.dialects import (
    UnsupportedOperation,
    _mysql_key_part_bytes,
    backend_for,
)

_URLS = {
    "mysql": "mysql+pymysql://u:p@127.0.0.1:1/db",
    "mariadb": "mariadb+pymysql://u:p@127.0.0.1:1/db",
}

_SHOW_CREATE = """CREATE TABLE `sf` (
  `id` int(11) DEFAULT NULL,
  `a` inet4 DEFAULT NULL,
  `b` inet6 DEFAULT NULL,
  `g` geometry DEFAULT NULL,
  `note` text DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci"""


def _table(**types) -> Table:
    columns = [Column(name, kind) for name, kind in types.items()]
    return Table("wide", MetaData(), *columns)


def _opened(backend: str):
    return backend_for(backend).open(_URLS[backend], writable=True)


def _reflected_types(dialect) -> dict:
    """What SQLAlchemy's own parser makes of the live server's table text."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SAWarning)
        state = dialect._tabledef_parser.parse(_SHOW_CREATE, "utf8mb4")
    assert [str(w.message) for w in caught] == [
        "Did not recognize type 'geometry' of column 'g'"
    ]
    return {column["name"]: column["type"] for column in state.columns}


@pytest.mark.parametrize(("kind", "expected"), [(INET4(), 4), (INET6(), 16)])
def test_an_inet_key_part_costs_what_the_engine_measured(kind, expected):
    assert _mysql_key_part_bytes(kind) == expected


@pytest.mark.parametrize("backend", ["mysql", "mariadb"])
def test_both_engines_of_an_attach_can_name_the_inet_types(backend):
    engines = _opened(backend)

    for engine in (engines.read, engines.write):
        assert engine.dialect.ischema_names == {
            **MySQLDialect.ischema_names,
            "inet4": INET4,
            "inet6": INET6,
        }


@pytest.mark.parametrize("backend", ["mysql", "mariadb"])
def test_the_dialect_class_mapping_is_left_alone(backend):
    before = dict(MySQLDialect.ischema_names)

    engines = _opened(backend)

    assert MySQLDialect.ischema_names == before
    assert "inet4" not in MySQLDialect.ischema_names
    assert "inet6" not in MySQLDialect.ischema_names
    assert engines.write.dialect.ischema_names is not MySQLDialect.ischema_names


@pytest.mark.parametrize("backend", ["mysql", "mariadb"])
def test_an_engine_this_server_did_not_open_is_not_changed(backend):
    _opened(backend)

    untouched = create_engine(_URLS[backend])

    assert "inet4" not in untouched.dialect.ischema_names
    assert "inet6" not in untouched.dialect.ischema_names


@pytest.mark.parametrize("backend", ["mysql", "mariadb"])
def test_reflection_names_inet_and_still_cannot_name_geometry(backend):
    types = _reflected_types(_opened(backend).write.dialect)

    assert isinstance(types["a"], INET4)
    assert isinstance(types["b"], INET6)
    assert isinstance(types["g"], NullType)


@pytest.mark.parametrize("backend", ["mysql", "mariadb"])
@pytest.mark.parametrize("kind", [INET4(), INET6()], ids=["inet4", "inet6"])
def test_an_inet_column_beside_a_text_column_leaves_the_text_its_prefix(backend, kind):
    table = _table(address=kind, some_text=Text())

    index, _ = backend_for(backend).build_index("ix", table, ["address", "some_text"])

    assert index.kwargs[f"{backend}_length"] == {"some_text": 255}


@pytest.mark.parametrize(
    ("kind", "expected"), [(INET4(), 167), (INET6(), 164)], ids=["inet4", "inet6"]
)
def test_an_inet_column_spends_the_key_budget(kind, expected):
    table = _table(address=kind, v=String(600), some_text=Text())

    index, _ = backend_for("mariadb").build_index(
        "ix", table, ["address", "v", "some_text"]
    )

    assert index.kwargs["mariadb_length"] == {"some_text": expected}


@pytest.mark.parametrize("backend", ["mysql", "mariadb"])
def test_a_true_nulltype_beside_a_text_column_stays_refused(backend):
    table = _table(payload=NullType(), some_text=Text())

    with pytest.raises(UnsupportedOperation) as refused:
        backend_for(backend).build_index("ix", table, ["payload", "some_text"])

    assert "payload" in str(refused.value)


@pytest.mark.parametrize("backend", ["mysql", "mariadb"])
def test_a_reflected_geometry_beside_a_text_column_stays_refused(backend):
    types = _reflected_types(_opened(backend).write.dialect)
    table = _table(g=types["g"], some_text=Text())

    with pytest.raises(UnsupportedOperation) as refused:
        backend_for(backend).build_index("ix", table, ["g", "some_text"])

    assert "g" in str(refused.value)
