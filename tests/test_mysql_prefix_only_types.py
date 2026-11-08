"""The rest of the TEXT and BLOB family needs a key prefix too (localdata#161).

tests/test_mysql_prefix_only_types.py — beside tests/test_mysql_key_budget.py,
which fixes what the other columns in a key cost. This file fixes which columns
are given a prefix at all.

SQLAlchemy's MEDIUMTEXT, LONGTEXT, TINYTEXT, TINYBLOB, MEDIUMBLOB and LONGBLOB
subclass neither ``Text`` nor ``LargeBinary``, so they used to be read as columns
that key whole. On MySQL 8.0.46 a lone one answered 1170, and beside a TEXT it
was refused as unsizable.

No container is needed: ``build_index`` is arithmetic over a ``Table``. The
expected numbers are not derived from that arithmetic. They were read off
MySQL 8.0.46 and MariaDB 11.8.9 with utf8mb4 columns, by the largest prefix each
engine accepted:

* ``TINYTEXT``: 63 characters is accepted by both. MySQL answers 1071 at 64, so
  the 255 every other type gets cannot be sent. MariaDB accepts 255, and 63 is
  used there too because a ``mysql+pymysql`` URL does not say which server it
  reaches.
* ``TINYBLOB``: 255 accepted, 256 refused on MySQL.
* The rest: 255 accepted on both. Beside ``id INT, v VARCHAR(600)`` the text
  types accepted 167 and answered 1071 at 168.
* ``VARCHAR(760)`` beside a ``TINYTEXT``: 8 accepted, 9 answered 1071 on both.
"""

from __future__ import annotations

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, Text
from sqlalchemy.dialects import mysql

from localdata_mcp.dialects import UnsupportedOperation, backend_for


def _table(**types) -> Table:
    columns = [Column(name, kind) for name, kind in types.items()]
    return Table("wide", MetaData(), *columns)


_PREFIX_ONLY_TYPES = [
    (mysql.MEDIUMTEXT(), 255),
    (mysql.LONGTEXT(), 255),
    (mysql.TINYTEXT(), 63),
    (mysql.MEDIUMBLOB(), 255),
    (mysql.LONGBLOB(), 255),
    (mysql.TINYBLOB(), 255),
]


@pytest.mark.parametrize("backend", ["mysql", "mariadb"])
@pytest.mark.parametrize(
    ("kind", "expected"), _PREFIX_ONLY_TYPES, ids=lambda value: repr(value)[:20]
)
def test_a_lone_column_of_the_family_takes_the_prefix_the_engine_accepted(
    backend, kind, expected
):
    table = _table(body=kind)

    index, notes = backend_for(backend).build_index("ix", table, ["body"])

    assert index.kwargs[f"{backend}_length"] == {"body": expected}
    assert str(expected) in " ".join(notes)
    assert "body" in " ".join(notes)


@pytest.mark.parametrize(
    "kind", [kind for kind, expected in _PREFIX_ONLY_TYPES if expected == 255]
)
def test_a_family_column_beside_a_text_column_is_prefixed_like_it(kind):
    """It used to be refused as a column nobody can size."""
    table = _table(body=kind, some_text=Text())

    index, _ = backend_for("mysql").build_index("ix", table, ["body", "some_text"])

    assert index.kwargs["mysql_length"] == {"body": 255, "some_text": 255}


@pytest.mark.parametrize("kind", [mysql.MEDIUMTEXT(), mysql.LONGTEXT()])
def test_the_family_shares_the_key_budget_with_the_other_columns(kind):
    """4 + 2400 bytes leaves 668: 167 accepted live, 168 answered 1071."""
    table = _table(id=Integer(), v=String(600), body=kind)

    index, notes = backend_for("mysql").build_index("ix", table, ["id", "v", "body"])

    assert index.kwargs["mysql_length"] == {"body": 167}
    assert "167" in " ".join(notes)


def test_two_family_columns_are_both_prefixed():
    table = _table(a=mysql.LONGTEXT(), b=mysql.MEDIUMBLOB())

    index, _ = backend_for("mysql").build_index("ix", table, ["a", "b"])

    assert index.kwargs["mysql_length"] == {"a": 255, "b": 255}


def test_a_tinytext_is_not_sent_the_255_its_neighbour_gets():
    """Beside a TEXT the two prefixes differ, and the note gives each its own."""
    table = _table(short=mysql.TINYTEXT(), some_text=Text())

    index, notes = backend_for("mysql").build_index("ix", table, ["short", "some_text"])

    assert index.kwargs["mysql_length"] == {"short": 63, "some_text": 255}
    note = " ".join(notes)
    assert "63" in note and "255" in note


def test_a_tinytext_under_a_tight_budget_takes_what_is_left():
    """3040 bytes spent leaves 32: 8 characters accepted live, 9 answered 1071."""
    table = _table(v=String(760), short=mysql.TINYTEXT())

    index, _ = backend_for("mysql").build_index("ix", table, ["v", "short"])

    assert index.kwargs["mysql_length"] == {"short": 8}


def test_a_family_column_with_no_room_left_is_refused_by_name():
    """3072 bytes spent leaves nothing, so it is refused rather than sent."""
    table = _table(v=String(768), body=mysql.MEDIUMTEXT())

    with pytest.raises(UnsupportedOperation) as refused:
        backend_for("mysql").build_index("ix", table, ["v", "body"])

    assert "body" in str(refused.value)
