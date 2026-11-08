"""What a MySQL or MariaDB index key costs, column by column (localdata#93).

tests/test_mysql_key_budget.py — beside tests/test_dialects.py, which holds the
first half of this behaviour: a bounded VARCHAR spends the key. This file adds
the rest of it, that every other column in the key spends it too.

No container is needed: ``build_index`` is arithmetic over a ``Table``. The
expected numbers are not derived from that arithmetic. They were read off a live
MySQL 8.0.46 and MariaDB 11, one column beside a BLOB, by the largest BLOB prefix
InnoDB accepted (a BLOB prefix is counted in bytes, so 3072 minus it is what the
column cost), and they agree with the MySQL manual's "Data Type Storage
Requirements". NULLable and NOT NULL columns measured the same.
"""

from __future__ import annotations

import pytest
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    Numeric,
    SmallInteger,
    String,
    Table,
    Text,
    Time,
    UUID,
)
from sqlalchemy.dialects import mysql
from sqlalchemy.types import NullType

from localdata_mcp.dialects import UnsupportedOperation, backend_for


def _table(**types) -> Table:
    columns = [Column(name, kind) for name, kind in types.items()]
    return Table("wide", MetaData(), *columns)


#: Every type as SQLAlchemy reflects it from the server (and the generic types a
#: table this server built carries), against the bytes the engine charged.
_MEASURED_KEY_BYTES = [
    (mysql.TINYINT(), 1),
    (mysql.TINYINT(display_width=1), 1),  # how BOOL reflects
    (mysql.SMALLINT(), 2),
    (mysql.MEDIUMINT(), 3),
    (mysql.INTEGER(), 4),
    (mysql.INTEGER(unsigned=True), 4),
    (mysql.BIGINT(), 8),
    (mysql.BIGINT(unsigned=True), 8),
    (mysql.FLOAT(), 4),
    (mysql.DOUBLE(asdecimal=True), 8),
    (mysql.DECIMAL(precision=1, scale=0), 1),
    (mysql.DECIMAL(precision=10, scale=2), 5),
    (mysql.DECIMAL(precision=18, scale=9), 8),
    (mysql.DECIMAL(precision=20, scale=6), 10),
    (mysql.DECIMAL(precision=65, scale=30), 30),
    (mysql.DATE(), 3),
    (mysql.YEAR(), 1),
    (mysql.TIME(), 3),
    (mysql.TIME(fsp=2), 4),
    (mysql.TIME(fsp=4), 5),
    (mysql.TIME(fsp=6), 6),
    (mysql.DATETIME(), 5),
    (mysql.DATETIME(fsp=2), 6),
    (mysql.DATETIME(fsp=6), 8),
    (mysql.TIMESTAMP(), 4),
    (mysql.TIMESTAMP(fsp=6), 7),
    (mysql.BIT(length=1), 1),
    (mysql.BIT(length=9), 2),
    (mysql.BIT(length=64), 8),
    (mysql.ENUM("a", "b"), 1),
    (mysql.ENUM(*[f"m{i}" for i in range(300)]), 2),
    (mysql.SET("a", "b"), 1),
    (mysql.SET(*[f"m{i}" for i in range(9)]), 2),
    (mysql.SET(*[f"m{i}" for i in range(17)]), 3),
    (mysql.SET(*[f"m{i}" for i in range(25)]), 4),
    (mysql.SET(*[f"m{i}" for i in range(33)]), 8),
    (mysql.SET(*[f"m{i}" for i in range(64)]), 8),
    (mysql.BINARY(length=10), 10),
    (mysql.VARBINARY(length=10), 10),
    # Four bytes a character: utf8mb4, the charset this budget is written in.
    (mysql.CHAR(length=10), 40),
    (mysql.VARCHAR(length=10), 40),
    # MariaDB's reflected UUID, 16 bytes on the engine.
    (UUID(), 16),
    # The generic types a table built by this server reflects from.
    (Integer(), 4),
    (BigInteger(), 8),
    (SmallInteger(), 2),
    (Float(), 4),
    (Boolean(), 1),
    (Date(), 3),
    (Time(), 3),
    (DateTime(), 5),
    (Numeric(10, 2), 5),
    (String(10), 40),
]


@pytest.mark.parametrize(
    ("kind", "expected"), _MEASURED_KEY_BYTES, ids=lambda value: repr(value)[:40]
)
def test_a_key_part_costs_what_the_engine_measured(kind, expected):
    from localdata_mcp.dialects import _mysql_key_part_bytes

    assert _mysql_key_part_bytes(kind) == expected


@pytest.mark.parametrize("kind", [mysql.JSON(), JSON(), NullType(), String()])
def test_a_type_with_no_known_width_is_not_sized(kind):
    from localdata_mcp.dialects import _mysql_key_part_bytes

    assert _mysql_key_part_bytes(kind) is None


def test_a_fixed_width_column_shrinks_the_prefix_so_the_key_fits():
    """The issue's case: 1071 on the live engine at some_text(168) beside an INT.

    id INT is 4 bytes and VARCHAR(600) is 2400, leaving 668 of 3072 for the TEXT:
    167 characters at four bytes each, which is 3072 bytes exactly.
    """
    table = _table(id=Integer(), wide_varchar=String(600), some_text=Text())

    index, notes = backend_for("mysql").build_index(
        "ix", table, ["id", "wide_varchar", "some_text"]
    )

    assert index.kwargs["mysql_length"] == {"some_text": 167}
    assert "167" in " ".join(notes)


def test_a_bigint_in_the_key_costs_eight_bytes_not_four():
    table = _table(big=BigInteger(), wide_varchar=String(600), some_text=Text())

    index, _ = backend_for("mysql").build_index(
        "ix", table, ["big", "wide_varchar", "some_text"]
    )

    assert index.kwargs["mysql_length"] == {"some_text": 166}


def test_the_prefix_is_unchanged_where_no_fixed_width_column_is_in_the_key():
    table = _table(id=Integer(), wide_varchar=String(600), some_text=Text())

    index, _ = backend_for("mysql").build_index(
        "ix", table, ["wide_varchar", "some_text"]
    )

    assert index.kwargs["mysql_length"] == {"some_text": 168}


def test_mariadb_charges_the_same_columns_on_its_own_length_keyword():
    table = _table(id=Integer(), wide_varchar=String(600), some_text=Text())

    index, _ = backend_for("mariadb").build_index(
        "ix", table, ["id", "wide_varchar", "some_text"]
    )

    assert index.kwargs["mariadb_length"] == {"some_text": 167}


def test_a_key_with_no_text_column_takes_the_generic_path_untouched():
    """No prefix, no note, and no sizing: nothing here needs a width."""
    table = _table(id=Integer(), payload=mysql.JSON())

    index, notes = backend_for("mysql").build_index("ix", table, ["id", "payload"])

    assert index.kwargs.get("mysql_length") is None
    assert notes == ()


def test_the_last_character_of_room_is_still_used():
    """Boundary, room for one: VARCHAR(766) + INT leaves 3072 - 3068 = 4 bytes."""
    table = _table(id=Integer(), narrow=String(766), some_text=Text())

    index, _ = backend_for("mysql").build_index(
        "ix", table, ["id", "narrow", "some_text"]
    )

    assert index.kwargs["mysql_length"] == {"some_text": 1}


def test_no_room_left_after_the_fixed_width_columns_is_refused_by_column():
    """Boundary, room for none: VARCHAR(767) + INT leaves 0 bytes for the TEXT.

    The refusal names the VARCHAR, the column worth dropping, rather than the INT
    that happened to tip it over.
    """
    table = _table(id=Integer(), wide=String(767), some_text=Text())

    with pytest.raises(UnsupportedOperation) as refused:
        backend_for("mysql").build_index("ix", table, ["id", "wide", "some_text"])

    message = str(refused.value)
    assert "Drop wide" in message
    assert "3072" in message


def test_a_key_of_nothing_but_prefixes_is_refused_without_crashing():
    """No sized column to blame: 769 TEXT columns cannot fit one character each.

    The engine's 16-part limit would stop this first, but the arithmetic must
    still answer with a refusal naming a real column, not a ValueError.
    """
    table = _table(**{f"t{i}": Text() for i in range(769)})

    with pytest.raises(UnsupportedOperation) as refused:
        backend_for("mysql").build_index("ix", table, list(table.c.keys()))

    assert "t768" in str(refused.value)


@pytest.mark.parametrize("kind", [mysql.JSON(), NullType()])
def test_a_column_that_cannot_be_sized_beside_a_text_column_is_refused(kind):
    """It must not count as free: the prefix would be derived from a wrong sum.

    Measured: MySQL refuses JSON in any key of this shape (3152) and reflection's
    NullType covers GEOMETRY (1070 on MySQL). It also covers MariaDB's INET4 and
    INET6, which the engine would accept at 4 and 16 bytes: an unrecognised type
    is sized by nobody here, so the refusal costs that case and says what to do.
    """
    table = _table(payload=kind, some_text=Text())

    with pytest.raises(UnsupportedOperation) as refused:
        backend_for("mysql").build_index("ix", table, ["payload", "some_text"])

    message = str(refused.value)
    assert "payload" in message
    assert "some_text" in message


def test_a_binary_column_is_charged_its_bytes_not_four_per_character():
    """VARBINARY(1000) is 1000 bytes, so it leaves room for the 255 cap."""
    table = _table(blob_key=mysql.VARBINARY(length=1000), some_text=Text())

    index, _ = backend_for("mysql").build_index("ix", table, ["blob_key", "some_text"])

    assert index.kwargs["mysql_length"] == {"some_text": 255}
