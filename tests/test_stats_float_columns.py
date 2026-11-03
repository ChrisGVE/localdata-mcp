"""A floating-point column is numeric, whichever SQLAlchemy release reflects it (#70).

SQLAlchemy 2.1 stopped making ``Float`` a subclass of ``Numeric``: ``Float``,
``REAL``, ``FLOAT``, ``DOUBLE_PRECISION`` and ``Double`` now sit under
``NumericCommon`` alone. ``stats`` decided numeric-ness with
``isinstance(type, (Numeric, Integer))``, so on 2.1 a ``REAL`` column lost its
range and average silently. The endpoint test that caught it landed a CSV with a
blank in a ``REAL`` salary column; on PostgreSQL and CockroachDB that column
reflects as a float type, and ``min``, ``max`` and ``avg`` were missing.

The fix classifies by the public type classes that cover every release
(``Numeric``, ``Float``, ``Integer``), so the answer does not depend on which
release's hierarchy is installed.
"""

from __future__ import annotations

import pytest
from sqlalchemy import REAL, Column, Double, Float, Integer, Numeric, String
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION
from sqlalchemy.types import FLOAT

from localdata_mcp.loader import Workspace


@pytest.mark.parametrize(
    "kind",
    [Numeric(), Float(), FLOAT(), REAL(), Double(), DOUBLE_PRECISION(), Integer()],
    ids=lambda kind: type(kind).__name__,
)
def test_a_numeric_column_is_offered_its_range_and_average(kind):
    column = Column("value", kind)
    aggregates = Workspace._aggregates_for(column, None, {})
    assert {"min", "max", "avg"} <= set(aggregates), type(kind).__name__


def test_a_text_column_is_still_offered_its_null_count_alone():
    column = Column("name", String())
    aggregates = Workspace._aggregates_for(column, None, {})
    assert set(aggregates) == {"non_nulls"}
