"""tests/v3/test_sql_validate_cache.py — E6.3 bounded LRU behavior.

The cache stores only classifications (and refusals), is bounded and
LRU-evicting, and is fail-closed against literal differences: two
statements differing only in a path literal never share an entry.
"""

from __future__ import annotations

import pytest

from localdata_mcp.nexus.chokepoint.sql_validate.cache import ValidationCache
from localdata_mcp.nexus.chokepoint.sql_validate.walker import SqlRefusedError


class TestClassificationCaching:
    def test_whitespace_only_difference_hits_the_same_entry(self) -> None:
        cache = ValidationCache(max_entries=8)
        cache.classify("SELECT   a FROM t", "sqlite")
        cache.classify("SELECT a FROM t", "sqlite")
        assert len(cache) == 1

    def test_literal_difference_never_shares_an_entry(self) -> None:
        cache = ValidationCache(max_entries=8)
        cache.classify("SELECT * FROM read_csv_auto('/a.csv')", "duckdb")
        cache.classify("SELECT * FROM read_csv_auto('/b.csv')", "duckdb")
        assert len(cache) == 2

    def test_case_is_not_folded(self) -> None:
        # A quoted identifier's case is semantic — never normalized away.
        cache = ValidationCache(max_entries=8)
        cache.classify('SELECT "A" FROM t', "sqlite")
        cache.classify('SELECT "a" FROM t', "sqlite")
        assert len(cache) == 2

    def test_dialect_is_part_of_the_key(self) -> None:
        cache = ValidationCache(max_entries=8)
        cache.classify("SELECT 1", "sqlite")
        cache.classify("SELECT 1", "postgresql")
        assert len(cache) == 2


class TestBound:
    def test_lru_evicts_the_least_recently_used(self) -> None:
        cache = ValidationCache(max_entries=2)
        cache.classify("SELECT 1", "sqlite")
        cache.classify("SELECT 2", "sqlite")
        cache.classify("SELECT 1", "sqlite")  # touch — now MRU
        cache.classify("SELECT 3", "sqlite")  # evicts SELECT 2
        assert len(cache) == 2

    def test_a_bound_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError):
            ValidationCache(max_entries=0)


class TestRefusalCaching:
    def test_a_refusal_is_cached_and_re_raised(self) -> None:
        cache = ValidationCache(max_entries=8)
        with pytest.raises(SqlRefusedError):
            cache.classify("DROP TABLE t", "sqlite")
        with pytest.raises(SqlRefusedError):
            cache.classify("DROP TABLE t", "sqlite")
        assert len(cache) == 1
