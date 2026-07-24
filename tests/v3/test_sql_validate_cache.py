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
    def test_byte_identical_statements_share_one_entry(self) -> None:
        cache = ValidationCache(max_entries=8)
        cache.classify("SELECT a FROM t", "sqlite")
        cache.classify("SELECT a FROM t", "sqlite")
        assert len(cache) == 1

    def test_exact_bytes_key_any_whitespace_difference_is_distinct(self) -> None:
        """CR-034: the key is the exact executed bytes — no normalization
        — so even an insignificant whitespace difference OUTSIDE a literal
        is a distinct entry (a miss, never a collision). This trades a few
        cache hits for removing the second lexer that reconstructed the
        statement."""
        cache = ValidationCache(max_entries=8)
        cache.classify("SELECT   a FROM t", "sqlite")
        cache.classify("SELECT a FROM t", "sqlite")
        assert len(cache) == 2

    def test_literal_difference_never_shares_an_entry(self) -> None:
        cache = ValidationCache(max_entries=8)
        cache.classify("SELECT * FROM read_csv_auto('/a.csv')", "duckdb")
        cache.classify("SELECT * FROM read_csv_auto('/b.csv')", "duckdb")
        assert len(cache) == 2

    def test_whitespace_inside_a_literal_never_shares_an_entry(self) -> None:
        """CR-011: two statements identical except for a whitespace run
        INSIDE a quoted literal are distinct executed bytes — and, for a
        path literal, distinct files — so the cached verdict must not be
        reused across them (guard-reads-a-model, NFR-108)."""
        cache = ValidationCache(max_entries=8)
        cache.classify("SELECT * FROM read_csv_auto('/a b.csv')", "duckdb")
        cache.classify("SELECT * FROM read_csv_auto('/a  b.csv')", "duckdb")
        assert len(cache) == 2

    def test_value_literal_interior_whitespace_is_preserved(self) -> None:
        cache = ValidationCache(max_entries=8)
        cache.classify("SELECT 'a b' FROM t", "sqlite")
        cache.classify("SELECT 'a  b' FROM t", "sqlite")
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
