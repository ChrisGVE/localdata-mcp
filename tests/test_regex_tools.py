"""Tests for safe regex execution engine with ReDoS prevention."""

import re

import pytest

from localdata_mcp.regex_tools import (
    RegexEngine,
    RegexSafetyValidator,
    get_regex_engine,
    search_data,
    transform_data,
)


class TestPatternValidation:
    """Tests for regex pattern validation."""

    def test_pattern_validation_valid(self):
        """Simple pattern passes validation."""
        validator = RegexSafetyValidator()
        result = validator.validate(r"\d+")
        assert result.is_valid is True
        assert result.error_message is None

    def test_pattern_validation_too_long(self):
        """Pattern exceeding MAX_PATTERN_LENGTH is rejected."""
        validator = RegexSafetyValidator()
        long_pattern = "a" * 300
        result = validator.validate(long_pattern)
        assert result.is_valid is False
        assert "too long" in result.error_message.lower()

    def test_pattern_validation_too_many_groups(self):
        """Pattern with more than MAX_CAPTURE_GROUPS is rejected."""
        validator = RegexSafetyValidator()
        # 15 capture groups
        pattern = "".join(f"(g{i})" for i in range(15))
        result = validator.validate(pattern)
        assert result.is_valid is False
        assert "Too many groups" in result.error_message

    def test_pattern_validation_invalid_syntax(self):
        """Syntactically invalid regex is rejected."""
        validator = RegexSafetyValidator()
        result = validator.validate("[unclosed")
        assert result.is_valid is False
        assert "Invalid regex" in result.error_message

    def test_redos_detection(self):
        """Nested quantifiers like (a+)+$ are flagged as dangerous."""
        validator = RegexSafetyValidator()
        result = validator.validate(r"(a+)+$")
        assert len(result.detected_issues) > 0
        assert any("Nested quantifier" in issue for issue in result.detected_issues)


class TestSearchData:
    """Tests for search_data function."""

    def setup_method(self):
        """Reset singleton between tests."""
        import localdata_mcp.regex_tools as mod

        mod._engine = None

    def test_search_basic(self):
        """Find email pattern in data."""
        data = [
            {"name": "Alice", "email": "alice@example.com"},
            {"name": "Bob", "email": "bob@test.org"},
            {"name": "Charlie", "email": "no-email-here"},
        ]
        result = search_data(data, r"[\w.+-]+@[\w-]+\.[\w.]+")
        assert result["total_matches"] == 2
        assert result["matches"][0]["column"] == "email"

    def test_search_case_insensitive(self):
        """Case-insensitive flag works."""
        data = [
            {"text": "Hello World"},
            {"text": "hello there"},
            {"text": "goodbye"},
        ]
        result = search_data(data, r"hello", case_sensitive=False)
        assert result["total_matches"] == 2

    def test_search_column_filter(self):
        """Only specified columns are searched."""
        data = [
            {"name": "test123", "code": "test456"},
        ]
        result = search_data(data, r"test\d+", columns=["code"])
        assert result["total_matches"] == 1
        assert result["matches"][0]["column"] == "code"

    def test_search_max_matches(self):
        """Max matches limit is enforced."""
        data = [{"val": "aaa"} for _ in range(50)]
        result = search_data(data, r"a", max_matches=3)
        assert result["total_matches"] == 3
        assert result["truncated"] is True

    def test_search_no_matches(self):
        """Empty results when no matches found."""
        data = [{"text": "hello"}]
        result = search_data(data, r"\d+")
        assert result["total_matches"] == 0
        assert result["matches"] == []
        assert result["truncated"] is False


class TestTransformData:
    """Tests for transform_data function."""

    def setup_method(self):
        """Reset singleton between tests."""
        import localdata_mcp.regex_tools as mod

        mod._engine = None

    def test_transform_basic(self):
        """Simple replacement works."""
        data = [{"name": "foo-bar"}, {"name": "baz-qux"}]
        result = transform_data(data, "name", r"-", "_")
        assert result["transformed_count"] == 2
        assert result["data"][0]["name"] == "foo_bar"

    def test_transform_capture_groups(self):
        r"""Backreference \1 replacement works."""
        data = [{"val": "2024-01-15"}]
        result = transform_data(data, "val", r"(\d{4})-(\d{2})-(\d{2})", r"\2/\3/\1")
        assert result["data"][0]["val"] == "01/15/2024"
        assert result["transformed_count"] == 1

    def test_transform_no_change(self):
        """No transformations when pattern not found."""
        data = [{"text": "hello"}]
        result = transform_data(data, "text", r"\d+", "NUM")
        assert result["transformed_count"] == 0
        assert result["data"][0]["text"] == "hello"

    def test_transform_sample(self):
        """Sample shows first 5 changes."""
        data = [{"v": f"item{i}"} for i in range(10)]
        result = transform_data(data, "v", r"item", "thing")
        assert len(result["sample"]) == 5
        assert result["sample"][0]["original"] == "item0"
        assert result["sample"][0]["transformed"] == "thing0"


class TestRegexEngine:
    """Tests for RegexEngine methods."""

    def setup_method(self):
        """Reset singleton between tests."""
        import localdata_mcp.regex_tools as mod

        mod._engine = None

    def test_engine_singleton(self):
        """get_regex_engine returns the same instance."""
        e1 = get_regex_engine()
        e2 = get_regex_engine()
        assert e1 is e2

    def test_engine_sub_method(self):
        """RegexEngine.sub performs substitution."""
        engine = RegexEngine()
        result = engine.sub(r"\d+", "NUM", "abc123def456")
        assert result == "abcNUMdefNUM"

    def test_engine_findall_method(self):
        """RegexEngine.findall returns all matches."""
        engine = RegexEngine()
        result = engine.findall("abc 123 def 456", r"\d+")
        assert result == ["123", "456"]
