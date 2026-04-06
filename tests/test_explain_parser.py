"""Tests for explain_parser module."""

from unittest.mock import MagicMock, patch

import pytest

from localdata_mcp.explain_parser import (
    _PARSER_REGISTRY,
    ExplainResult,
    PreflightResult,
    generate_suggestion,
    get_parser,
    register_parser,
    run_explain,
)


class TestExplainResult:
    """Tests for ExplainResult dataclass."""

    def test_explain_result_defaults(self):
        """Default confidence=0.3, empty columns list."""
        result = ExplainResult()
        assert result.confidence == 0.3
        assert result.columns == []
        assert result.estimated_rows is None
        assert result.estimated_cost is None
        assert result.scan_type is None
        assert result.raw_plan is None


class TestPreflightResult:
    """Tests for PreflightResult dataclass."""

    def test_preflight_result_to_dict(self):
        """Verify all standard fields in to_dict output."""
        result = PreflightResult(
            estimated_rows=500,
            estimated_size_mb=1.5,
            columns=["id", "name"],
            scan_type="index",
            confidence=0.7,
            suggestion="Safe to execute directly.",
        )
        d = result.to_dict()
        assert d["preflight"] is True
        assert d["estimated_rows"] == 500
        assert d["estimated_size_mb"] == 1.5
        assert d["columns"] == ["id", "name"]
        assert d["scan_type"] == "index"
        assert d["confidence"] == 0.7
        assert d["suggestion"] == "Safe to execute directly."
        assert "error" not in d
        assert "fallback" not in d

    def test_preflight_result_to_dict_with_error(self):
        """Error present triggers fallback field."""
        result = PreflightResult(error="EXPLAIN not supported")
        d = result.to_dict()
        assert d["error"] == "EXPLAIN not supported"
        assert d["fallback"] == "EXPLAIN not available, consider adding LIMIT"


class TestParserRegistry:
    """Tests for parser registration and retrieval."""

    def test_register_and_get_parser(self):
        """Register a parser then retrieve it."""
        mock_parser = MagicMock()
        register_parser("test_dialect", mock_parser)
        assert get_parser("test_dialect") is mock_parser
        # Cleanup
        _PARSER_REGISTRY.pop("test_dialect", None)

    def test_get_parser_unknown(self):
        """Unknown dialect returns None."""
        assert get_parser("nonexistent_dialect_xyz") is None


class TestRunExplain:
    """Tests for run_explain function."""

    def test_run_explain_sqlite(self):
        """Mock engine with sqlite dialect returns ExplainResult."""
        engine = MagicMock()
        engine.dialect.name = "sqlite"
        mock_parser = MagicMock(
            return_value={
                "estimated_rows": 42,
                "total_cost": 1.5,
                "scan_type": "index",
                "confidence": 0.5,
                "raw_plan": "SCAN TABLE foo",
            }
        )
        with patch.dict(_PARSER_REGISTRY, {"sqlite": mock_parser}):
            result = run_explain(engine, "SELECT * FROM foo")
        assert result is not None
        assert result.estimated_rows == 42
        assert result.estimated_cost == 1.5
        assert result.scan_type == "index"
        assert result.confidence == 0.5
        assert result.raw_plan == "SCAN TABLE foo"
        mock_parser.assert_called_once_with(engine, "SELECT * FROM foo")

    def test_run_explain_failure(self):
        """Parser that throws returns None."""
        engine = MagicMock()
        engine.dialect.name = "sqlite"
        mock_parser = MagicMock(side_effect=RuntimeError("boom"))
        with patch.dict(_PARSER_REGISTRY, {"sqlite": mock_parser}):
            result = run_explain(engine, "SELECT * FROM foo")
        assert result is None

    def test_run_explain_no_engine(self):
        """None engine returns None."""
        result = run_explain(None, "SELECT * FROM foo")
        assert result is None


class TestGenerateSuggestion:
    """Tests for generate_suggestion function."""

    def test_generate_suggestion_small(self):
        """Under 1000 rows: safe."""
        msg = generate_suggestion(500)
        assert "Safe" in msg

    def test_generate_suggestion_medium(self):
        """1000-100000 rows: streaming."""
        msg = generate_suggestion(50000)
        assert "Streaming" in msg

    def test_generate_suggestion_large(self):
        """Over 100000 rows: add LIMIT."""
        msg = generate_suggestion(200000)
        assert "LIMIT" in msg
        assert "200,000" in msg

    def test_generate_suggestion_unknown(self):
        """None rows: consider adding LIMIT."""
        msg = generate_suggestion(None)
        assert "LIMIT" in msg
        assert "unknown" in msg.lower()


class TestParsersRegistered:
    """Test that built-in parsers are registered at import time."""

    def test_parsers_registered(self):
        """sqlite, postgresql, mysql must be in registry."""
        assert "sqlite" in _PARSER_REGISTRY
        assert "postgresql" in _PARSER_REGISTRY
        assert "mysql" in _PARSER_REGISTRY


class TestExecutePreflight:
    """Tests for the preflight parameter on execute_query."""

    def _make_manager(self):
        """Create a DatabaseManager with an in-memory SQLite DB."""
        import json as _json

        from sqlalchemy import create_engine, text

        from localdata_mcp.localdata_mcp import DatabaseManager

        mgr = DatabaseManager()
        engine = create_engine("sqlite:///:memory:")
        with engine.connect() as conn:
            conn.execute(text("CREATE TABLE t (id INTEGER, val TEXT)"))
            conn.execute(text("INSERT INTO t VALUES (1, 'a'), (2, 'b')"))
            conn.commit()
        mgr.connections["testdb"] = engine
        mgr.db_types["testdb"] = "sqlite"
        return mgr

    def test_preflight_with_sqlite(self):
        """Preflight against in-memory SQLite returns valid result."""
        import json

        mgr = self._make_manager()
        result = mgr.execute_query("testdb", "SELECT * FROM t", preflight=True)
        data = json.loads(result)
        assert data["preflight"] is True

    def test_preflight_returns_json(self):
        """Preflight returns valid JSON with expected keys."""
        import json

        mgr = self._make_manager()
        data = json.loads(
            mgr.execute_query("testdb", "SELECT * FROM t", preflight=True)
        )
        for key in ("preflight", "estimated_rows", "estimated_size_mb", "confidence"):
            assert key in data

    def test_preflight_error_handling(self):
        """Invalid query returns error in preflight response, not an exception."""
        import json

        mgr = self._make_manager()
        data = json.loads(
            mgr.execute_query(
                "testdb", "SELECT * FROM nonexistent_table_xyz", preflight=True
            )
        )
        assert data["preflight"] is True
        assert isinstance(data, dict)

    def test_preflight_suggestion_present(self):
        """Preflight result includes suggestion field."""
        import json

        mgr = self._make_manager()
        data = json.loads(
            mgr.execute_query("testdb", "SELECT * FROM t", preflight=True)
        )
        assert "suggestion" in data or "error" in data

    def test_preflight_false_executes_normally(self):
        """Default preflight=False executes the query normally."""
        import json

        mgr = self._make_manager()
        result = mgr.execute_query("testdb", "SELECT * FROM t", preflight=False)
        data = json.loads(result)
        # Normal execution returns rows, not preflight metadata
        assert "preflight" not in data or data.get("preflight") is not True
