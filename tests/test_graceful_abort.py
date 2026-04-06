"""Tests for graceful abort handler (subtask 135.14).

Covers handle_graceful_abort, add_truncation_metadata, and the
execute_streaming_core abort path with disk monitor integration.
"""

import pandas as pd
import pytest
from unittest.mock import Mock, patch

from localdata_mcp.streaming.execution import (
    _ABORT_SUGGESTION,
    add_truncation_metadata,
    handle_graceful_abort,
)
from localdata_mcp.streaming.models import ResultBuffer


# ------------------------------------------------------------------
# handle_graceful_abort unit tests
# ------------------------------------------------------------------


class TestHandleGracefulAbort:
    """Test handle_graceful_abort sets buffer state correctly."""

    def test_marks_buffer_incomplete(self):
        """Buffer is_complete must be False after abort."""
        buf = ResultBuffer("q1", "db", "SELECT 1")
        buf.is_complete = True
        handle_graceful_abort(buf, 500, "disk full")
        assert buf.is_complete is False

    def test_logs_abort_event(self):
        """Abort event is logged with context."""
        buf = ResultBuffer("q1", "db", "SELECT 1")
        with patch("localdata_mcp.streaming.execution.logger") as mock_logger:
            handle_graceful_abort(buf, 1200, "Staging limit")
            mock_logger.warning.assert_called_once()
            msg = mock_logger.warning.call_args[0][0]
            assert "Graceful abort" in msg

    def test_log_includes_row_count(self):
        """Log message includes rows_processed."""
        buf = ResultBuffer("q1", "db", "SELECT 1")
        with patch("localdata_mcp.streaming.execution.logger") as mock_logger:
            handle_graceful_abort(buf, 3000, "reason")
            args = mock_logger.warning.call_args[0]
            assert 3000 in args

    def test_log_includes_query_id(self):
        """Log message includes the buffer query_id."""
        buf = ResultBuffer("my_query", "db", "SELECT 1")
        with patch("localdata_mcp.streaming.execution.logger") as mock_logger:
            handle_graceful_abort(buf, 100, "reason")
            args = mock_logger.warning.call_args[0]
            assert "my_query" in args

    def test_none_abort_reason(self):
        """None abort_reason does not raise."""
        buf = ResultBuffer("q1", "db", "SELECT 1")
        handle_graceful_abort(buf, 0, None)
        assert buf.is_complete is False


# ------------------------------------------------------------------
# add_truncation_metadata unit tests
# ------------------------------------------------------------------


class TestAddTruncationMetadata:
    """Test add_truncation_metadata populates all required fields."""

    def _base_metadata(self) -> dict:
        return {"query_id": "q1", "streaming": True}

    def test_sets_truncated_flag(self):
        """Metadata must contain truncated=True."""
        meta = self._base_metadata()
        add_truncation_metadata(meta, "disk full", 500)
        assert meta["truncated"] is True

    def test_sets_partial_result(self):
        """Metadata must contain partial_result=True."""
        meta = self._base_metadata()
        add_truncation_metadata(meta, "disk full", 500)
        assert meta["partial_result"] is True

    def test_sets_total_rows_before_abort(self):
        """Metadata must contain total_rows_before_abort."""
        meta = self._base_metadata()
        add_truncation_metadata(meta, "disk full", 1234)
        assert meta["total_rows_before_abort"] == 1234

    def test_sets_abort_reason(self):
        """Metadata must contain the abort_reason string."""
        meta = self._base_metadata()
        reason = "Staging limit reached (450.0MB of 500MB)"
        add_truncation_metadata(meta, reason, 100)
        assert meta["abort_reason"] == reason

    def test_sets_suggestion(self):
        """Suggestion must mention LIMIT, WHERE, and aggregation."""
        meta = self._base_metadata()
        add_truncation_metadata(meta, "reason", 100)
        assert "LIMIT" in meta["suggestion"]
        assert "WHERE" in meta["suggestion"]
        assert "GROUP BY" in meta["suggestion"]
        assert "COUNT" in meta["suggestion"]

    def test_sets_disk_monitor_active(self):
        """Metadata must contain disk_monitor_active=True."""
        meta = self._base_metadata()
        add_truncation_metadata(meta, "reason", 100)
        assert meta["disk_monitor_active"] is True

    def test_suggestion_matches_constant(self):
        """Suggestion value matches the module-level constant."""
        meta = self._base_metadata()
        add_truncation_metadata(meta, "reason", 100)
        assert meta["suggestion"] == _ABORT_SUGGESTION

    def test_preserves_existing_keys(self):
        """Existing metadata keys are not removed."""
        meta = {"query_id": "q1", "custom": "value"}
        add_truncation_metadata(meta, "reason", 50)
        assert meta["custom"] == "value"
        assert meta["query_id"] == "q1"


# ------------------------------------------------------------------
# execute_streaming_core abort-path integration
# ------------------------------------------------------------------


class TestExecuteStreamingCoreAbort:
    """Test that execute_streaming_core calls handle_graceful_abort."""

    def _make_source(self, chunks):
        """Create a mock streaming data source."""
        source = Mock()
        source.get_chunk_iterator.return_value = iter(chunks)
        source.estimate_total_rows.return_value = sum(len(c) for c in chunks)
        return source

    def _make_config(self):
        """Create a minimal PerformanceConfig."""
        from localdata_mcp.config_manager import PerformanceConfig

        return PerformanceConfig()

    def _make_memory_status(self):
        """Create a normal memory status."""
        from localdata_mcp.streaming.models import MemoryStatus

        return MemoryStatus(
            total_gb=16.0,
            available_gb=8.0,
            used_percent=50.0,
            is_low_memory=False,
            recommended_chunk_size=5000,
            max_safe_chunk_size=10000,
        )

    @patch("localdata_mcp.streaming.execution.get_memory_status")
    def test_buffer_incomplete_on_abort(self, mock_mem):
        """Buffer is_complete is False when disk monitor aborts."""
        from localdata_mcp.streaming.execution import (
            execute_streaming_core,
        )

        mock_mem.return_value = self._make_memory_status()
        chunks = [
            pd.DataFrame({"a": range(100)}),
            pd.DataFrame({"a": range(100)}),
        ]
        source = self._make_source(chunks)
        config = self._make_config()
        buf = ResultBuffer("q1", "db", "SELECT 1")

        monitor = Mock()
        monitor.check_can_continue.side_effect = [
            (True, None),
            (False, "disk full"),
        ]

        result = execute_streaming_core(
            source,
            "q1",
            5000,
            self._make_memory_status(),
            buf,
            config,
            [],
            disk_monitor=monitor,
        )
        _, total_rows, _, truncated, reason, _ = result

        assert truncated is True
        assert buf.is_complete is False
        assert reason == "disk full"

    @patch("localdata_mcp.streaming.execution.get_memory_status")
    def test_buffer_complete_without_abort(self, mock_mem):
        """Buffer is_complete is True when no abort occurs."""
        from localdata_mcp.streaming.execution import (
            execute_streaming_core,
        )

        mock_mem.return_value = self._make_memory_status()
        chunks = [pd.DataFrame({"a": range(50)})]
        source = self._make_source(chunks)
        config = self._make_config()
        buf = ResultBuffer("q2", "db", "SELECT 1")

        result = execute_streaming_core(
            source,
            "q2",
            5000,
            self._make_memory_status(),
            buf,
            config,
            [],
        )
        _, _, _, truncated, _, _ = result

        assert truncated is False
        assert buf.is_complete is True
