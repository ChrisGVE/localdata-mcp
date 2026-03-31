"""Tests for DiskMonitor integration with StreamingQueryExecutor."""

import logging
import sys
import pandas as pd
import pytest
from unittest.mock import Mock, patch, MagicMock

# Work around a pre-existing circular-import bug in the package:
# size_estimator.py uses ``from localdata_mcp._size_types import …``
# (absolute import), which re-enters __init__.py and creates a second
# LoggingManager before the first finishes, causing a prometheus
# "Duplicated timeseries" ValueError.
#
# Fix: make MetricsCollector.__init__ silently reuse existing collectors
# instead of crashing on duplicate registration.
import prometheus_client

_orig_counter = prometheus_client.Counter
_orig_histogram = prometheus_client.Histogram
_orig_gauge = prometheus_client.Gauge


def _safe_counter(*args, registry=prometheus_client.REGISTRY, **kwargs):
    try:
        return _orig_counter(*args, registry=registry, **kwargs)
    except ValueError:
        # Already registered — return existing collector
        name = args[0] if args else kwargs.get("name", "")
        for key, col in registry._names_to_collectors.items():
            if key == name:
                return col
        return _orig_counter(*args, registry=registry, **kwargs)


def _safe_histogram(*args, registry=prometheus_client.REGISTRY, **kwargs):
    try:
        return _orig_histogram(*args, registry=registry, **kwargs)
    except ValueError:
        name = args[0] if args else kwargs.get("name", "")
        for key, col in registry._names_to_collectors.items():
            if key == name:
                return col
        return _orig_histogram(*args, registry=registry, **kwargs)


def _safe_gauge(*args, registry=prometheus_client.REGISTRY, **kwargs):
    try:
        return _orig_gauge(*args, registry=registry, **kwargs)
    except ValueError:
        name = args[0] if args else kwargs.get("name", "")
        for key, col in registry._names_to_collectors.items():
            if key == name:
                return col
        return _orig_gauge(*args, registry=registry, **kwargs)


prometheus_client.Counter = _safe_counter
prometheus_client.Histogram = _safe_histogram
prometheus_client.Gauge = _safe_gauge

# Also patch inside the metrics module that MetricsCollector imports from
import prometheus_client.metrics as _pm

_pm.Counter = _safe_counter
_pm.Histogram = _safe_histogram
_pm.Gauge = _safe_gauge

from localdata_mcp.streaming_executor import (  # noqa: E402
    StreamingQueryExecutor,
    StreamingDataSource,
    MemoryStatus,
)
from localdata_mcp.config_manager import PerformanceConfig  # noqa: E402
from localdata_mcp.disk_monitor import DiskMonitor  # noqa: E402


def _make_memory_status(low=False):
    """Create a MemoryStatus for mocking."""
    return MemoryStatus(
        total_gb=16.0,
        available_gb=8.0 if not low else 1.0,
        used_percent=50.0 if not low else 93.0,
        is_low_memory=low,
        recommended_chunk_size=5000,
        max_safe_chunk_size=10000,
    )


def _make_chunks(n_chunks=3, rows_per_chunk=500):
    """Build a list of DataFrames to be yielded by a mock source."""
    return [
        pd.DataFrame(
            {
                "id": range(i * rows_per_chunk, (i + 1) * rows_per_chunk),
                "value": range(rows_per_chunk),
            }
        )
        for i in range(n_chunks)
    ]


def _build_executor_and_source(chunks):
    """Return a patched StreamingQueryExecutor and a mock StreamingDataSource.

    The executor's memory check and token metadata helpers are stubbed so
    that the tests focus solely on the disk-monitor integration path.
    """
    config = PerformanceConfig()
    executor = StreamingQueryExecutor(config=config)

    source = Mock(spec=StreamingDataSource)
    source.get_chunk_iterator.return_value = iter(chunks)
    source.estimate_total_rows.return_value = sum(len(c) for c in chunks)
    source.estimate_memory_per_row.return_value = 128.0

    return executor, source


class TestStreamingWithDiskMonitorNoAbort:
    """135.17 - test_streaming_with_disk_monitor_no_abort."""

    @patch("src.localdata_mcp.streaming_executor.get_logging_manager")
    @patch("src.localdata_mcp.streaming_executor.get_token_manager")
    @patch(
        "src.localdata_mcp.streaming_executor.StreamingQueryExecutor._get_memory_status"
    )
    def test_monitor_always_allows_continuation(
        self, mock_mem, mock_token_mgr, mock_log_mgr
    ):
        mock_mem.return_value = _make_memory_status()
        mock_log_mgr.return_value = MagicMock()
        mock_token_mgr.return_value = MagicMock()

        chunks = _make_chunks(n_chunks=3, rows_per_chunk=500)
        executor, source = _build_executor_and_source(chunks)

        monitor = Mock(spec=DiskMonitor)
        monitor.check_can_continue.return_value = (True, None)

        result_df, metadata = executor.execute_streaming(
            source=source,
            query_id="test_no_abort",
            initial_chunk_size=5000,
            disk_monitor=monitor,
        )

        # All chunks processed, no truncation
        assert metadata["total_rows_processed"] == 1500
        assert "truncated" not in metadata
        assert monitor.check_can_continue.call_count >= 1


class TestStreamingWithDiskMonitorAbort:
    """135.17 - test_streaming_with_disk_monitor_abort."""

    @patch("src.localdata_mcp.streaming_executor.get_logging_manager")
    @patch("src.localdata_mcp.streaming_executor.get_token_manager")
    @patch(
        "src.localdata_mcp.streaming_executor.StreamingQueryExecutor._get_memory_status"
    )
    def test_monitor_aborts_at_row_threshold(
        self, mock_mem, mock_token_mgr, mock_log_mgr
    ):
        mock_mem.return_value = _make_memory_status()
        mock_log_mgr.return_value = MagicMock()
        mock_token_mgr.return_value = MagicMock()

        # 4 chunks of 500 rows = 2000 rows total
        chunks = _make_chunks(n_chunks=4, rows_per_chunk=500)
        executor, source = _build_executor_and_source(chunks)

        monitor = Mock(spec=DiskMonitor)
        # Allow first two chunks (rows 500, 1000), abort on third (1500)
        monitor.check_can_continue.side_effect = [
            (True, None),
            (False, "Staging limit reached (450.0MB of 500MB)"),
            (True, None),  # should never be reached
        ]

        result_df, metadata = executor.execute_streaming(
            source=source,
            query_id="test_abort",
            initial_chunk_size=5000,
            disk_monitor=monitor,
        )

        # Should have stopped after 2 chunks (1000 rows)
        assert metadata["total_rows_processed"] == 1000
        assert metadata["truncated"] is True
        assert "Staging limit" in metadata["abort_reason"]


class TestStreamingWithoutDiskMonitor:
    """135.17 - test_streaming_without_disk_monitor (backward compat)."""

    @patch("src.localdata_mcp.streaming_executor.get_logging_manager")
    @patch("src.localdata_mcp.streaming_executor.get_token_manager")
    @patch(
        "src.localdata_mcp.streaming_executor.StreamingQueryExecutor._get_memory_status"
    )
    def test_no_monitor_processes_all_chunks(
        self, mock_mem, mock_token_mgr, mock_log_mgr
    ):
        mock_mem.return_value = _make_memory_status()
        mock_log_mgr.return_value = MagicMock()
        mock_token_mgr.return_value = MagicMock()

        chunks = _make_chunks(n_chunks=3, rows_per_chunk=500)
        executor, source = _build_executor_and_source(chunks)

        # No disk_monitor argument at all
        result_df, metadata = executor.execute_streaming(
            source=source,
            query_id="test_no_monitor",
            initial_chunk_size=5000,
        )

        assert metadata["total_rows_processed"] == 1500
        assert "truncated" not in metadata
        assert "disk_monitor_active" not in metadata


class TestTruncationMetadataOnAbort:
    """135.17 - test_truncation_metadata_on_abort."""

    @patch("src.localdata_mcp.streaming_executor.get_logging_manager")
    @patch("src.localdata_mcp.streaming_executor.get_token_manager")
    @patch(
        "src.localdata_mcp.streaming_executor.StreamingQueryExecutor._get_memory_status"
    )
    def test_metadata_fields_present_on_abort(
        self, mock_mem, mock_token_mgr, mock_log_mgr
    ):
        mock_mem.return_value = _make_memory_status()
        mock_log_mgr.return_value = MagicMock()
        mock_token_mgr.return_value = MagicMock()

        chunks = _make_chunks(n_chunks=3, rows_per_chunk=500)
        executor, source = _build_executor_and_source(chunks)

        abort_msg = "System disk critical (200MB free, need 500MB)"
        monitor = Mock(spec=DiskMonitor)
        monitor.check_can_continue.return_value = (False, abort_msg)

        result_df, metadata = executor.execute_streaming(
            source=source,
            query_id="test_meta",
            initial_chunk_size=5000,
            disk_monitor=monitor,
        )

        assert metadata["truncated"] is True
        assert metadata["abort_reason"] == abort_msg
        assert metadata["disk_monitor_active"] is True
        assert "suggestion" in metadata


class TestAbortSuggestionPresent:
    """135.17 - test_abort_suggestion_present."""

    @patch("src.localdata_mcp.streaming_executor.get_logging_manager")
    @patch("src.localdata_mcp.streaming_executor.get_token_manager")
    @patch(
        "src.localdata_mcp.streaming_executor.StreamingQueryExecutor._get_memory_status"
    )
    def test_suggestion_text_content(self, mock_mem, mock_token_mgr, mock_log_mgr):
        mock_mem.return_value = _make_memory_status()
        mock_log_mgr.return_value = MagicMock()
        mock_token_mgr.return_value = MagicMock()

        chunks = _make_chunks(n_chunks=2, rows_per_chunk=500)
        executor, source = _build_executor_and_source(chunks)

        monitor = Mock(spec=DiskMonitor)
        monitor.check_can_continue.return_value = (
            False,
            "Disk usage above 95% (96.2%)",
        )

        _, metadata = executor.execute_streaming(
            source=source,
            query_id="test_suggestion",
            initial_chunk_size=5000,
            disk_monitor=monitor,
        )

        assert metadata["suggestion"] == ("Add LIMIT, WHERE clause, or use aggregation")
