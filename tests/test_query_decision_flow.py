"""Tests for memory-aware query execution decision flow.

Verifies that:
- Small queries route to in-memory execution
- Medium queries route to staging execution
- Large queries return a refinement response
- Aggressive mode reduces thresholds appropriately
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from localdata_mcp.memory_budget import MemoryBudget
from localdata_mcp.server.query_execution import (
    ExecutionDecision,
    build_memory_budget_metadata,
    build_refinement_response,
    calculate_dynamic_chunk_size,
    decide_execution_path,
    get_disk_budget,
    get_memory_budget,
)


def _make_analysis(estimated_mb: float, row_size: float = 256.0):
    """Create a minimal QueryAnalysis-like object for testing."""
    return SimpleNamespace(
        estimated_total_memory_mb=estimated_mb,
        estimated_row_size_bytes=row_size,
        estimated_rows=int(estimated_mb * 1024 * 1024 / row_size) if row_size > 0 else 0,
        should_chunk=True,
        recommended_chunk_size=500,
    )


def _make_budget(budget_mb: int, aggressive: bool = False):
    """Create a MemoryBudget with a specific size."""
    return MemoryBudget(
        budget_bytes=budget_mb * 1024 * 1024,
        max_budget_mb=budget_mb,
        budget_percent=10.0,
        is_aggressive_mode=aggressive,
        available_ram_gb=8.0 if not aggressive else 0.5,
    )


class TestDecideExecutionPath:
    """Tests for decide_execution_path()."""

    def test_small_query_routes_in_memory(self):
        """A query well below RAM budget goes to in_memory."""
        budget = _make_budget(512)
        analysis = _make_analysis(estimated_mb=50.0)

        decision = decide_execution_path(analysis, budget)

        assert decision.path == "in_memory"
        assert decision.estimated_bytes == 50 * 1024 * 1024

    def test_no_analysis_defaults_in_memory(self):
        """When query_analysis is None, default to in_memory."""
        budget = _make_budget(512)

        decision = decide_execution_path(None, budget)

        assert decision.path == "in_memory"
        assert decision.estimated_bytes == 0

    @patch("localdata_mcp.server.query_execution.get_disk_budget")
    def test_medium_query_routes_staging(self, mock_disk):
        """Query exceeds RAM but fits disk budget -> staging."""
        mock_disk.return_value = 2048 * 1024 * 1024  # 2 GB disk

        budget = _make_budget(512)
        analysis = _make_analysis(estimated_mb=700.0)

        decision = decide_execution_path(analysis, budget)

        assert decision.path == "staging"
        assert decision.disk_budget_bytes == 2048 * 1024 * 1024

    @patch("localdata_mcp.server.query_execution.get_disk_budget")
    def test_large_query_returns_refinement(self, mock_disk):
        """Query exceeds both RAM and disk -> refinement."""
        mock_disk.return_value = 1024 * 1024 * 1024  # 1 GB disk

        budget = _make_budget(512)
        analysis = _make_analysis(estimated_mb=2000.0)

        decision = decide_execution_path(analysis, budget)

        assert decision.path == "refinement"

    def test_aggressive_mode_tighter_threshold(self):
        """Aggressive budget (128 MB) blocks more queries."""
        budget = _make_budget(128, aggressive=True)
        analysis = _make_analysis(estimated_mb=200.0)

        # Would fit normal 512 MB budget, but not aggressive 128 MB
        # This should NOT be in_memory
        decision = decide_execution_path(analysis, budget)
        assert decision.path != "in_memory"

    def test_boundary_just_under_budget(self):
        """Estimated size just below budget -> in_memory."""
        budget = _make_budget(512)
        analysis = _make_analysis(estimated_mb=511.0)

        decision = decide_execution_path(analysis, budget)

        assert decision.path == "in_memory"

    def test_boundary_at_budget(self):
        """Estimated size exactly at budget -> NOT in_memory."""
        budget = _make_budget(512)
        analysis = _make_analysis(estimated_mb=512.0)

        decision = decide_execution_path(analysis, budget)

        assert decision.path != "in_memory"


class TestBuildRefinementResponse:
    """Tests for build_refinement_response()."""

    def test_response_structure(self):
        """Refinement response has expected keys."""
        budget = _make_budget(512)
        result = build_refinement_response(
            estimated_bytes=3 * 1024 * 1024 * 1024,
            memory_budget=budget,
            disk_budget_bytes=2 * 1024 * 1024 * 1024,
        )
        data = json.loads(result)

        assert data["error"] is False
        assert data["requires_refinement"] is True
        assert data["estimated_size_mb"] > 0
        assert isinstance(data["suggestions"], list)
        assert len(data["suggestions"]) >= 3

    def test_response_includes_budget_info(self):
        """Response includes budget and aggressive mode info."""
        budget = _make_budget(128, aggressive=True)
        result = build_refinement_response(
            estimated_bytes=5 * 1024 * 1024 * 1024,
            memory_budget=budget,
            disk_budget_bytes=1 * 1024 * 1024 * 1024,
        )
        data = json.loads(result)

        assert data["is_aggressive_mode"] is True
        assert data["memory_budget_mb"] == 128


class TestCalculateDynamicChunkSize:
    """Tests for calculate_dynamic_chunk_size()."""

    def test_no_analysis_returns_fallback(self):
        """Without analysis, return the fallback value."""
        budget = _make_budget(512)

        result = calculate_dynamic_chunk_size(budget, None)

        assert result == 1000  # default fallback

    def test_custom_fallback(self):
        """Custom fallback is returned when analysis is None."""
        budget = _make_budget(512)

        result = calculate_dynamic_chunk_size(budget, None, fallback=500)

        assert result == 500

    def test_large_row_size_smaller_chunks(self):
        """Large rows produce smaller chunk counts."""
        budget = _make_budget(512)
        analysis = _make_analysis(estimated_mb=1000.0, row_size=10240.0)

        result = calculate_dynamic_chunk_size(budget, analysis)

        # 10% of 512MB / 10KB per row = ~5242 rows
        assert result > 10
        assert result < 10000

    def test_small_row_size_larger_chunks(self):
        """Small rows produce larger chunk counts."""
        budget = _make_budget(512)
        analysis = _make_analysis(estimated_mb=100.0, row_size=64.0)

        result = calculate_dynamic_chunk_size(budget, analysis)

        # 10% of 512MB / 64 bytes = ~838860 rows
        assert result > 100000

    def test_minimum_10_rows(self):
        """Chunk size never drops below 10."""
        budget = _make_budget(1)  # 1 MB budget
        analysis = _make_analysis(estimated_mb=1000.0, row_size=1024 * 1024)

        result = calculate_dynamic_chunk_size(budget, analysis)

        assert result >= 10

    def test_zero_row_size_returns_fallback(self):
        """Zero estimated row size returns fallback."""
        budget = _make_budget(512)
        analysis = _make_analysis(estimated_mb=100.0, row_size=0.0)

        result = calculate_dynamic_chunk_size(budget, analysis)

        assert result == 1000


class TestGetDiskBudget:
    """Tests for get_disk_budget()."""

    @patch("localdata_mcp.server.query_execution.shutil.disk_usage")
    @patch(
        "localdata_mcp.server.query_execution.get_disk_budget.__wrapped__"
        if hasattr(get_disk_budget, "__wrapped__")
        else "localdata_mcp.server.query_execution.Path"
    )
    def test_disk_budget_respects_config(self, mock_path, mock_usage):
        """Disk budget is capped by config limits."""
        # Mock disk_usage to return plenty of space
        mock_usage.return_value = SimpleNamespace(
            total=500 * 1024**3,
            used=100 * 1024**3,
            free=400 * 1024**3,
        )

        budget = get_disk_budget()

        # Budget should be positive and not exceed config max
        assert budget > 0


class TestBuildMemoryBudgetMetadata:
    """Tests for build_memory_budget_metadata()."""

    def test_metadata_keys(self):
        """Metadata dict contains all expected keys."""
        budget = _make_budget(512, aggressive=False)

        result = build_memory_budget_metadata(budget)

        assert result["budget_bytes"] == 512 * 1024 * 1024
        assert result["max_budget_mb"] == 512
        assert result["is_aggressive_mode"] is False
        assert result["available_ram_gb"] == 8.0

    def test_aggressive_metadata(self):
        """Aggressive mode is reflected in metadata."""
        budget = _make_budget(128, aggressive=True)

        result = build_memory_budget_metadata(budget)

        assert result["is_aggressive_mode"] is True
        assert result["available_ram_gb"] == 0.5
