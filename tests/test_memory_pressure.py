"""Tests for memory pressure scenarios and aggressive mode.

Verifies that:
- Aggressive mode activates when available RAM < 1 GB
- Budget reduces under memory pressure
- Clean error messages are produced
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from localdata_mcp.memory_budget import MemoryBudget
from localdata_mcp.server.query_execution import (
    build_refinement_response,
    decide_execution_path,
)


def _fake_vmem(available_gb: float):
    """Return a mock virtual_memory() result."""
    available = int(available_gb * 1024**3)
    return SimpleNamespace(
        total=16 * 1024**3,
        available=available,
        percent=100 - (available_gb / 16) * 100,
        used=int((16 - available_gb) * 1024**3),
        free=available,
    )


def _make_analysis(estimated_mb: float):
    """Create a minimal QueryAnalysis-like object."""
    return SimpleNamespace(
        estimated_total_memory_mb=estimated_mb,
        estimated_row_size_bytes=256.0,
        estimated_rows=int(estimated_mb * 1024 * 1024 / 256),
        should_chunk=True,
        recommended_chunk_size=500,
    )


class TestAggressiveModeActivation:
    """Tests that aggressive mode activates under low memory."""

    @patch("localdata_mcp.memory_budget.PSUTIL_AVAILABLE", True)
    @patch("localdata_mcp.memory_budget.psutil")
    def test_aggressive_activates_below_1gb(self, mock_psutil):
        """With 0.8 GB available, aggressive mode activates."""
        mock_psutil.virtual_memory.return_value = _fake_vmem(0.8)

        budget = MemoryBudget.calculate()

        assert budget.is_aggressive_mode
        assert budget.available_ram_gb == 0.8
        assert budget.max_budget_mb == 128

    @patch("localdata_mcp.memory_budget.PSUTIL_AVAILABLE", True)
    @patch("localdata_mcp.memory_budget.psutil")
    def test_normal_mode_above_1gb(self, mock_psutil):
        """With 2 GB available, normal mode is used."""
        mock_psutil.virtual_memory.return_value = _fake_vmem(2.0)

        budget = MemoryBudget.calculate()

        assert not budget.is_aggressive_mode
        assert budget.max_budget_mb == 512

    @patch("localdata_mcp.memory_budget.PSUTIL_AVAILABLE", True)
    @patch("localdata_mcp.memory_budget.psutil")
    def test_aggressive_with_very_low_memory(self, mock_psutil):
        """With 0.2 GB available, aggressive mode still works."""
        mock_psutil.virtual_memory.return_value = _fake_vmem(0.2)

        budget = MemoryBudget.calculate()

        assert budget.is_aggressive_mode
        # 5% of 0.2 GB = ~10 MB
        expected = int(0.2 * 1024**3 * 0.05)
        assert budget.budget_bytes == expected
        assert budget.budget_bytes < 128 * 1024 * 1024


class TestBudgetReducesUnderPressure:
    """Tests that budgets shrink when RAM is limited."""

    @patch("localdata_mcp.memory_budget.PSUTIL_AVAILABLE", True)
    @patch("localdata_mcp.memory_budget.psutil")
    def test_budget_smaller_in_aggressive(self, mock_psutil):
        """Aggressive budget is smaller than normal budget."""
        mock_psutil.virtual_memory.return_value = _fake_vmem(8.0)
        normal_budget = MemoryBudget.calculate()

        mock_psutil.virtual_memory.return_value = _fake_vmem(0.5)
        aggressive_budget = MemoryBudget.calculate()

        assert aggressive_budget.budget_bytes < normal_budget.budget_bytes

    @patch("localdata_mcp.memory_budget.PSUTIL_AVAILABLE", True)
    @patch("localdata_mcp.memory_budget.psutil")
    def test_aggressive_blocks_more_queries(self, mock_psutil):
        """A query that fits normal budget may not fit aggressive."""
        mock_psutil.virtual_memory.return_value = _fake_vmem(8.0)
        normal = MemoryBudget.calculate()

        mock_psutil.virtual_memory.return_value = _fake_vmem(0.5)
        aggressive = MemoryBudget.calculate()

        # 200 MB query: fits 512 MB normal, may not fit aggressive
        analysis = _make_analysis(estimated_mb=200.0)

        normal_decision = decide_execution_path(analysis, normal)
        aggressive_decision = decide_execution_path(analysis, aggressive)

        assert normal_decision.path == "in_memory"
        assert aggressive_decision.path != "in_memory"

    @patch("localdata_mcp.memory_budget.PSUTIL_AVAILABLE", True)
    @patch("localdata_mcp.memory_budget.psutil")
    def test_custom_threshold(self, mock_psutil):
        """Custom low_memory_threshold_gb is respected."""
        mock_psutil.virtual_memory.return_value = _fake_vmem(1.5)

        # Default threshold is 1.0 GB, so 1.5 GB is normal
        default_budget = MemoryBudget.calculate()
        assert not default_budget.is_aggressive_mode

        # With 2.0 GB threshold, 1.5 GB triggers aggressive
        custom_budget = MemoryBudget.calculate(low_memory_threshold_gb=2.0)
        assert custom_budget.is_aggressive_mode


class TestCleanErrorMessages:
    """Tests that error/refinement messages are clear."""

    def test_refinement_has_suggestions(self):
        """Refinement response includes actionable suggestions."""
        budget = MemoryBudget(
            budget_bytes=128 * 1024 * 1024,
            max_budget_mb=128,
            budget_percent=5.0,
            is_aggressive_mode=True,
            available_ram_gb=0.5,
        )

        import json

        result = json.loads(
            build_refinement_response(
                estimated_bytes=5 * 1024 * 1024 * 1024,
                memory_budget=budget,
                disk_budget_bytes=2 * 1024 * 1024 * 1024,
            )
        )

        assert result["error"] is False
        assert result["requires_refinement"] is True
        assert "Add LIMIT clause" in result["suggestions"]
        assert "Add WHERE clause" in result["suggestions"]
        assert result["is_aggressive_mode"] is True

    def test_refinement_includes_size_estimate(self):
        """Refinement response shows the estimated size."""
        budget = MemoryBudget(
            budget_bytes=512 * 1024 * 1024,
            max_budget_mb=512,
            budget_percent=10.0,
            is_aggressive_mode=False,
            available_ram_gb=8.0,
        )

        import json

        result = json.loads(
            build_refinement_response(
                estimated_bytes=3 * 1024 * 1024 * 1024,
                memory_budget=budget,
                disk_budget_bytes=2 * 1024 * 1024 * 1024,
            )
        )

        assert result["estimated_size_mb"] == 3072.0
        assert result["memory_budget_mb"] == 512
        assert result["disk_budget_mb"] == 2048.0

    def test_decision_reason_is_descriptive(self):
        """ExecutionDecision reason explains the routing."""
        budget = MemoryBudget(
            budget_bytes=512 * 1024 * 1024,
            max_budget_mb=512,
            budget_percent=10.0,
            is_aggressive_mode=False,
            available_ram_gb=8.0,
        )
        analysis = _make_analysis(estimated_mb=50.0)

        decision = decide_execution_path(analysis, budget)

        assert "fits within" in decision.reason
        assert "RAM budget" in decision.reason
