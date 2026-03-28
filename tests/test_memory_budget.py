"""Tests for MemoryBudget calculation and aggressive mode."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from localdata_mcp.memory_budget import MemoryBudget


def _fake_vmem(available_gb: float):
    """Return a mock virtual_memory() result."""
    available = int(available_gb * 1024**3)
    return SimpleNamespace(
        total=16 * 1024**3,
        available=available,
        percent=100 - (available_gb / 16) * 100,
        used=(16 - available_gb) * 1024**3,
        free=available,
    )


class TestMemoryBudgetCalculate:
    """Tests for MemoryBudget.calculate()."""

    @patch("localdata_mcp.memory_budget.PSUTIL_AVAILABLE", True)
    @patch("localdata_mcp.memory_budget.psutil")
    def test_calculate_normal_mode(self, mock_psutil):
        """With 8 GB available the budget should use normal-mode params."""
        mock_psutil.virtual_memory.return_value = _fake_vmem(8.0)

        budget = MemoryBudget.calculate()

        assert not budget.is_aggressive_mode
        assert budget.available_ram_gb == 8.0
        assert budget.max_budget_mb == 512
        assert budget.budget_percent == 10.0
        # 10% of 8 GB = 819 MB, capped at 512 MB
        assert budget.budget_bytes == 512 * 1024 * 1024

    @patch("localdata_mcp.memory_budget.PSUTIL_AVAILABLE", True)
    @patch("localdata_mcp.memory_budget.psutil")
    def test_calculate_aggressive_mode(self, mock_psutil):
        """With 0.5 GB available the budget must switch to aggressive."""
        mock_psutil.virtual_memory.return_value = _fake_vmem(0.5)

        budget = MemoryBudget.calculate()

        assert budget.is_aggressive_mode
        assert budget.available_ram_gb == 0.5
        assert budget.max_budget_mb == 128
        assert budget.budget_percent == 5.0
        # 5% of 0.5 GB ≈ 26 MB, below 128 MB cap
        expected = int(0.5 * 1024**3 * 0.05)
        assert budget.budget_bytes == expected

    @patch("localdata_mcp.memory_budget.PSUTIL_AVAILABLE", True)
    @patch("localdata_mcp.memory_budget.psutil")
    def test_calculate_threshold_boundary(self, mock_psutil):
        """Exactly 1.0 GB available should NOT be aggressive (not < 1.0)."""
        mock_psutil.virtual_memory.return_value = _fake_vmem(1.0)

        budget = MemoryBudget.calculate()

        assert not budget.is_aggressive_mode
        assert budget.available_ram_gb == 1.0

    @patch("localdata_mcp.memory_budget.PSUTIL_AVAILABLE", False)
    def test_calculate_no_psutil(self):
        """Without psutil, fall back to default max_budget_mb."""
        budget = MemoryBudget.calculate()

        assert not budget.is_aggressive_mode
        assert budget.available_ram_gb == 0.0
        assert budget.budget_bytes == 512 * 1024 * 1024
        assert budget.budget_percent == 10.0

    @patch("localdata_mcp.memory_budget.PSUTIL_AVAILABLE", True)
    @patch("localdata_mcp.memory_budget.psutil")
    def test_calculate_custom_config(self, mock_psutil):
        """Custom max_budget_mb and percent are respected."""
        mock_psutil.virtual_memory.return_value = _fake_vmem(4.0)

        budget = MemoryBudget.calculate(max_budget_mb=256, budget_percent=20)

        assert not budget.is_aggressive_mode
        assert budget.budget_percent == 20.0
        assert budget.max_budget_mb == 256
        # 20% of 4 GB = 819 MB, capped at 256 MB
        assert budget.budget_bytes == 256 * 1024 * 1024

    @patch("localdata_mcp.memory_budget.PSUTIL_AVAILABLE", True)
    @patch("localdata_mcp.memory_budget.psutil")
    def test_budget_capped_at_max(self, mock_psutil):
        """When available * pct exceeds cap, budget is capped."""
        mock_psutil.virtual_memory.return_value = _fake_vmem(16.0)

        budget = MemoryBudget.calculate(max_budget_mb=100, budget_percent=50)

        # 50% of 16 GB = 8 GB, far above 100 MB cap
        assert budget.budget_bytes == 100 * 1024 * 1024

    @patch("localdata_mcp.memory_budget.PSUTIL_AVAILABLE", True)
    @patch("localdata_mcp.memory_budget.psutil")
    def test_budget_below_cap_uses_percent(self, mock_psutil):
        """When available * pct is below cap, the percent-based value wins."""
        mock_psutil.virtual_memory.return_value = _fake_vmem(1.0)

        budget = MemoryBudget.calculate(max_budget_mb=512, budget_percent=5)

        # 5% of 1 GB ≈ 51 MB, well below 512 MB cap
        expected = int(1.0 * 1024**3 * 0.05)
        assert budget.budget_bytes == expected


class TestMemoryConfigAggressive:
    """Tests for aggressive fields on MemoryConfig."""

    def test_aggressive_config_fields_defaults(self):
        """Default aggressive values are valid."""
        from localdata_mcp.config_schemas import MemoryConfig

        cfg = MemoryConfig()
        assert cfg.aggressive_budget_percent == 5
        assert cfg.aggressive_max_mb == 128

    def test_aggressive_config_fields_custom(self):
        """Custom aggressive values are accepted."""
        from localdata_mcp.config_schemas import MemoryConfig

        cfg = MemoryConfig(aggressive_budget_percent=3, aggressive_max_mb=64)
        assert cfg.aggressive_budget_percent == 3
        assert cfg.aggressive_max_mb == 64

    def test_aggressive_percent_validation(self):
        """aggressive_budget_percent outside 1-100 raises."""
        from localdata_mcp.config_schemas import MemoryConfig

        with pytest.raises(ValueError, match="aggressive_budget_percent"):
            MemoryConfig(aggressive_budget_percent=0)

        with pytest.raises(ValueError, match="aggressive_budget_percent"):
            MemoryConfig(aggressive_budget_percent=101)

    def test_aggressive_max_mb_validation(self):
        """aggressive_max_mb <= 0 raises."""
        from localdata_mcp.config_schemas import MemoryConfig

        with pytest.raises(ValueError, match="aggressive_max_mb"):
            MemoryConfig(aggressive_max_mb=0)
