"""Memory budget calculation for query execution decisions.

Determines how much RAM a query is allowed to consume, switching to
aggressive (smaller) budgets when available memory drops below a
configurable threshold.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class MemoryBudget:
    """Calculated memory budget for query execution."""

    budget_bytes: int
    max_budget_mb: int
    budget_percent: float
    is_aggressive_mode: bool
    available_ram_gb: float

    @classmethod
    def calculate(
        cls,
        max_budget_mb: int = 512,
        budget_percent: int = 10,
        low_memory_threshold_gb: float = 1.0,
        aggressive_percent: int = 5,
        aggressive_max_mb: int = 128,
    ) -> "MemoryBudget":
        """Calculate a memory budget based on current system resources.

        When *psutil* is not installed the budget falls back to
        ``max_budget_mb`` without querying the OS.

        Args:
            max_budget_mb: Hard cap in normal mode (megabytes).
            budget_percent: Percentage of available RAM in normal mode.
            low_memory_threshold_gb: Available RAM below this triggers
                aggressive mode.
            aggressive_percent: Percentage of available RAM in aggressive
                mode.
            aggressive_max_mb: Hard cap in aggressive mode (megabytes).

        Returns:
            A populated ``MemoryBudget`` instance.
        """
        if not PSUTIL_AVAILABLE:
            return cls(
                budget_bytes=max_budget_mb * 1024 * 1024,
                max_budget_mb=max_budget_mb,
                budget_percent=float(budget_percent),
                is_aggressive_mode=False,
                available_ram_gb=0.0,
            )

        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        aggressive = available_gb < low_memory_threshold_gb

        if aggressive:
            pct = aggressive_percent / 100.0
            cap = aggressive_max_mb
            logger.info(
                "Low memory (%.2f GB available) — using aggressive budget "
                "(%d%%, cap %d MB)",
                available_gb,
                aggressive_percent,
                aggressive_max_mb,
            )
        else:
            pct = budget_percent / 100.0
            cap = max_budget_mb

        budget = int(min(mem.available * pct, cap * 1024 * 1024))
        return cls(
            budget_bytes=budget,
            max_budget_mb=cap,
            budget_percent=pct * 100,
            is_aggressive_mode=aggressive,
            available_ram_gb=round(available_gb, 2),
        )

    @classmethod
    def from_config(cls) -> "MemoryBudget":
        """Build a ``MemoryBudget`` using values from the config system."""
        from .config_manager import get_config_manager

        mgr = get_config_manager()
        cfg = mgr.get_memory_config()
        return cls.calculate(
            max_budget_mb=cfg.max_budget_mb,
            budget_percent=cfg.budget_percent,
            low_memory_threshold_gb=cfg.low_memory_threshold_gb,
            aggressive_percent=cfg.aggressive_budget_percent,
            aggressive_max_mb=cfg.aggressive_max_mb,
        )
