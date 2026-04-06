"""Memory-aware query execution decision logic.

Replaces the fixed 100-row threshold with a byte-based memory budget
that auto-calculates from available RAM. Provides three execution paths:

1. **In-memory**: estimated size fits in RAM budget
2. **Staging**: estimated size exceeds RAM but fits on disk
3. **Refinement**: estimated size exceeds both RAM and disk budgets
"""

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config_schemas import DiskBudgetConfig
from ..memory_budget import MemoryBudget
from ..query_analyzer.models import QueryAnalysis

logger = logging.getLogger(__name__)

# Refinement suggestions presented to the LLM agent
_REFINEMENT_SUGGESTIONS: List[str] = [
    "Add LIMIT clause",
    "Add WHERE clause",
    "Use aggregation (GROUP BY, COUNT, SUM)",
    "Select fewer columns",
]


@dataclass
class ExecutionDecision:
    """Result of the memory-aware execution routing decision."""

    path: str  # "in_memory", "staging", "refinement"
    estimated_bytes: int
    memory_budget: MemoryBudget
    disk_budget_bytes: Optional[int] = None
    reason: str = ""


def get_memory_budget() -> MemoryBudget:
    """Build a MemoryBudget from config, used after query analysis."""
    return MemoryBudget.from_config()


def get_disk_budget() -> int:
    """Calculate available disk budget in bytes.

    Returns the minimum of:
    - max_staging_size_mb from config (converted to bytes)
    - available disk space minus headroom
    - max_total_staging_mb from config (converted to bytes)
    """
    from ..config_manager import get_config_manager

    cfg = get_config_manager().get_disk_budget_config()
    max_single = cfg.max_staging_size_mb * 1024 * 1024
    max_total = cfg.max_total_staging_mb * 1024 * 1024

    try:
        usage = shutil.disk_usage(Path.home())
        available = usage.free - (cfg.headroom_mb * 1024 * 1024)
        available = max(available, 0)
    except OSError:
        logger.warning("Cannot read disk usage; using config limits only")
        available = max_single

    budget = min(max_single, available, max_total)
    logger.debug(
        "Disk budget: %d bytes (single=%d, available=%d, total=%d)",
        budget,
        max_single,
        available,
        max_total,
    )
    return budget


def _decide_in_memory(
    estimated_bytes: int, memory_budget: MemoryBudget
) -> Optional[ExecutionDecision]:
    """Return an in-memory decision if the estimate fits RAM."""
    if estimated_bytes >= memory_budget.budget_bytes:
        return None
    logger.info(
        "Query fits in memory budget: %d bytes < %d bytes",
        estimated_bytes,
        memory_budget.budget_bytes,
    )
    return ExecutionDecision(
        path="in_memory",
        estimated_bytes=estimated_bytes,
        memory_budget=memory_budget,
        reason=(
            f"Estimated {estimated_bytes} bytes fits within "
            f"{memory_budget.budget_bytes} byte RAM budget"
        ),
    )


def _decide_staging_or_refinement(
    estimated_bytes: int, memory_budget: MemoryBudget
) -> ExecutionDecision:
    """Return staging or refinement based on disk budget."""
    disk_budget_bytes = get_disk_budget()

    if estimated_bytes < disk_budget_bytes:
        logger.info(
            "Query exceeds RAM but fits disk: %d < %d bytes",
            estimated_bytes,
            disk_budget_bytes,
        )
        return ExecutionDecision(
            path="staging",
            estimated_bytes=estimated_bytes,
            memory_budget=memory_budget,
            disk_budget_bytes=disk_budget_bytes,
            reason=(
                f"Estimated {estimated_bytes} bytes exceeds RAM budget "
                f"({memory_budget.budget_bytes}) but fits disk "
                f"({disk_budget_bytes})"
            ),
        )

    logger.warning(
        "Query exceeds both RAM and disk budgets: %d bytes",
        estimated_bytes,
    )
    return ExecutionDecision(
        path="refinement",
        estimated_bytes=estimated_bytes,
        memory_budget=memory_budget,
        disk_budget_bytes=disk_budget_bytes,
        reason=(
            f"Estimated {estimated_bytes} bytes exceeds both RAM "
            f"({memory_budget.budget_bytes}) and disk ({disk_budget_bytes})"
        ),
    )


def decide_execution_path(
    query_analysis: Optional[QueryAnalysis],
    memory_budget: MemoryBudget,
) -> ExecutionDecision:
    """Choose in-memory, staging, or refinement based on budgets."""
    if query_analysis is None:
        return ExecutionDecision(
            path="in_memory",
            estimated_bytes=0,
            memory_budget=memory_budget,
            reason="No query analysis available; defaulting to in-memory",
        )

    estimated_bytes = int(query_analysis.estimated_total_memory_mb * 1024 * 1024)

    in_memory = _decide_in_memory(estimated_bytes, memory_budget)
    if in_memory is not None:
        return in_memory

    return _decide_staging_or_refinement(estimated_bytes, memory_budget)


def build_refinement_response(
    estimated_bytes: int,
    memory_budget: MemoryBudget,
    disk_budget_bytes: int,
) -> str:
    """Build a JSON response asking the LLM to refine the query."""
    estimated_mb = round(estimated_bytes / (1024 * 1024), 1)
    return json.dumps(
        {
            "error": False,
            "requires_refinement": True,
            "estimated_size_mb": estimated_mb,
            "memory_budget_mb": memory_budget.max_budget_mb,
            "disk_budget_mb": round(disk_budget_bytes / (1024 * 1024), 1),
            "is_aggressive_mode": memory_budget.is_aggressive_mode,
            "suggestions": list(_REFINEMENT_SUGGESTIONS),
        },
        indent=2,
    )


def calculate_dynamic_chunk_size(
    memory_budget: MemoryBudget,
    query_analysis: Optional[QueryAnalysis],
    fallback: int = 1000,
) -> int:
    """Calculate chunk size from memory budget and estimated row size.

    Targets using at most 10% of the memory budget per chunk.

    Args:
        memory_budget: Current RAM budget.
        query_analysis: Analysis with row size estimate.
        fallback: Returned when no analysis is available.

    Returns:
        Number of rows per chunk (at least 10).
    """
    if query_analysis is None or query_analysis.estimated_row_size_bytes <= 0:
        return fallback

    chunk_budget = memory_budget.budget_bytes * 0.10
    rows = int(chunk_budget / query_analysis.estimated_row_size_bytes)
    return max(rows, 10)


def build_memory_budget_metadata(budget: MemoryBudget) -> Dict[str, Any]:
    """Return a dict of memory budget info for streaming metadata."""
    return {
        "budget_bytes": budget.budget_bytes,
        "max_budget_mb": budget.max_budget_mb,
        "is_aggressive_mode": budget.is_aggressive_mode,
        "available_ram_gb": budget.available_ram_gb,
    }
