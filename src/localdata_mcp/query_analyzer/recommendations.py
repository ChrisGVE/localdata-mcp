"""Recommendation generation for query analysis.

Provides functions that generate actionable recommendations based on
memory, token, and timeout analysis results.
"""

from typing import Any, Dict


def generate_recommendations(
    row_count: int,
    memory_analysis: Dict[str, Any],
    token_analysis: Dict[str, Any],
    timeout_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate recommendations based on analysis results.

    Args:
        row_count: Estimated row count.
        memory_analysis: Memory usage analysis.
        token_analysis: Token count analysis.
        timeout_analysis: Execution time analysis.

    Returns:
        Dictionary with recommendations and chunking suggestions.
    """
    recommendations: list[str] = []
    should_chunk = False
    chunk_size = None

    # Memory-based recommendations
    if memory_analysis["risk_level"] in ["high", "critical"]:
        should_chunk = True
        recommendations.append(
            f"High memory usage expected ({memory_analysis['total_memory']:.1f}MB). "
            "Consider chunking the query results."
        )

    # Token-based recommendations
    if token_analysis["risk_level"] in ["high", "critical"]:
        should_chunk = True
        recommendations.append(
            f"Large token count expected ({token_analysis['total_tokens']:,} tokens). "
            "Consider processing results in chunks to avoid context limits."
        )

    # Timeout-based recommendations
    if timeout_analysis["risk_level"] in ["high", "critical"]:
        recommendations.append(
            f"Long execution time expected ({timeout_analysis['estimated_time']:.1f}s). "
            "Consider adding LIMIT clause or optimizing the query."
        )

    # Row count recommendations
    if row_count > 1000:
        should_chunk = True
        recommendations.append(
            f"Large result set ({row_count:,} rows). Automatic chunking recommended."
        )

    # Determine chunk size if needed
    if should_chunk:
        chunk_size = _determine_chunk_size(row_count, memory_analysis, token_analysis)

    # Add general recommendations
    if not recommendations:
        recommendations.append("Query appears safe to execute without chunking.")

    return {
        "messages": recommendations,
        "should_chunk": should_chunk,
        "chunk_size": chunk_size,
    }


def _determine_chunk_size(
    row_count: int,
    memory_analysis: Dict[str, Any],
    token_analysis: Dict[str, Any],
) -> int:
    """Determine the optimal chunk size based on risk levels.

    Args:
        row_count: Estimated row count.
        memory_analysis: Memory usage analysis.
        token_analysis: Token count analysis.

    Returns:
        Recommended chunk size.
    """
    if memory_analysis["risk_level"] == "critical":
        return 50
    elif memory_analysis["risk_level"] == "high":
        return 100
    elif token_analysis["risk_level"] in ["high", "critical"]:
        return 200
    else:
        return min(500, max(100, row_count // 10))
