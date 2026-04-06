"""Query Analysis System for LocalData MCP.

This sub-package provides intelligent pre-query analysis using COUNT(*)
and LIMIT 1 sampling to estimate resource usage before executing queries.
This prevents memory overflows and provides metadata for LLM decision
making.

Key Features:
- Pre-execution sampling with COUNT(*) for row estimation
- LIMIT 1 sampling for row structure/size analysis
- Memory usage estimation with buffer factors
- Token count estimation using tiktoken
- Query timeout prediction based on complexity
- Integration with existing query execution pipeline
"""

from .analyzer import QueryAnalyzer, analyze_query, get_query_analyzer
from .models import QueryAnalysis

__all__ = [
    "QueryAnalysis",
    "QueryAnalyzer",
    "analyze_query",
    "get_query_analyzer",
]
