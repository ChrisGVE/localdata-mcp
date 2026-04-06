"""Intelligent Token Management System for LocalData MCP.

This sub-package provides efficient DataFrame-based token counting that
enables LLMs to make intelligent decisions about data handling without
expensive token counting operations.

Key Features:
- DataFrame-structure-aware token estimation (numeric vs text columns)
- Sample-based analysis for performance (text columns sampled, not fully counted)
- Rich metadata generation for LLM decision-making
- Context window awareness for different model limits
- Chunking recommendations for large responses
- Integration with Query Analyzer and Streaming Pipeline
"""

from .manager import TokenManager, get_token_manager
from .models import (
    MODEL_CONTEXT_WINDOWS,
    TOKEN_RISK_THRESHOLDS,
    ChunkingRecommendation,
    ResponseMetadata,
    TokenEstimation,
)

__all__ = [
    "ChunkingRecommendation",
    "MODEL_CONTEXT_WINDOWS",
    "ResponseMetadata",
    "TOKEN_RISK_THRESHOLDS",
    "TokenEstimation",
    "TokenManager",
    "get_token_manager",
]
