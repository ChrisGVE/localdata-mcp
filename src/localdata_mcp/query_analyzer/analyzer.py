"""Core query analyzer with pre-execution sampling.

Contains the QueryAnalyzer class, the singleton accessor, and the
convenience ``analyze_query`` wrapper function.
"""

import hashlib
import logging
import re
import time
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import tiktoken
from sqlalchemy import text
from sqlalchemy.engine import Engine

from ..query_parser import SQLSecurityError, parse_and_validate_sql
from .estimation import (
    estimate_execution_time,
    estimate_memory_usage,
    estimate_token_count,
    fallback_row_estimation,
)
from .models import QueryAnalysis
from .recommendations import generate_recommendations

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """Intelligent query analysis system with pre-execution sampling.

    This analyzer performs proactive analysis using COUNT(*) and LIMIT 1
    sampling to understand query resource requirements before full
    execution.
    """

    def __init__(self) -> None:
        """Initialize the query analyzer with tiktoken encoding."""
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to initialize tiktoken encoder: {e}")
            self.encoding = None

    def analyze_query(self, query: str, engine: Engine, db_name: str) -> QueryAnalysis:
        """Analyze a query for resource requirements and complexity.

        Args:
            query: SQL query to analyze.
            engine: Database engine connection.
            db_name: Name of the database for logging.

        Returns:
            QueryAnalysis object with complete analysis results.

        Raises:
            SQLSecurityError: If query fails security validation.
            Exception: If analysis fails.
        """
        analysis_start = time.time()
        logger.info(f"Starting query analysis for database '{db_name}'")

        try:
            validated_query = parse_and_validate_sql(query)
            query_hash = self._generate_query_hash(query)

            complexity_analysis = self._analyze_query_complexity(validated_query)

            count_start = time.time()
            estimated_rows = self._get_row_count(validated_query, engine)
            count_time = time.time() - count_start

            sample_start = time.time()
            sample_row, column_info = self._get_sample_row(validated_query, engine)
            sample_time = time.time() - sample_start

            memory_analysis = estimate_memory_usage(
                estimated_rows,
                sample_row,
                column_info,
                engine=engine,
                query=validated_query,
            )

            token_analysis = estimate_token_count(
                estimated_rows, sample_row, column_info
            )

            timeout_analysis = estimate_execution_time(
                estimated_rows, complexity_analysis, engine
            )

            recommendations = generate_recommendations(
                estimated_rows,
                memory_analysis,
                token_analysis,
                timeout_analysis,
            )

            analysis_time = time.time() - analysis_start

            analysis = QueryAnalysis(
                query=query,
                query_hash=query_hash,
                validated_query=validated_query,
                estimated_rows=estimated_rows,
                count_query_time=count_time,
                sample_row=sample_row,
                sample_query_time=sample_time,
                column_count=column_info["count"],
                column_types=column_info["types"],
                estimated_row_size_bytes=memory_analysis["row_size"],
                estimated_total_memory_mb=memory_analysis["total_memory"],
                memory_risk_level=memory_analysis["risk_level"],
                estimated_tokens_per_row=token_analysis["tokens_per_row"],
                estimated_total_tokens=token_analysis["total_tokens"],
                token_risk_level=token_analysis["risk_level"],
                estimated_execution_time_seconds=timeout_analysis["estimated_time"],
                timeout_risk_level=timeout_analysis["risk_level"],
                complexity_score=complexity_analysis["score"],
                has_joins=complexity_analysis["has_joins"],
                has_aggregations=complexity_analysis["has_aggregations"],
                has_subqueries=complexity_analysis["has_subqueries"],
                has_window_functions=complexity_analysis["has_window_functions"],
                recommendations=recommendations["messages"],
                should_chunk=recommendations["should_chunk"],
                recommended_chunk_size=recommendations["chunk_size"],
                analysis_time_seconds=analysis_time,
                timestamp=time.time(),
            )

            logger.info(
                f"Query analysis completed in {analysis_time:.3f}s: "
                f"{estimated_rows} rows, "
                f"{memory_analysis['total_memory']:.1f}MB, "
                f"{token_analysis['total_tokens']} tokens"
            )

            return analysis

        except Exception as e:
            logger.error(f"Query analysis failed for database '{db_name}': {e}")
            raise

    def _estimate_memory_usage(self, *args, **kw):
        """Delegate to standalone function for backward compatibility."""
        return estimate_memory_usage(*args, **kw)

    def _estimate_token_count(self, *args, **kw):
        """Delegate to standalone function for backward compatibility."""
        return estimate_token_count(*args, **kw)

    def _estimate_execution_time(self, *args, **kw):
        """Delegate to standalone function for backward compatibility."""
        return estimate_execution_time(*args, **kw)

    def _generate_recommendations(self, *args, **kw):
        """Delegate to standalone function for backward compatibility."""
        return generate_recommendations(*args, **kw)

    def get_result_preview(self, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Get a preview of what the query results will look like.

        Args:
            analysis: QueryAnalysis object from analyze_query.

        Returns:
            Dictionary with result preview information.
        """
        preview: Dict[str, Any] = {
            "estimated_rows": analysis.estimated_rows,
            "estimated_columns": analysis.column_count,
            "column_types": analysis.column_types,
            "estimated_size_mb": analysis.estimated_total_memory_mb,
            "estimated_tokens": analysis.estimated_total_tokens,
            "risk_assessment": {
                "memory": analysis.memory_risk_level,
                "tokens": analysis.token_risk_level,
                "timeout": analysis.timeout_risk_level,
            },
            "recommendations": analysis.recommendations,
            "should_chunk": analysis.should_chunk,
            "chunk_size": analysis.recommended_chunk_size,
        }

        if analysis.sample_row is not None:
            preview["sample_structure"] = {
                col: f"<{dtype}>" for col, dtype in analysis.column_types.items()
            }

        return preview

    # -- Private helpers ---------------------------------------------------

    @staticmethod
    def _generate_query_hash(query: str) -> str:
        """Generate a short hash for the query."""
        return hashlib.md5(query.encode(), usedforsecurity=False).hexdigest()[:8]

    @staticmethod
    def _analyze_query_complexity(query: str) -> Dict[str, Any]:
        """Analyze query complexity and identify key features.

        Args:
            query: Validated SQL query string.

        Returns:
            Dictionary with complexity analysis results.
        """
        query_upper = query.upper()

        has_joins = bool(re.search(r"\bJOIN\b", query_upper))
        has_aggregations = bool(
            re.search(r"\b(COUNT|SUM|AVG|MIN|MAX|GROUP BY)\b", query_upper)
        )
        has_subqueries = bool(re.search(r"\(\s*SELECT\b", query_upper))
        has_window_functions = bool(re.search(r"\bOVER\s*\(", query_upper))

        score = 1  # Base score for simple SELECT

        if has_joins:
            join_count = len(re.findall(r"\bJOIN\b", query_upper))
            score += min(join_count * 2, 4)

        if has_aggregations:
            score += 2

        if has_subqueries:
            subquery_count = len(re.findall(r"\(\s*SELECT\b", query_upper))
            score += min(subquery_count * 2, 3)

        if has_window_functions:
            score += 2

        if re.search(r"\bUNION\b", query_upper):
            score += 1

        if re.search(r"\bORDER BY\b", query_upper):
            score += 1

        score = min(score, 10)

        return {
            "score": score,
            "has_joins": has_joins,
            "has_aggregations": has_aggregations,
            "has_subqueries": has_subqueries,
            "has_window_functions": has_window_functions,
        }

    @staticmethod
    def _get_row_count(query: str, engine: Engine) -> int:
        """Get estimated row count using COUNT(*) wrapper.

        Args:
            query: Validated SQL query.
            engine: Database engine connection.

        Returns:
            Estimated number of rows the query will return.
        """
        count_query = f"SELECT COUNT(*) as row_count FROM ({query}) as count_subquery"

        try:
            with engine.connect() as conn:
                result = conn.execute(text(count_query))
                row_count = result.scalar()
                return int(row_count) if row_count is not None else 0

        except Exception as e:
            logger.warning(f"Failed to get row count, using fallback estimation: {e}")
            return fallback_row_estimation(query, engine)

    @staticmethod
    def _get_sample_row(
        query: str, engine: Engine
    ) -> Tuple[Optional[pd.Series], Dict[str, Any]]:
        """Get sample row using LIMIT 1 to analyze structure.

        Args:
            query: Validated SQL query.
            engine: Database engine connection.

        Returns:
            Tuple of (sample_row_as_series, column_info_dict).
        """
        sample_query = f"SELECT * FROM ({query}) as sample_subquery LIMIT 1"

        try:
            with engine.connect() as conn:
                df = pd.read_sql_query(sample_query, conn)

                if df.empty:
                    return None, {"count": 0, "types": {}}

                sample_row = df.iloc[0]
                column_info = {
                    "count": len(df.columns),
                    "types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                }

                return sample_row, column_info

        except Exception as e:
            logger.warning(f"Failed to get sample row: {e}")
            return None, {"count": 0, "types": {}}


# -- Singleton and convenience wrapper ------------------------------------

_query_analyzer: Optional[QueryAnalyzer] = None


def get_query_analyzer() -> QueryAnalyzer:
    """Get the global QueryAnalyzer instance (singleton pattern).

    Returns:
        QueryAnalyzer instance.
    """
    global _query_analyzer
    if _query_analyzer is None:
        _query_analyzer = QueryAnalyzer()
    return _query_analyzer


def analyze_query(query: str, engine: Engine, db_name: str) -> QueryAnalysis:
    """Analyze query using the global analyzer instance.

    Args:
        query: SQL query to analyze.
        engine: Database engine connection.
        db_name: Database name for logging.

    Returns:
        QueryAnalysis object with complete analysis results.
    """
    analyzer = get_query_analyzer()
    return analyzer.analyze_query(query, engine, db_name)
