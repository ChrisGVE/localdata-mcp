"""Query Analysis System for LocalData MCP v1.3.1.

This module provides intelligent pre-query analysis using COUNT(*) and LIMIT 1 sampling
to estimate resource usage before executing queries. This prevents memory overflows 
and provides metadata for LLM decision making.

Key Features:
- Pre-execution sampling with COUNT(*) for row estimation
- LIMIT 1 sampling for row structure/size analysis  
- Memory usage estimation with buffer factors
- Token count estimation using tiktoken
- Query timeout prediction based on complexity
- Integration with existing query execution pipeline
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import tiktoken
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .query_parser import parse_and_validate_sql, SQLSecurityError
from .token_manager import get_token_manager

logger = logging.getLogger(__name__)


@dataclass
class QueryAnalysis:
    """Results of query analysis containing resource estimates and metadata."""
    
    # Basic query information
    query: str
    query_hash: str
    validated_query: str
    
    # Row count analysis from COUNT(*)
    estimated_rows: int
    count_query_time: float
    
    # Sample analysis from LIMIT 1
    sample_row: Optional[pd.Series]
    sample_query_time: float
    column_count: int
    column_types: Dict[str, str]
    
    # Memory estimation
    estimated_row_size_bytes: float
    estimated_total_memory_mb: float
    memory_risk_level: str  # 'low', 'medium', 'high', 'critical'
    
    # Token estimation
    estimated_tokens_per_row: int
    estimated_total_tokens: int
    token_risk_level: str  # 'low', 'medium', 'high', 'critical'
    
    # Timeout estimation
    estimated_execution_time_seconds: float
    timeout_risk_level: str  # 'low', 'medium', 'high', 'critical'
    
    # Query complexity analysis
    complexity_score: int  # 1-10 scale
    has_joins: bool
    has_aggregations: bool
    has_subqueries: bool
    has_window_functions: bool
    
    # Recommendations
    recommendations: List[str]
    should_chunk: bool
    recommended_chunk_size: Optional[int]
    
    # Analysis metadata
    analysis_time_seconds: float
    timestamp: float


class QueryAnalyzer:
    """Intelligent query analysis system with pre-execution sampling.
    
    This analyzer performs proactive analysis using COUNT(*) and LIMIT 1 sampling
    to understand query resource requirements before full execution.
    """
    
    # Memory risk thresholds (MB)
    MEMORY_THRESHOLDS = {
        'low': 10,      # < 10MB
        'medium': 50,   # 10-50MB  
        'high': 200,    # 50-200MB
        'critical': 500 # > 200MB
    }
    
    # Token risk thresholds
    TOKEN_THRESHOLDS = {
        'low': 1000,      # < 1K tokens
        'medium': 10000,  # 1K-10K tokens
        'high': 50000,    # 10K-50K tokens  
        'critical': 100000 # > 50K tokens
    }
    
    # Timeout risk thresholds (seconds)
    TIMEOUT_THRESHOLDS = {
        'low': 1,      # < 1 second
        'medium': 5,   # 1-5 seconds
        'high': 30,    # 5-30 seconds
        'critical': 60 # > 30 seconds
    }
    
    def __init__(self):
        """Initialize the query analyzer with tiktoken encoding."""
        try:
            # Initialize tiktoken encoder for token counting
            self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        except Exception as e:
            logger.warning(f"Failed to initialize tiktoken encoder: {e}")
            self.encoding = None
    
    def analyze_query(self, query: str, engine: Engine, db_name: str) -> QueryAnalysis:
        """Analyze a query for resource requirements and complexity.
        
        Args:
            query: SQL query to analyze
            engine: Database engine connection
            db_name: Name of the database for logging
            
        Returns:
            QueryAnalysis object with complete analysis results
            
        Raises:
            SQLSecurityError: If query fails security validation
            Exception: If analysis fails
        """
        analysis_start = time.time()
        logger.info(f"Starting query analysis for database '{db_name}'")
        
        try:
            # Step 1: Validate query security
            validated_query = parse_and_validate_sql(query)
            query_hash = self._generate_query_hash(query)
            
            # Step 2: Analyze query complexity
            complexity_analysis = self._analyze_query_complexity(validated_query)
            
            # Step 3: Execute COUNT(*) to get row count
            count_start = time.time()
            estimated_rows = self._get_row_count(validated_query, engine)
            count_time = time.time() - count_start
            
            # Step 4: Execute LIMIT 1 to sample row structure
            sample_start = time.time()
            sample_row, column_info = self._get_sample_row(validated_query, engine)
            sample_time = time.time() - sample_start
            
            # Step 5: Estimate memory usage
            memory_analysis = self._estimate_memory_usage(
                estimated_rows, sample_row, column_info
            )
            
            # Step 6: Estimate token count
            token_analysis = self._estimate_token_count(
                estimated_rows, sample_row, column_info
            )
            
            # Step 7: Estimate execution timeout
            timeout_analysis = self._estimate_execution_time(
                estimated_rows, complexity_analysis, engine
            )
            
            # Step 8: Generate recommendations
            recommendations = self._generate_recommendations(
                estimated_rows, memory_analysis, token_analysis, timeout_analysis
            )
            
            analysis_time = time.time() - analysis_start
            
            # Create final analysis object
            analysis = QueryAnalysis(
                query=query,
                query_hash=query_hash,
                validated_query=validated_query,
                estimated_rows=estimated_rows,
                count_query_time=count_time,
                sample_row=sample_row,
                sample_query_time=sample_time,
                column_count=column_info['count'],
                column_types=column_info['types'],
                estimated_row_size_bytes=memory_analysis['row_size'],
                estimated_total_memory_mb=memory_analysis['total_memory'],
                memory_risk_level=memory_analysis['risk_level'],
                estimated_tokens_per_row=token_analysis['tokens_per_row'],
                estimated_total_tokens=token_analysis['total_tokens'],
                token_risk_level=token_analysis['risk_level'],
                estimated_execution_time_seconds=timeout_analysis['estimated_time'],
                timeout_risk_level=timeout_analysis['risk_level'],
                complexity_score=complexity_analysis['score'],
                has_joins=complexity_analysis['has_joins'],
                has_aggregations=complexity_analysis['has_aggregations'],
                has_subqueries=complexity_analysis['has_subqueries'],
                has_window_functions=complexity_analysis['has_window_functions'],
                recommendations=recommendations['messages'],
                should_chunk=recommendations['should_chunk'],
                recommended_chunk_size=recommendations['chunk_size'],
                analysis_time_seconds=analysis_time,
                timestamp=time.time()
            )
            
            logger.info(f"Query analysis completed in {analysis_time:.3f}s: "
                       f"{estimated_rows} rows, {memory_analysis['total_memory']:.1f}MB, "
                       f"{token_analysis['total_tokens']} tokens")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Query analysis failed for database '{db_name}': {e}")
            raise
    
    def _generate_query_hash(self, query: str) -> str:
        """Generate a short hash for the query."""
        import hashlib
        return hashlib.md5(query.encode(), usedforsecurity=False).hexdigest()[:8]
    
    def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query complexity and identify key features.
        
        Args:
            query: Validated SQL query string
            
        Returns:
            Dictionary with complexity analysis results
        """
        query_upper = query.upper()
        
        # Detect query features
        has_joins = bool(re.search(r'\bJOIN\b', query_upper))
        has_aggregations = bool(re.search(r'\b(COUNT|SUM|AVG|MIN|MAX|GROUP BY)\b', query_upper))
        has_subqueries = bool(re.search(r'\(\s*SELECT\b', query_upper))
        has_window_functions = bool(re.search(r'\bOVER\s*\(', query_upper))
        
        # Calculate complexity score (1-10)
        score = 1  # Base score for simple SELECT
        
        if has_joins:
            # Count joins
            join_count = len(re.findall(r'\bJOIN\b', query_upper))
            score += min(join_count * 2, 4)  # Up to +4 for joins
        
        if has_aggregations:
            score += 2  # +2 for aggregations
        
        if has_subqueries:
            subquery_count = len(re.findall(r'\(\s*SELECT\b', query_upper))
            score += min(subquery_count * 2, 3)  # Up to +3 for subqueries
        
        if has_window_functions:
            score += 2  # +2 for window functions
        
        # Additional complexity indicators
        if re.search(r'\bUNION\b', query_upper):
            score += 1
        
        if re.search(r'\bORDER BY\b', query_upper):
            score += 1
        
        score = min(score, 10)  # Cap at 10
        
        return {
            'score': score,
            'has_joins': has_joins,
            'has_aggregations': has_aggregations,
            'has_subqueries': has_subqueries,
            'has_window_functions': has_window_functions
        }
    
    def _get_row_count(self, query: str, engine: Engine) -> int:
        """Get estimated row count using COUNT(*) wrapper.
        
        Args:
            query: Validated SQL query
            engine: Database engine connection
            
        Returns:
            Estimated number of rows the query will return
        """
        count_query = f"SELECT COUNT(*) as row_count FROM ({query}) as count_subquery"
        
        try:
            with engine.connect() as conn:
                result = conn.execute(text(count_query))
                row_count = result.scalar()
                return int(row_count) if row_count is not None else 0
                
        except Exception as e:
            logger.warning(f"Failed to get row count, using fallback estimation: {e}")
            # Fallback: try to estimate from EXPLAIN or use default
            return self._fallback_row_estimation(query, engine)
    
    def _get_sample_row(self, query: str, engine: Engine) -> Tuple[Optional[pd.Series], Dict[str, Any]]:
        """Get sample row using LIMIT 1 to analyze structure.
        
        Args:
            query: Validated SQL query
            engine: Database engine connection
            
        Returns:
            Tuple of (sample_row_as_series, column_info_dict)
        """
        sample_query = f"SELECT * FROM ({query}) as sample_subquery LIMIT 1"
        
        try:
            with engine.connect() as conn:
                df = pd.read_sql_query(sample_query, conn)
                
                if df.empty:
                    return None, {'count': 0, 'types': {}}
                
                sample_row = df.iloc[0]
                column_info = {
                    'count': len(df.columns),
                    'types': {col: str(dtype) for col, dtype in df.dtypes.items()}
                }
                
                return sample_row, column_info
                
        except Exception as e:
            logger.warning(f"Failed to get sample row: {e}")
            return None, {'count': 0, 'types': {}}
    
    def _estimate_memory_usage(self, row_count: int, sample_row: Optional[pd.Series], 
                             column_info: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate memory usage based on row count and sample data.
        
        Args:
            row_count: Estimated number of rows
            sample_row: Sample row data
            column_info: Column metadata
            
        Returns:
            Dictionary with memory usage estimates
        """
        if sample_row is None or row_count == 0:
            return {
                'row_size': 0,
                'total_memory': 0,
                'risk_level': 'low'
            }
        
        # Calculate sample row size in bytes
        row_size_bytes = 0
        
        for col_name, value in sample_row.items():
            if pd.isna(value):
                row_size_bytes += 8  # NULL value overhead
            elif isinstance(value, str):
                row_size_bytes += len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                row_size_bytes += 8  # Numeric types
            elif isinstance(value, bool):
                row_size_bytes += 1  # Boolean
            else:
                # Complex types - estimate based on string representation
                row_size_bytes += len(str(value).encode('utf-8'))
        
        # Add DataFrame overhead (index, column headers, etc.)
        overhead_per_row = 24  # Estimated pandas overhead per row
        total_row_size = row_size_bytes + overhead_per_row
        
        # Apply buffer factor (1.5x) for memory allocation overhead
        buffer_factor = 1.5
        estimated_memory_bytes = row_count * total_row_size * buffer_factor
        estimated_memory_mb = estimated_memory_bytes / (1024 * 1024)
        
        # Determine risk level
        risk_level = 'low'
        for level, threshold in self.MEMORY_THRESHOLDS.items():
            if estimated_memory_mb > threshold:
                risk_level = level
        
        return {
            'row_size': total_row_size,
            'total_memory': estimated_memory_mb,
            'risk_level': risk_level
        }
    
    def _estimate_token_count(self, row_count: int, sample_row: Optional[pd.Series],
                            column_info: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate token count using enhanced TokenManager.
        
        Args:
            row_count: Estimated number of rows
            sample_row: Sample row data
            column_info: Column metadata
            
        Returns:
            Dictionary with token count estimates
        """
        if sample_row is None or row_count == 0:
            return {
                'tokens_per_row': 0,
                'total_tokens': 0,
                'risk_level': 'low'
            }
        
        # Create DataFrame from sample for TokenManager
        sample_df = pd.DataFrame([sample_row])
        
        # Use TokenManager for intelligent estimation
        token_manager = get_token_manager()
        estimation = token_manager.estimate_tokens_for_query_result(row_count, sample_df)
        
        return {
            'tokens_per_row': estimation.tokens_per_row,
            'total_tokens': estimation.total_tokens,
            'risk_level': estimation.risk_level
        }
    
    def _estimate_execution_time(self, row_count: int, complexity_analysis: Dict[str, Any],
                               engine: Engine) -> Dict[str, Any]:
        """Estimate query execution time based on complexity and row count.
        
        Args:
            row_count: Estimated number of rows
            complexity_analysis: Query complexity metadata
            engine: Database engine connection
            
        Returns:
            Dictionary with execution time estimates
        """
        # Base time estimation (very rough heuristics)
        base_time = 0.001  # 1ms base time
        
        # Time per row based on complexity
        if complexity_analysis['score'] <= 3:
            time_per_1k_rows = 0.01  # Simple queries
        elif complexity_analysis['score'] <= 6:
            time_per_1k_rows = 0.05  # Medium complexity
        else:
            time_per_1k_rows = 0.1   # High complexity
        
        # Additional time for specific features
        complexity_multiplier = 1.0
        
        if complexity_analysis['has_joins']:
            complexity_multiplier *= 1.5
        
        if complexity_analysis['has_aggregations']:
            complexity_multiplier *= 1.3
        
        if complexity_analysis['has_subqueries']:
            complexity_multiplier *= 1.4
        
        if complexity_analysis['has_window_functions']:
            complexity_multiplier *= 1.6
        
        # Calculate estimated time
        estimated_time = base_time + (row_count / 1000.0) * time_per_1k_rows * complexity_multiplier
        
        # Determine risk level
        risk_level = 'low'
        for level, threshold in self.TIMEOUT_THRESHOLDS.items():
            if estimated_time > threshold:
                risk_level = level
        
        return {
            'estimated_time': estimated_time,
            'risk_level': risk_level
        }
    
    def _generate_recommendations(self, row_count: int, memory_analysis: Dict[str, Any],
                                token_analysis: Dict[str, Any], timeout_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations based on analysis results.
        
        Args:
            row_count: Estimated row count
            memory_analysis: Memory usage analysis
            token_analysis: Token count analysis
            timeout_analysis: Execution time analysis
            
        Returns:
            Dictionary with recommendations and chunking suggestions
        """
        recommendations = []
        should_chunk = False
        chunk_size = None
        
        # Memory-based recommendations
        if memory_analysis['risk_level'] in ['high', 'critical']:
            should_chunk = True
            recommendations.append(
                f"High memory usage expected ({memory_analysis['total_memory']:.1f}MB). "
                "Consider chunking the query results."
            )
            
        # Token-based recommendations
        if token_analysis['risk_level'] in ['high', 'critical']:
            should_chunk = True
            recommendations.append(
                f"Large token count expected ({token_analysis['total_tokens']:,} tokens). "
                "Consider processing results in chunks to avoid context limits."
            )
        
        # Timeout-based recommendations
        if timeout_analysis['risk_level'] in ['high', 'critical']:
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
            if memory_analysis['risk_level'] == 'critical':
                chunk_size = 50
            elif memory_analysis['risk_level'] == 'high':
                chunk_size = 100
            elif token_analysis['risk_level'] in ['high', 'critical']:
                chunk_size = 200
            else:
                chunk_size = min(500, max(100, row_count // 10))
        
        # Add general recommendations
        if not recommendations:
            recommendations.append("Query appears safe to execute without chunking.")
        
        return {
            'messages': recommendations,
            'should_chunk': should_chunk,
            'chunk_size': chunk_size
        }
    
    def _fallback_row_estimation(self, query: str, engine: Engine) -> int:
        """Fallback row estimation when COUNT(*) fails.
        
        Args:
            query: SQL query
            engine: Database engine connection
            
        Returns:
            Estimated row count (conservative estimate)
        """
        try:
            # Try EXPLAIN if supported
            explain_query = f"EXPLAIN {query}"
            with engine.connect() as conn:
                result = conn.execute(text(explain_query))
                # This is database-specific - for now, return conservative estimate
                return 1000  # Conservative default
                
        except Exception:
            # Last resort: return very conservative estimate
            logger.warning("All row estimation methods failed, using conservative default")
            return 100
    
    def get_result_preview(self, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Get a preview of what the query results will look like.
        
        Args:
            analysis: QueryAnalysis object from analyze_query
            
        Returns:
            Dictionary with result preview information
        """
        preview = {
            'estimated_rows': analysis.estimated_rows,
            'estimated_columns': analysis.column_count,
            'column_types': analysis.column_types,
            'estimated_size_mb': analysis.estimated_total_memory_mb,
            'estimated_tokens': analysis.estimated_total_tokens,
            'risk_assessment': {
                'memory': analysis.memory_risk_level,
                'tokens': analysis.token_risk_level,
                'timeout': analysis.timeout_risk_level
            },
            'recommendations': analysis.recommendations,
            'should_chunk': analysis.should_chunk,
            'chunk_size': analysis.recommended_chunk_size
        }
        
        if analysis.sample_row is not None:
            # Include sample data structure (without actual values for privacy)
            preview['sample_structure'] = {
                col: f"<{dtype}>" for col, dtype in analysis.column_types.items()
            }
        
        return preview


# Global analyzer instance for efficient reuse
_query_analyzer = None


def get_query_analyzer() -> QueryAnalyzer:
    """Get the global QueryAnalyzer instance (singleton pattern).
    
    Returns:
        QueryAnalyzer instance
    """
    global _query_analyzer
    if _query_analyzer is None:
        _query_analyzer = QueryAnalyzer()
    return _query_analyzer


def analyze_query(query: str, engine: Engine, db_name: str) -> QueryAnalysis:
    """Analyze query using the global analyzer instance.
    
    Args:
        query: SQL query to analyze
        engine: Database engine connection
        db_name: Database name for logging
        
    Returns:
        QueryAnalysis object with complete analysis results
    """
    analyzer = get_query_analyzer()
    return analyzer.analyze_query(query, engine, db_name)