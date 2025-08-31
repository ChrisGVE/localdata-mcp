"""Enhanced Response Metadata and LLM Communication Protocol for LocalData MCP v1.3.1.

This module provides comprehensive response metadata that enables intelligent LLM decision-making
about large datasets, including progressive data loading, query complexity analysis, and rich
statistical summaries.

Key Features:
- Enhanced ResponseMetadata with query complexity scoring and data quality metrics
- LLMCommunicationProtocol for progressive data loading and cancellation
- Statistical summaries and schema information for data understanding
- Intelligent chunk recommendations based on LLM context windows
- Data preview capabilities with sample rows
- Query result caching with modification-based invalidation
"""

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple, Iterator
from enum import Enum
from pathlib import Path

import pandas as pd
import numpy as np

# Import existing components
from .token_manager import get_token_manager, TokenEstimation, ChunkingRecommendation
from .query_analyzer import QueryAnalysis

logger = logging.getLogger(__name__)


class DataQualityLevel(Enum):
    """Data quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"  
    FAIR = "fair"
    POOR = "poor"
    UNKNOWN = "unknown"


class QueryComplexity(Enum):
    """Query complexity levels for processing estimation."""
    SIMPLE = "simple"      # Basic SELECT, small results
    MODERATE = "moderate"  # JOINs, aggregations, medium results
    COMPLEX = "complex"    # Multi-table JOINs, subqueries, large results
    INTENSIVE = "intensive" # Complex analytics, very large results


@dataclass
class StatisticalSummary:
    """Statistical summary for a dataset or column."""
    
    # Basic statistics
    total_rows: int
    non_null_rows: int
    null_percentage: float
    
    # Data type distribution
    data_types: Dict[str, int]  # dtype -> count of columns
    
    # Numeric column statistics (if any numeric columns exist)
    numeric_summary: Optional[Dict[str, Any]] = None
    
    # Text column statistics (if any text columns exist)  
    text_summary: Optional[Dict[str, Any]] = None
    
    # Sample data preview
    sample_rows: List[Dict[str, Any]] = field(default_factory=list)
    
    # Data quality indicators
    duplicate_rows: int = 0
    duplicate_percentage: float = 0.0
    
    # Memory and storage estimates
    estimated_memory_usage_mb: float = 0.0


@dataclass
class SchemaInformation:
    """Detailed schema information for data understanding."""
    
    # Column information
    columns: List[Dict[str, Any]]  # name, dtype, nullable, sample_values
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[Dict[str, str]] = field(default_factory=list)
    
    # Index information
    indexes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Constraints
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    
    # Table metadata
    table_name: Optional[str] = None
    last_modified: Optional[float] = None


@dataclass
class ChunkAvailability:
    """Information about available data chunks for progressive loading."""
    
    total_chunks: int
    available_chunks: List[int]  # Which chunks are ready/cached
    chunk_size: int  # Rows per chunk
    chunk_overlap: int = 0
    
    # Chunk metadata
    chunk_metadata: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    # Loading status
    loading_status: Dict[int, str] = field(default_factory=dict)  # chunk_id -> status
    estimated_load_time: Dict[int, float] = field(default_factory=dict)


@dataclass
class DataQualityMetrics:
    """Comprehensive data quality assessment."""
    
    overall_quality: DataQualityLevel
    quality_score: float  # 0.0 to 1.0
    
    # Specific quality dimensions
    completeness: float  # Percentage of non-null values
    consistency: float   # Data format consistency
    validity: float      # Values within expected ranges/formats
    accuracy: float      # Estimated accuracy (based on patterns)
    
    # Quality issues detected
    issues: List[Dict[str, Any]] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)


@dataclass 
class EnhancedResponseMetadata:
    """Comprehensive response metadata for intelligent LLM decision-making."""
    
    # Basic response information
    query_id: str
    timestamp: float
    
    # Enhanced size and complexity estimates
    query_complexity_score: float  # 0.0 to 1.0
    query_complexity_level: QueryComplexity
    estimated_processing_time: float  # Seconds
    memory_footprint: float  # MB
    
    # Token and response size analysis (from existing TokenManager)
    token_estimation: TokenEstimation
    
    # Data characteristics and quality
    statistical_summary: StatisticalSummary
    data_quality_metrics: DataQualityMetrics
    schema_information: SchemaInformation
    
    # Progressive loading capabilities
    chunk_availability: ChunkAvailability
    supports_streaming: bool
    supports_cancellation: bool
    
    # Caching information
    is_cached: bool = False
    cache_key: Optional[str] = None
    cache_expiry: Optional[float] = None
    
    # LLM guidance
    recommended_action: str = "proceed"  # proceed, chunk, sample, cancel
    action_rationale: str = ""
    llm_friendly_summary: str = ""


class LLMCommunicationProtocol:
    """Protocol for progressive data interaction and intelligent communication with LLMs."""
    
    def __init__(self, response_metadata: EnhancedResponseMetadata, 
                 data_source: Any = None):
        """Initialize the communication protocol.
        
        Args:
            response_metadata: Comprehensive metadata about the response
            data_source: Source of data (QueryBuffer, file path, database connection, etc.)
        """
        self.metadata = response_metadata
        self.data_source = data_source
        self.cancelled = False
        self.active_chunks: Dict[int, pd.DataFrame] = {}
        
        logger.debug(f"Initialized LLM communication protocol for query {response_metadata.query_id}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary suitable for LLM understanding.
        
        Returns:
            Dictionary with key information for LLM decision-making
        """
        summary = {
            "query_id": self.metadata.query_id,
            "data_overview": {
                "total_rows": self.metadata.statistical_summary.total_rows,
                "columns": len(self.metadata.schema_information.columns),
                "estimated_tokens": self.metadata.token_estimation.total_tokens,
                "memory_footprint_mb": self.metadata.memory_footprint,
                "data_quality": self.metadata.data_quality_metrics.overall_quality.value
            },
            "complexity_assessment": {
                "query_complexity": self.metadata.query_complexity_level.value,
                "complexity_score": self.metadata.query_complexity_score,
                "processing_time_estimate": self.metadata.estimated_processing_time
            },
            "loading_options": {
                "supports_chunking": self.metadata.chunk_availability.total_chunks > 1,
                "supports_streaming": self.metadata.supports_streaming,
                "supports_cancellation": self.metadata.supports_cancellation,
                "recommended_action": self.metadata.recommended_action
            },
            "sample_data": self.metadata.statistical_summary.sample_rows[:3],  # First 3 rows
            "schema_preview": [
                {
                    "column": col["name"], 
                    "type": col["dtype"],
                    "sample_values": col.get("sample_values", [])[:3]
                }
                for col in self.metadata.schema_information.columns[:10]  # First 10 columns
            ],
            "recommendations": {
                "action": self.metadata.recommended_action,
                "rationale": self.metadata.action_rationale,
                "chunking_strategy": self.metadata.token_estimation.recommended_chunk_size,
                "quality_recommendations": self.metadata.data_quality_metrics.recommendations[:3]
            }
        }
        
        return summary
    
    def request_chunk(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """Request a specific chunk of data.
        
        Args:
            chunk_id: ID of the chunk to retrieve
            
        Returns:
            Chunk data with metadata, or None if chunk not available
        """
        if self.cancelled:
            logger.warning(f"Cannot request chunk {chunk_id}: operation cancelled")
            return None
        
        if chunk_id not in self.metadata.chunk_availability.available_chunks:
            logger.warning(f"Chunk {chunk_id} not available")
            return None
        
        # Simulate chunk loading (in real implementation, this would fetch from data_source)
        logger.info(f"Loading chunk {chunk_id} for query {self.metadata.query_id}")
        
        try:
            # In real implementation, this would load the actual chunk from data source
            chunk_data = self._load_chunk_from_source(chunk_id)
            
            if chunk_data is not None:
                self.active_chunks[chunk_id] = chunk_data
                
                # Generate token estimation for this chunk
                token_manager = get_token_manager()
                chunk_token_estimate = token_manager.estimate_tokens_from_dataframe(chunk_data)
                
                return {
                    "chunk_id": chunk_id,
                    "data": chunk_data.to_dict(orient='records'),
                    "metadata": {
                        "rows": len(chunk_data),
                        "columns": len(chunk_data.columns),
                        "estimated_tokens": chunk_token_estimate.total_tokens,
                        "memory_mb": chunk_data.memory_usage(deep=True).sum() / (1024 * 1024),
                        "chunk_size": self.metadata.chunk_availability.chunk_size,
                        "total_chunks": self.metadata.chunk_availability.total_chunks
                    }
                }
        except Exception as e:
            logger.error(f"Error loading chunk {chunk_id}: {e}")
            return None
        
        return None
    
    def request_multiple_chunks(self, chunk_ids: List[int]) -> Dict[int, Optional[Dict[str, Any]]]:
        """Request multiple chunks efficiently.
        
        Args:
            chunk_ids: List of chunk IDs to retrieve
            
        Returns:
            Dictionary mapping chunk_id to chunk data (or None if failed)
        """
        results = {}
        
        for chunk_id in chunk_ids:
            if self.cancelled:
                break
            results[chunk_id] = self.request_chunk(chunk_id)
        
        return results
    
    def cancel_operation(self, reason: str = "User requested") -> bool:
        """Cancel the ongoing operation.
        
        Args:
            reason: Reason for cancellation
            
        Returns:
            True if cancellation was successful
        """
        if not self.metadata.supports_cancellation:
            logger.warning(f"Operation {self.metadata.query_id} does not support cancellation")
            return False
        
        self.cancelled = True
        logger.info(f"Operation {self.metadata.query_id} cancelled: {reason}")
        
        # Clean up active chunks
        self.active_chunks.clear()
        
        return True
    
    def get_schema_details(self) -> Dict[str, Any]:
        """Get detailed schema information for data understanding.
        
        Returns:
            Detailed schema information
        """
        return {
            "columns": self.metadata.schema_information.columns,
            "primary_keys": self.metadata.schema_information.primary_keys,
            "foreign_keys": self.metadata.schema_information.foreign_keys,
            "indexes": self.metadata.schema_information.indexes,
            "constraints": self.metadata.schema_information.constraints,
            "table_metadata": {
                "name": self.metadata.schema_information.table_name,
                "last_modified": self.metadata.schema_information.last_modified
            }
        }
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive data quality assessment.
        
        Returns:
            Data quality report with metrics and recommendations
        """
        return {
            "overall_quality": self.metadata.data_quality_metrics.overall_quality.value,
            "quality_score": self.metadata.data_quality_metrics.quality_score,
            "dimensions": {
                "completeness": self.metadata.data_quality_metrics.completeness,
                "consistency": self.metadata.data_quality_metrics.consistency,
                "validity": self.metadata.data_quality_metrics.validity,
                "accuracy": self.metadata.data_quality_metrics.accuracy
            },
            "issues": self.metadata.data_quality_metrics.issues,
            "recommendations": self.metadata.data_quality_metrics.recommendations,
            "statistical_summary": {
                "null_percentage": self.metadata.statistical_summary.null_percentage,
                "duplicate_percentage": self.metadata.statistical_summary.duplicate_percentage,
                "data_types": self.metadata.statistical_summary.data_types
            }
        }
    
    def _load_chunk_from_source(self, chunk_id: int) -> Optional[pd.DataFrame]:
        """Load a chunk from the data source.
        
        This is a placeholder implementation. In practice, this would:
        - Load from QueryBuffer if data is cached
        - Execute chunked query if loading from database
        - Read chunk from file if data source is a file
        
        Args:
            chunk_id: ID of chunk to load
            
        Returns:
            DataFrame chunk or None if loading failed
        """
        # Placeholder implementation
        # In real implementation, this would interact with the actual data source
        logger.debug(f"Loading chunk {chunk_id} from data source")
        
        # For now, return None to indicate chunk loading needs to be implemented
        # based on the specific data source type
        return None


class ResponseMetadataGenerator:
    """Factory for generating enhanced response metadata."""
    
    def __init__(self):
        self.token_manager = get_token_manager()
        self._cache: Dict[str, Tuple[EnhancedResponseMetadata, float]] = {}
        
    def generate_metadata(self, 
                         query_id: str,
                         df: pd.DataFrame, 
                         query: str,
                         query_analysis: Optional[QueryAnalysis] = None,
                         db_name: Optional[str] = None) -> EnhancedResponseMetadata:
        """Generate comprehensive response metadata for a dataset.
        
        Args:
            query_id: Unique identifier for the query
            df: DataFrame to analyze
            query: Original SQL query
            query_analysis: Pre-computed query analysis
            db_name: Database name
            
        Returns:
            Comprehensive response metadata
        """
        # Check cache first
        cache_key = self._generate_cache_key(query_id, query, df)
        if cache_key in self._cache:
            cached_metadata, cache_time = self._cache[cache_key]
            # Cache valid for 5 minutes
            if time.time() - cache_time < 300:
                logger.debug(f"Using cached metadata for query {query_id}")
                return cached_metadata
        
        # Generate fresh metadata
        logger.info(f"Generating enhanced response metadata for query {query_id}")
        
        # Get token estimation
        token_estimation = self.token_manager.estimate_tokens_from_dataframe(df)
        
        # Generate statistical summary
        statistical_summary = self._generate_statistical_summary(df)
        
        # Assess data quality
        data_quality = self._assess_data_quality(df)
        
        # Generate schema information
        schema_info = self._generate_schema_information(df, db_name)
        
        # Determine query complexity
        complexity_level, complexity_score = self._assess_query_complexity(
            query, df, query_analysis
        )
        
        # Estimate processing time
        processing_time = self._estimate_processing_time(
            df, complexity_level, token_estimation
        )
        
        # Calculate memory footprint
        memory_footprint = self._calculate_memory_footprint(df, token_estimation)
        
        # Generate chunk availability
        chunk_availability = self._generate_chunk_availability(df, token_estimation)
        
        # Determine recommended action
        recommended_action, action_rationale = self._determine_recommended_action(
            token_estimation, data_quality, complexity_level, len(df)
        )
        
        # Generate LLM-friendly summary
        llm_summary = self._generate_llm_friendly_summary(
            df, token_estimation, data_quality, complexity_level
        )
        
        # Create comprehensive metadata
        metadata = EnhancedResponseMetadata(
            query_id=query_id,
            timestamp=time.time(),
            query_complexity_score=complexity_score,
            query_complexity_level=complexity_level,
            estimated_processing_time=processing_time,
            memory_footprint=memory_footprint,
            token_estimation=token_estimation,
            statistical_summary=statistical_summary,
            data_quality_metrics=data_quality,
            schema_information=schema_info,
            chunk_availability=chunk_availability,
            supports_streaming=len(df) > 1000,  # Streaming for large datasets
            supports_cancellation=True,  # Always support cancellation
            cache_key=cache_key,
            recommended_action=recommended_action,
            action_rationale=action_rationale,
            llm_friendly_summary=llm_summary
        )
        
        # Cache the result
        self._cache[cache_key] = (metadata, time.time())
        
        logger.info(f"Generated metadata for {len(df)} rows, {len(df.columns)} columns, "
                   f"complexity: {complexity_level.value}, quality: {data_quality.overall_quality.value}")
        
        return metadata
    
    def _generate_statistical_summary(self, df: pd.DataFrame) -> StatisticalSummary:
        """Generate comprehensive statistical summary."""
        if df.empty:
            return StatisticalSummary(
                total_rows=0,
                non_null_rows=0,
                null_percentage=0.0,
                data_types={}
            )
        
        # Basic statistics
        total_rows = len(df)
        non_null_rows = len(df.dropna())
        null_percentage = ((total_rows - non_null_rows) / total_rows * 100) if total_rows > 0 else 0.0
        
        # Data type distribution
        data_types = df.dtypes.value_counts().to_dict()
        data_types = {str(k): int(v) for k, v in data_types.items()}
        
        # Numeric summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_summary = None
        if numeric_cols:
            numeric_summary = {
                "columns": numeric_cols,
                "statistics": df[numeric_cols].describe().to_dict()
            }
        
        # Text summary
        text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        text_summary = None
        if text_cols:
            text_summary = {
                "columns": text_cols,
                "avg_length": {col: df[col].astype(str).str.len().mean() for col in text_cols[:5]},
                "unique_values": {col: df[col].nunique() for col in text_cols[:5]}
            }
        
        # Sample rows (first 5 rows)
        sample_rows = df.head(5).to_dict(orient='records')
        
        # Duplicate analysis
        duplicate_rows = len(df) - len(df.drop_duplicates())
        duplicate_percentage = (duplicate_rows / len(df) * 100) if len(df) > 0 else 0.0
        
        # Memory estimate
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        return StatisticalSummary(
            total_rows=total_rows,
            non_null_rows=non_null_rows,
            null_percentage=null_percentage,
            data_types=data_types,
            numeric_summary=numeric_summary,
            text_summary=text_summary,
            sample_rows=sample_rows,
            duplicate_rows=duplicate_rows,
            duplicate_percentage=duplicate_percentage,
            estimated_memory_usage_mb=memory_mb
        )
    
    def _assess_data_quality(self, df: pd.DataFrame) -> DataQualityMetrics:
        """Assess comprehensive data quality metrics."""
        if df.empty:
            return DataQualityMetrics(
                overall_quality=DataQualityLevel.UNKNOWN,
                quality_score=0.0,
                completeness=0.0,
                consistency=0.0,
                validity=0.0,
                accuracy=0.0
            )
        
        # Completeness: percentage of non-null values
        total_values = df.size
        non_null_values = df.count().sum()
        completeness = (non_null_values / total_values) if total_values > 0 else 0.0
        
        # Consistency: data format consistency within columns
        consistency_scores = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check format consistency for text columns
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    # Simple consistency check: length variance
                    lengths = non_null_values.astype(str).str.len()
                    if len(lengths) > 1:
                        cv = lengths.std() / lengths.mean() if lengths.mean() > 0 else 1.0
                        consistency_scores.append(max(0.0, 1.0 - min(1.0, cv)))
                    else:
                        consistency_scores.append(1.0)
            else:
                consistency_scores.append(1.0)  # Numeric columns are consistent by definition
        
        consistency = np.mean(consistency_scores) if consistency_scores else 0.0
        
        # Validity: basic range/format checks
        validity_scores = []
        issues = []
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check for extreme outliers
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    outliers = ((df[col] < (q1 - 3 * iqr)) | (df[col] > (q3 + 3 * iqr))).sum()
                    validity_score = max(0.0, 1.0 - (outliers / len(df)))
                    validity_scores.append(validity_score)
                    if outliers > len(df) * 0.05:  # More than 5% outliers
                        issues.append({
                            "type": "outliers",
                            "column": col,
                            "count": int(outliers),
                            "percentage": outliers / len(df) * 100
                        })
                else:
                    validity_scores.append(1.0)
            else:
                validity_scores.append(1.0)  # Assume text columns are valid
        
        validity = np.mean(validity_scores) if validity_scores else 1.0
        
        # Accuracy: estimated based on data patterns (simplified)
        accuracy = (completeness + consistency + validity) / 3.0
        
        # Overall quality score
        quality_score = (completeness * 0.4 + consistency * 0.2 + validity * 0.2 + accuracy * 0.2)
        
        # Determine overall quality level
        if quality_score >= 0.9:
            overall_quality = DataQualityLevel.EXCELLENT
        elif quality_score >= 0.7:
            overall_quality = DataQualityLevel.GOOD
        elif quality_score >= 0.5:
            overall_quality = DataQualityLevel.FAIR
        else:
            overall_quality = DataQualityLevel.POOR
        
        # Generate recommendations
        recommendations = []
        if completeness < 0.8:
            recommendations.append("Consider handling missing values")
        if consistency < 0.7:
            recommendations.append("Check data format consistency")
        if len(issues) > 0:
            recommendations.append("Review data for outliers and anomalies")
        
        return DataQualityMetrics(
            overall_quality=overall_quality,
            quality_score=quality_score,
            completeness=completeness,
            consistency=consistency,
            validity=validity,
            accuracy=accuracy,
            issues=issues,
            recommendations=recommendations
        )
    
    def _generate_schema_information(self, df: pd.DataFrame, 
                                   db_name: Optional[str] = None) -> SchemaInformation:
        """Generate detailed schema information."""
        columns = []
        
        for col in df.columns:
            col_info = {
                "name": str(col),
                "dtype": str(df[col].dtype),
                "nullable": df[col].isnull().any(),
                "unique_values": int(df[col].nunique()),
                "sample_values": df[col].dropna().astype(str).head(3).tolist()
            }
            columns.append(col_info)
        
        return SchemaInformation(
            columns=columns,
            table_name=db_name
        )
    
    def _assess_query_complexity(self, query: str, df: pd.DataFrame,
                               query_analysis: Optional[QueryAnalysis] = None) -> Tuple[QueryComplexity, float]:
        """Assess query complexity level and score."""
        complexity_score = 0.0
        
        # Basic query analysis
        query_lower = query.lower()
        
        # Size factor (0.0 - 0.4)
        row_factor = min(0.4, len(df) / 100000)  # Up to 0.4 for 100k+ rows
        complexity_score += row_factor
        
        # Query pattern analysis (0.0 - 0.6)
        pattern_score = 0.0
        
        if 'join' in query_lower:
            pattern_score += 0.2
        if query_lower.count('join') > 1:
            pattern_score += 0.1
        if 'group by' in query_lower:
            pattern_score += 0.1
        if 'order by' in query_lower:
            pattern_score += 0.05
        if 'having' in query_lower:
            pattern_score += 0.1
        if any(keyword in query_lower for keyword in ['window', 'partition', 'over']):
            pattern_score += 0.15
        
        complexity_score += min(0.6, pattern_score)
        
        # Determine complexity level
        if complexity_score < 0.2:
            complexity_level = QueryComplexity.SIMPLE
        elif complexity_score < 0.5:
            complexity_level = QueryComplexity.MODERATE
        elif complexity_score < 0.8:
            complexity_level = QueryComplexity.COMPLEX
        else:
            complexity_level = QueryComplexity.INTENSIVE
        
        return complexity_level, complexity_score
    
    def _estimate_processing_time(self, df: pd.DataFrame, 
                                complexity_level: QueryComplexity,
                                token_estimation: TokenEstimation) -> float:
        """Estimate processing time in seconds."""
        base_time = 0.1  # Base processing time
        
        # Size-based time
        size_time = len(df) * 0.0001  # 0.1ms per row
        
        # Complexity multiplier
        complexity_multipliers = {
            QueryComplexity.SIMPLE: 1.0,
            QueryComplexity.MODERATE: 2.0,
            QueryComplexity.COMPLEX: 4.0,
            QueryComplexity.INTENSIVE: 8.0
        }
        
        complexity_time = size_time * complexity_multipliers[complexity_level]
        
        # Token serialization time
        token_time = token_estimation.total_tokens * 0.000001  # 1μs per token
        
        return round(base_time + complexity_time + token_time, 3)
    
    def _calculate_memory_footprint(self, df: pd.DataFrame, 
                                  token_estimation: TokenEstimation) -> float:
        """Calculate memory footprint in MB."""
        # DataFrame memory usage
        df_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Token serialization overhead (rough estimate)
        token_memory = token_estimation.total_tokens * 4 / (1024 * 1024)
        
        return round(df_memory + token_memory, 2)
    
    def _generate_chunk_availability(self, df: pd.DataFrame, 
                                   token_estimation: TokenEstimation) -> ChunkAvailability:
        """Generate chunk availability information."""
        chunk_size = token_estimation.recommended_chunk_size or 1000
        total_chunks = math.ceil(len(df) / chunk_size) if chunk_size > 0 else 1
        
        # For now, assume all chunks are available (in practice, this would depend on the data source)
        available_chunks = list(range(total_chunks))
        
        return ChunkAvailability(
            total_chunks=total_chunks,
            available_chunks=available_chunks,
            chunk_size=chunk_size
        )
    
    def _determine_recommended_action(self, token_estimation: TokenEstimation,
                                    data_quality: DataQualityMetrics,
                                    complexity_level: QueryComplexity,
                                    row_count: int) -> Tuple[str, str]:
        """Determine recommended action for LLM."""
        
        # High-level decision logic
        if token_estimation.total_tokens > 100000:
            return "chunk", f"Dataset is very large ({token_estimation.total_tokens:,} tokens). Recommend chunking for better handling."
        
        elif token_estimation.total_tokens > 20000:
            return "sample", f"Dataset is large ({token_estimation.total_tokens:,} tokens). Consider sampling or chunking."
        
        elif data_quality.overall_quality == DataQualityLevel.POOR:
            return "review", f"Data quality is {data_quality.overall_quality.value} (score: {data_quality.quality_score:.2f}). Review before processing."
        
        elif complexity_level == QueryComplexity.INTENSIVE:
            return "stream", f"Query complexity is {complexity_level.value}. Consider streaming processing."
        
        else:
            return "proceed", f"Dataset is manageable ({token_estimation.total_tokens:,} tokens, {row_count:,} rows). Safe to proceed."
    
    def _generate_llm_friendly_summary(self, df: pd.DataFrame,
                                     token_estimation: TokenEstimation,
                                     data_quality: DataQualityMetrics,
                                     complexity_level: QueryComplexity) -> str:
        """Generate a human-readable summary for LLM understanding."""
        
        size_desc = "small" if len(df) < 1000 else "medium" if len(df) < 10000 else "large"
        quality_desc = data_quality.overall_quality.value
        complexity_desc = complexity_level.value
        
        summary = (
            f"This is a {size_desc} dataset with {len(df):,} rows and {len(df.columns)} columns. "
            f"Data quality is {quality_desc} with {data_quality.quality_score:.1%} quality score. "
            f"Query complexity is {complexity_desc}. "
            f"Estimated response size: {token_estimation.total_tokens:,} tokens "
            f"({token_estimation.total_tokens/1000:.1f}K). "
        )
        
        if token_estimation.total_tokens > 10000:
            summary += "Consider chunking or sampling for optimal processing. "
            
        if data_quality.recommendations:
            summary += f"Data recommendations: {'; '.join(data_quality.recommendations[:2])}."
        
        return summary
    
    def _generate_cache_key(self, query_id: str, query: str, df: pd.DataFrame) -> str:
        """Generate cache key for metadata."""
        # Create hash from query, data shape, and basic characteristics
        key_data = f"{query}_{len(df)}_{len(df.columns)}_{df.dtypes.to_string()}"
        return hashlib.md5(key_data.encode()).hexdigest()


# Singleton instance
_metadata_generator = None

def get_metadata_generator() -> ResponseMetadataGenerator:
    """Get the global ResponseMetadataGenerator instance."""
    global _metadata_generator
    if _metadata_generator is None:
        _metadata_generator = ResponseMetadataGenerator()
    return _metadata_generator