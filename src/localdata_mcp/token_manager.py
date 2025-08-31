"""Intelligent Token Management System for LocalData MCP v1.3.1.

This module provides efficient DataFrame-based token counting that enables LLMs to make
intelligent decisions about data handling without expensive token counting operations.

Key Features:
- DataFrame-structure-aware token estimation (numeric vs text columns)
- Sample-based analysis for performance (text columns sampled, not fully counted)
- Rich metadata generation for LLM decision-making
- Context window awareness for different model limits
- Chunking recommendations for large responses
- Integration with Query Analyzer and Streaming Pipeline
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import tiktoken

logger = logging.getLogger(__name__)


@dataclass
class TokenEstimation:
    """Complete token estimation results with rich metadata."""
    
    # Core estimation
    total_tokens: int
    tokens_per_row: float
    confidence: float  # 0.0-1.0 based on sample quality
    
    # Column analysis
    numeric_columns: List[str]
    text_columns: List[str]
    other_columns: List[str]
    column_token_breakdown: Dict[str, int]
    
    # Performance metadata
    estimation_method: str  # 'full', 'sampled', 'extrapolated'
    sample_size: int
    total_rows: int
    
    # JSON overhead
    json_overhead_per_row: int
    json_overhead_total: int
    
    # Risk assessment
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    memory_risk: str  # 'low', 'medium', 'high'
    
    # Context window analysis
    fits_in_context: Dict[str, bool] = field(default_factory=dict)  # model -> fits
    recommended_chunk_size: Optional[int] = None


@dataclass
class ChunkingRecommendation:
    """Chunking strategy recommendations for large responses."""
    
    should_chunk: bool
    recommended_chunk_size: int
    estimated_chunks: int
    chunk_overlap_rows: int
    strategy: str  # 'row_based', 'column_based', 'mixed'
    
    # Metadata for LLM
    chunk_size_rationale: str
    performance_impact: str
    memory_benefits: str
    

@dataclass
class ResponseMetadata:
    """Rich metadata provided to LLMs for intelligent decision-making."""
    
    # Size estimates
    estimated_tokens: int
    estimated_memory_mb: float
    response_size_category: str  # 'small', 'medium', 'large', 'xlarge'
    
    # Data characteristics
    row_count: int
    column_count: int
    data_density: str  # 'sparse', 'moderate', 'dense'
    text_heavy: bool
    
    # Processing recommendations
    chunking_recommendation: Optional[ChunkingRecommendation]
    streaming_recommended: bool
    sampling_options: Dict[str, Any]
    
    # Context window compatibility
    model_compatibility: Dict[str, Dict[str, Any]]
    
    # Performance indicators
    processing_complexity: str  # 'low', 'medium', 'high'
    estimated_response_time: float


# Context window sizes for popular models
MODEL_CONTEXT_WINDOWS = {
    'gpt-4': 8192,
    'gpt-4-32k': 32768,
    'gpt-4-turbo': 128000,
    'gpt-3.5-turbo': 16385,
    'claude-3-haiku': 200000,
    'claude-3-sonnet': 200000,
    'claude-3-opus': 200000,
    'claude-3.5-sonnet': 200000,
    'gemini-pro': 32768,
    'default': 8192
}

# Token thresholds for risk assessment
TOKEN_RISK_THRESHOLDS = {
    'low': 1000,
    'medium': 10000,
    'high': 50000,
    'critical': 100000
}


class TokenManager:
    """Intelligent token management with DataFrame-based estimation."""
    
    def __init__(self, encoding_name: str = "cl100k_base", sample_size: int = 100):
        """Initialize the token manager.
        
        Args:
            encoding_name: Tiktoken encoding to use (default: cl100k_base for GPT-4)
            sample_size: Number of rows to sample for text analysis
        """
        self.sample_size = sample_size
        self.encoding = None
        
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
            logger.debug(f"Initialized TokenManager with {encoding_name} encoding")
        except Exception as e:
            logger.warning(f"Failed to initialize tiktoken encoder: {e}")
            logger.warning("Token estimation will use fallback methods")
    
    def estimate_tokens_from_dataframe(self, df: pd.DataFrame, 
                                     confidence_boost: float = 0.0) -> TokenEstimation:
        """Estimate tokens from a complete DataFrame using efficient sampling.
        
        Args:
            df: DataFrame to analyze
            confidence_boost: Additional confidence for known complete data
            
        Returns:
            Complete token estimation with metadata
        """
        if df.empty:
            return self._create_empty_estimation()
        
        # Analyze DataFrame structure
        column_analysis = self._analyze_column_types(df)
        
        # Estimate tokens per column type
        column_tokens = {}
        total_tokens_per_row = 0
        
        # Handle numeric columns (1 token per value)
        for col in column_analysis['numeric']:
            col_tokens = 1  # Numeric values = 1 token each
            column_tokens[col] = col_tokens
            total_tokens_per_row += col_tokens
        
        # Handle text columns with sampling
        text_token_analysis = self._estimate_text_column_tokens(df, column_analysis['text'])
        for col, tokens in text_token_analysis.items():
            column_tokens[col] = tokens
            total_tokens_per_row += tokens
        
        # Handle other columns
        for col in column_analysis['other']:
            col_tokens = self._estimate_other_column_tokens(df[col])
            column_tokens[col] = col_tokens
            total_tokens_per_row += col_tokens
        
        # Calculate JSON serialization overhead
        json_overhead = self._calculate_json_overhead(df)
        total_tokens_per_row += json_overhead
        
        # Calculate total tokens
        total_tokens = int(len(df) * total_tokens_per_row)
        
        # Determine confidence based on sampling
        confidence = self._calculate_confidence(df, column_analysis, confidence_boost)
        
        return TokenEstimation(
            total_tokens=total_tokens,
            tokens_per_row=total_tokens_per_row,
            confidence=confidence,
            numeric_columns=column_analysis['numeric'],
            text_columns=column_analysis['text'],
            other_columns=column_analysis['other'],
            column_token_breakdown=column_tokens,
            estimation_method='sampled' if len(df) > self.sample_size else 'full',
            sample_size=min(len(df), self.sample_size),
            total_rows=len(df),
            json_overhead_per_row=json_overhead,
            json_overhead_total=int(len(df) * json_overhead),
            risk_level=self._assess_token_risk(total_tokens),
            memory_risk=self._assess_memory_risk(total_tokens, len(df)),
            fits_in_context=self._assess_context_compatibility(total_tokens),
            recommended_chunk_size=self._calculate_recommended_chunk_size(total_tokens, total_tokens_per_row)
        )
    
    def estimate_tokens_for_query_result(self, row_count: int, 
                                       sample_df: pd.DataFrame) -> TokenEstimation:
        """Estimate tokens for a large query result using sample data.
        
        This is used by QueryAnalyzer for pre-execution estimation.
        
        Args:
            row_count: Total number of rows expected
            sample_df: Sample DataFrame (e.g., from LIMIT 1)
            
        Returns:
            Token estimation extrapolated from sample
        """
        if sample_df.empty or row_count == 0:
            return self._create_empty_estimation()
        
        # Get per-row estimation from sample
        sample_estimation = self.estimate_tokens_from_dataframe(sample_df)
        tokens_per_row = sample_estimation.tokens_per_row
        
        # Extrapolate to full dataset
        total_tokens = int(row_count * tokens_per_row)
        
        # Reduce confidence for extrapolation
        extrapolation_confidence = max(0.1, sample_estimation.confidence * 0.6)
        
        return TokenEstimation(
            total_tokens=total_tokens,
            tokens_per_row=tokens_per_row,
            confidence=extrapolation_confidence,
            numeric_columns=sample_estimation.numeric_columns,
            text_columns=sample_estimation.text_columns,
            other_columns=sample_estimation.other_columns,
            column_token_breakdown=sample_estimation.column_token_breakdown,
            estimation_method='extrapolated',
            sample_size=len(sample_df),
            total_rows=row_count,
            json_overhead_per_row=sample_estimation.json_overhead_per_row,
            json_overhead_total=int(row_count * sample_estimation.json_overhead_per_row),
            risk_level=self._assess_token_risk(total_tokens),
            memory_risk=self._assess_memory_risk(total_tokens, row_count),
            fits_in_context=self._assess_context_compatibility(total_tokens),
            recommended_chunk_size=self._calculate_recommended_chunk_size(total_tokens, tokens_per_row)
        )
    
    def get_response_metadata(self, estimation: TokenEstimation, 
                            include_chunking: bool = True) -> ResponseMetadata:
        """Generate rich metadata for LLM decision-making.
        
        Args:
            estimation: Token estimation results
            include_chunking: Whether to include chunking recommendations
            
        Returns:
            Complete response metadata
        """
        # Categorize response size
        size_category = self._categorize_response_size(estimation.total_tokens, estimation.total_rows)
        
        # Assess data characteristics
        text_heavy = len(estimation.text_columns) > len(estimation.numeric_columns)
        data_density = self._assess_data_density(estimation)
        
        # Generate chunking recommendation
        chunking_rec = None
        if include_chunking and estimation.total_tokens > 5000:
            chunking_rec = self._generate_chunking_recommendation(estimation)
        
        # Model compatibility analysis
        model_compat = self._analyze_model_compatibility(estimation)
        
        # Estimate memory usage (rough approximation)
        estimated_memory = estimation.total_tokens * 4 / (1024 * 1024)  # ~4 bytes per token
        
        return ResponseMetadata(
            estimated_tokens=estimation.total_tokens,
            estimated_memory_mb=estimated_memory,
            response_size_category=size_category,
            row_count=estimation.total_rows,
            column_count=len(estimation.numeric_columns) + len(estimation.text_columns) + len(estimation.other_columns),
            data_density=data_density,
            text_heavy=text_heavy,
            chunking_recommendation=chunking_rec,
            streaming_recommended=estimation.total_tokens > 10000 or estimation.total_rows > 1000,
            sampling_options=self._generate_sampling_options(estimation),
            model_compatibility=model_compat,
            processing_complexity=self._assess_processing_complexity(estimation),
            estimated_response_time=self._estimate_response_time(estimation)
        )
    
    def _analyze_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Analyze DataFrame columns by type for token estimation strategy."""
        numeric_cols = []
        text_cols = []
        other_cols = []
        
        for col in df.columns:
            dtype = df[col].dtype
            
            # Numeric types (int, float, etc.)
            if pd.api.types.is_numeric_dtype(dtype):
                numeric_cols.append(col)
            # String/object types that likely contain text
            elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
                text_cols.append(col)
            # Everything else (datetime, boolean, etc.)
            else:
                other_cols.append(col)
        
        return {
            'numeric': numeric_cols,
            'text': text_cols,
            'other': other_cols
        }
    
    def _estimate_text_column_tokens(self, df: pd.DataFrame, 
                                   text_columns: List[str]) -> Dict[str, int]:
        """Estimate tokens for text columns using sampling."""
        column_tokens = {}
        
        for col in text_columns:
            if col not in df.columns:
                continue
            
            # Sample data for analysis
            sample_data = df[col].dropna().head(self.sample_size)
            
            if sample_data.empty:
                column_tokens[col] = 1  # Placeholder for null/empty
                continue
            
            total_tokens = 0
            valid_samples = 0
            
            for value in sample_data:
                if pd.isna(value) or value == '':
                    total_tokens += 1  # "null" or empty string
                else:
                    str_value = str(value)
                    if self.encoding:
                        try:
                            token_count = len(self.encoding.encode(str_value))
                            total_tokens += token_count
                        except Exception:
                            # Fallback: 1 token per 4 characters
                            total_tokens += max(1, len(str_value) // 4)
                    else:
                        # Fallback method
                        total_tokens += max(1, len(str_value) // 4)
                
                valid_samples += 1
            
            # Average tokens per non-null value in this column
            avg_tokens = total_tokens / valid_samples if valid_samples > 0 else 1
            column_tokens[col] = int(avg_tokens)
        
        return column_tokens
    
    def _estimate_other_column_tokens(self, column: pd.Series) -> int:
        """Estimate tokens for non-numeric, non-text columns."""
        dtype = column.dtype
        
        # Boolean columns
        if pd.api.types.is_bool_dtype(dtype):
            return 1  # "true"/"false" = 1 token each
        
        # Datetime columns
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return 3  # Typical datetime string ~3 tokens
        
        # Default
        else:
            return 2  # Conservative estimate
    
    def _calculate_json_overhead(self, df: pd.DataFrame) -> float:
        """Calculate JSON serialization overhead per row."""
        # Field names + brackets, commas, quotes
        # Roughly: {"field1": value1, "field2": value2, ...}
        # Each field adds ~2 tokens for quotes and structure
        return len(df.columns) * 2 + 2  # +2 for opening/closing braces
    
    def _calculate_confidence(self, df: pd.DataFrame, 
                            column_analysis: Dict[str, List[str]],
                            confidence_boost: float) -> float:
        """Calculate confidence in token estimation."""
        base_confidence = 0.8  # Start with high confidence
        
        # Reduce confidence for sampling
        if len(df) > self.sample_size:
            sampling_factor = min(1.0, self.sample_size / len(df))
            base_confidence *= (0.6 + 0.4 * sampling_factor)
        
        # Reduce confidence for text-heavy data (more variable)
        text_ratio = len(column_analysis['text']) / len(df.columns) if len(df.columns) > 0 else 0
        base_confidence *= (1.0 - 0.3 * text_ratio)
        
        # Apply confidence boost
        final_confidence = min(1.0, base_confidence + confidence_boost)
        
        return round(final_confidence, 2)
    
    def _assess_token_risk(self, total_tokens: int) -> str:
        """Assess token count risk level."""
        for risk_level, threshold in TOKEN_RISK_THRESHOLDS.items():
            if total_tokens < threshold:
                return risk_level
        return 'critical'
    
    def _assess_memory_risk(self, total_tokens: int, row_count: int) -> str:
        """Assess memory usage risk."""
        # Rough memory estimate (tokens * 4 bytes)
        estimated_mb = total_tokens * 4 / (1024 * 1024)
        
        if estimated_mb < 50:
            return 'low'
        elif estimated_mb < 200:
            return 'medium'
        else:
            return 'high'
    
    def _assess_context_compatibility(self, total_tokens: int) -> Dict[str, bool]:
        """Check if response fits in various model context windows."""
        compatibility = {}
        
        for model, window_size in MODEL_CONTEXT_WINDOWS.items():
            if model == 'default':
                continue
            # Reserve 20% for prompt and response space
            available_tokens = int(window_size * 0.8)
            compatibility[model] = total_tokens <= available_tokens
        
        return compatibility
    
    def _calculate_recommended_chunk_size(self, total_tokens: int, tokens_per_row: float) -> Optional[int]:
        """Calculate recommended chunk size in rows."""
        if total_tokens <= 5000:
            return None  # No chunking needed
        
        # Target ~5000 tokens per chunk for good balance
        target_tokens_per_chunk = 5000
        rows_per_chunk = int(target_tokens_per_chunk / tokens_per_row) if tokens_per_row > 0 else 100
        
        # Minimum 10 rows, maximum 10000 rows per chunk
        return max(10, min(10000, rows_per_chunk))
    
    def _categorize_response_size(self, total_tokens: int, row_count: int) -> str:
        """Categorize response size for LLM understanding."""
        if total_tokens < 1000:
            return 'small'
        elif total_tokens < 10000:
            return 'medium'
        elif total_tokens < 50000:
            return 'large'
        else:
            return 'xlarge'
    
    def _assess_data_density(self, estimation: TokenEstimation) -> str:
        """Assess how dense the data is (text content vs structure)."""
        total_cols = len(estimation.numeric_columns) + len(estimation.text_columns) + len(estimation.other_columns)
        if total_cols == 0:
            return 'sparse'
        
        # Calculate ratio of content tokens to overhead
        content_ratio = (estimation.tokens_per_row - estimation.json_overhead_per_row) / estimation.tokens_per_row
        
        if content_ratio < 0.3:
            return 'sparse'  # Mostly structure
        elif content_ratio < 0.7:
            return 'moderate'
        else:
            return 'dense'  # Mostly content
    
    def _generate_chunking_recommendation(self, estimation: TokenEstimation) -> ChunkingRecommendation:
        """Generate intelligent chunking recommendations."""
        should_chunk = estimation.total_tokens > 5000
        
        if not should_chunk:
            return ChunkingRecommendation(
                should_chunk=False,
                recommended_chunk_size=0,
                estimated_chunks=1,
                chunk_overlap_rows=0,
                strategy='none',
                chunk_size_rationale="Response is small enough to send as single chunk",
                performance_impact="Minimal",
                memory_benefits="None needed"
            )
        
        chunk_size = estimation.recommended_chunk_size or 1000
        estimated_chunks = math.ceil(estimation.total_rows / chunk_size)
        
        # Strategy depends on data characteristics
        if len(estimation.text_columns) > len(estimation.numeric_columns):
            strategy = 'row_based'  # Text-heavy data benefits from row chunking
        else:
            strategy = 'row_based'  # Most common strategy
        
        return ChunkingRecommendation(
            should_chunk=True,
            recommended_chunk_size=chunk_size,
            estimated_chunks=estimated_chunks,
            chunk_overlap_rows=0,  # No overlap needed for database results
            strategy=strategy,
            chunk_size_rationale=f"Optimized for ~5000 tokens per chunk with {estimation.tokens_per_row:.1f} tokens per row",
            performance_impact="Reduced memory usage, streaming capability",
            memory_benefits=f"Reduces peak memory from ~{estimation.total_tokens*4//1024//1024}MB to ~{chunk_size*estimation.tokens_per_row*4//1024//1024}MB per chunk"
        )
    
    def _analyze_model_compatibility(self, estimation: TokenEstimation) -> Dict[str, Dict[str, Any]]:
        """Analyze compatibility with different language models."""
        compatibility = {}
        
        for model, window_size in MODEL_CONTEXT_WINDOWS.items():
            if model == 'default':
                continue
            
            available_tokens = int(window_size * 0.8)  # Reserve space
            fits = estimation.total_tokens <= available_tokens
            
            compatibility[model] = {
                'fits_in_context': fits,
                'context_window': window_size,
                'available_tokens': available_tokens,
                'utilization_percent': (estimation.total_tokens / available_tokens) * 100,
                'recommended_chunk_count': max(1, math.ceil(estimation.total_tokens / available_tokens))
            }
        
        return compatibility
    
    def _generate_sampling_options(self, estimation: TokenEstimation) -> Dict[str, Any]:
        """Generate sampling options for large datasets."""
        if estimation.total_rows <= 1000:
            return {'recommended': False, 'reason': 'Dataset is small enough to return in full'}
        
        # Calculate sampling options
        sample_sizes = [100, 500, 1000, 5000]
        options = {}
        
        for size in sample_sizes:
            if size < estimation.total_rows:
                sample_tokens = int(size * estimation.tokens_per_row)
                options[f'sample_{size}'] = {
                    'rows': size,
                    'estimated_tokens': sample_tokens,
                    'percentage': (size / estimation.total_rows) * 100
                }
        
        return {
            'recommended': estimation.total_tokens > 20000,
            'options': options,
            'reason': 'Large dataset - sampling can provide quick overview'
        }
    
    def _assess_processing_complexity(self, estimation: TokenEstimation) -> str:
        """Assess processing complexity for performance estimation."""
        # Based on size and text content
        complexity_score = 0
        
        # Size factor
        if estimation.total_rows > 10000:
            complexity_score += 2
        elif estimation.total_rows > 1000:
            complexity_score += 1
        
        # Text processing factor
        text_ratio = len(estimation.text_columns) / max(1, len(estimation.text_columns) + len(estimation.numeric_columns))
        if text_ratio > 0.5:
            complexity_score += 2
        elif text_ratio > 0.2:
            complexity_score += 1
        
        # Token density factor
        if estimation.tokens_per_row > 100:
            complexity_score += 1
        
        if complexity_score >= 4:
            return 'high'
        elif complexity_score >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _estimate_response_time(self, estimation: TokenEstimation) -> float:
        """Estimate response time based on data characteristics."""
        # Base time for small datasets
        base_time = 0.1
        
        # Add time based on row count
        row_time = estimation.total_rows * 0.0001  # 0.1ms per row
        
        # Add time for text processing
        text_time = len(estimation.text_columns) * estimation.total_rows * 0.0001
        
        # Add time for JSON serialization
        json_time = estimation.total_tokens * 0.000001  # 1μs per token
        
        return round(base_time + row_time + text_time + json_time, 2)
    
    def _create_empty_estimation(self) -> TokenEstimation:
        """Create estimation for empty dataset."""
        return TokenEstimation(
            total_tokens=0,
            tokens_per_row=0.0,
            confidence=1.0,
            numeric_columns=[],
            text_columns=[],
            other_columns=[],
            column_token_breakdown={},
            estimation_method='empty',
            sample_size=0,
            total_rows=0,
            json_overhead_per_row=0,
            json_overhead_total=0,
            risk_level='low',
            memory_risk='low',
            fits_in_context={model: True for model in MODEL_CONTEXT_WINDOWS.keys() if model != 'default'},
            recommended_chunk_size=None
        )


# Singleton instance for global use
_token_manager = None

def get_token_manager() -> TokenManager:
    """Get the global TokenManager instance."""
    global _token_manager
    if _token_manager is None:
        _token_manager = TokenManager()
    return _token_manager