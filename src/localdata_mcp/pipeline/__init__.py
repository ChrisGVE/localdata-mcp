"""
Core Pipeline Framework for LocalData MCP v2.0

This module provides the foundational pipeline architecture that implements
the First Principles Architecture for comprehensive data science capabilities.

Key Components:
- Input pipelines with streaming compatibility
- Preprocessing pipelines with progressive disclosure  
- Analysis pipelines with intention-driven interfaces
- Output standardization across domains
- Error handling with graceful degradation

The framework enables natural language analytics where LLM agents express
analytical intentions rather than technical procedures, with automatic
streaming for large datasets and composition-aware results for tool chaining.
"""

from .base import (
    AnalysisPipelineBase,
    PipelineError,
    PipelineState,
    CompositionMetadata
)

from .input import (
    DataInputPipeline,
    DataSourceType,
    StreamingConfig
)

from .preprocessing import (
    DataPreprocessingPipeline,  
    PreprocessingIntent,
    TransformationStrategy
)

from .output import (
    AnalysisOutputStandardizer,
    StandardizedResult,
    ResultQuality
)

from .error_handling import (
    PipelineErrorHandler,
    ErrorClassification,
    RecoveryStrategy
)

from .factory import (
    PipelineFactory,
    create_analysis_pipeline,
    register_domain_pipeline
)

__all__ = [
    # Base classes
    "AnalysisPipelineBase",
    "PipelineError", 
    "PipelineState",
    "CompositionMetadata",
    
    # Input pipeline
    "DataInputPipeline",
    "DataSourceType",
    "StreamingConfig",
    
    # Preprocessing pipeline
    "DataPreprocessingPipeline",
    "PreprocessingIntent", 
    "TransformationStrategy",
    
    # Output standardization
    "AnalysisOutputStandardizer",
    "StandardizedResult",
    "ResultQuality",
    
    # Error handling
    "PipelineErrorHandler",
    "ErrorClassification",
    "RecoveryStrategy",
    
    # Pipeline factory
    "PipelineFactory",
    "create_analysis_pipeline",
    "register_domain_pipeline"
]

# Version info
__version__ = "2.0.0"
__author__ = "LocalData MCP Team"
__description__ = "Core Pipeline Framework implementing First Principles Architecture"