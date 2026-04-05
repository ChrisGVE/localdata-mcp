"""
Automatic Shim Insertion Logic for LocalData MCP v2.0 Integration Shims Framework.

This package provides intelligent pipeline analysis and automatic shim insertion capabilities
for seamless cross-domain data science workflows with minimal user intervention.

Key Features:
- PipelineAnalyzer: Identify incompatible connections in pipeline chains
- ShimInjector: Automatic adapter insertion with optimal selection
- PipelineValidator: Complete pipeline composition verification
- Cost-based optimization for efficient shim selection
- Integration with existing compatibility matrix and shim registry

Design Principles:
- Intention-Driven Interface: Analyze pipelines by analytical goals
- Context-Aware Composition: Consider upstream/downstream context
- Progressive Disclosure: Simple analysis with detailed breakdowns available
- Streaming-First: Memory-efficient for large pipeline chains
- Modular Domain Integration: Seamless integration with existing infrastructure
"""

from .types import (
    AnalysisType,
    InjectionStrategy,
    PipelineStep,
    PipelineConnection,
    IncompatibilityIssue,
    ShimRecommendation,
    PipelineAnalysisResult,
    OptimizationCriteria,
)

from .analyzer import PipelineAnalyzer

from .injector import ShimInjector

from .validator import PipelineValidator

from .factories import (
    create_pipeline_analyzer,
    create_shim_injector,
    create_pipeline_validator,
    create_optimization_criteria,
    create_pipeline_step,
    analyze_and_fix_pipeline,
)

__all__ = [
    # Core pipeline analysis classes
    "PipelineAnalyzer",
    "ShimInjector",
    "PipelineValidator",
    # Data structures
    "PipelineStep",
    "PipelineConnection",
    "IncompatibilityIssue",
    "ShimRecommendation",
    "PipelineAnalysisResult",
    "OptimizationCriteria",
    # Enums
    "AnalysisType",
    "InjectionStrategy",
    # Factory functions
    "create_pipeline_analyzer",
    "create_shim_injector",
    "create_pipeline_validator",
    "create_optimization_criteria",
    "create_pipeline_step",
    # Utility functions
    "analyze_and_fix_pipeline",
]
