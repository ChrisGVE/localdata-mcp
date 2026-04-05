"""
Factory and utility functions for pipeline analysis components.

Provides convenient creation functions for PipelineAnalyzer, ShimInjector,
PipelineValidator, and related data structures.
"""

from typing import Any, Dict, List

from ..interfaces import DataFormat
from ..compatibility_matrix import PipelineCompatibilityMatrix
from ..shim_registry import ShimRegistry

from .types import (
    PipelineStep,
    OptimizationCriteria,
)
from .analyzer import PipelineAnalyzer
from .injector import ShimInjector
from .validator import PipelineValidator


# Factory Functions


def create_pipeline_analyzer(
    compatibility_matrix: PipelineCompatibilityMatrix,
    shim_registry: ShimRegistry,
    **kwargs,
) -> PipelineAnalyzer:
    """Create a PipelineAnalyzer with standard configuration."""
    return PipelineAnalyzer(
        compatibility_matrix=compatibility_matrix, shim_registry=shim_registry, **kwargs
    )


def create_shim_injector(
    shim_registry: ShimRegistry,
    compatibility_matrix: PipelineCompatibilityMatrix,
    **kwargs,
) -> ShimInjector:
    """Create a ShimInjector with standard configuration."""
    return ShimInjector(
        shim_registry=shim_registry, compatibility_matrix=compatibility_matrix, **kwargs
    )


def create_pipeline_validator(
    compatibility_matrix: PipelineCompatibilityMatrix,
    shim_registry: ShimRegistry,
    **kwargs,
) -> PipelineValidator:
    """Create a PipelineValidator with complete analysis capabilities."""
    analyzer = create_pipeline_analyzer(compatibility_matrix, shim_registry)
    injector = create_shim_injector(shim_registry, compatibility_matrix)

    return PipelineValidator(
        compatibility_matrix=compatibility_matrix,
        shim_registry=shim_registry,
        analyzer=analyzer,
        injector=injector,
        **kwargs,
    )


def create_optimization_criteria(**kwargs) -> OptimizationCriteria:
    """Create OptimizationCriteria with custom parameters."""
    return OptimizationCriteria(**kwargs)


# Utility Functions


def create_pipeline_step(
    step_id: str,
    domain: str,
    operation: str,
    input_format: DataFormat,
    output_format: DataFormat,
    **kwargs,
) -> PipelineStep:
    """Factory function to create a PipelineStep."""
    return PipelineStep(
        step_id=step_id,
        domain=domain,
        operation=operation,
        input_format=input_format,
        output_format=output_format,
        **kwargs,
    )


def analyze_and_fix_pipeline(
    pipeline_steps: List[PipelineStep],
    compatibility_matrix: PipelineCompatibilityMatrix,
    shim_registry: ShimRegistry,
    auto_fix: bool = True,
) -> Dict[str, Any]:
    """
    High-level utility function to analyze and fix a pipeline in one call.

    Args:
        pipeline_steps: Pipeline steps to analyze and fix
        compatibility_matrix: Compatibility matrix for analysis
        shim_registry: Registry of available shims
        auto_fix: Whether to automatically fix detected issues

    Returns:
        Complete analysis and fix results
    """
    validator = create_pipeline_validator(compatibility_matrix, shim_registry)

    return validator.validate_and_fix_pipeline(
        pipeline_steps=pipeline_steps, auto_fix=auto_fix, validation_level="balanced"
    )
