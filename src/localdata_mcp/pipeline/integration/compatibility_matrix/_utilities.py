"""
Factory and utility functions for the Pipeline Compatibility Matrix.

Provides convenient creation functions and pipeline analysis utilities.
"""

from typing import List, Optional

from ..interfaces import DataFormat, ValidationResult
from ._types import DomainProfile
from ._matrix import PipelineCompatibilityMatrix


# Factory Functions


def create_compatibility_matrix(
    enable_caching: bool = True, cache_size: int = 500
) -> PipelineCompatibilityMatrix:
    """Create a PipelineCompatibilityMatrix with standard configuration."""
    return PipelineCompatibilityMatrix(
        enable_caching=enable_caching, cache_size=cache_size
    )


def create_minimal_compatibility_matrix() -> PipelineCompatibilityMatrix:
    """Create a minimal compatibility matrix without caching."""
    return PipelineCompatibilityMatrix(enable_caching=False, cache_size=0)


# Utility Functions


def assess_pipeline_compatibility(
    pipeline_domains: List[str], matrix: Optional[PipelineCompatibilityMatrix] = None
) -> ValidationResult:
    """Assess compatibility of a pipeline with given domain sequence."""
    if matrix is None:
        matrix = create_compatibility_matrix()

    return matrix.validate_pipeline(pipeline_domains)


def find_optimal_format_for_domains(
    domains: List[str], matrix: Optional[PipelineCompatibilityMatrix] = None
) -> Optional[DataFormat]:
    """Find optimal data format that works well across multiple domains."""
    if matrix is None:
        matrix = create_compatibility_matrix()

    if not domains:
        return None

    # Find formats supported by all domains
    common_formats = None
    for domain_name in domains:
        profile = matrix.get_domain_profile(domain_name)
        if profile is None:
            continue

        domain_formats = set(
            profile.base_requirements.input_formats
            + profile.base_requirements.output_formats
        )

        if common_formats is None:
            common_formats = domain_formats
        else:
            common_formats &= domain_formats

    if not common_formats:
        # No common format, return most versatile format
        return DataFormat.PANDAS_DATAFRAME

    # Score each common format based on domain preferences
    format_scores = {}
    for fmt in common_formats:
        total_score = 0.0
        for domain_name in domains:
            profile = matrix.get_domain_profile(domain_name)
            if profile and fmt in profile.compatibility_preferences:
                total_score += profile.compatibility_preferences[fmt]
            else:
                total_score += 0.5  # Default score

        format_scores[fmt] = total_score / len(domains)

    # Return format with highest average score
    return max(format_scores.keys(), key=lambda f: format_scores[f])


def suggest_pipeline_improvements(
    pipeline_domains: List[str], matrix: Optional[PipelineCompatibilityMatrix] = None
) -> List[str]:
    """Suggest improvements for pipeline compatibility."""
    if matrix is None:
        matrix = create_compatibility_matrix()

    validation = matrix.validate_pipeline(pipeline_domains)
    suggestions = list(validation.suggestions)

    # Add format-specific suggestions
    optimal_format = find_optimal_format_for_domains(pipeline_domains, matrix)
    if optimal_format:
        suggestions.append(
            f"Consider using {optimal_format.value} as the primary data format"
        )

    # Analyze problematic transitions
    if hasattr(validation.details, "items"):
        for step_key, step_info in validation.details.items():
            if (
                isinstance(step_info, dict)
                and step_info.get("compatibility_score", 1.0) < 0.6
            ):
                source_domain = step_info.get("source_domain")
                target_domain = step_info.get("target_domain")
                if source_domain and target_domain:
                    suggestions.append(
                        f"Consider adding an intermediate step between {source_domain} and {target_domain}"
                    )

    return suggestions
