"""
Pipeline Compatibility Matrix for LocalData MCP v2.0 Integration Shims Framework.

This module provides domain compatibility mapping, format specification management,
and automated conversion pathway discovery to enable seamless cross-domain pipeline
composition for LLM agents.

Key Features:
- PipelineCompatibilityMatrix for comprehensive domain compatibility assessment
- Domain profiles for Statistical, Regression, Time Series, and Pattern Recognition domains
- Automatic compatibility scoring and pathway discovery
- Pipeline validation with detailed error reporting and recommendations
- Integration with existing TypeDetectionEngine and converter framework
- Extensible architecture for future domain additions

Design Principles:
- Intention-Driven Interface: Score compatibility based on analytical goals
- Context-Aware Composition: Consider upstream/downstream context in validation
- Progressive Disclosure: Simple scoring with detailed breakdowns available
- Streaming-First: Memory-efficient compatibility checking
- Modular Integration: Easy addition of new domains and formats
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from functools import lru_cache

from .interfaces import (
    DataFormat, CompatibilityMatrix, CompatibilityScore, ConversionPath,
    ConversionStep, ConversionCost, ValidationResult, DomainRequirements,
    DataFormatSpec, MemoryConstraints, PerformanceRequirements
)
from .type_detection import TypeDetectionEngine
from ...logging_manager import get_logger

logger = get_logger(__name__)


class CompatibilityLevel(Enum):
    """Compatibility levels between formats and domains."""
    PERFECT = "perfect"         # Direct compatibility, no conversion needed (0.95-1.0)
    HIGH = "high"              # Compatible with minimal conversion (0.8-0.94)
    MODERATE = "moderate"       # Compatible with standard conversion (0.6-0.79)
    LOW = "low"                # Compatible with complex conversion (0.3-0.59)
    INCOMPATIBLE = "incompatible"  # Not compatible (0.0-0.29)

    @property
    def score_threshold(self) -> float:
        """Get minimum score threshold for this level."""
        return {
            CompatibilityLevel.PERFECT: 0.95,
            CompatibilityLevel.HIGH: 0.8,
            CompatibilityLevel.MODERATE: 0.6,
            CompatibilityLevel.LOW: 0.3,
            CompatibilityLevel.INCOMPATIBLE: 0.0
        }[self]


@dataclass
class DomainProfile:
    """Extended domain profile with tool-specific requirements."""
    domain_name: str
    base_requirements: DomainRequirements
    tool_specifications: Dict[str, DataFormatSpec] = field(default_factory=dict)
    compatibility_preferences: Dict[DataFormat, float] = field(default_factory=dict)
    conversion_costs: Dict[Tuple[DataFormat, DataFormat], float] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)


class PipelineCompatibilityMatrix(CompatibilityMatrix):
    """
    Core compatibility matrix implementation for cross-domain pipeline composition.
    
    Provides comprehensive compatibility scoring, pathway discovery, and pipeline
    validation for LocalData MCP domains with LLM-friendly interfaces.
    """
    
    def __init__(self, enable_caching: bool = True, cache_size: int = 500):
        """
        Initialize Pipeline Compatibility Matrix.
        
        Args:
            enable_caching: Enable LRU caching for compatibility scores
            cache_size: Maximum cache size for compatibility assessments
        """
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        
        # Domain registry
        self._domain_profiles: Dict[str, DomainProfile] = {}
        
        # Compatibility scoring
        self._format_compatibility_matrix = self._build_format_compatibility_matrix()
        
        # Caching
        if enable_caching:
            self._compatibility_cache: Dict[str, CompatibilityScore] = {}
        
        # Statistics
        self._stats = {
            'assessments': 0,
            'cache_hits': 0,
            'validations': 0
        }
        
        # Initialize with standard LocalData MCP domains
        self._initialize_standard_domains()
        
        logger.info("PipelineCompatibilityMatrix initialized",
                   domains=len(self._domain_profiles),
                   caching_enabled=enable_caching)
    
    def _build_format_compatibility_matrix(self) -> Dict[Tuple[DataFormat, DataFormat], float]:
        """Build the base format compatibility matrix."""
        matrix = {}
        
        # Perfect compatibility (same format)
        for fmt in DataFormat:
            matrix[(fmt, fmt)] = 1.0
        
        # High compatibility conversions (0.8-0.94)
        high_compat = [
            # Pandas DataFrame conversions
            (DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY, 0.9),
            (DataFormat.PANDAS_DATAFRAME, DataFormat.TIME_SERIES, 0.92),
            (DataFormat.PANDAS_DATAFRAME, DataFormat.PYTHON_DICT, 0.85),
            (DataFormat.TIME_SERIES, DataFormat.PANDAS_DATAFRAME, 0.95),
            
            # NumPy array conversions
            (DataFormat.NUMPY_ARRAY, DataFormat.PANDAS_DATAFRAME, 0.85),
            (DataFormat.NUMPY_ARRAY, DataFormat.SCIPY_SPARSE, 0.88),
            (DataFormat.NUMPY_ARRAY, DataFormat.PYTHON_LIST, 0.92),
            
            # Result format conversions
            (DataFormat.STATISTICAL_RESULT, DataFormat.PYTHON_DICT, 0.9),
            (DataFormat.REGRESSION_MODEL, DataFormat.PYTHON_DICT, 0.85),
            (DataFormat.CLUSTERING_RESULT, DataFormat.PANDAS_DATAFRAME, 0.8),
            (DataFormat.FORECAST_RESULT, DataFormat.TIME_SERIES, 0.9),
        ]
        
        # Moderate compatibility conversions (0.6-0.79)
        moderate_compat = [
            (DataFormat.PANDAS_DATAFRAME, DataFormat.SCIPY_SPARSE, 0.75),
            (DataFormat.SCIPY_SPARSE, DataFormat.PANDAS_DATAFRAME, 0.7),
            (DataFormat.TIME_SERIES, DataFormat.NUMPY_ARRAY, 0.75),
            (DataFormat.CATEGORICAL, DataFormat.PANDAS_DATAFRAME, 0.8),
            (DataFormat.CATEGORICAL, DataFormat.NUMPY_ARRAY, 0.65),
            (DataFormat.PYTHON_DICT, DataFormat.PANDAS_DATAFRAME, 0.7),
        ]
        
        # Low compatibility conversions (0.3-0.59)
        low_compat = [
            (DataFormat.STATISTICAL_RESULT, DataFormat.NUMPY_ARRAY, 0.4),
            (DataFormat.REGRESSION_MODEL, DataFormat.PANDAS_DATAFRAME, 0.45),
            (DataFormat.PYTHON_LIST, DataFormat.SCIPY_SPARSE, 0.5),
            (DataFormat.JSON, DataFormat.PANDAS_DATAFRAME, 0.55),
            (DataFormat.CSV, DataFormat.NUMPY_ARRAY, 0.5),
        ]
        
        # Add all compatibility scores
        for source, target, score in high_compat + moderate_compat + low_compat:
            matrix[(source, target)] = score
            # Add reverse with slight penalty
            matrix[(target, source)] = score * 0.95
        
        return matrix
    
    def _initialize_standard_domains(self) -> None:
        """Initialize compatibility profiles for standard LocalData MCP domains."""
        
        # Statistical Analysis Domain
        stats_requirements = DomainRequirements(
            domain_name="statistical_analysis",
            input_formats=[DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY, 
                          DataFormat.TIME_SERIES, DataFormat.PYTHON_LIST],
            output_formats=[DataFormat.STATISTICAL_RESULT, DataFormat.PANDAS_DATAFRAME],
            preferred_format=DataFormat.PANDAS_DATAFRAME,
            metadata_requirements=['column_names', 'data_types', 'sample_size'],
            quality_requirements={
                'min_sample_size': 30,
                'max_missing_ratio': 0.2,
                'numeric_data_required': True
            }
        )
        
        stats_profile = DomainProfile(
            domain_name="statistical_analysis",
            base_requirements=stats_requirements,
            tool_specifications={
                'hypothesis_testing': DataFormatSpec(
                    format_type=DataFormat.PANDAS_DATAFRAME,
                    schema_requirements={'min_columns': 2, 'numeric_columns': True}
                ),
                'correlation_analysis': DataFormatSpec(
                    format_type=DataFormat.PANDAS_DATAFRAME,
                    schema_requirements={'all_numeric': True, 'min_columns': 2}
                ),
                'descriptive_stats': DataFormatSpec(
                    format_type=DataFormat.PANDAS_DATAFRAME,
                    schema_requirements={'numeric_columns': True}
                )
            },
            compatibility_preferences={
                DataFormat.PANDAS_DATAFRAME: 1.0,
                DataFormat.NUMPY_ARRAY: 0.8,
                DataFormat.TIME_SERIES: 0.9,
                DataFormat.PYTHON_LIST: 0.6
            }
        )
        
        self.register_domain_requirements("statistical_analysis", stats_profile)
        
        # Regression & Modeling Domain
        regression_requirements = DomainRequirements(
            domain_name="regression_modeling",
            input_formats=[DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY,
                          DataFormat.SCIPY_SPARSE],
            output_formats=[DataFormat.REGRESSION_MODEL, DataFormat.PANDAS_DATAFRAME],
            preferred_format=DataFormat.PANDAS_DATAFRAME,
            metadata_requirements=['feature_names', 'target_column', 'data_types'],
            quality_requirements={
                'min_samples_per_feature': 10,
                'feature_target_separation': True
            }
        )
        
        regression_profile = DomainProfile(
            domain_name="regression_modeling",
            base_requirements=regression_requirements,
            tool_specifications={
                'linear_regression': DataFormatSpec(
                    format_type=DataFormat.PANDAS_DATAFRAME,
                    schema_requirements={'features_numeric': True, 'target_numeric': True}
                ),
                'logistic_regression': DataFormatSpec(
                    format_type=DataFormat.PANDAS_DATAFRAME,
                    schema_requirements={'features_numeric': True, 'target_categorical': True}
                )
            },
            compatibility_preferences={
                DataFormat.PANDAS_DATAFRAME: 1.0,
                DataFormat.NUMPY_ARRAY: 0.85,
                DataFormat.SCIPY_SPARSE: 0.9
            }
        )
        
        self.register_domain_requirements("regression_modeling", regression_profile)
        
        # Time Series Analysis Domain
        timeseries_requirements = DomainRequirements(
            domain_name="time_series",
            input_formats=[DataFormat.TIME_SERIES, DataFormat.PANDAS_DATAFRAME],
            output_formats=[DataFormat.FORECAST_RESULT, DataFormat.TIME_SERIES],
            preferred_format=DataFormat.TIME_SERIES,
            metadata_requirements=['datetime_index', 'frequency', 'seasonality'],
            quality_requirements={
                'min_periods': 24,
                'regular_frequency': True,
                'datetime_sorted': True
            }
        )
        
        timeseries_profile = DomainProfile(
            domain_name="time_series",
            base_requirements=timeseries_requirements,
            tool_specifications={
                'arima_forecasting': DataFormatSpec(
                    format_type=DataFormat.TIME_SERIES,
                    schema_requirements={'univariate': True, 'regular_intervals': True}
                ),
                'seasonal_decomposition': DataFormatSpec(
                    format_type=DataFormat.TIME_SERIES,
                    schema_requirements={'min_periods_per_season': 2}
                )
            },
            compatibility_preferences={
                DataFormat.TIME_SERIES: 1.0,
                DataFormat.PANDAS_DATAFRAME: 0.9
            }
        )
        
        self.register_domain_requirements("time_series", timeseries_profile)
        
        # Advanced Pattern Recognition Domain
        pattern_requirements = DomainRequirements(
            domain_name="pattern_recognition",
            input_formats=[DataFormat.NUMPY_ARRAY, DataFormat.PANDAS_DATAFRAME,
                          DataFormat.SCIPY_SPARSE],
            output_formats=[DataFormat.CLUSTERING_RESULT, DataFormat.PATTERN_RECOGNITION_RESULT],
            preferred_format=DataFormat.NUMPY_ARRAY,
            metadata_requirements=['feature_names', 'dimensionality'],
            quality_requirements={
                'min_samples': 50,
                'feature_scaling_recommended': True
            }
        )
        
        pattern_profile = DomainProfile(
            domain_name="pattern_recognition",
            base_requirements=pattern_requirements,
            tool_specifications={
                'clustering': DataFormatSpec(
                    format_type=DataFormat.NUMPY_ARRAY,
                    schema_requirements={'numeric_features': True, 'feature_scaling': True}
                ),
                'dimensionality_reduction': DataFormatSpec(
                    format_type=DataFormat.NUMPY_ARRAY,
                    schema_requirements={'high_dimensional': True}
                )
            },
            compatibility_preferences={
                DataFormat.NUMPY_ARRAY: 1.0,
                DataFormat.PANDAS_DATAFRAME: 0.8,
                DataFormat.SCIPY_SPARSE: 0.9
            }
        )
        
        self.register_domain_requirements("pattern_recognition", pattern_profile)
        
        logger.info("Initialized standard domain profiles",
                   domains=list(self._domain_profiles.keys()))
    
    def get_compatibility(self, source_format: DataFormat, 
                         target_format: DataFormat) -> CompatibilityScore:
        """
        Get compatibility score between two data formats.
        
        Args:
            source_format: Source data format
            target_format: Target data format
            
        Returns:
            Comprehensive compatibility score with conversion details
        """
        self._stats['assessments'] += 1
        
        # Check cache
        cache_key = f"{source_format.value}->{target_format.value}"
        if self.enable_caching and cache_key in self._compatibility_cache:
            self._stats['cache_hits'] += 1
            return self._compatibility_cache[cache_key]
        
        # Calculate compatibility
        score = self._calculate_compatibility_score(source_format, target_format)
        
        # Determine if conversion is required
        direct_compatible = source_format == target_format
        conversion_required = not direct_compatible and score > 0.0
        
        # Find conversion path if needed
        conversion_path = None
        if conversion_required:
            conversion_path = self._find_conversion_path(source_format, target_format)
        
        # Identify issues and recommendations
        issues = self._identify_compatibility_issues(source_format, target_format, score)
        recommendations = self._generate_recommendations(source_format, target_format, score)
        
        # Create compatibility score
        compatibility_score = CompatibilityScore(
            score=score,
            direct_compatible=direct_compatible,
            conversion_required=conversion_required,
            conversion_path=conversion_path,
            compatibility_issues=issues,
            recommendations=recommendations
        )
        
        # Cache result
        if self.enable_caching:
            if len(self._compatibility_cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self._compatibility_cache))
                del self._compatibility_cache[oldest_key]
            self._compatibility_cache[cache_key] = compatibility_score
        
        return compatibility_score
    
    def _calculate_compatibility_score(self, source_format: DataFormat,
                                     target_format: DataFormat) -> float:
        """Calculate base compatibility score between formats."""
        # Direct lookup in compatibility matrix
        score = self._format_compatibility_matrix.get((source_format, target_format))
        if score is not None:
            return score
        
        # Format family compatibility fallback
        return self._calculate_family_compatibility(source_format, target_format)
    
    def _calculate_family_compatibility(self, source_format: DataFormat,
                                      target_format: DataFormat) -> float:
        """Calculate compatibility based on format families."""
        # Define format families
        tabular_formats = {
            DataFormat.PANDAS_DATAFRAME, DataFormat.TIME_SERIES, 
            DataFormat.CATEGORICAL, DataFormat.MULTI_INDEX
        }
        
        array_formats = {
            DataFormat.NUMPY_ARRAY, DataFormat.SCIPY_SPARSE
        }
        
        result_formats = {
            DataFormat.STATISTICAL_RESULT, DataFormat.REGRESSION_MODEL,
            DataFormat.CLUSTERING_RESULT, DataFormat.FORECAST_RESULT,
            DataFormat.PATTERN_RECOGNITION_RESULT
        }
        
        collection_formats = {
            DataFormat.PYTHON_LIST, DataFormat.PYTHON_DICT
        }
        
        # Same family compatibility
        families = [tabular_formats, array_formats, result_formats, collection_formats]
        for family in families:
            if source_format in family and target_format in family:
                return 0.7
        
        # Cross-family compatibility
        if ((source_format in tabular_formats and target_format in array_formats) or
            (source_format in array_formats and target_format in tabular_formats)):
            return 0.6
        
        if ((source_format in collection_formats and target_format in tabular_formats) or
            (source_format in tabular_formats and target_format in collection_formats)):
            return 0.5
        
        # Default low compatibility for unknown combinations
        return 0.2
    
    def _find_conversion_path(self, source_format: DataFormat,
                            target_format: DataFormat) -> Optional[ConversionPath]:
        """Find conversion path between formats."""
        # For now, create a simple direct conversion path
        # This can be enhanced with multi-step path finding later
        
        if source_format == target_format:
            return None
        
        # Estimate conversion cost based on compatibility score
        compatibility_score = self._calculate_compatibility_score(source_format, target_format)
        
        # Base cost increases as compatibility decreases
        base_cost = 1.0 - compatibility_score
        
        cost = ConversionCost(
            computational_cost=base_cost * 0.5,
            memory_cost_mb=base_cost * 100,
            time_estimate_seconds=base_cost * 2.0,
            quality_impact=max(0.0, 0.1 - compatibility_score * 0.1)
        )
        
        step = ConversionStep(
            adapter_id=f"converter_{source_format.value}_to_{target_format.value}",
            source_format=source_format,
            target_format=target_format,
            estimated_cost=cost,
            confidence=min(compatibility_score + 0.1, 1.0)
        )
        
        return ConversionPath(
            source_format=source_format,
            target_format=target_format,
            steps=[step],
            total_cost=cost,
            success_probability=min(compatibility_score + 0.2, 1.0)
        )
    
    def _identify_compatibility_issues(self, source_format: DataFormat,
                                     target_format: DataFormat, score: float) -> List[str]:
        """Identify potential compatibility issues."""
        issues = []
        
        if score < 0.3:
            issues.append(f"Very low compatibility between {source_format.value} and {target_format.value}")
        
        # Format-specific issues
        if source_format == DataFormat.SCIPY_SPARSE and target_format == DataFormat.PANDAS_DATAFRAME:
            issues.append("Sparse to DataFrame conversion may significantly increase memory usage")
        
        if source_format == DataFormat.TIME_SERIES and target_format == DataFormat.NUMPY_ARRAY:
            issues.append("Time series to array conversion will lose temporal indexing")
        
        if target_format in [DataFormat.STATISTICAL_RESULT, DataFormat.REGRESSION_MODEL] and source_format in [DataFormat.PYTHON_LIST, DataFormat.PYTHON_DICT]:
            issues.append("Converting simple data structures to complex result formats may not be meaningful")
        
        return issues
    
    def _generate_recommendations(self, source_format: DataFormat,
                                target_format: DataFormat, score: float) -> List[str]:
        """Generate recommendations for improving compatibility."""
        recommendations = []
        
        if score < 0.6:
            recommendations.append(f"Consider using {DataFormat.PANDAS_DATAFRAME.value} as an intermediate format")
        
        if source_format == DataFormat.PYTHON_LIST and target_format != DataFormat.NUMPY_ARRAY:
            recommendations.append("Convert Python lists to NumPy arrays for better data science compatibility")
        
        if target_format == DataFormat.TIME_SERIES:
            recommendations.append("Ensure source data has proper datetime indexing for time series conversion")
        
        if source_format in [DataFormat.NUMPY_ARRAY, DataFormat.SCIPY_SPARSE] and target_format == DataFormat.PANDAS_DATAFRAME:
            recommendations.append("Consider providing column names for better DataFrame structure")
        
        return recommendations
    
    def register_domain_requirements(self, domain_name: str,
                                   requirements: Union[DomainRequirements, DomainProfile]) -> None:
        """Register domain requirements in the compatibility matrix."""
        if isinstance(requirements, DomainRequirements):
            profile = DomainProfile(
                domain_name=domain_name,
                base_requirements=requirements
            )
        else:
            profile = requirements
        
        self._domain_profiles[domain_name] = profile
        logger.info(f"Registered domain requirements for {domain_name}")
    
    def validate_pipeline(self, pipeline_steps: List[str]) -> ValidationResult:
        """
        Validate pipeline compatibility across all steps.
        
        Args:
            pipeline_steps: List of domain names in pipeline order
            
        Returns:
            Validation result with detailed compatibility assessment
        """
        self._stats['validations'] += 1
        
        errors = []
        warnings = []
        details = {}
        suggestions = []
        
        if len(pipeline_steps) < 2:
            return ValidationResult(
                is_valid=True,
                score=1.0,
                details={'message': 'Single step pipeline requires no compatibility validation'}
            )
        
        total_score = 0.0
        step_count = 0
        
        # Validate each step transition
        for i in range(len(pipeline_steps) - 1):
            current_domain = pipeline_steps[i]
            next_domain = pipeline_steps[i + 1]
            
            # Get domain profiles
            current_profile = self._domain_profiles.get(current_domain)
            next_profile = self._domain_profiles.get(next_domain)
            
            if not current_profile:
                errors.append(f"Unknown domain: {current_domain}")
                continue
            
            if not next_profile:
                errors.append(f"Unknown domain: {next_domain}")
                continue
            
            # Find best compatibility between domain outputs and inputs
            best_score = 0.0
            best_source_format = None
            best_target_format = None
            
            for output_format in current_profile.base_requirements.output_formats:
                for input_format in next_profile.base_requirements.input_formats:
                    compatibility = self.get_compatibility(output_format, input_format)
                    if compatibility.score > best_score:
                        best_score = compatibility.score
                        best_source_format = output_format
                        best_target_format = input_format
            
            total_score += best_score
            step_count += 1
            
            # Collect issues based on compatibility score
            step_key = f"step_{i}_to_{i+1}"
            details[step_key] = {
                'source_domain': current_domain,
                'target_domain': next_domain,
                'best_source_format': best_source_format.value if best_source_format else None,
                'best_target_format': best_target_format.value if best_target_format else None,
                'compatibility_score': best_score,
                'conversion_required': best_score < 1.0
            }
            
            if best_score < 0.3:
                errors.append(f"Very low compatibility between {current_domain} and {next_domain} (score: {best_score:.2f})")
            elif best_score < 0.6:
                warnings.append(f"Moderate compatibility issues between {current_domain} and {next_domain} (score: {best_score:.2f})")
        
        # Calculate overall score
        overall_score = total_score / step_count if step_count > 0 else 0.0
        
        # Generate suggestions
        if overall_score < 0.7:
            suggestions.append("Pipeline has compatibility issues - consider reordering steps or adding intermediate conversions")
        
        if any(details[key]['conversion_required'] for key in details):
            suggestions.append("Some steps require data format conversion - ensure proper adapters are available")
        
        low_compat_steps = [key for key in details if details[key]['compatibility_score'] < 0.5]
        if low_compat_steps:
            suggestions.append(f"Steps with very low compatibility: {', '.join(low_compat_steps)}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            score=overall_score,
            errors=errors,
            warnings=warnings,
            details=details,
            suggestions=suggestions
        )
    
    def find_conversion_path(self, source_format: DataFormat,
                           target_format: DataFormat) -> Optional[ConversionPath]:
        """Find optimal conversion path between formats."""
        return self._find_conversion_path(source_format, target_format)
    
    def get_compatible_adapters(self, request) -> List[Tuple[Any, float]]:
        """Get adapters compatible with the conversion request."""
        # This would integrate with the ShimAdapter registry
        # For now, return empty list as adapters are implemented separately
        return []
    
    def get_domain_profile(self, domain_name: str) -> Optional[DomainProfile]:
        """Get domain profile by name."""
        return self._domain_profiles.get(domain_name)
    
    def list_domains(self) -> List[str]:
        """Get list of registered domain names."""
        return list(self._domain_profiles.keys())
    
    def get_compatibility_level(self, score: float) -> CompatibilityLevel:
        """Get compatibility level for a given score."""
        if score >= CompatibilityLevel.PERFECT.score_threshold:
            return CompatibilityLevel.PERFECT
        elif score >= CompatibilityLevel.HIGH.score_threshold:
            return CompatibilityLevel.HIGH
        elif score >= CompatibilityLevel.MODERATE.score_threshold:
            return CompatibilityLevel.MODERATE
        elif score >= CompatibilityLevel.LOW.score_threshold:
            return CompatibilityLevel.LOW
        else:
            return CompatibilityLevel.INCOMPATIBLE
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get compatibility matrix usage statistics."""
        cache_stats = {}
        if self.enable_caching:
            cache_stats = {
                'cache_size': len(self._compatibility_cache),
                'cache_hit_rate': self._stats['cache_hits'] / max(self._stats['assessments'], 1)
            }
        
        return {
            'registered_domains': len(self._domain_profiles),
            'total_assessments': self._stats['assessments'],
            'total_validations': self._stats['validations'],
            **cache_stats
        }


# Factory Functions

def create_compatibility_matrix(enable_caching: bool = True,
                              cache_size: int = 500) -> PipelineCompatibilityMatrix:
    """Create a PipelineCompatibilityMatrix with standard configuration."""
    return PipelineCompatibilityMatrix(
        enable_caching=enable_caching,
        cache_size=cache_size
    )


def create_minimal_compatibility_matrix() -> PipelineCompatibilityMatrix:
    """Create a minimal compatibility matrix without caching."""
    return PipelineCompatibilityMatrix(
        enable_caching=False,
        cache_size=0
    )


# Utility Functions

def assess_pipeline_compatibility(pipeline_domains: List[str],
                                 matrix: Optional[PipelineCompatibilityMatrix] = None) -> ValidationResult:
    """Assess compatibility of a pipeline with given domain sequence."""
    if matrix is None:
        matrix = create_compatibility_matrix()
    
    return matrix.validate_pipeline(pipeline_domains)


def find_optimal_format_for_domains(domains: List[str],
                                   matrix: Optional[PipelineCompatibilityMatrix] = None) -> Optional[DataFormat]:
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
        
        domain_formats = set(profile.base_requirements.input_formats + 
                           profile.base_requirements.output_formats)
        
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


def suggest_pipeline_improvements(pipeline_domains: List[str],
                                 matrix: Optional[PipelineCompatibilityMatrix] = None) -> List[str]:
    """Suggest improvements for pipeline compatibility."""
    if matrix is None:
        matrix = create_compatibility_matrix()
    
    validation = matrix.validate_pipeline(pipeline_domains)
    suggestions = list(validation.suggestions)
    
    # Add format-specific suggestions
    optimal_format = find_optimal_format_for_domains(pipeline_domains, matrix)
    if optimal_format:
        suggestions.append(f"Consider using {optimal_format.value} as the primary data format")
    
    # Analyze problematic transitions
    if hasattr(validation.details, 'items'):
        for step_key, step_info in validation.details.items():
            if isinstance(step_info, dict) and step_info.get('compatibility_score', 1.0) < 0.6:
                source_domain = step_info.get('source_domain')
                target_domain = step_info.get('target_domain')
                if source_domain and target_domain:
                    suggestions.append(f"Consider adding an intermediate step between {source_domain} and {target_domain}")
    
    return suggestions