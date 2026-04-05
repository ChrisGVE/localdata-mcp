"""
Core PipelineCompatibilityMatrix implementation.

Provides comprehensive compatibility scoring, pathway discovery, and pipeline
validation for LocalData MCP domains with LLM-friendly interfaces.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from ..interfaces import (
    DataFormat,
    CompatibilityMatrix,
    CompatibilityScore,
    ConversionPath,
    ValidationResult,
    DomainRequirements,
)
from ....logging_manager import get_logger
from ._types import CompatibilityLevel, DomainProfile
from ._domains import build_standard_domain_profiles
from ._scoring import (
    build_format_compatibility_matrix,
    calculate_compatibility_score,
    find_conversion_path,
    identify_compatibility_issues,
    generate_recommendations,
)

logger = get_logger(__name__)


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
        self._format_compatibility_matrix = build_format_compatibility_matrix()

        # Caching
        if enable_caching:
            self._compatibility_cache: Dict[str, CompatibilityScore] = {}

        # Statistics
        self._stats = {"assessments": 0, "cache_hits": 0, "validations": 0}

        # Initialize with standard LocalData MCP domains
        self._initialize_standard_domains()

        logger.info(
            "PipelineCompatibilityMatrix initialized",
            domains=len(self._domain_profiles),
            caching_enabled=enable_caching,
        )

    def _initialize_standard_domains(self) -> None:
        """Initialize compatibility profiles for standard LocalData MCP domains."""
        profiles = build_standard_domain_profiles()
        for domain_name, profile in profiles.items():
            self.register_domain_requirements(domain_name, profile)

        logger.info(
            "Initialized standard domain profiles",
            domains=list(self._domain_profiles.keys()),
        )

    def get_compatibility(
        self, source_format: DataFormat, target_format: DataFormat
    ) -> CompatibilityScore:
        """
        Get compatibility score between two data formats.

        Args:
            source_format: Source data format
            target_format: Target data format

        Returns:
            Comprehensive compatibility score with conversion details
        """
        self._stats["assessments"] += 1

        # Check cache
        cache_key = f"{source_format.value}->{target_format.value}"
        if self.enable_caching and cache_key in self._compatibility_cache:
            self._stats["cache_hits"] += 1
            return self._compatibility_cache[cache_key]

        # Calculate compatibility
        score = calculate_compatibility_score(
            self._format_compatibility_matrix, source_format, target_format
        )

        # Determine if conversion is required
        direct_compatible = source_format == target_format
        conversion_required = not direct_compatible and score > 0.0

        # Find conversion path if needed
        conversion_path = None
        if conversion_required:
            conversion_path = find_conversion_path(
                self._format_compatibility_matrix, source_format, target_format
            )

        # Identify issues and recommendations
        issues = identify_compatibility_issues(source_format, target_format, score)
        recommendations = generate_recommendations(source_format, target_format, score)

        # Create compatibility score
        compatibility_score = CompatibilityScore(
            score=score,
            direct_compatible=direct_compatible,
            conversion_required=conversion_required,
            conversion_path=conversion_path,
            compatibility_issues=issues,
            recommendations=recommendations,
        )

        # Cache result
        if self.enable_caching:
            if len(self._compatibility_cache) >= self.cache_size:
                oldest_key = next(iter(self._compatibility_cache))
                del self._compatibility_cache[oldest_key]
            self._compatibility_cache[cache_key] = compatibility_score

        return compatibility_score

    def register_domain_requirements(
        self, domain_name: str, requirements: Union[DomainRequirements, DomainProfile]
    ) -> None:
        """Register domain requirements in the compatibility matrix."""
        if isinstance(requirements, DomainRequirements):
            profile = DomainProfile(
                domain_name=domain_name, base_requirements=requirements
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
        self._stats["validations"] += 1

        errors: List[str] = []
        warnings: List[str] = []
        details: Dict[str, Any] = {}
        suggestions: List[str] = []

        if len(pipeline_steps) < 2:
            return ValidationResult(
                is_valid=True,
                score=1.0,
                details={
                    "message": "Single step pipeline requires no compatibility validation"
                },
            )

        total_score = 0.0
        step_count = 0

        for i in range(len(pipeline_steps) - 1):
            current_domain = pipeline_steps[i]
            next_domain = pipeline_steps[i + 1]

            current_profile = self._domain_profiles.get(current_domain)
            next_profile = self._domain_profiles.get(next_domain)

            if not current_profile:
                errors.append(f"Unknown domain: {current_domain}")
                continue
            if not next_profile:
                errors.append(f"Unknown domain: {next_domain}")
                continue

            best_score, best_src, best_tgt = self._best_transition_score(
                current_profile, next_profile
            )

            total_score += best_score
            step_count += 1

            step_key = f"step_{i}_to_{i + 1}"
            details[step_key] = {
                "source_domain": current_domain,
                "target_domain": next_domain,
                "best_source_format": best_src.value if best_src else None,
                "best_target_format": best_tgt.value if best_tgt else None,
                "compatibility_score": best_score,
                "conversion_required": best_score < 1.0,
            }

            if best_score < 0.3:
                errors.append(
                    f"Very low compatibility between {current_domain} and {next_domain} (score: {best_score:.2f})"
                )
            elif best_score < 0.6:
                warnings.append(
                    f"Moderate compatibility issues between {current_domain} and {next_domain} (score: {best_score:.2f})"
                )

        overall_score = total_score / step_count if step_count > 0 else 0.0
        suggestions = self._build_pipeline_suggestions(overall_score, details)

        return ValidationResult(
            is_valid=len(errors) == 0,
            score=overall_score,
            errors=errors,
            warnings=warnings,
            details=details,
            suggestions=suggestions,
        )

    def _best_transition_score(
        self, current_profile: DomainProfile, next_profile: DomainProfile
    ) -> Tuple[float, Optional[DataFormat], Optional[DataFormat]]:
        """Find best compatibility score between two domain profiles."""
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

        return best_score, best_source_format, best_target_format

    @staticmethod
    def _build_pipeline_suggestions(
        overall_score: float, details: Dict[str, Any]
    ) -> List[str]:
        """Build suggestion list from pipeline validation results."""
        suggestions: List[str] = []

        if overall_score < 0.7:
            suggestions.append(
                "Pipeline has compatibility issues - consider reordering steps or adding intermediate conversions"
            )

        if any(details[key]["conversion_required"] for key in details):
            suggestions.append(
                "Some steps require data format conversion - ensure proper adapters are available"
            )

        low_compat_steps = [
            key for key in details if details[key]["compatibility_score"] < 0.5
        ]
        if low_compat_steps:
            suggestions.append(
                f"Steps with very low compatibility: {', '.join(low_compat_steps)}"
            )

        return suggestions

    def find_conversion_path(
        self, source_format: DataFormat, target_format: DataFormat
    ) -> Optional[ConversionPath]:
        """Find optimal conversion path between formats."""
        return find_conversion_path(
            self._format_compatibility_matrix, source_format, target_format
        )

    def get_compatible_adapters(self, request) -> List[Tuple[Any, float]]:
        """Get adapters compatible with the conversion request."""
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
                "cache_size": len(self._compatibility_cache),
                "cache_hit_rate": self._stats["cache_hits"]
                / max(self._stats["assessments"], 1),
            }

        return {
            "registered_domains": len(self._domain_profiles),
            "total_assessments": self._stats["assessments"],
            "total_validations": self._stats["validations"],
            **cache_stats,
        }
