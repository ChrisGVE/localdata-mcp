"""
Factory and utility functions for domain shim creation and management.

Provides convenience functions for creating individual shims, creating all shims,
finding compatible shims, and validating shim configurations.
"""

from typing import Dict, List, Optional

from ..shim_registry import AdapterConfig
from ..interfaces import ValidationResult
from ....logging_manager import get_logger

from ._base import BaseDomainShim
from ._statistical import StatisticalShim
from ._regression import RegressionShim
from ._time_series import TimeSeriesShim
from ._pattern_recognition import PatternRecognitionShim

logger = get_logger(__name__)


# Factory functions for creating domain shims


def create_statistical_shim(
    adapter_id: str = "statistical_shim",
    config: Optional[AdapterConfig] = None,
    **kwargs,
) -> StatisticalShim:
    """Create a StatisticalShim with optional configuration."""
    return StatisticalShim(adapter_id=adapter_id, config=config, **kwargs)


def create_regression_shim(
    adapter_id: str = "regression_shim",
    config: Optional[AdapterConfig] = None,
    **kwargs,
) -> RegressionShim:
    """Create a RegressionShim with optional configuration."""
    return RegressionShim(adapter_id=adapter_id, config=config, **kwargs)


def create_time_series_shim(
    adapter_id: str = "time_series_shim",
    config: Optional[AdapterConfig] = None,
    **kwargs,
) -> TimeSeriesShim:
    """Create a TimeSeriesShim with optional configuration."""
    return TimeSeriesShim(adapter_id=adapter_id, config=config, **kwargs)


def create_pattern_recognition_shim(
    adapter_id: str = "pattern_recognition_shim",
    config: Optional[AdapterConfig] = None,
    **kwargs,
) -> PatternRecognitionShim:
    """Create a PatternRecognitionShim with optional configuration."""
    return PatternRecognitionShim(adapter_id=adapter_id, config=config, **kwargs)


# Utility functions for domain shim management


def create_all_domain_shims(
    registry: Optional["ShimRegistry"] = None,  # noqa: F821
    auto_register: bool = True,
) -> Dict[str, BaseDomainShim]:
    """
    Create all available domain shims.

    Args:
        registry: Optional registry to register shims with
        auto_register: Whether to automatically register shims with registry

    Returns:
        Dictionary mapping shim names to shim instances
    """
    shims = {
        "statistical": create_statistical_shim(),
        "regression": create_regression_shim(),
        "time_series": create_time_series_shim(),
        "pattern_recognition": create_pattern_recognition_shim(),
    }

    if registry and auto_register:
        for shim_name, shim in shims.items():
            registry.register_adapter(shim)
            logger.info(f"Registered domain shim: {shim_name}")

    return shims


def get_compatible_domain_shims(
    source_domain: str,
    target_domain: str,
    available_shims: Optional[Dict[str, BaseDomainShim]] = None,
) -> List[BaseDomainShim]:
    """
    Find compatible domain shims for a specific domain conversion.

    Args:
        source_domain: Source domain name
        target_domain: Target domain name
        available_shims: Optional dictionary of available shims

    Returns:
        List of compatible domain shims
    """
    if available_shims is None:
        available_shims = create_all_domain_shims(auto_register=False)

    compatible_shims = []

    for shim in available_shims.values():
        for mapping in shim.supported_mappings:
            if (
                mapping.source_domain == source_domain
                and mapping.target_domain == target_domain
            ):
                compatible_shims.append(shim)
                break

    return compatible_shims


def validate_domain_shim_configuration(
    shims: Dict[str, BaseDomainShim],
) -> ValidationResult:
    """
    Validate domain shim configuration for completeness and consistency.

    Args:
        shims: Dictionary of domain shims to validate

    Returns:
        Validation result with any configuration issues
    """
    errors = []
    warnings = []

    expected_shims = {"statistical", "regression", "time_series", "pattern_recognition"}
    available_shims = set(shims.keys())

    # Check for missing shims
    missing_shims = expected_shims - available_shims
    if missing_shims:
        warnings.append(f"Missing domain shims: {list(missing_shims)}")

    # Check shim configurations
    for shim_name, shim in shims.items():
        if not shim.supported_mappings:
            errors.append(f"Domain shim '{shim_name}' has no supported mappings")

        # Check for bidirectional mappings
        source_domains = {m.source_domain for m in shim.supported_mappings}
        target_domains = {m.target_domain for m in shim.supported_mappings}

        if len(source_domains) == 1 and len(target_domains) == 1:
            warnings.append(
                f"Domain shim '{shim_name}' only supports unidirectional conversion"
            )

    return ValidationResult(
        is_valid=len(errors) == 0,
        score=1.0 - len(errors) * 0.2 - len(warnings) * 0.1,
        errors=errors,
        warnings=warnings,
        details={
            "total_shims": len(shims),
            "expected_shims": len(expected_shims),
            "available_shims": list(available_shims),
            "missing_shims": list(missing_shims),
        },
    )
