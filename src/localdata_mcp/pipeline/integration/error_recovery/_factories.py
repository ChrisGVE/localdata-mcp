"""
Factory and utility functions for error recovery system creation.
"""

import logging
from typing import Any, Callable, Dict, Optional, Tuple

from ....logging_manager import get_logger
from ..interfaces import ConversionRequest
from ..shim_registry import ShimRegistry
from ._error_handler import ConversionErrorHandler
from ._pathway_engine import AlternativePathwayEngine
from ._recovery_engine import RecoveryStrategyEngine
from ._rollback_manager import RollbackManager

logger = get_logger(__name__)


# Factory Functions


def create_conversion_error_handler(**kwargs) -> ConversionErrorHandler:
    """Create ConversionErrorHandler with standard configuration."""
    return ConversionErrorHandler(**kwargs)


def create_alternative_pathway_engine(
    registry: Optional[ShimRegistry] = None, **kwargs
) -> AlternativePathwayEngine:
    """Create AlternativePathwayEngine with required dependencies."""
    return AlternativePathwayEngine(registry=registry, **kwargs)


def create_rollback_manager(**kwargs) -> RollbackManager:
    """Create RollbackManager with standard configuration."""
    return RollbackManager(**kwargs)


def create_recovery_strategy_engine(
    error_handler: ConversionErrorHandler,
    pathway_engine: AlternativePathwayEngine,
    rollback_manager: RollbackManager,
) -> RecoveryStrategyEngine:
    """Create RecoveryStrategyEngine with all components."""
    return RecoveryStrategyEngine(error_handler, pathway_engine, rollback_manager)


def create_complete_error_recovery_system(
    registry: Optional[ShimRegistry] = None, **kwargs
) -> Dict[str, Any]:
    """
    Create complete error recovery system with all components.

    Args:
        registry: Optional ShimRegistry for adapter discovery
        **kwargs: Additional configuration options

    Returns:
        Dictionary with all recovery system components
    """
    # Create individual components
    error_handler = create_conversion_error_handler(**kwargs.get("error_handler", {}))
    pathway_engine = create_alternative_pathway_engine(
        registry=registry, **kwargs.get("pathway_engine", {})
    )
    rollback_manager = create_rollback_manager(**kwargs.get("rollback_manager", {}))
    recovery_engine = create_recovery_strategy_engine(
        error_handler, pathway_engine, rollback_manager
    )

    logger.info("Complete error recovery system created")

    return {
        "error_handler": error_handler,
        "pathway_engine": pathway_engine,
        "rollback_manager": rollback_manager,
        "recovery_engine": recovery_engine,
    }


def create_error_recovery_framework(
    registry: Optional[ShimRegistry] = None, **kwargs
) -> Tuple[
    ConversionErrorHandler,
    AlternativePathwayEngine,
    RollbackManager,
    RecoveryStrategyEngine,
]:
    """
    Create complete error recovery framework components as tuple.

    Args:
        registry: Optional ShimRegistry for pathway discovery
        **kwargs: Component-specific configuration options

    Returns:
        Tuple of (error_handler, pathway_engine, rollback_manager, recovery_engine)
    """
    # Create individual components
    error_handler = create_conversion_error_handler(**kwargs.get("error_handler", {}))

    pathway_engine = create_alternative_pathway_engine(
        registry=registry, **kwargs.get("pathway_engine", {})
    )

    rollback_manager = create_rollback_manager(**kwargs.get("rollback_manager", {}))

    recovery_engine = create_recovery_strategy_engine(
        error_handler=error_handler,
        pathway_engine=pathway_engine,
        rollback_manager=rollback_manager,
        **kwargs.get("recovery_engine", {}),
    )

    return error_handler, pathway_engine, rollback_manager, recovery_engine


# Utility Functions


def handle_pipeline_error_with_recovery(
    error: Exception,
    conversion_request: ConversionRequest,
    recovery_system: Dict[str, Any],
    operation: Callable,
    *args,
    **kwargs,
) -> Tuple[bool, Any, Dict[str, Any]]:
    """
    High-level utility function to handle pipeline error with full recovery.

    Args:
        error: The error that occurred
        conversion_request: The conversion request that failed
        recovery_system: Complete recovery system from create_complete_error_recovery_system
        operation: Original operation to recover
        *args, **kwargs: Operation arguments

    Returns:
        Tuple of (success, result, recovery_metadata)
    """
    error_handler = recovery_system["error_handler"]
    recovery_engine = recovery_system["recovery_engine"]

    # Handle error and create context
    error_context = error_handler.handle_conversion_error(error, conversion_request)

    # Execute recovery
    return recovery_engine.execute_recovery(error_context, operation, *args, **kwargs)
