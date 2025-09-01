"""
Integration Registry - Unified Non-sklearn Library Integration Management
LocalData MCP v2.0

This module provides the central registry for all library integration adapters,
creating a unified interface for the pipeline system to access any non-sklearn
library seamlessly through the established adapter patterns.

Key Features:
- Centralized adapter registration and management
- Unified function lookup across all library categories
- Automatic fallback chain resolution
- Dependency status reporting and installation guidance
- Rich metadata aggregation for pipeline composition
- Performance monitoring and optimization suggestions

Integration with Pipeline Framework:
- Extends the existing ToolRegistry protocol
- Provides CompositionStage integration
- Supports streaming execution strategies
- Maintains validation and error handling patterns
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, OrderedDict
import importlib
from datetime import datetime

# Import base integration architecture
from library_integration_shims import (
    BaseLibraryAdapter,
    LibraryCategory,
    LibraryDependency,
    IntegrationStrategy,
    IntegrationMetadata,
    LibraryIntegrationResult,
    DependencyManager,
    CompositionError
)

# Import specific adapters
from geospatial_integration_adapter import GeospatialAdapter
from timeseries_integration_adapter import TimeSeriesAdapter  
from statistics_integration_adapter import StatisticsAdapter

# Import pipeline framework components
from pipeline_composition_framework import (
    CompositionStage,
    ToolRegistry,
    ValidationResult
)

logger = logging.getLogger(__name__)


# ============================================================================
# Registry Data Structures
# ============================================================================

@dataclass
class AdapterInfo:
    """Information about a registered adapter."""
    adapter: BaseLibraryAdapter
    category: LibraryCategory
    priority: int = 1  # Higher priority adapters tried first
    enabled: bool = True
    last_used: Optional[datetime] = None
    success_rate: float = 1.0
    average_execution_time: float = 0.0
    
    def update_performance(self, success: bool, execution_time: float):
        """Update performance metrics."""
        # Simple exponential moving average
        alpha = 0.1
        self.success_rate = (1 - alpha) * self.success_rate + alpha * (1.0 if success else 0.0)
        self.average_execution_time = (1 - alpha) * self.average_execution_time + alpha * execution_time
        self.last_used = datetime.now()


@dataclass  
class FunctionMapping:
    """Mapping of function names to adapters."""
    function_name: str
    adapter_category: LibraryCategory
    primary_library: str
    fallback_libraries: List[str] = field(default_factory=list)
    description: str = ""
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    streaming_compatible: bool = True
    estimated_complexity: int = 1  # 1=simple, 5=complex


@dataclass
class IntegrationReport:
    """Comprehensive integration status report."""
    total_adapters: int
    active_adapters: int
    total_functions: int
    available_functions: int
    library_status: Dict[str, bool] = field(default_factory=dict)
    missing_libraries: List[str] = field(default_factory=list)
    integration_coverage: Dict[LibraryCategory, float] = field(default_factory=dict)
    performance_summary: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Unified Integration Registry
# ============================================================================

class IntegrationRegistry:
    """
    Central registry for all library integration adapters.
    
    Manages:
    - Adapter registration and lifecycle
    - Function discovery and routing
    - Dependency checking and fallback resolution
    - Performance monitoring and optimization
    - Integration with pipeline composition system
    """
    
    def __init__(self):
        self.adapters: Dict[LibraryCategory, AdapterInfo] = {}
        self.function_mappings: Dict[str, FunctionMapping] = {}
        self.dependency_manager = DependencyManager()
        
        # Performance tracking
        self.execution_stats = defaultdict(list)
        self.error_history = defaultdict(list)
        
        # Auto-register core adapters
        self._register_core_adapters()
        self._build_function_mappings()
        
    def _register_core_adapters(self):
        """Register the core adapter implementations."""
        core_adapters = [
            (GeospatialAdapter(), LibraryCategory.GEOSPATIAL, 1),
            (TimeSeriesAdapter(), LibraryCategory.TIME_SERIES, 1),
            (StatisticsAdapter(), LibraryCategory.STATISTICS, 1)
        ]
        
        for adapter, category, priority in core_adapters:
            self.register_adapter(adapter, category, priority)
    
    def register_adapter(self, 
                        adapter: BaseLibraryAdapter,
                        category: LibraryCategory,
                        priority: int = 1):
        """Register a new integration adapter."""
        adapter_info = AdapterInfo(
            adapter=adapter,
            category=category,
            priority=priority
        )
        
        self.adapters[category] = adapter_info
        logger.info(f"Registered {category.value} adapter with priority {priority}")
        
        # Update function mappings
        self._update_function_mappings_for_adapter(adapter, category)
    
    def get_adapter(self, category: LibraryCategory) -> Optional[BaseLibraryAdapter]:
        """Get adapter for a specific category."""
        adapter_info = self.adapters.get(category)
        if adapter_info and adapter_info.enabled:
            return adapter_info.adapter
        return None
    
    def get_function_adapter(self, function_name: str) -> Optional[Tuple[BaseLibraryAdapter, FunctionMapping]]:
        """Get adapter and mapping for a specific function."""
        if function_name not in self.function_mappings:
            return None
        
        mapping = self.function_mappings[function_name]
        adapter = self.get_adapter(mapping.adapter_category)
        
        if adapter:
            return adapter, mapping
        return None
    
    def list_available_functions(self, 
                                category: Optional[LibraryCategory] = None,
                                include_unavailable: bool = False) -> Dict[str, FunctionMapping]:
        """List all available functions, optionally filtered by category."""
        functions = {}
        
        for func_name, mapping in self.function_mappings.items():
            # Filter by category if specified
            if category and mapping.adapter_category != category:
                continue
            
            # Check if adapter is available
            adapter = self.get_adapter(mapping.adapter_category)
            if adapter or include_unavailable:
                functions[func_name] = mapping
        
        return functions
    
    def execute_function(self,
                        function_name: str,
                        data: Any,
                        parameters: Dict[str, Any],
                        fallback_on_error: bool = True) -> LibraryIntegrationResult:
        """
        Execute a function through the appropriate adapter.
        
        Provides unified interface for all non-sklearn library functions
        with automatic fallback handling and performance tracking.
        """
        start_time = datetime.now()
        
        try:
            # Get adapter and mapping
            adapter_mapping = self.get_function_adapter(function_name)
            if not adapter_mapping:
                raise CompositionError(
                    f"Function {function_name} not found in registry",
                    error_type="function_not_found"
                )
            
            adapter, mapping = adapter_mapping
            
            # Execute the function
            result_data, metadata = adapter.adapt_function_call(
                function_name, data, parameters
            )
            
            # Track performance
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_stats(function_name, True, execution_time)
            
            # Create comprehensive result
            result = LibraryIntegrationResult(
                data=result_data,
                metadata=metadata,
                performance_info={
                    'execution_time_seconds': execution_time,
                    'adapter_used': mapping.adapter_category.value,
                    'primary_library': mapping.primary_library
                }
            )
            
            # Add composition hints based on output type
            self._add_composition_hints(result, mapping)
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_stats(function_name, False, execution_time)
            
            if fallback_on_error:
                return self._attempt_fallback_execution(
                    function_name, data, parameters, e
                )
            else:
                raise CompositionError(
                    f"Function execution failed: {e}",
                    error_type="execution_failed"
                )
    
    def validate_function_composition(self,
                                    functions: List[str],
                                    data_flow: List[str]) -> ValidationResult:
        """
        Validate that a sequence of functions can be composed together.
        
        Checks:
        - Function availability
        - Data type compatibility
        - Adapter dependency resolution
        """
        errors = []
        warnings = []
        
        for i, function_name in enumerate(functions):
            # Check function exists
            if function_name not in self.function_mappings:
                errors.append({
                    'function': function_name,
                    'error': 'Function not found in registry',
                    'position': i
                })
                continue
            
            mapping = self.function_mappings[function_name]
            adapter = self.get_adapter(mapping.adapter_category)
            
            # Check adapter availability
            if not adapter:
                errors.append({
                    'function': function_name,
                    'error': f'Adapter for {mapping.adapter_category.value} not available',
                    'position': i,
                    'suggestions': mapping.fallback_libraries
                })
                continue
            
            # Check data type compatibility with next function
            if i < len(functions) - 1:
                next_function = functions[i + 1]
                if next_function in self.function_mappings:
                    next_mapping = self.function_mappings[next_function]
                    
                    # Check if output types are compatible with input types
                    if not self._types_compatible(mapping.output_types, next_mapping.input_types):
                        warnings.append({
                            'from_function': function_name,
                            'to_function': next_function,
                            'warning': 'Potential type incompatibility',
                            'output_types': mapping.output_types,
                            'input_types': next_mapping.input_types
                        })
        
        return ValidationResult(
            valid=len(errors) == 0,
            category='function_composition',
            errors=errors,
            warnings=warnings
        )
    
    def get_integration_report(self) -> IntegrationReport:
        """Generate comprehensive integration status report."""
        total_adapters = len(self.adapters)
        active_adapters = sum(1 for info in self.adapters.values() if info.enabled)
        total_functions = len(self.function_mappings)
        
        # Check function availability
        available_functions = 0
        for func_name in self.function_mappings:
            if self.get_function_adapter(func_name):
                available_functions += 1
        
        # Library status
        library_status = {}
        missing_libraries = []
        
        for adapter_info in self.adapters.values():
            for dep in adapter_info.adapter.dependencies:
                is_available = adapter_info.adapter.is_library_available(dep.name)
                library_status[dep.name] = is_available
                if not is_available and not dep.is_optional:
                    missing_libraries.append(dep.name)
        
        # Integration coverage by category
        coverage = {}
        for category in LibraryCategory:
            if category in self.adapters:
                adapter_info = self.adapters[category]
                available_deps = sum(1 for dep in adapter_info.adapter.dependencies
                                   if adapter_info.adapter.is_library_available(dep.name))
                total_deps = len(adapter_info.adapter.dependencies)
                coverage[category] = available_deps / total_deps if total_deps > 0 else 0.0
            else:
                coverage[category] = 0.0
        
        # Performance summary
        performance_summary = {}
        for category, adapter_info in self.adapters.items():
            performance_summary[category.value] = {
                'success_rate': adapter_info.success_rate,
                'avg_execution_time': adapter_info.average_execution_time,
                'last_used': adapter_info.last_used.isoformat() if adapter_info.last_used else None
            }
        
        return IntegrationReport(
            total_adapters=total_adapters,
            active_adapters=active_adapters,
            total_functions=total_functions,
            available_functions=available_functions,
            library_status=library_status,
            missing_libraries=missing_libraries,
            integration_coverage=coverage,
            performance_summary=performance_summary
        )
    
    def suggest_installation_commands(self) -> Dict[str, str]:
        """Generate installation commands for missing libraries."""
        commands = {}
        
        for adapter_info in self.adapters.values():
            for dep in adapter_info.adapter.dependencies:
                if not adapter_info.adapter.is_library_available(dep.name):
                    commands[dep.name] = dep.installation_hint
        
        return commands
    
    def optimize_adapter_priorities(self):
        """Optimize adapter priorities based on performance history."""
        for category, adapter_info in self.adapters.items():
            # Adjust priority based on success rate and execution time
            performance_score = adapter_info.success_rate / (1 + adapter_info.average_execution_time)
            
            # Update priority (higher performance = higher priority)
            if performance_score > 0.8:
                adapter_info.priority = min(5, adapter_info.priority + 1)
            elif performance_score < 0.5:
                adapter_info.priority = max(1, adapter_info.priority - 1)
    
    # ========================================================================
    # Pipeline Framework Integration
    # ========================================================================
    
    def create_composition_stage(self,
                               function_name: str,
                               parameters: Dict[str, Any]) -> Optional[CompositionStage]:
        """Create a CompositionStage for pipeline integration."""
        adapter_mapping = self.get_function_adapter(function_name)
        if not adapter_mapping:
            return None
        
        adapter, mapping = adapter_mapping
        
        return CompositionStage(
            tool_name="integration_registry",
            function=function_name,
            parameters=parameters,
            expected_input_types=mapping.input_types,
            output_types=mapping.output_types,
            is_streaming=mapping.streaming_compatible,
            estimated_duration_seconds=float(mapping.estimated_complexity),
            estimated_memory_mb=float(mapping.estimated_complexity * 100)  # Rough estimate
        )
    
    def get_output_types(self, function_name: str) -> List[str]:
        """Get output types for ToolRegistry protocol."""
        if function_name in self.function_mappings:
            return self.function_mappings[function_name].output_types
        return []
    
    def get_input_types(self, function_name: str) -> List[str]:
        """Get input types for ToolRegistry protocol."""
        if function_name in self.function_mappings:
            return self.function_mappings[function_name].input_types
        return []
    
    def is_function_available(self, function_name: str) -> bool:
        """Check if function is available for ToolRegistry protocol."""
        return self.get_function_adapter(function_name) is not None
    
    # ========================================================================
    # Private Helper Methods
    # ========================================================================
    
    def _build_function_mappings(self):
        """Build comprehensive function mappings from all adapters."""
        for category, adapter_info in self.adapters.items():
            self._update_function_mappings_for_adapter(adapter_info.adapter, category)
    
    def _update_function_mappings_for_adapter(self, 
                                            adapter: BaseLibraryAdapter,
                                            category: LibraryCategory):
        """Update function mappings for a specific adapter."""
        supported_functions = adapter.get_supported_functions()
        
        for func_name in supported_functions.keys():
            # Extract metadata about the function
            input_types, output_types = self._infer_function_types(func_name, category)
            streaming_compatible = self._is_function_streaming_compatible(func_name, adapter)
            
            mapping = FunctionMapping(
                function_name=func_name,
                adapter_category=category,
                primary_library=self._get_primary_library_for_function(func_name, category),
                fallback_libraries=self._get_fallback_libraries_for_function(func_name, category),
                description=self._get_function_description(func_name),
                input_types=input_types,
                output_types=output_types,
                streaming_compatible=streaming_compatible,
                estimated_complexity=self._estimate_function_complexity(func_name)
            )
            
            self.function_mappings[func_name] = mapping
    
    def _infer_function_types(self, 
                            function_name: str,
                            category: LibraryCategory) -> Tuple[List[str], List[str]]:
        """Infer input and output types for a function."""
        # Default types based on category
        if category == LibraryCategory.GEOSPATIAL:
            input_types = ["pandas.DataFrame", "geopandas.GeoDataFrame"]
            output_types = ["geopandas.GeoDataFrame", "pandas.DataFrame"]
        elif category == LibraryCategory.TIME_SERIES:
            input_types = ["pandas.DataFrame", "pandas.Series"]
            output_types = ["pandas.DataFrame", "dict"]
        elif category == LibraryCategory.STATISTICS:
            input_types = ["pandas.DataFrame", "numpy.ndarray"]
            output_types = ["pandas.DataFrame", "dict", "float"]
        else:
            input_types = ["pandas.DataFrame"]
            output_types = ["pandas.DataFrame"]
        
        return input_types, output_types
    
    def _is_function_streaming_compatible(self, 
                                        function_name: str,
                                        adapter: BaseLibraryAdapter) -> bool:
        """Check if function supports streaming execution."""
        # This could be enhanced with more sophisticated detection
        streaming_patterns = [
            'prepare', 'handle', 'create', 'detect', 'transform'
        ]
        
        for pattern in streaming_patterns:
            if pattern in function_name.lower():
                return True
        
        return False
    
    def _get_primary_library_for_function(self, 
                                        function_name: str,
                                        category: LibraryCategory) -> str:
        """Get primary library for a function."""
        library_mapping = {
            LibraryCategory.GEOSPATIAL: "geopandas",
            LibraryCategory.TIME_SERIES: "statsmodels", 
            LibraryCategory.STATISTICS: "scipy"
        }
        
        return library_mapping.get(category, "unknown")
    
    def _get_fallback_libraries_for_function(self,
                                           function_name: str,
                                           category: LibraryCategory) -> List[str]:
        """Get fallback libraries for a function."""
        fallback_mapping = {
            LibraryCategory.GEOSPATIAL: ["shapely", "sklearn"],
            LibraryCategory.TIME_SERIES: ["scipy", "sklearn"],
            LibraryCategory.STATISTICS: ["numpy", "sklearn"]
        }
        
        return fallback_mapping.get(category, [])
    
    def _get_function_description(self, function_name: str) -> str:
        """Get description for a function."""
        # This could be enhanced by extracting from docstrings
        return f"Integrated {function_name.replace('_', ' ')} function"
    
    def _estimate_function_complexity(self, function_name: str) -> int:
        """Estimate computational complexity of function."""
        complex_patterns = ['forecast', 'regression', 'clustering', 'decomposition']
        simple_patterns = ['prepare', 'detect', 'create']
        
        for pattern in complex_patterns:
            if pattern in function_name.lower():
                return 4
        
        for pattern in simple_patterns:
            if pattern in function_name.lower():
                return 1
        
        return 2  # Default complexity
    
    def _types_compatible(self, output_types: List[str], input_types: List[str]) -> bool:
        """Check type compatibility between functions."""
        # Simple compatibility check - could be enhanced
        for output_type in output_types:
            if output_type in input_types:
                return True
            # Check for common compatible types
            if output_type == "pandas.DataFrame" and "geopandas.GeoDataFrame" in input_types:
                return True
        
        return False
    
    def _update_performance_stats(self, 
                                function_name: str,
                                success: bool,
                                execution_time: float):
        """Update performance statistics for a function."""
        if function_name in self.function_mappings:
            mapping = self.function_mappings[function_name]
            adapter_info = self.adapters.get(mapping.adapter_category)
            if adapter_info:
                adapter_info.update_performance(success, execution_time)
    
    def _add_composition_hints(self, 
                             result: LibraryIntegrationResult,
                             mapping: FunctionMapping):
        """Add hints for downstream tool composition."""
        # Add common downstream tools based on function type
        if 'forecast' in mapping.function_name:
            result.add_composition_hint('visualization', {'chart_type': 'timeseries'})
        elif 'correlation' in mapping.function_name:
            result.add_composition_hint('visualization', {'chart_type': 'heatmap'})
        elif 'spatial' in mapping.function_name:
            result.add_composition_hint('visualization', {'chart_type': 'map'})
    
    def _attempt_fallback_execution(self,
                                  function_name: str,
                                  data: Any,
                                  parameters: Dict[str, Any],
                                  original_error: Exception) -> LibraryIntegrationResult:
        """Attempt fallback execution when primary adapter fails."""
        logger.warning(f"Primary execution failed for {function_name}: {original_error}")
        
        # This could be enhanced with more sophisticated fallback logic
        # For now, return error information
        raise CompositionError(
            f"Function execution failed and no fallback available: {original_error}",
            error_type="execution_failed_no_fallback"
        )


# ============================================================================
# Global Registry Instance
# ============================================================================

# Create global registry instance for easy access
integration_registry = IntegrationRegistry()


# ============================================================================
# Convenience Functions
# ============================================================================

def execute_integrated_function(function_name: str,
                               data: Any,
                               **parameters) -> LibraryIntegrationResult:
    """Convenience function for executing integrated functions."""
    return integration_registry.execute_function(function_name, data, parameters)


def list_available_integrations(category: Optional[LibraryCategory] = None) -> Dict[str, Any]:
    """List all available integration functions."""
    functions = integration_registry.list_available_functions(category)
    
    return {
        'functions': {name: {
            'category': mapping.adapter_category.value,
            'primary_library': mapping.primary_library,
            'streaming_compatible': mapping.streaming_compatible,
            'description': mapping.description
        } for name, mapping in functions.items()},
        'integration_report': integration_registry.get_integration_report()
    }


def get_installation_guide() -> Dict[str, Any]:
    """Get installation guide for missing libraries."""
    report = integration_registry.get_integration_report()
    commands = integration_registry.suggest_installation_commands()
    
    return {
        'missing_libraries': report.missing_libraries,
        'installation_commands': commands,
        'coverage_by_category': {cat.value: cov for cat, cov in report.integration_coverage.items()}
    }


if __name__ == "__main__":
    # Example usage and testing
    print("LocalData MCP Integration Registry")
    print("=" * 50)
    
    # Get integration report
    report = integration_registry.get_integration_report()
    print(f"Total adapters: {report.total_adapters}")
    print(f"Available functions: {report.available_functions}")
    print(f"Missing libraries: {report.missing_libraries}")
    
    # List some available functions
    functions = integration_registry.list_available_functions()
    print(f"\nAvailable functions ({len(functions)}):")
    for name, mapping in list(functions.items())[:5]:  # Show first 5
        print(f"  {name}: {mapping.description} [{mapping.primary_library}]")
    
    # Show installation guide
    if report.missing_libraries:
        print(f"\nInstallation guide:")
        commands = integration_registry.suggest_installation_commands()
        for lib, cmd in commands.items():
            print(f"  {lib}: {cmd}")