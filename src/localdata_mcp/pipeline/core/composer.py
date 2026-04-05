"""
PipelineComposer: Multi-pipeline workflow orchestration for LocalData MCP v2.0

This module provides the PipelineComposer class that orchestrates multiple
DataSciencePipeline instances for complex multi-stage workflows with
dependency resolution, parallel execution, and metadata enrichment.
"""

import time
import uuid
import concurrent.futures
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from ..base import (
    CompositionMetadata,
    PipelineError,
    PipelineResult,
    PipelineState,
    ErrorClassification,
    StreamingConfig,
)
from ...streaming import StreamingQueryExecutor, MemoryStatus
from ...logging_manager import get_logging_manager, get_logger

from .pipeline_class import DataSciencePipeline

logger = get_logger(__name__)


class PipelineComposer:
    """
    Orchestrates multiple DataSciencePipeline instances for complex multi-stage workflows.

    The PipelineComposer implements sophisticated workflow orchestration capabilities:
    - Sequential execution with metadata flow between pipeline stages
    - Parallel execution for independent analyses
    - Intelligent dependency resolution and execution scheduling
    - Automatic composition metadata enrichment for LLM tool chaining
    - Error propagation with graceful recovery and partial results
    - Streaming integration with memory-aware execution

    Key Features:
    - Context-aware composition with rich metadata for intelligent tool chaining
    - Progressive disclosure from simple workflows to complex orchestration
    - Intention-driven workflow design ("Analyze customer data" -> multi-stage workflow)
    - Streaming-first architecture supporting large dataset workflows
    - Modular integration with easy addition of new pipeline stages

    Example Usage:
        # Sequential workflow
        composer = PipelineComposer('sequential')
        composer.add_pipeline('cleaning', data_cleaning_pipeline)
        composer.add_pipeline('analysis', statistical_analysis_pipeline, depends_on='cleaning')
        composer.add_pipeline('visualization', viz_pipeline, depends_on='analysis')
        results = composer.execute(data)

        # Parallel analysis
        composer = PipelineComposer('parallel')
        composer.add_pipeline('stats', stats_pipeline)
        composer.add_pipeline('ml', ml_pipeline)
        composer.add_pipeline('viz', viz_pipeline)
        results = composer.execute(data)
    """

    def __init__(
        self,
        composition_strategy: str = "sequential",
        metadata_enrichment: bool = True,
        streaming_aware: bool = True,
        error_recovery_mode: str = "partial",
        max_parallel_pipelines: int = 4,
        composition_timeout_seconds: int = 3600,
    ):
        """
        Initialize PipelineComposer with workflow orchestration capabilities.

        Args:
            composition_strategy: Strategy for pipeline execution ('sequential', 'parallel', 'adaptive')
            metadata_enrichment: Whether to generate enriched composition metadata
            streaming_aware: Enable streaming integration for large datasets
            error_recovery_mode: Error handling strategy ('strict', 'partial', 'continue')
            max_parallel_pipelines: Maximum number of pipelines to run in parallel
            composition_timeout_seconds: Timeout for entire composition workflow
        """
        self.composition_strategy = composition_strategy
        self.metadata_enrichment = metadata_enrichment
        self.streaming_aware = streaming_aware
        self.error_recovery_mode = error_recovery_mode
        self.max_parallel_pipelines = max_parallel_pipelines
        self.composition_timeout_seconds = composition_timeout_seconds

        # Pipeline registry and dependency graph
        self._pipelines: Dict[str, DataSciencePipeline] = {}
        self._pipeline_metadata: Dict[str, Dict[str, Any]] = {}
        self._dependency_graph: Dict[str, List[str]] = defaultdict(
            list
        )  # name -> [dependencies]
        self._reverse_dependencies: Dict[str, List[str]] = defaultdict(
            list
        )  # name -> [dependents]

        # Execution state management
        self._composer_id = str(uuid.uuid4())
        self._execution_order: List[str] = []
        self._parallel_groups: List[List[str]] = []
        self._composition_metadata: Optional[CompositionMetadata] = None

        # Results and error tracking
        self._pipeline_results: Dict[str, PipelineResult] = {}
        self._pipeline_errors: Dict[str, Exception] = {}
        self._partial_results: Dict[str, Any] = {}
        self._execution_metrics: Dict[str, Dict[str, Any]] = {}

        # Streaming and performance management
        self._streaming_executor: Optional[StreamingQueryExecutor] = None
        self._memory_tracker: List[MemoryStatus] = []
        self._optimization_cache: Dict[str, Any] = {}

        # Logging integration
        self._logging_manager = get_logging_manager()
        self._request_id: Optional[str] = None

        logger.info(
            "PipelineComposer initialized",
            composer_id=self._composer_id,
            strategy=composition_strategy,
            metadata_enrichment=metadata_enrichment,
            streaming_aware=streaming_aware,
            error_recovery=error_recovery_mode,
        )

    @property
    def composer_id(self) -> str:
        """Get unique composer identifier."""
        return self._composer_id

    @property
    def registered_pipelines(self) -> Dict[str, str]:
        """Get registry of pipeline names and their types."""
        return {
            name: type(pipeline).__name__ for name, pipeline in self._pipelines.items()
        }

    @property
    def composition_metadata(self) -> Optional[CompositionMetadata]:
        """Get enriched composition metadata for tool chaining."""
        return self._composition_metadata

    def add_pipeline(
        self,
        name: str,
        pipeline: DataSciencePipeline,
        depends_on: Optional[Union[str, List[str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        data_transformation: Optional[Callable] = None,
    ) -> "PipelineComposer":
        """
        Add a pipeline to the composition with optional dependencies.

        Args:
            name: Unique name for the pipeline
            pipeline: DataSciencePipeline instance to add
            depends_on: Pipeline name(s) this pipeline depends on
            metadata: Additional metadata for the pipeline
            data_transformation: Function to transform data between pipelines

        Returns:
            Self for method chaining

        Raises:
            ValueError: If pipeline name already exists or invalid dependencies
        """
        if name in self._pipelines:
            raise ValueError(f"Pipeline '{name}' already exists in composition")

        # Validate dependencies
        dependencies = []
        if depends_on:
            if isinstance(depends_on, str):
                dependencies = [depends_on]
            else:
                dependencies = list(depends_on)

            for dep in dependencies:
                if dep not in self._pipelines:
                    raise ValueError(f"Dependency '{dep}' not found in composition")

        # Register pipeline
        self._pipelines[name] = pipeline
        self._pipeline_metadata[name] = {
            "dependencies": dependencies,
            "data_transformation": data_transformation,
            "metadata": metadata or {},
            "registration_time": time.time(),
            "pipeline_type": type(pipeline).__name__,
            "analytical_intention": getattr(
                pipeline, "analytical_intention", "Unknown"
            ),
        }

        # Update dependency graph
        self._dependency_graph[name] = dependencies
        for dep in dependencies:
            self._reverse_dependencies[dep].append(name)

        # Invalidate cached execution order
        self._execution_order = []
        self._parallel_groups = []

        logger.info(
            f"Pipeline '{name}' added to composition",
            composer_id=self._composer_id,
            pipeline_type=type(pipeline).__name__,
            dependencies=dependencies,
            total_pipelines=len(self._pipelines),
        )

        return self

    def resolve_dependencies(self) -> Dict[str, Any]:
        """
        Analyze and resolve pipeline dependencies to create execution plan.

        Returns:
            Dependency analysis report with execution order and parallel groups

        Raises:
            ValueError: If circular dependencies are detected
        """
        if not self._pipelines:
            return {
                "execution_order": [],
                "parallel_groups": [],
                "dependency_analysis": "No pipelines registered",
            }

        # Detect circular dependencies using DFS
        def has_cycle(node: str, visited: set, rec_stack: set) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in self._dependency_graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        visited = set()
        for pipeline_name in self._pipelines:
            if pipeline_name not in visited:
                if has_cycle(pipeline_name, visited, set()):
                    raise ValueError(
                        f"Circular dependency detected involving pipeline '{pipeline_name}'"
                    )

        # Topological sort for execution order
        execution_order = self._topological_sort()
        self._execution_order = execution_order

        # Identify parallel groups (pipelines with no interdependencies)
        parallel_groups = self._identify_parallel_groups(execution_order)
        self._parallel_groups = parallel_groups

        dependency_report = {
            "execution_order": execution_order,
            "parallel_groups": parallel_groups,
            "dependency_graph": dict(self._dependency_graph),
            "reverse_dependencies": dict(self._reverse_dependencies),
            "total_pipelines": len(self._pipelines),
            "parallelizable_pipelines": sum(
                len(group) for group in parallel_groups if len(group) > 1
            ),
            "dependency_analysis": "Valid dependency graph - no cycles detected",
        }

        logger.info(
            "Dependencies resolved",
            composer_id=self._composer_id,
            execution_order=execution_order,
            parallel_groups_count=len([g for g in parallel_groups if len(g) > 1]),
            total_stages=len(parallel_groups),
        )

        return dependency_report

    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, PipelineResult]:
        """
        Execute the composition using the configured strategy.

        Args:
            data: Input data for the composition
            **kwargs: Additional parameters for pipeline execution

        Returns:
            Dictionary of pipeline results keyed by pipeline name
        """
        strategy = kwargs.pop("strategy", self.composition_strategy)
        return self._execute_composition(strategy, data, **kwargs)

    def _topological_sort(self) -> List[str]:
        """
        Perform topological sort to determine pipeline execution order.

        Returns:
            List of pipeline names in execution order
        """
        # Kahn's algorithm for topological sorting
        in_degree = {name: 0 for name in self._pipelines}
        for name in self._pipelines:
            for dep in self._dependency_graph.get(name, []):
                in_degree[name] += 1

        # Find all nodes with no incoming edges
        queue = deque([name for name, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)

            # For each dependent of current node
            for dependent in self._reverse_dependencies.get(current, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # Check if all nodes were processed (no cycle)
        if len(result) != len(self._pipelines):
            raise ValueError("Circular dependency detected in pipeline composition")

        return result

    def _identify_parallel_groups(self, execution_order: List[str]) -> List[List[str]]:
        """
        Identify groups of pipelines that can be executed in parallel.

        Args:
            execution_order: Topologically sorted pipeline names

        Returns:
            List of pipeline groups, each group can be executed in parallel
        """
        groups = []
        processed = set()

        for pipeline_name in execution_order:
            if pipeline_name in processed:
                continue

            # Start a new group with this pipeline
            parallel_group = [pipeline_name]
            dependencies = set(self._dependency_graph.get(pipeline_name, []))

            # Check remaining pipelines to see which can run in parallel
            for other_name in execution_order:
                if other_name == pipeline_name or other_name in processed:
                    continue

                other_deps = set(self._dependency_graph.get(other_name, []))

                # Can run in parallel if they don't depend on each other
                # and have compatible dependency requirements
                if (
                    pipeline_name not in other_deps
                    and other_name not in dependencies
                    and dependencies == other_deps
                ):  # Same level in dependency hierarchy
                    parallel_group.append(other_name)

            groups.append(parallel_group)
            processed.update(parallel_group)

        return groups

    def _execute_composition(
        self, strategy: str, data: pd.DataFrame, **kwargs
    ) -> Dict[str, PipelineResult]:
        """
        Execute pipeline composition with specified strategy.

        Args:
            strategy: Execution strategy ('sequential', 'parallel', 'adaptive')
            data: Input data
            **kwargs: Additional execution parameters

        Returns:
            Dictionary of pipeline results
        """
        if not self._pipelines:
            raise ValueError("No pipelines registered in composition")

        # Start logging context
        self._request_id = self._logging_manager.log_query_start(
            database_name="composition",
            query=f"execute_{strategy}_{self._composer_id}",
            database_type="pipeline_composition",
        )

        start_time = time.time()

        with self._logging_manager.context(
            request_id=self._request_id,
            operation="pipeline_composition",
            component="pipeline_composer",
        ):
            try:
                # Resolve dependencies and create execution plan
                dependency_report = self.resolve_dependencies()

                # Initialize execution state
                self._pipeline_results = {}
                self._pipeline_errors = {}
                self._partial_results = {}
                self._execution_metrics = {}

                # Build initial composition metadata
                if self.metadata_enrichment:
                    self._composition_metadata = self._build_composition_metadata(
                        data, strategy
                    )

                # Execute based on strategy
                if strategy == "sequential":
                    results = self._execute_sequential_workflow(
                        data, dependency_report, **kwargs
                    )
                elif strategy == "parallel":
                    results = self._execute_parallel_workflow(
                        data, dependency_report, **kwargs
                    )
                elif strategy == "adaptive":
                    results = self._execute_adaptive_workflow(
                        data, dependency_report, **kwargs
                    )
                else:
                    raise ValueError(f"Unsupported composition strategy: {strategy}")

                # Enrich composition metadata with execution results
                if self.metadata_enrichment and self._composition_metadata:
                    self._enrich_metadata_with_results(results)

                execution_time = time.time() - start_time

                # Log successful completion
                self._logging_manager.log_query_complete(
                    request_id=self._request_id,
                    database_name="composition",
                    database_type="pipeline_composition",
                    duration=execution_time,
                    success=True,
                )

                logger.info(
                    "Pipeline composition executed successfully",
                    composer_id=self._composer_id,
                    strategy=strategy,
                    execution_time=execution_time,
                    successful_pipelines=len(
                        [r for r in results.values() if r.success]
                    ),
                    total_pipelines=len(results),
                )

                return results

            except Exception as e:
                execution_time = time.time() - start_time

                # Handle composition error
                error_info = self._handle_composition_error(
                    e,
                    strategy,
                    {
                        "execution_time": execution_time,
                        "data_shape": data.shape,
                        "total_pipelines": len(self._pipelines),
                    },
                )

                self._logging_manager.log_query_complete(
                    request_id=self._request_id,
                    database_name="composition",
                    database_type="pipeline_composition",
                    duration=execution_time,
                    success=False,
                )

                # Return partial results if available in recovery mode
                if self.error_recovery_mode in ["partial", "continue"]:
                    return self._pipeline_results
                else:
                    raise PipelineError(
                        f"Pipeline composition failed: {str(e)}",
                        classification=ErrorClassification.CONFIGURATION_ERROR,
                        pipeline_stage="composition",
                        context=error_info,
                        partial_results=self._pipeline_results,
                        recovery_suggestions=[
                            "Check pipeline dependencies and configuration",
                            "Enable error recovery mode for partial results",
                            "Verify input data format and size",
                        ],
                    ) from e

    def _execute_sequential_workflow(
        self, data: pd.DataFrame, dependency_report: Dict[str, Any], **kwargs
    ) -> Dict[str, PipelineResult]:
        """
        Execute pipelines sequentially with metadata flow between stages.

        Args:
            data: Input data
            dependency_report: Resolved dependency information
            **kwargs: Additional execution parameters

        Returns:
            Dictionary of pipeline results
        """
        execution_order = dependency_report["execution_order"]
        current_data = data
        results = {}

        for pipeline_name in execution_order:
            try:
                pipeline = self._pipelines[pipeline_name]
                metadata = self._pipeline_metadata[pipeline_name]

                # Apply data transformation if specified
                if metadata.get("data_transformation"):
                    current_data = metadata["data_transformation"](current_data)

                # Execute pipeline
                logger.debug(
                    f"Executing pipeline '{pipeline_name}' sequentially",
                    composer_id=self._composer_id,
                    data_shape=current_data.shape,
                )

                # Fit and transform with the pipeline
                pipeline_start_time = time.time()
                pipeline.fit(current_data)
                result = pipeline.transform(current_data)

                # If the result is a PipelineResult, use it directly
                if isinstance(result, PipelineResult):
                    pipeline_result = result
                else:
                    # Create PipelineResult wrapper
                    pipeline_result = PipelineResult(
                        success=True,
                        data=result,
                        metadata={
                            "pipeline_name": pipeline_name,
                            "sequential_execution": True,
                            "execution_order": execution_order.index(pipeline_name),
                        },
                        execution_time_seconds=time.time() - pipeline_start_time,
                        memory_used_mb=self._calculate_pipeline_memory_usage(pipeline),
                        pipeline_stage=pipeline.state.value
                        if hasattr(pipeline, "state")
                        else "completed",
                        composition_metadata=pipeline.composition_metadata
                        if hasattr(pipeline, "composition_metadata")
                        else None,
                    )

                results[pipeline_name] = pipeline_result

                # Use pipeline output as input for next stage if it's DataFrame-like
                if isinstance(result, pd.DataFrame):
                    current_data = result
                elif hasattr(result, "data") and isinstance(result.data, pd.DataFrame):
                    current_data = result.data

                # Propagate metadata between pipelines
                self._propagate_metadata_between_pipelines(
                    pipeline_name, pipeline_result
                )

                logger.info(
                    f"Pipeline '{pipeline_name}' completed successfully in sequential workflow",
                    composer_id=self._composer_id,
                    execution_time=pipeline_result.execution_time_seconds,
                    output_shape=getattr(result, "shape", "N/A"),
                )

            except Exception as e:
                # Handle pipeline error in sequential workflow
                error_result = self._handle_pipeline_error_in_composition(
                    pipeline_name, e, "sequential"
                )
                results[pipeline_name] = error_result

                if self.error_recovery_mode == "strict":
                    # Stop execution on first error in strict mode
                    break
                elif self.error_recovery_mode == "continue":
                    # Continue with next pipeline, using original data
                    continue
                # For 'partial' mode, continue with whatever data we have

        return results

    def _execute_parallel_workflow(
        self, data: pd.DataFrame, dependency_report: Dict[str, Any], **kwargs
    ) -> Dict[str, PipelineResult]:
        """
        Execute pipelines in parallel for independent analyses.

        Args:
            data: Input data (copied to each parallel pipeline)
            dependency_report: Resolved dependency information
            **kwargs: Additional execution parameters

        Returns:
            Dictionary of pipeline results
        """
        parallel_groups = dependency_report["parallel_groups"]
        results = {}

        for group_idx, group in enumerate(parallel_groups):
            if len(group) == 1:
                # Single pipeline execution
                pipeline_name = group[0]
                result = self._execute_single_pipeline(pipeline_name, data, kwargs)
                results[pipeline_name] = result
            else:
                # Parallel execution within group
                group_results = self._execute_parallel_group(group, data, kwargs)
                results.update(group_results)

        return results

    def _execute_adaptive_workflow(
        self, data: pd.DataFrame, dependency_report: Dict[str, Any], **kwargs
    ) -> Dict[str, PipelineResult]:
        """
        Execute pipelines using adaptive strategy based on data characteristics and dependencies.

        Args:
            data: Input data
            dependency_report: Resolved dependency information
            **kwargs: Additional execution parameters

        Returns:
            Dictionary of pipeline results
        """
        # Analyze data characteristics to choose strategy
        data_size_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
        has_dependencies = any(
            len(deps) > 0 for deps in self._dependency_graph.values()
        )
        parallelizable_count = dependency_report["parallelizable_pipelines"]

        # Adaptive strategy decision
        if has_dependencies and data_size_mb > 100:
            # Large data with dependencies - use sequential with streaming
            logger.info(
                "Adaptive strategy: Sequential execution with streaming for large data",
                composer_id=self._composer_id,
                data_size_mb=data_size_mb,
                has_dependencies=has_dependencies,
            )
            return self._execute_sequential_workflow(data, dependency_report, **kwargs)
        elif parallelizable_count > 1 and data_size_mb < 500:
            # Moderate data size with parallelizable pipelines
            logger.info(
                "Adaptive strategy: Parallel execution for independent pipelines",
                composer_id=self._composer_id,
                parallelizable_count=parallelizable_count,
                data_size_mb=data_size_mb,
            )
            return self._execute_parallel_workflow(data, dependency_report, **kwargs)
        else:
            # Default to sequential for complex dependency graphs or very large data
            logger.info(
                "Adaptive strategy: Sequential execution (default)",
                composer_id=self._composer_id,
                has_dependencies=has_dependencies,
                data_size_mb=data_size_mb,
            )
            return self._execute_sequential_workflow(data, dependency_report, **kwargs)

    def _execute_single_pipeline(
        self, pipeline_name: str, data: pd.DataFrame, kwargs: Dict[str, Any]
    ) -> PipelineResult:
        """
        Execute a single pipeline with error handling.

        Args:
            pipeline_name: Name of pipeline to execute
            data: Input data
            kwargs: Additional parameters

        Returns:
            PipelineResult from execution
        """
        try:
            pipeline = self._pipelines[pipeline_name]
            metadata = self._pipeline_metadata[pipeline_name]

            # Apply data transformation if specified
            if metadata.get("data_transformation"):
                data = metadata["data_transformation"](data)

            logger.debug(
                f"Executing single pipeline '{pipeline_name}'",
                composer_id=self._composer_id,
                data_shape=data.shape,
            )

            # Execute pipeline
            pipeline_start_time = time.time()
            pipeline.fit(data)
            result = pipeline.transform(data)

            # Ensure we return a PipelineResult
            if isinstance(result, PipelineResult):
                return result
            else:
                return PipelineResult(
                    success=True,
                    data=result,
                    metadata={"pipeline_name": pipeline_name, "single_execution": True},
                    execution_time_seconds=time.time() - pipeline_start_time,
                    memory_used_mb=self._calculate_pipeline_memory_usage(pipeline),
                    pipeline_stage=pipeline.state.value
                    if hasattr(pipeline, "state")
                    else "completed",
                    composition_metadata=pipeline.composition_metadata
                    if hasattr(pipeline, "composition_metadata")
                    else None,
                )

        except Exception as e:
            return self._handle_pipeline_error_in_composition(
                pipeline_name, e, "single"
            )

    def _execute_parallel_group(
        self, group: List[str], data: pd.DataFrame, kwargs: Dict[str, Any]
    ) -> Dict[str, PipelineResult]:
        """
        Execute a group of pipelines in parallel.

        Args:
            group: List of pipeline names to execute in parallel
            data: Input data (copied to each pipeline)
            kwargs: Additional parameters

        Returns:
            Dictionary of pipeline results
        """
        results = {}

        # Use ThreadPoolExecutor for I/O bound operations
        max_workers = min(len(group), self.max_parallel_pipelines)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all pipelines for parallel execution
            future_to_pipeline = {}
            for pipeline_name in group:
                # Create a copy of data for each pipeline to avoid conflicts
                pipeline_data = data.copy() if hasattr(data, "copy") else data
                future = executor.submit(
                    self._execute_single_pipeline, pipeline_name, pipeline_data, kwargs
                )
                future_to_pipeline[future] = pipeline_name

            # Collect results as they complete
            for future in concurrent.futures.as_completed(
                future_to_pipeline, timeout=self.composition_timeout_seconds
            ):
                pipeline_name = future_to_pipeline[future]
                try:
                    result = future.result()
                    results[pipeline_name] = result

                    logger.info(
                        f"Pipeline '{pipeline_name}' completed in parallel group",
                        composer_id=self._composer_id,
                        success=result.success,
                        execution_time=result.execution_time_seconds,
                    )

                except Exception as e:
                    error_result = self._handle_pipeline_error_in_composition(
                        pipeline_name, e, "parallel"
                    )
                    results[pipeline_name] = error_result

        return results

    def _build_composition_metadata(
        self, data: pd.DataFrame, strategy: str
    ) -> CompositionMetadata:
        """
        Build comprehensive composition metadata for tool chaining.

        Args:
            data: Input data
            strategy: Execution strategy

        Returns:
            Rich composition metadata for LLM tool chaining
        """
        # Analyze the overall composition domain and type
        domains = set()
        analysis_types = set()
        compatible_tools = set()

        for pipeline_name, pipeline in self._pipelines.items():
            # Extract domain information from pipelines
            if (
                hasattr(pipeline, "composition_metadata")
                and pipeline.composition_metadata
            ):
                meta = pipeline.composition_metadata
                domains.add(meta.domain)
                analysis_types.add(meta.analysis_type)
                compatible_tools.update(meta.compatible_tools)

        # Determine composite domain and analysis type
        if len(domains) == 1:
            composite_domain = list(domains)[0]
        elif "ml" in domains or "machine_learning" in domains:
            composite_domain = "ml"
        else:
            composite_domain = "multi_domain"

        if len(analysis_types) == 1:
            composite_analysis_type = list(analysis_types)[0]
        else:
            composite_analysis_type = "multi_stage_analysis"

        # Generate tool suggestions for the composition
        composition_tools = list(compatible_tools)
        composition_tools.extend(
            [
                "pipeline_visualization",
                "workflow_monitoring",
                "result_comparison",
                "performance_analysis",
            ]
        )

        # Create suggested next steps
        suggested_compositions = [
            {
                "tool": "visualize_pipeline_results",
                "purpose": "Visualize outputs from multiple pipeline stages",
                "priority": "high",
            },
            {
                "tool": "compare_pipeline_performance",
                "purpose": "Compare execution metrics across pipelines",
                "priority": "medium",
            },
            {
                "tool": "export_workflow_results",
                "purpose": "Export complete workflow results for reporting",
                "priority": "medium",
            },
        ]

        return CompositionMetadata(
            domain=composite_domain,
            analysis_type=composite_analysis_type,
            result_type="multi_pipeline_composition",
            compatible_tools=composition_tools,
            suggested_compositions=suggested_compositions,
            data_artifacts={
                "total_pipelines": len(self._pipelines),
                "execution_strategy": strategy,
                "dependency_complexity": len(
                    [d for d in self._dependency_graph.values() if len(d) > 0]
                ),
                "parallelizable_stages": len(
                    [g for g in self._parallel_groups if len(g) > 1]
                ),
            },
            input_schema={
                "columns": list(data.columns),
                "dtypes": {str(col): str(dtype) for col, dtype in data.dtypes.items()},
                "shape": data.shape,
                "composition_input": True,
            },
            transformation_summary={
                "multi_stage_workflow": True,
                "composition_strategy": strategy,
                "registered_pipelines": list(self._pipelines.keys()),
                "execution_order": self._execution_order,
                "streaming_aware": self.streaming_aware,
                "error_recovery_mode": self.error_recovery_mode,
            },
            confidence_level=0.8,  # High confidence for composed workflows
            quality_score=0.75,  # Good quality baseline for orchestration
            recommended_next_steps=[
                {
                    "action": "analyze_workflow_performance",
                    "description": "Review execution metrics and optimize pipeline ordering",
                    "priority": "high",
                },
                {
                    "action": "visualize_pipeline_outputs",
                    "description": "Create visualizations comparing outputs from different stages",
                    "priority": "high",
                },
                {
                    "action": "export_comprehensive_report",
                    "description": "Generate detailed report with all pipeline results",
                    "priority": "medium",
                },
            ],
        )

    def _enrich_metadata_with_results(self, results: Dict[str, PipelineResult]):
        """
        Enrich composition metadata with execution results.

        Args:
            results: Dictionary of pipeline results
        """
        if not self._composition_metadata:
            return

        successful_pipelines = [
            name for name, result in results.items() if result.success
        ]
        failed_pipelines = [
            name for name, result in results.items() if not result.success
        ]

        # Update quality score based on success rate
        success_rate = len(successful_pipelines) / len(results) if results else 0
        self._composition_metadata.quality_score = 0.5 + (success_rate * 0.5)

        # Update confidence level
        self._composition_metadata.confidence_level = min(
            0.95, success_rate * 0.9 + 0.1
        )

        # Add execution artifacts
        self._composition_metadata.data_artifacts.update(
            {
                "successful_pipelines": successful_pipelines,
                "failed_pipelines": failed_pipelines,
                "success_rate": success_rate,
                "total_execution_time": sum(
                    r.execution_time_seconds for r in results.values()
                ),
                "total_memory_used": sum(r.memory_used_mb for r in results.values()),
                "execution_timestamp": time.time(),
            }
        )

        # Update limitations based on failures
        if failed_pipelines:
            self._composition_metadata.limitations.extend(
                [
                    f"Pipeline failures: {', '.join(failed_pipelines)}",
                    "Partial results available - some analysis stages incomplete",
                ]
            )

    def _propagate_metadata_between_pipelines(
        self, pipeline_name: str, result: PipelineResult
    ):
        """
        Propagate metadata between pipeline stages for enhanced composition context.

        Args:
            pipeline_name: Name of the pipeline that just completed
            result: Result from the completed pipeline
        """
        # Store pipeline-specific metadata for downstream use
        self._execution_metrics[pipeline_name] = {
            "execution_time": result.execution_time_seconds,
            "memory_used": result.memory_used_mb,
            "success": result.success,
            "stage": result.pipeline_stage,
            "timestamp": time.time(),
        }

        # Update dependent pipelines with context from this pipeline
        dependents = self._reverse_dependencies.get(pipeline_name, [])
        for dependent_name in dependents:
            if dependent_name in self._pipelines:
                dependent_pipeline = self._pipelines[dependent_name]

                # Update custom parameters with upstream context
                if hasattr(dependent_pipeline, "custom_parameters"):
                    dependent_pipeline.custom_parameters.update(
                        {
                            f"upstream_{pipeline_name}_success": result.success,
                            f"upstream_{pipeline_name}_execution_time": result.execution_time_seconds,
                            f"upstream_{pipeline_name}_stage": result.pipeline_stage,
                        }
                    )

    def _handle_pipeline_error_in_composition(
        self, pipeline_name: str, error: Exception, context: str
    ) -> PipelineResult:
        """
        Handle errors that occur during pipeline execution within composition.

        Args:
            pipeline_name: Name of the failed pipeline
            error: Exception that occurred
            context: Execution context ('sequential', 'parallel', 'single')

        Returns:
            PipelineResult representing the error
        """
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "pipeline_name": pipeline_name,
            "composition_context": context,
            "composer_id": self._composer_id,
            "timestamp": time.time(),
        }

        self._pipeline_errors[pipeline_name] = error

        # Log the error
        logger.error(
            f"Pipeline '{pipeline_name}' failed in {context} composition",
            composer_id=self._composer_id,
            error_type=type(error).__name__,
            error_message=str(error),
            context=context,
        )

        return PipelineResult(
            success=False,
            data=None,
            metadata={"error_info": error_info, "pipeline_name": pipeline_name},
            execution_time_seconds=0,
            memory_used_mb=0,
            pipeline_stage="error",
            error=error_info,
        )

    def _handle_composition_error(
        self, error: Exception, strategy: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle errors that occur during composition execution.

        Args:
            error: Exception that occurred
            strategy: Execution strategy that failed
            context: Additional context information

        Returns:
            Error information dictionary
        """
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "composition_strategy": strategy,
            "composer_id": self._composer_id,
            "timestamp": time.time(),
            "context": context,
        }

        # Log the composition error
        self._logging_manager.log_error(
            error,
            "pipeline_composer",
            composer_id=self._composer_id,
            strategy=strategy,
            **error_info,
        )

        return error_info

    def _calculate_pipeline_memory_usage(self, pipeline: DataSciencePipeline) -> float:
        """
        Calculate memory usage for a specific pipeline.

        Args:
            pipeline: Pipeline to calculate memory for

        Returns:
            Memory usage in MB
        """
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def get_composition_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of the pipeline composition.

        Returns:
            Detailed composition summary
        """
        summary = {
            "composer_id": self._composer_id,
            "composition_strategy": self.composition_strategy,
            "total_pipelines": len(self._pipelines),
            "registered_pipelines": self.registered_pipelines,
            "dependency_graph": dict(self._dependency_graph),
            "execution_order": self._execution_order,
            "parallel_groups": self._parallel_groups,
            "metadata_enrichment_enabled": self.metadata_enrichment,
            "streaming_aware": self.streaming_aware,
            "error_recovery_mode": self.error_recovery_mode,
            "max_parallel_pipelines": self.max_parallel_pipelines,
            "composition_timeout_seconds": self.composition_timeout_seconds,
        }

        # Add execution results if available
        if self._pipeline_results:
            summary["execution_results"] = {
                "successful_pipelines": [
                    name
                    for name, result in self._pipeline_results.items()
                    if result.success
                ],
                "failed_pipelines": [
                    name
                    for name, result in self._pipeline_results.items()
                    if not result.success
                ],
                "total_execution_time": sum(
                    r.execution_time_seconds for r in self._pipeline_results.values()
                ),
                "total_memory_used": sum(
                    r.memory_used_mb for r in self._pipeline_results.values()
                ),
            }

        # Add composition metadata if available
        if self._composition_metadata:
            summary["composition_metadata"] = {
                "domain": self._composition_metadata.domain,
                "analysis_type": self._composition_metadata.analysis_type,
                "result_type": self._composition_metadata.result_type,
                "compatible_tools": self._composition_metadata.compatible_tools,
                "confidence_level": self._composition_metadata.confidence_level,
                "quality_score": self._composition_metadata.quality_score,
            }

        return summary
