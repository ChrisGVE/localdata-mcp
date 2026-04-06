"""
Complete pipeline composition validator with execution plan generation.

Validates entire pipeline chains, checks for circular dependencies,
identifies bottlenecks, and generates optimized execution plans.
"""

import time
from typing import Any, Dict, List

import networkx as nx

from ....logging_manager import get_logger
from ..compatibility_matrix import PipelineCompatibilityMatrix
from ..shim_registry import ShimRegistry
from .analyzer import PipelineAnalyzer
from .injector import ShimInjector
from .types import (
    AnalysisType,
    InjectionStrategy,
    PipelineStep,
)

logger = get_logger(__name__)


class PipelineValidator:
    """
    Complete pipeline composition validator with execution plan generation.

    Validates entire pipeline chains, checks for circular dependencies,
    identifies bottlenecks, and generates optimized execution plans.
    """

    def __init__(
        self,
        compatibility_matrix: PipelineCompatibilityMatrix,
        shim_registry: ShimRegistry,
        analyzer: PipelineAnalyzer,
        injector: ShimInjector,
    ):
        """
        Initialize PipelineValidator.

        Args:
            compatibility_matrix: Matrix for compatibility assessment
            shim_registry: Registry of available shim adapters
            analyzer: Pipeline analyzer for issue detection
            injector: Shim injector for automatic fixes
        """
        self.compatibility_matrix = compatibility_matrix
        self.shim_registry = shim_registry
        self.analyzer = analyzer
        self.injector = injector

        # Validation statistics
        self._stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "pipelines_fixed": 0,
        }

        logger.info("PipelineValidator initialized")

    def validate_and_fix_pipeline(
        self,
        pipeline_steps: List[PipelineStep],
        auto_fix: bool = True,
        validation_level: str = "strict",
    ) -> Dict[str, Any]:
        """
        Validate pipeline and optionally auto-fix issues.

        Args:
            pipeline_steps: Pipeline steps to validate
            auto_fix: Whether to automatically fix detected issues
            validation_level: 'strict', 'moderate', or 'lenient'

        Returns:
            Comprehensive validation result with fixes applied
        """
        start_time = time.time()
        self._stats["total_validations"] += 1

        pipeline_id = f"validation_{int(time.time() * 1000)}"

        logger.info(
            f"Starting pipeline validation",
            pipeline_id=pipeline_id,
            steps_count=len(pipeline_steps),
            auto_fix=auto_fix,
            validation_level=validation_level,
        )

        try:
            validation_result = {
                "pipeline_id": pipeline_id,
                "original_steps_count": len(pipeline_steps),
                "validation_level": validation_level,
                "auto_fix_enabled": auto_fix,
                "is_valid": False,
                "validation_errors": [],
                "validation_warnings": [],
                "structural_issues": [],
                "compatibility_issues": [],
                "performance_issues": [],
                "fixes_applied": [],
                "final_pipeline": pipeline_steps.copy(),
                "execution_plan": None,
                "validation_score": 0.0,
                "execution_time": 0.0,
            }

            # 1. Structural validation
            structural_issues = self._validate_pipeline_structure(pipeline_steps)
            validation_result["structural_issues"] = structural_issues

            if structural_issues and validation_level == "strict":
                validation_result["validation_errors"].extend(
                    [f"Structural issue: {issue}" for issue in structural_issues]
                )

            # 2. Compatibility analysis
            analysis_result = self.analyzer.analyze_pipeline(
                pipeline_steps, AnalysisType.COMPLETE, pipeline_id
            )

            validation_result["compatibility_issues"] = (
                analysis_result.identified_issues
            )
            validation_result["validation_score"] = analysis_result.compatibility_score

            # 3. Auto-fix if enabled and issues found
            fixed_pipeline = pipeline_steps.copy()
            if auto_fix and not analysis_result.is_compatible:
                try:
                    fixed_pipeline, injection_metadata = (
                        self.injector.inject_shims_for_pipeline(
                            pipeline_steps, analysis_result, InjectionStrategy.BALANCED
                        )
                    )

                    validation_result["fixes_applied"] = injection_metadata.get(
                        "injections", []
                    )
                    validation_result["final_pipeline"] = fixed_pipeline

                    if injection_metadata.get("injections"):
                        self._stats["pipelines_fixed"] += 1

                    # Re-analyze fixed pipeline
                    fixed_analysis = self.analyzer.analyze_pipeline(
                        fixed_pipeline,
                        AnalysisType.COMPATIBILITY,
                        f"{pipeline_id}_fixed",
                    )

                    validation_result["is_valid"] = fixed_analysis.is_compatible
                    validation_result["validation_score"] = (
                        fixed_analysis.compatibility_score
                    )

                except Exception as e:
                    logger.error(f"Auto-fix failed: {e}")
                    validation_result["validation_errors"].append(
                        f"Auto-fix failed: {str(e)}"
                    )
            else:
                validation_result["is_valid"] = analysis_result.is_compatible

            # 4. Generate execution plan
            if validation_result["is_valid"] or validation_level == "lenient":
                execution_plan = self._generate_execution_plan(fixed_pipeline)
                validation_result["execution_plan"] = execution_plan

            # 5. Performance validation
            performance_issues = self._validate_performance(fixed_pipeline)
            validation_result["performance_issues"] = performance_issues

            if performance_issues and validation_level == "strict":
                validation_result["validation_warnings"].extend(
                    [f"Performance issue: {issue}" for issue in performance_issues]
                )

            self._stats["successful_validations"] += 1

            validation_result["execution_time"] = time.time() - start_time

            logger.info(
                f"Pipeline validation completed",
                pipeline_id=pipeline_id,
                is_valid=validation_result["is_valid"],
                fixes_applied=len(validation_result["fixes_applied"]),
                final_score=validation_result["validation_score"],
            )

            return validation_result

        except Exception as e:
            self._stats["failed_validations"] += 1
            logger.error(f"Pipeline validation failed: {e}")

            return {
                "pipeline_id": pipeline_id,
                "is_valid": False,
                "validation_errors": [f"Validation failed: {str(e)}"],
                "execution_time": time.time() - start_time,
            }

    def _validate_pipeline_structure(
        self, pipeline_steps: List[PipelineStep]
    ) -> List[str]:
        """Validate structural integrity of pipeline."""
        issues = []

        if not pipeline_steps:
            issues.append("Pipeline is empty")
            return issues

        if len(pipeline_steps) < 2:
            issues.append("Pipeline must have at least 2 steps")

        # Check for duplicate step IDs
        step_ids = [step.step_id for step in pipeline_steps]
        if len(step_ids) != len(set(step_ids)):
            issues.append("Duplicate step IDs found")

        # Check for circular dependencies
        if self._has_circular_dependencies(pipeline_steps):
            issues.append("Circular dependencies detected")

        # Validate format flow continuity
        for i in range(len(pipeline_steps) - 1):
            current_step = pipeline_steps[i]
            next_step = pipeline_steps[i + 1]

            if current_step.output_format != next_step.input_format:
                # This is expected and should be handled by compatibility analysis
                continue

        return issues

    def _has_circular_dependencies(self, pipeline_steps: List[PipelineStep]) -> bool:
        """Check for circular dependencies in pipeline."""
        # Create directed graph
        graph = nx.DiGraph()

        for step in pipeline_steps:
            graph.add_node(step.step_id)

        # Add edges based on step sequence
        for i in range(len(pipeline_steps) - 1):
            graph.add_edge(pipeline_steps[i].step_id, pipeline_steps[i + 1].step_id)

        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(graph))
            return len(cycles) > 0
        except:
            return False

    def _validate_performance(self, pipeline_steps: List[PipelineStep]) -> List[str]:
        """Validate pipeline performance characteristics."""
        issues = []

        # Check for potential bottlenecks
        memory_intensive_steps = 0
        compute_intensive_steps = 0

        for step in pipeline_steps:
            # Check for known memory-intensive operations
            if any(
                keyword in step.operation.lower()
                for keyword in ["sparse", "matrix", "large", "dense"]
            ):
                memory_intensive_steps += 1

            # Check for compute-intensive operations
            if any(
                keyword in step.operation.lower()
                for keyword in ["model", "train", "optimize", "search"]
            ):
                compute_intensive_steps += 1

        if memory_intensive_steps > len(pipeline_steps) * 0.5:
            issues.append(
                "High number of memory-intensive operations may cause performance issues"
            )

        if compute_intensive_steps > len(pipeline_steps) * 0.3:
            issues.append(
                "High number of compute-intensive operations may cause long execution times"
            )

        return issues

    def _generate_execution_plan(
        self, pipeline_steps: List[PipelineStep]
    ) -> Dict[str, Any]:
        """Generate optimized execution plan for pipeline."""
        plan = {
            "steps": [],
            "estimated_total_time": 0.0,
            "estimated_total_memory": 0.0,
            "parallel_opportunities": [],
            "optimization_suggestions": [],
        }

        for i, step in enumerate(pipeline_steps):
            step_plan = {
                "step_index": i,
                "step_id": step.step_id,
                "operation": step.operation,
                "domain": step.domain,
                "input_format": step.input_format.value,
                "output_format": step.output_format.value,
                "estimated_time": self._estimate_step_time(step),
                "estimated_memory": self._estimate_step_memory(step),
                "dependencies": [pipeline_steps[i - 1].step_id] if i > 0 else [],
                "can_parallelize": self._can_step_parallelize(step),
            }

            plan["steps"].append(step_plan)
            plan["estimated_total_time"] += step_plan["estimated_time"]
            plan["estimated_total_memory"] = max(
                plan["estimated_total_memory"], step_plan["estimated_memory"]
            )

        # Identify parallel opportunities
        plan["parallel_opportunities"] = self._identify_parallel_opportunities(
            plan["steps"]
        )

        # Generate optimization suggestions
        plan["optimization_suggestions"] = self._generate_optimization_suggestions(plan)

        return plan

    def _estimate_step_time(self, step: PipelineStep) -> float:
        """Estimate execution time for a pipeline step."""
        # Simple heuristic based on operation type
        base_time = 1.0  # seconds

        if "convert" in step.operation.lower():
            base_time *= 2.0
        elif "model" in step.operation.lower():
            base_time *= 10.0
        elif "analysis" in step.operation.lower():
            base_time *= 3.0

        return base_time

    def _estimate_step_memory(self, step: PipelineStep) -> float:
        """Estimate memory usage for a pipeline step."""
        # Simple heuristic based on operation type
        base_memory = 100.0  # MB

        if "sparse" in step.operation.lower():
            base_memory *= 0.5
        elif "dense" in step.operation.lower():
            base_memory *= 5.0
        elif "matrix" in step.operation.lower():
            base_memory *= 3.0

        return base_memory

    def _can_step_parallelize(self, step: PipelineStep) -> bool:
        """Check if step can be parallelized."""
        # Simple heuristic
        parallel_operations = ["analysis", "transform", "process"]
        return any(op in step.operation.lower() for op in parallel_operations)

    def _identify_parallel_opportunities(
        self, steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify opportunities for parallel execution."""
        opportunities = []

        # Look for independent steps that can run in parallel
        for i in range(len(steps) - 1):
            current_step = steps[i]
            next_step = steps[i + 1]

            if (
                current_step["can_parallelize"]
                and next_step["can_parallelize"]
                and len(next_step["dependencies"]) == 1
            ):
                opportunity = {
                    "type": "parallel_sequence",
                    "steps": [current_step["step_id"], next_step["step_id"]],
                    "potential_time_saving": min(
                        current_step["estimated_time"], next_step["estimated_time"]
                    )
                    * 0.8,
                }
                opportunities.append(opportunity)

        return opportunities

    def _generate_optimization_suggestions(self, plan: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions for execution plan."""
        suggestions = []

        total_time = plan["estimated_total_time"]
        total_memory = plan["estimated_total_memory"]

        if total_time > 300:  # 5 minutes
            suggestions.append(
                "Consider implementing parallel processing to reduce execution time"
            )

        if total_memory > 2000:  # 2GB
            suggestions.append(
                "Consider using streaming processing to reduce memory usage"
            )

        if len(plan["parallel_opportunities"]) > 0:
            suggestions.append(
                f"Found {len(plan['parallel_opportunities'])} opportunities for parallel execution"
            )

        # Check for consecutive conversion steps
        conversion_steps = [
            step for step in plan["steps"] if "convert" in step["operation"].lower()
        ]
        if len(conversion_steps) > 2:
            suggestions.append(
                "Multiple consecutive conversions detected - consider combining or optimizing"
            )

        return suggestions

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return self._stats.copy()
