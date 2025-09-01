#!/usr/bin/env python3
"""
Baseline Establisher - Performance Baseline Management

Establishes and manages performance baselines for LocalData MCP:
- Creates v1.3.0 baseline metrics from benchmark results
- Stores baseline data with version tracking
- Provides baseline comparison functionality
- Manages baseline evolution across versions
- Supports multiple baseline contexts (development, production, etc.)
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import statistics
from enum import Enum


class BaselineType(Enum):
    """Types of baselines that can be established"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    REGRESSION = "regression"


@dataclass
class BaselineMetric:
    """Individual baseline metric with statistical data"""
    name: str
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    percentile_95: float
    percentile_99: float
    unit: str
    sample_size: int
    confidence_interval: Tuple[float, float]


@dataclass
class BaselineData:
    """Complete baseline data for a specific version and context"""
    version: str
    baseline_type: BaselineType
    created_at: str
    system_info: Dict[str, Any]
    dataset_baselines: Dict[str, Dict[str, BaselineMetric]]
    performance_baselines: Dict[str, BaselineMetric]
    memory_baselines: Dict[str, BaselineMetric]
    concurrent_baselines: Dict[str, BaselineMetric]
    metadata: Dict[str, Any]


@dataclass
class BaselineComparison:
    """Comparison between current results and baseline"""
    baseline_version: str
    current_version: str
    comparison_timestamp: str
    metric_comparisons: Dict[str, Dict[str, Any]]
    overall_assessment: Dict[str, Any]
    recommendations: List[str]


class BaselineEstablisher:
    """Manages performance baseline establishment and comparison"""
    
    def __init__(self, baselines_dir: Path):
        """Initialize baseline establisher
        
        Args:
            baselines_dir: Directory for storing baseline data
        """
        self.baselines_dir = baselines_dir
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("BaselineEstablisher")
        
        # Load existing baselines
        self.baselines: Dict[str, Dict[BaselineType, BaselineData]] = {}
        self.load_existing_baselines()
        
        self.logger.info("BaselineEstablisher initialized")
    
    def load_existing_baselines(self):
        """Load all existing baseline files"""
        baseline_files = list(self.baselines_dir.glob("*.json"))
        
        for baseline_file in baseline_files:
            try:
                with open(baseline_file, 'r') as f:
                    data = json.load(f)
                
                # Parse baseline data
                baseline_data = self._parse_baseline_file(data)
                
                version = baseline_data.version
                baseline_type = baseline_data.baseline_type
                
                if version not in self.baselines:
                    self.baselines[version] = {}
                
                self.baselines[version][baseline_type] = baseline_data
                
                self.logger.debug(f"Loaded baseline: {version} ({baseline_type.value})")
                
            except Exception as e:
                self.logger.error(f"Error loading baseline {baseline_file}: {str(e)}")
        
        self.logger.info(f"Loaded {len(baseline_files)} baseline files")
    
    def _parse_baseline_file(self, data: Dict) -> BaselineData:
        """Parse baseline file data into BaselineData object"""
        # Convert metric dictionaries back to BaselineMetric objects
        dataset_baselines = {}
        for dataset_name, metrics in data.get("dataset_baselines", {}).items():
            dataset_baselines[dataset_name] = {
                metric_name: BaselineMetric(**metric_data)
                for metric_name, metric_data in metrics.items()
            }
        
        performance_baselines = {
            name: BaselineMetric(**metric_data)
            for name, metric_data in data.get("performance_baselines", {}).items()
        }
        
        memory_baselines = {
            name: BaselineMetric(**metric_data)
            for name, metric_data in data.get("memory_baselines", {}).items()
        }
        
        concurrent_baselines = {
            name: BaselineMetric(**metric_data)
            for name, metric_data in data.get("concurrent_baselines", {}).items()
        }
        
        return BaselineData(
            version=data["version"],
            baseline_type=BaselineType(data["baseline_type"]),
            created_at=data["created_at"],
            system_info=data["system_info"],
            dataset_baselines=dataset_baselines,
            performance_baselines=performance_baselines,
            memory_baselines=memory_baselines,
            concurrent_baselines=concurrent_baselines,
            metadata=data.get("metadata", {})
        )
    
    def establish_baseline(self, benchmark_result, baseline_type: BaselineType = BaselineType.DEVELOPMENT) -> BaselineData:
        """Establish a new baseline from benchmark results"""
        self.logger.info(f"Establishing {baseline_type.value} baseline for v{benchmark_result.localdata_version}")
        
        # Create baseline metrics from benchmark results
        baseline_data = BaselineData(
            version=benchmark_result.localdata_version,
            baseline_type=baseline_type,
            created_at=datetime.now().isoformat(),
            system_info=benchmark_result.system_info,
            dataset_baselines=self._create_dataset_baselines(benchmark_result.dataset_results),
            performance_baselines=self._create_performance_baselines(benchmark_result.performance_summary),
            memory_baselines=self._create_memory_baselines(benchmark_result.memory_results),
            concurrent_baselines=self._create_concurrent_baselines(benchmark_result.concurrent_results),
            metadata={
                "benchmark_id": benchmark_result.benchmark_id,
                "total_duration_seconds": benchmark_result.total_duration_seconds,
                "success_rate": benchmark_result.success_rate,
                "recommendations_count": len(benchmark_result.recommendations)
            }
        )
        
        # Store baseline
        self._save_baseline(baseline_data)
        
        # Add to memory
        version = baseline_data.version
        if version not in self.baselines:
            self.baselines[version] = {}
        
        self.baselines[version][baseline_type] = baseline_data
        
        self.logger.info(f"Baseline established and saved for v{version}")
        return baseline_data
    
    def _create_dataset_baselines(self, dataset_results: Dict[str, Any]) -> Dict[str, Dict[str, BaselineMetric]]:
        """Create baseline metrics from dataset results"""
        dataset_baselines = {}
        
        for dataset_name, result in dataset_results.items():
            if not result.get("success", False):
                continue
            
            metrics = {}
            
            # Generation time baseline
            generation_time = result.get("generation_time_seconds", 0)
            if generation_time > 0:
                metrics["generation_time_seconds"] = self._create_single_value_baseline(
                    "generation_time_seconds", generation_time, "seconds"
                )
            
            # Dataset size baseline
            dataset_size = result.get("dataset_size_mb", 0)
            if dataset_size > 0:
                metrics["dataset_size_mb"] = self._create_single_value_baseline(
                    "dataset_size_mb", dataset_size, "MB"
                )
            
            # Throughput baseline
            throughput = result.get("throughput_mb_per_second", 0)
            if throughput > 0:
                metrics["throughput_mb_per_second"] = self._create_single_value_baseline(
                    "throughput_mb_per_second", throughput, "MB/s"
                )
            
            # Query performance baselines
            if "query_results" in result:
                query_results = result["query_results"]
                if query_results.get("avg_query_time_ms"):
                    metrics["avg_query_time_ms"] = self._create_single_value_baseline(
                        "avg_query_time_ms", query_results["avg_query_time_ms"], "ms"
                    )
            
            if metrics:
                dataset_baselines[dataset_name] = metrics
        
        return dataset_baselines
    
    def _create_performance_baselines(self, performance_summary: Dict[str, Any]) -> Dict[str, BaselineMetric]:
        """Create baseline metrics from performance summary"""
        baselines = {}
        
        # Overall performance score
        overall_score = performance_summary.get("overall_performance_score")
        if overall_score is not None:
            baselines["overall_performance_score"] = self._create_single_value_baseline(
                "overall_performance_score", overall_score, "score"
            )
        
        # Performance characteristics
        characteristics = performance_summary.get("performance_characteristics", {})
        for char_name, char_value in characteristics.items():
            if isinstance(char_value, (int, float)):
                baselines[char_name] = self._create_single_value_baseline(
                    char_name, char_value, "percent"
                )
        
        return baselines
    
    def _create_memory_baselines(self, memory_results: Dict[str, Any]) -> Dict[str, BaselineMetric]:
        """Create baseline metrics from memory results"""
        baselines = {}
        
        # Max memory usage
        max_memory = memory_results.get("max_memory_usage_mb")
        if max_memory is not None:
            baselines["max_memory_usage_mb"] = self._create_single_value_baseline(
                "max_memory_usage_mb", max_memory, "MB"
            )
        
        # Streaming threshold
        streaming_threshold = memory_results.get("streaming_threshold_mb")
        if streaming_threshold is not None:
            baselines["streaming_threshold_mb"] = self._create_single_value_baseline(
                "streaming_threshold_mb", streaming_threshold, "MB"
            )
        
        return baselines
    
    def _create_concurrent_baselines(self, concurrent_results: Dict[str, Any]) -> Dict[str, BaselineMetric]:
        """Create baseline metrics from concurrent results"""
        baselines = {}
        
        # Max concurrent operations
        max_concurrent = concurrent_results.get("max_concurrent_operations")
        if max_concurrent is not None:
            baselines["max_concurrent_operations"] = self._create_single_value_baseline(
                "max_concurrent_operations", max_concurrent, "operations"
            )
        
        # Average response time
        avg_response_time = concurrent_results.get("avg_response_time_ms")
        if avg_response_time is not None:
            baselines["avg_response_time_ms"] = self._create_single_value_baseline(
                "avg_response_time_ms", avg_response_time, "ms"
            )
        
        return baselines
    
    def _create_single_value_baseline(self, name: str, value: float, unit: str) -> BaselineMetric:
        """Create a baseline metric from a single value"""
        # For single values, set statistical measures assuming some variance
        estimated_std_dev = value * 0.1  # Assume 10% standard deviation
        
        return BaselineMetric(
            name=name,
            mean=value,
            median=value,
            std_dev=estimated_std_dev,
            min_value=value * 0.8,  # Assume 20% variance range
            max_value=value * 1.2,
            percentile_95=value * 1.1,
            percentile_99=value * 1.15,
            unit=unit,
            sample_size=1,
            confidence_interval=(value * 0.9, value * 1.1)
        )
    
    def _save_baseline(self, baseline_data: BaselineData):
        """Save baseline data to file"""
        filename = f"{baseline_data.version}_{baseline_data.baseline_type.value}_baseline.json"
        filepath = self.baselines_dir / filename
        
        # Convert to serializable format
        serializable_data = self._baseline_to_dict(baseline_data)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2, default=str)
        
        self.logger.info(f"Baseline saved to {filepath}")
    
    def _baseline_to_dict(self, baseline_data: BaselineData) -> Dict:
        """Convert BaselineData to serializable dictionary"""
        return {
            "version": baseline_data.version,
            "baseline_type": baseline_data.baseline_type.value,
            "created_at": baseline_data.created_at,
            "system_info": baseline_data.system_info,
            "dataset_baselines": {
                dataset_name: {
                    metric_name: asdict(metric)
                    for metric_name, metric in metrics.items()
                }
                for dataset_name, metrics in baseline_data.dataset_baselines.items()
            },
            "performance_baselines": {
                name: asdict(metric)
                for name, metric in baseline_data.performance_baselines.items()
            },
            "memory_baselines": {
                name: asdict(metric)
                for name, metric in baseline_data.memory_baselines.items()
            },
            "concurrent_baselines": {
                name: asdict(metric)
                for name, metric in baseline_data.concurrent_baselines.items()
            },
            "metadata": baseline_data.metadata
        }
    
    def compare_to_baseline(self, benchmark_result, 
                           baseline_version: str = None,
                           baseline_type: BaselineType = BaselineType.DEVELOPMENT) -> BaselineComparison:
        """Compare current benchmark results to baseline"""
        self.logger.info(f"Comparing to baseline: {baseline_version} ({baseline_type.value})")
        
        # Find appropriate baseline
        if baseline_version:
            baseline = self.baselines.get(baseline_version, {}).get(baseline_type)
        else:
            # Use most recent baseline of the specified type
            baseline = self._get_latest_baseline(baseline_type)
        
        if not baseline:
            raise ValueError(f"No baseline found for version {baseline_version} type {baseline_type.value}")
        
        # Perform comparison
        metric_comparisons = {}
        
        # Compare dataset metrics
        dataset_comparisons = self._compare_dataset_metrics(
            benchmark_result.dataset_results, 
            baseline.dataset_baselines
        )
        metric_comparisons.update(dataset_comparisons)
        
        # Compare performance metrics
        performance_comparisons = self._compare_performance_metrics(
            benchmark_result.performance_summary,
            baseline.performance_baselines
        )
        metric_comparisons.update(performance_comparisons)
        
        # Compare memory metrics
        memory_comparisons = self._compare_memory_metrics(
            benchmark_result.memory_results,
            baseline.memory_baselines
        )
        metric_comparisons.update(memory_comparisons)
        
        # Compare concurrent metrics
        concurrent_comparisons = self._compare_concurrent_metrics(
            benchmark_result.concurrent_results,
            baseline.concurrent_baselines
        )
        metric_comparisons.update(concurrent_comparisons)
        
        # Calculate overall assessment
        overall_assessment = self._calculate_overall_assessment(metric_comparisons)
        
        # Generate recommendations
        recommendations = self._generate_comparison_recommendations(metric_comparisons, overall_assessment)
        
        return BaselineComparison(
            baseline_version=baseline.version,
            current_version=benchmark_result.localdata_version,
            comparison_timestamp=datetime.now().isoformat(),
            metric_comparisons=metric_comparisons,
            overall_assessment=overall_assessment,
            recommendations=recommendations
        )
    
    def _get_latest_baseline(self, baseline_type: BaselineType) -> Optional[BaselineData]:
        """Get the most recent baseline of specified type"""
        latest_baseline = None
        latest_date = None
        
        for version, baselines in self.baselines.items():
            if baseline_type in baselines:
                baseline = baselines[baseline_type]
                created_date = datetime.fromisoformat(baseline.created_at)
                
                if latest_date is None or created_date > latest_date:
                    latest_baseline = baseline
                    latest_date = created_date
        
        return latest_baseline
    
    def _compare_dataset_metrics(self, current_results: Dict, baseline_metrics: Dict) -> Dict[str, Any]:
        """Compare dataset metrics to baseline"""
        comparisons = {}
        
        for dataset_name, current_result in current_results.items():
            if not current_result.get("success", False):
                continue
            
            if dataset_name not in baseline_metrics:
                continue
            
            baseline_dataset = baseline_metrics[dataset_name]
            
            # Compare generation time
            current_time = current_result.get("generation_time_seconds")
            if current_time and "generation_time_seconds" in baseline_dataset:
                baseline_metric = baseline_dataset["generation_time_seconds"]
                comparison = self._compare_metric_value(current_time, baseline_metric)
                comparisons[f"{dataset_name}_generation_time"] = comparison
            
            # Compare throughput
            current_throughput = current_result.get("throughput_mb_per_second")
            if current_throughput and "throughput_mb_per_second" in baseline_dataset:
                baseline_metric = baseline_dataset["throughput_mb_per_second"]
                comparison = self._compare_metric_value(current_throughput, baseline_metric, higher_is_better=True)
                comparisons[f"{dataset_name}_throughput"] = comparison
            
            # Compare dataset size
            current_size = current_result.get("dataset_size_mb")
            if current_size and "dataset_size_mb" in baseline_dataset:
                baseline_metric = baseline_dataset["dataset_size_mb"]
                comparison = self._compare_metric_value(current_size, baseline_metric)
                comparisons[f"{dataset_name}_size"] = comparison
        
        return comparisons
    
    def _compare_performance_metrics(self, current_summary: Dict, baseline_metrics: Dict) -> Dict[str, Any]:
        """Compare performance metrics to baseline"""
        comparisons = {}
        
        # Overall performance score
        current_score = current_summary.get("overall_performance_score")
        if current_score and "overall_performance_score" in baseline_metrics:
            baseline_metric = baseline_metrics["overall_performance_score"]
            comparison = self._compare_metric_value(current_score, baseline_metric, higher_is_better=True)
            comparisons["overall_performance_score"] = comparison
        
        # Performance characteristics
        characteristics = current_summary.get("performance_characteristics", {})
        for char_name, current_value in characteristics.items():
            if char_name in baseline_metrics:
                baseline_metric = baseline_metrics[char_name]
                comparison = self._compare_metric_value(current_value, baseline_metric)
                comparisons[f"performance_{char_name}"] = comparison
        
        return comparisons
    
    def _compare_memory_metrics(self, current_results: Dict, baseline_metrics: Dict) -> Dict[str, Any]:
        """Compare memory metrics to baseline"""
        comparisons = {}
        
        # Max memory usage
        current_memory = current_results.get("max_memory_usage_mb")
        if current_memory and "max_memory_usage_mb" in baseline_metrics:
            baseline_metric = baseline_metrics["max_memory_usage_mb"]
            comparison = self._compare_metric_value(current_memory, baseline_metric)
            comparisons["max_memory_usage"] = comparison
        
        # Streaming threshold
        current_threshold = current_results.get("streaming_threshold_mb")
        if current_threshold and "streaming_threshold_mb" in baseline_metrics:
            baseline_metric = baseline_metrics["streaming_threshold_mb"]
            comparison = self._compare_metric_value(current_threshold, baseline_metric)
            comparisons["streaming_threshold"] = comparison
        
        return comparisons
    
    def _compare_concurrent_metrics(self, current_results: Dict, baseline_metrics: Dict) -> Dict[str, Any]:
        """Compare concurrent metrics to baseline"""
        comparisons = {}
        
        # Max concurrent operations
        current_concurrent = current_results.get("max_concurrent_operations")
        if current_concurrent and "max_concurrent_operations" in baseline_metrics:
            baseline_metric = baseline_metrics["max_concurrent_operations"]
            comparison = self._compare_metric_value(current_concurrent, baseline_metric, higher_is_better=True)
            comparisons["max_concurrent_operations"] = comparison
        
        # Average response time
        current_response_time = current_results.get("avg_response_time_ms")
        if current_response_time and "avg_response_time_ms" in baseline_metrics:
            baseline_metric = baseline_metrics["avg_response_time_ms"]
            comparison = self._compare_metric_value(current_response_time, baseline_metric)
            comparisons["avg_response_time"] = comparison
        
        return comparisons
    
    def _compare_metric_value(self, current_value: float, baseline_metric: BaselineMetric,
                             higher_is_better: bool = False) -> Dict[str, Any]:
        """Compare a current value to baseline metric"""
        baseline_mean = baseline_metric.mean
        baseline_std = baseline_metric.std_dev
        
        # Calculate percentage difference
        pct_diff = ((current_value - baseline_mean) / baseline_mean) * 100 if baseline_mean != 0 else 0
        
        # Determine significance using standard deviation
        if baseline_std > 0:
            z_score = abs(current_value - baseline_mean) / baseline_std
            significance = "high" if z_score > 2 else "medium" if z_score > 1 else "low"
        else:
            significance = "low"
        
        # Determine trend
        if higher_is_better:
            trend = "improvement" if current_value > baseline_mean else "regression"
        else:
            trend = "regression" if current_value > baseline_mean else "improvement"
        
        # Adjust trend based on significance
        if significance == "low":
            trend = "stable"
        
        return {
            "current_value": current_value,
            "baseline_mean": baseline_mean,
            "percentage_change": pct_diff,
            "absolute_change": current_value - baseline_mean,
            "trend": trend,
            "significance": significance,
            "unit": baseline_metric.unit,
            "within_confidence_interval": (
                baseline_metric.confidence_interval[0] <= current_value <= baseline_metric.confidence_interval[1]
            )
        }
    
    def _calculate_overall_assessment(self, metric_comparisons: Dict) -> Dict[str, Any]:
        """Calculate overall assessment from metric comparisons"""
        improvements = sum(1 for comp in metric_comparisons.values() if comp["trend"] == "improvement")
        regressions = sum(1 for comp in metric_comparisons.values() if comp["trend"] == "regression")
        stable = sum(1 for comp in metric_comparisons.values() if comp["trend"] == "stable")
        
        total_metrics = len(metric_comparisons)
        
        # Calculate overall trend
        if regressions > improvements:
            overall_trend = "regression"
        elif improvements > regressions:
            overall_trend = "improvement"
        else:
            overall_trend = "stable"
        
        # Calculate severity
        high_significance_regressions = sum(
            1 for comp in metric_comparisons.values()
            if comp["trend"] == "regression" and comp["significance"] == "high"
        )
        
        if high_significance_regressions > 0:
            severity = "critical"
        elif regressions > total_metrics * 0.5:
            severity = "high"
        elif regressions > total_metrics * 0.3:
            severity = "medium"
        else:
            severity = "low"
        
        return {
            "overall_trend": overall_trend,
            "severity": severity,
            "total_metrics_compared": total_metrics,
            "improvements": improvements,
            "regressions": regressions,
            "stable_metrics": stable,
            "improvement_rate": (improvements / total_metrics) * 100 if total_metrics > 0 else 0,
            "regression_rate": (regressions / total_metrics) * 100 if total_metrics > 0 else 0
        }
    
    def _generate_comparison_recommendations(self, metric_comparisons: Dict, 
                                          overall_assessment: Dict) -> List[str]:
        """Generate recommendations based on comparison results"""
        recommendations = []
        
        # Overall assessment recommendations
        if overall_assessment["overall_trend"] == "regression":
            if overall_assessment["severity"] == "critical":
                recommendations.append("CRITICAL: Significant performance regressions detected - immediate investigation required")
            elif overall_assessment["severity"] == "high":
                recommendations.append("High number of performance regressions - review recent changes")
            else:
                recommendations.append("Minor performance regressions detected - monitor trends")
        
        elif overall_assessment["overall_trend"] == "improvement":
            recommendations.append("Performance improvements detected - document optimizations for future reference")
        
        # Specific metric recommendations
        for metric_name, comparison in metric_comparisons.items():
            if comparison["trend"] == "regression" and comparison["significance"] == "high":
                if "memory" in metric_name.lower():
                    recommendations.append(f"Memory usage regression in {metric_name} - review memory management")
                elif "time" in metric_name.lower() or "throughput" in metric_name.lower():
                    recommendations.append(f"Performance regression in {metric_name} - investigate bottlenecks")
                elif "concurrent" in metric_name.lower():
                    recommendations.append(f"Concurrency regression in {metric_name} - review threading model")
        
        # Baseline-specific recommendations
        regression_rate = overall_assessment["regression_rate"]
        if regression_rate > 50:
            recommendations.append("High regression rate suggests systematic performance issues")
        elif regression_rate > 30:
            recommendations.append("Moderate regression rate - review optimization strategies")
        
        # If no specific recommendations, provide general guidance
        if not recommendations:
            recommendations.append("Performance is stable compared to baseline - continue monitoring")
        
        return recommendations
    
    def list_available_baselines(self) -> Dict[str, List[str]]:
        """List all available baselines by version and type"""
        baseline_summary = {}
        
        for version, baselines in self.baselines.items():
            baseline_types = [baseline_type.value for baseline_type in baselines.keys()]
            baseline_summary[version] = baseline_types
        
        return baseline_summary
    
    def get_baseline(self, version: str, baseline_type: BaselineType) -> Optional[BaselineData]:
        """Get specific baseline data"""
        return self.baselines.get(version, {}).get(baseline_type)
    
    def update_baseline(self, baseline_data: BaselineData) -> None:
        """Update an existing baseline"""
        self._save_baseline(baseline_data)
        
        # Update in memory
        version = baseline_data.version
        if version not in self.baselines:
            self.baselines[version] = {}
        
        self.baselines[version][baseline_data.baseline_type] = baseline_data
        
        self.logger.info(f"Baseline updated: {version} ({baseline_data.baseline_type.value})")
    
    def delete_baseline(self, version: str, baseline_type: BaselineType) -> bool:
        """Delete a baseline"""
        if version not in self.baselines or baseline_type not in self.baselines[version]:
            return False
        
        # Remove file
        filename = f"{version}_{baseline_type.value}_baseline.json"
        filepath = self.baselines_dir / filename
        
        if filepath.exists():
            filepath.unlink()
        
        # Remove from memory
        del self.baselines[version][baseline_type]
        
        # Clean up empty version dict
        if not self.baselines[version]:
            del self.baselines[version]
        
        self.logger.info(f"Baseline deleted: {version} ({baseline_type.value})")
        return True