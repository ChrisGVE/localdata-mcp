"""
LocalData MCP Benchmarking & Reporting Framework

Comprehensive benchmarking system that validates LocalData MCP v1.3.0
architecture and provides baseline metrics for v1.4.0+ development.

Components:
- BenchmarkOrchestrator: Main coordination and execution
- DatasetBenchmark: Individual dataset benchmark runners  
- PerformanceCollector: Metrics collection and analysis
- ReportGenerator: HTML/JSON reporting with visualizations
- BaselineEstablisher: v1.3.0 baseline metric creation
- RegressionDetector: Performance regression detection
"""

__version__ = "1.0.0"
__author__ = "LocalData MCP Team"

from .benchmark_orchestrator import BenchmarkOrchestrator, BenchmarkResult, DatasetSpec
from .dataset_benchmark import DatasetBenchmark, DatasetBenchmarkResult, QueryBenchmark
from .performance_collector import PerformanceCollector, PerformanceReport, PerformanceMetric
from .report_generator import ReportGenerator, ReportConfiguration
from .baseline_establisher import BaselineEstablisher, BaselineData, BaselineComparison, BaselineType
from .regression_detector import RegressionDetector, RegressionReport, RegressionAlert, RegressionSeverity

__all__ = [
    "BenchmarkOrchestrator",
    "BenchmarkResult",
    "DatasetSpec",
    "DatasetBenchmark", 
    "DatasetBenchmarkResult",
    "QueryBenchmark",
    "PerformanceCollector",
    "PerformanceReport",
    "PerformanceMetric",
    "ReportGenerator",
    "ReportConfiguration",
    "BaselineEstablisher",
    "BaselineData",
    "BaselineComparison", 
    "BaselineType",
    "RegressionDetector",
    "RegressionReport",
    "RegressionAlert",
    "RegressionSeverity"
]