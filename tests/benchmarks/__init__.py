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

from .benchmark_orchestrator import BenchmarkOrchestrator
from .dataset_benchmark import DatasetBenchmark
from .performance_collector import PerformanceCollector
from .report_generator import ReportGenerator
from .baseline_establisher import BaselineEstablisher
from .regression_detector import RegressionDetector

__all__ = [
    "BenchmarkOrchestrator",
    "DatasetBenchmark", 
    "PerformanceCollector",
    "ReportGenerator",
    "BaselineEstablisher",
    "RegressionDetector"
]