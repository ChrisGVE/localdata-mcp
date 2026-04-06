"""Data models for the performance benchmarking suite.

This module defines the configuration, result, and comparison dataclasses
used throughout the benchmarking system.
"""

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    # Test data configuration
    small_dataset_rows: int = 1000
    medium_dataset_rows: int = 50000
    large_dataset_rows: int = 500000

    # Memory configuration
    memory_limit_mb: int = 512
    chunk_sizes: List[int] = field(default_factory=lambda: [100, 500, 1000, 5000])

    # Test repetition
    test_iterations: int = 3
    warmup_iterations: int = 1

    # Database types to test
    database_types: List[str] = field(
        default_factory=lambda: ["sqlite", "postgresql", "mysql"]
    )

    # Output configuration
    results_dir: Path = field(default_factory=lambda: Path("benchmark_results"))
    generate_reports: bool = True
    save_detailed_metrics: bool = True


@dataclass
class BenchmarkResult:
    """Results from a single benchmark test."""

    # Test identification
    test_name: str
    test_category: str
    dataset_size: str  # 'small', 'medium', 'large'
    database_type: str
    processing_mode: str  # 'streaming', 'batch'

    # Performance metrics
    execution_time_seconds: float
    peak_memory_mb: float
    average_memory_mb: float
    memory_efficiency: float  # 0-1 ratio

    # Data processing metrics
    rows_processed: int
    processing_rate_rows_per_second: float
    chunk_count: int
    average_chunk_size: int

    # Token metrics (if applicable)
    token_count: Optional[int] = None
    token_estimation_time_seconds: Optional[float] = None
    token_estimation_accuracy: Optional[float] = None

    # Memory usage profile
    memory_samples: List[float] = field(default_factory=list)
    memory_timestamps: List[float] = field(default_factory=list)

    # Error information
    success: bool = True
    error_message: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return asdict(self)


@dataclass
class ComparisonResult:
    """Results comparing different approaches."""

    baseline_result: BenchmarkResult
    comparison_result: BenchmarkResult

    # Performance improvements (positive = improvement)
    execution_time_improvement_percent: float
    memory_improvement_percent: float
    processing_rate_improvement_percent: float

    # Statistical significance
    is_significant: bool
    confidence_level: float

    # Summary
    summary: str
    recommendations: List[str] = field(default_factory=list)
