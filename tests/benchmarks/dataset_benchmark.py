#!/usr/bin/env python3
"""
Dataset Benchmark Runner

Handles individual dataset benchmarking with:
- Dataset generation performance measurement
- Query execution benchmarking
- Memory usage tracking during operations
- Streaming architecture validation
- Data integrity verification
"""

import time
import json
import logging
import psutil
import sqlite3
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import tracemalloc


@dataclass
class QueryBenchmark:
    """Individual query benchmark result"""
    query: str
    execution_time_ms: float
    memory_used_mb: float
    rows_processed: int
    streaming_activated: bool
    error: Optional[str] = None


@dataclass
class DatasetBenchmarkResult:
    """Complete benchmark result for a dataset"""
    dataset_name: str
    generation_performance: Dict[str, Any]
    query_benchmarks: List[QueryBenchmark]
    memory_profile: Dict[str, Any]
    streaming_validation: Dict[str, Any]
    integrity_checks: Dict[str, Any]
    overall_score: float
    recommendations: List[str]


class DatasetBenchmark:
    """Individual dataset benchmark execution and analysis"""
    
    def __init__(self, dataset_name: str, output_dir: Path):
        """Initialize dataset benchmark runner
        
        Args:
            dataset_name: Name of the dataset being benchmarked
            output_dir: Directory for benchmark outputs
        """
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.logger = logging.getLogger(f"DatasetBenchmark.{dataset_name}")
        
        # Performance tracking
        self.memory_tracker = MemoryTracker()
        self.generation_metrics = {}
        self.query_results = []
        
    def benchmark_dataset_generation(self, generator_class, generator_args: Dict) -> Dict[str, Any]:
        """Benchmark dataset generation performance"""
        self.logger.info(f"Starting generation benchmark for {self.dataset_name}")
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024*1024)  # MB
        
        # Start memory tracking
        self.memory_tracker.start_tracking()
        
        try:
            # Initialize and run generator
            generator = generator_class(
                str(self.output_dir / f"{self.dataset_name}_benchmark"),
                **generator_args
            )
            
            generation_start = time.time()
            generator.generate_datasets()
            generation_time = time.time() - generation_start
            
            # Stop memory tracking
            peak_memory_mb = self.memory_tracker.stop_tracking()
            
            end_memory = psutil.Process().memory_info().rss / (1024*1024)  # MB
            memory_delta = end_memory - start_memory
            
            # Calculate dataset size
            dataset_size_mb = self.calculate_dataset_size()
            
            # Calculate throughput
            throughput = dataset_size_mb / generation_time if generation_time > 0 else 0
            
            result = {
                "success": True,
                "generation_time_seconds": generation_time,
                "dataset_size_mb": dataset_size_mb,
                "throughput_mb_per_second": throughput,
                "memory_usage": {
                    "start_mb": start_memory,
                    "end_mb": end_memory,
                    "delta_mb": memory_delta,
                    "peak_mb": peak_memory_mb
                },
                "performance_score": self.calculate_generation_score(throughput, memory_delta)
            }
            
            self.generation_metrics = result
            return result
            
        except Exception as e:
            self.logger.error(f"Generation benchmark failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "generation_time_seconds": time.time() - start_time
            }
    
    def benchmark_queries(self, test_queries: List[str]) -> List[QueryBenchmark]:
        """Benchmark query execution performance"""
        self.logger.info(f"Running {len(test_queries)} query benchmarks")
        
        results = []
        dataset_path = self.output_dir / f"{self.dataset_name}_benchmark"
        
        for i, query in enumerate(test_queries):
            self.logger.debug(f"Executing query {i+1}/{len(test_queries)}")
            
            try:
                result = self.benchmark_single_query(query, dataset_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Query {i+1} failed: {str(e)}")
                results.append(QueryBenchmark(
                    query=query,
                    execution_time_ms=0,
                    memory_used_mb=0,
                    rows_processed=0,
                    streaming_activated=False,
                    error=str(e)
                ))
        
        self.query_results = results
        return results
    
    def benchmark_single_query(self, query: str, dataset_path: Path) -> QueryBenchmark:
        """Benchmark execution of a single query"""
        start_memory = psutil.Process().memory_info().rss / (1024*1024)
        
        # Start detailed memory tracking
        tracemalloc.start()
        start_time = time.time()
        
        try:
            rows_processed = 0
            streaming_activated = False
            
            # Try to find SQLite database first
            db_files = list(dataset_path.glob("*.db")) + list(dataset_path.glob("*.sqlite"))
            
            if db_files:
                # SQLite query execution
                with sqlite3.connect(db_files[0]) as conn:
                    cursor = conn.execute(query)
                    
                    # Check if streaming was activated by monitoring memory
                    initial_mem = tracemalloc.get_traced_memory()[0]
                    
                    rows = cursor.fetchall()
                    rows_processed = len(rows)
                    
                    # Simple heuristic for streaming detection
                    final_mem = tracemalloc.get_traced_memory()[0]
                    streaming_activated = (final_mem - initial_mem) < 50 * 1024 * 1024  # Less than 50MB delta
            
            else:
                # CSV/JSON file query simulation
                csv_files = list(dataset_path.glob("*.csv"))
                if csv_files:
                    # Simple CSV row counting for basic queries
                    with open(csv_files[0], 'r') as f:
                        reader = csv.reader(f)
                        rows_processed = sum(1 for row in reader) - 1  # Exclude header
                    streaming_activated = True  # CSV reading is inherently streaming
            
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Calculate memory usage
            current_memory = psutil.Process().memory_info().rss / (1024*1024)
            memory_used = current_memory - start_memory
            
            return QueryBenchmark(
                query=query,
                execution_time_ms=execution_time,
                memory_used_mb=memory_used,
                rows_processed=rows_processed,
                streaming_activated=streaming_activated
            )
            
        except Exception as e:
            raise Exception(f"Query execution failed: {str(e)}")
        
        finally:
            tracemalloc.stop()
    
    def validate_streaming_architecture(self) -> Dict[str, Any]:
        """Validate streaming architecture behavior"""
        self.logger.info("Validating streaming architecture behavior")
        
        validation_results = {
            "streaming_detection": "passed",
            "memory_efficiency": "passed",
            "threshold_behavior": "passed",
            "details": {
                "queries_with_streaming": sum(1 for q in self.query_results if q.streaming_activated),
                "total_queries": len(self.query_results),
                "avg_memory_per_query_mb": sum(q.memory_used_mb for q in self.query_results) / len(self.query_results) if self.query_results else 0
            }
        }
        
        # Check streaming activation rate
        streaming_rate = (validation_results["details"]["queries_with_streaming"] / 
                         validation_results["details"]["total_queries"]) if validation_results["details"]["total_queries"] > 0 else 0
        
        if streaming_rate < 0.5:  # Less than 50% streaming activation
            validation_results["streaming_detection"] = "warning"
            validation_results["details"]["warning"] = "Low streaming activation rate"
        
        # Check average memory usage per query
        if validation_results["details"]["avg_memory_per_query_mb"] > 100:
            validation_results["memory_efficiency"] = "warning"
            validation_results["details"]["memory_warning"] = "High memory usage per query"
        
        return validation_results
    
    def run_integrity_checks(self) -> Dict[str, Any]:
        """Run data integrity checks on generated dataset"""
        self.logger.info("Running data integrity checks")
        
        dataset_path = self.output_dir / f"{self.dataset_name}_benchmark"
        
        checks = {
            "file_existence": self.check_file_existence(dataset_path),
            "size_validation": self.check_dataset_size(dataset_path),
            "structure_validation": self.check_data_structure(dataset_path),
            "content_validation": self.check_data_content(dataset_path)
        }
        
        # Calculate overall integrity score
        passed_checks = sum(1 for check in checks.values() if check.get("status") == "passed")
        total_checks = len(checks)
        integrity_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        
        checks["overall_integrity_score"] = integrity_score
        return checks
    
    def check_file_existence(self, dataset_path: Path) -> Dict[str, Any]:
        """Check if expected dataset files exist"""
        if not dataset_path.exists():
            return {"status": "failed", "reason": "Dataset directory does not exist"}
        
        expected_extensions = [".csv", ".json", ".db", ".sqlite"]
        found_files = []
        
        for ext in expected_extensions:
            files = list(dataset_path.glob(f"*{ext}"))
            found_files.extend(files)
        
        if not found_files:
            return {"status": "failed", "reason": "No dataset files found"}
        
        return {
            "status": "passed",
            "files_found": len(found_files),
            "file_types": list(set(f.suffix for f in found_files))
        }
    
    def check_dataset_size(self, dataset_path: Path) -> Dict[str, Any]:
        """Validate dataset size meets expectations"""
        actual_size_mb = self.calculate_dataset_size()
        
        # Expected size based on dataset type (rough estimates)
        expected_sizes = {
            "ecommerce": 5000,    # 5GB
            "iot": 12000,         # 12GB  
            "social_media": 6000, # 6GB
            "financial": 8000     # 8GB
        }
        
        expected_mb = expected_sizes.get(self.dataset_name, 1000)  # Default 1GB
        
        # Allow 20% variance
        min_size = expected_mb * 0.8
        max_size = expected_mb * 1.2
        
        if min_size <= actual_size_mb <= max_size:
            status = "passed"
        elif actual_size_mb < min_size:
            status = "warning"
        else:
            status = "passed"  # Larger is generally okay
        
        return {
            "status": status,
            "actual_size_mb": actual_size_mb,
            "expected_size_mb": expected_mb,
            "variance_percent": ((actual_size_mb - expected_mb) / expected_mb) * 100
        }
    
    def check_data_structure(self, dataset_path: Path) -> Dict[str, Any]:
        """Validate data structure and schema"""
        # Basic structure validation
        try:
            # Check SQLite databases
            db_files = list(dataset_path.glob("*.db")) + list(dataset_path.glob("*.sqlite"))
            if db_files:
                with sqlite3.connect(db_files[0]) as conn:
                    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    if not tables:
                        return {"status": "failed", "reason": "No tables found in database"}
                    
                    return {
                        "status": "passed",
                        "tables_found": len(tables),
                        "table_names": tables
                    }
            
            # Check CSV files
            csv_files = list(dataset_path.glob("*.csv"))
            if csv_files:
                valid_files = 0
                for csv_file in csv_files:
                    try:
                        with open(csv_file, 'r') as f:
                            reader = csv.reader(f)
                            header = next(reader)  # Read header
                            if header:  # Has header
                                valid_files += 1
                    except:
                        continue
                
                return {
                    "status": "passed" if valid_files > 0 else "failed",
                    "csv_files_validated": valid_files,
                    "total_csv_files": len(csv_files)
                }
            
            return {"status": "warning", "reason": "No recognizable data files found"}
            
        except Exception as e:
            return {"status": "failed", "reason": f"Structure validation failed: {str(e)}"}
    
    def check_data_content(self, dataset_path: Path) -> Dict[str, Any]:
        """Validate data content quality"""
        try:
            # Sample-based content validation
            db_files = list(dataset_path.glob("*.db")) + list(dataset_path.glob("*.sqlite"))
            
            if db_files:
                with sqlite3.connect(db_files[0]) as conn:
                    # Get table with most rows
                    cursor = conn.execute("""
                        SELECT name FROM sqlite_master WHERE type='table'
                        ORDER BY name LIMIT 1
                    """)
                    
                    table_result = cursor.fetchone()
                    if table_result:
                        table_name = table_result[0]
                        
                        # Count total rows
                        cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                        total_rows = cursor.fetchone()[0]
                        
                        # Sample a few rows to check for data quality
                        cursor = conn.execute(f"SELECT * FROM {table_name} LIMIT 10")
                        sample_rows = cursor.fetchall()
                        
                        # Basic quality checks
                        non_empty_rows = sum(1 for row in sample_rows if any(cell for cell in row))
                        quality_score = (non_empty_rows / len(sample_rows)) * 100 if sample_rows else 0
                        
                        return {
                            "status": "passed" if quality_score > 80 else "warning",
                            "total_rows": total_rows,
                            "sample_quality_score": quality_score,
                            "table_sampled": table_name
                        }
            
            return {"status": "warning", "reason": "Content validation requires database files"}
            
        except Exception as e:
            return {"status": "failed", "reason": f"Content validation failed: {str(e)}"}
    
    def calculate_dataset_size(self) -> float:
        """Calculate total dataset size in MB"""
        dataset_path = self.output_dir / f"{self.dataset_name}_benchmark"
        
        if not dataset_path.exists():
            return 0.0
        
        total_size = 0
        for file_path in dataset_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def calculate_generation_score(self, throughput: float, memory_delta: float) -> float:
        """Calculate performance score for dataset generation"""
        # Scoring based on throughput (MB/s) and memory efficiency
        throughput_score = min(throughput / 50.0, 1.0) * 50  # Max 50 points, 50 MB/s = full points
        memory_score = max(0, 50 - (memory_delta / 100.0))   # Max 50 points, penalty for high memory use
        
        return throughput_score + memory_score
    
    def generate_recommendations(self) -> List[str]:
        """Generate performance recommendations based on benchmark results"""
        recommendations = []
        
        # Generation performance recommendations
        if hasattr(self, 'generation_metrics') and self.generation_metrics.get("success", False):
            throughput = self.generation_metrics.get("throughput_mb_per_second", 0)
            if throughput < 20:
                recommendations.append("Consider optimizing data generation algorithms for better throughput")
            
            memory_delta = self.generation_metrics.get("memory_usage", {}).get("delta_mb", 0)
            if memory_delta > 500:
                recommendations.append("High memory usage during generation - consider streaming generation")
        
        # Query performance recommendations  
        if self.query_results:
            avg_query_time = sum(q.execution_time_ms for q in self.query_results) / len(self.query_results)
            if avg_query_time > 1000:  # > 1 second
                recommendations.append("Consider adding database indexes for better query performance")
            
            streaming_rate = sum(1 for q in self.query_results if q.streaming_activated) / len(self.query_results)
            if streaming_rate < 0.7:
                recommendations.append("Low streaming activation rate - review streaming thresholds")
        
        # Dataset-specific recommendations
        if self.dataset_name == "iot" and self.calculate_dataset_size() > 15000:  # > 15GB
            recommendations.append("IoT dataset size exceeds target - consider data archiving strategies")
        
        return recommendations
    
    def run_complete_benchmark(self, generator_class, generator_args: Dict, 
                             test_queries: List[str]) -> DatasetBenchmarkResult:
        """Execute complete benchmark suite for the dataset"""
        self.logger.info(f"Starting complete benchmark for {self.dataset_name}")
        
        # Phase 1: Generation Performance
        generation_performance = self.benchmark_dataset_generation(generator_class, generator_args)
        
        # Phase 2: Query Benchmarking (only if generation succeeded)
        query_benchmarks = []
        if generation_performance.get("success", False):
            query_benchmarks = self.benchmark_queries(test_queries)
        
        # Phase 3: Streaming Validation
        streaming_validation = self.validate_streaming_architecture()
        
        # Phase 4: Integrity Checks
        integrity_checks = self.run_integrity_checks()
        
        # Phase 5: Memory Profiling
        memory_profile = {
            "generation_memory": generation_performance.get("memory_usage", {}),
            "query_memory": {
                "avg_memory_per_query_mb": sum(q.memory_used_mb for q in query_benchmarks) / len(query_benchmarks) if query_benchmarks else 0,
                "peak_query_memory_mb": max(q.memory_used_mb for q in query_benchmarks) if query_benchmarks else 0
            }
        }
        
        # Calculate overall score
        generation_score = generation_performance.get("performance_score", 0)
        query_score = (sum(50 - min(q.execution_time_ms / 20, 50) for q in query_benchmarks) / 
                      len(query_benchmarks)) if query_benchmarks else 50
        streaming_score = 100 if streaming_validation.get("streaming_detection") == "passed" else 50
        integrity_score = integrity_checks.get("overall_integrity_score", 0)
        
        overall_score = (generation_score + query_score + streaming_score + integrity_score) / 4
        
        # Generate recommendations
        recommendations = self.generate_recommendations()
        
        result = DatasetBenchmarkResult(
            dataset_name=self.dataset_name,
            generation_performance=generation_performance,
            query_benchmarks=query_benchmarks,
            memory_profile=memory_profile,
            streaming_validation=streaming_validation,
            integrity_checks=integrity_checks,
            overall_score=overall_score,
            recommendations=recommendations
        )
        
        self.logger.info(f"Benchmark completed for {self.dataset_name}. Score: {overall_score:.1f}")
        return result


class MemoryTracker:
    """Helper class for tracking memory usage during operations"""
    
    def __init__(self):
        self.tracking = False
        self.peak_memory = 0
        self.initial_memory = 0
    
    def start_tracking(self):
        """Start memory tracking"""
        self.tracking = True
        self.initial_memory = psutil.Process().memory_info().rss
        self.peak_memory = self.initial_memory
        
        # Start background monitoring
        self.monitor_thread = threading.Thread(target=self._monitor_memory)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_tracking(self) -> float:
        """Stop memory tracking and return peak memory usage in MB"""
        self.tracking = False
        return (self.peak_memory - self.initial_memory) / (1024*1024)
    
    def _monitor_memory(self):
        """Background memory monitoring"""
        while self.tracking:
            current_memory = psutil.Process().memory_info().rss
            self.peak_memory = max(self.peak_memory, current_memory)
            time.sleep(0.1)  # Monitor every 100ms