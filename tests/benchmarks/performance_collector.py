#!/usr/bin/env python3
"""
Performance Metrics Collection System

Comprehensive performance data collection and analysis:
- System resource monitoring (CPU, Memory, Disk, Network)
- Application-level metrics (query times, throughput, errors)
- Streaming architecture performance tracking  
- Memory safety validation metrics
- Concurrent operation performance measurement
- Statistical analysis and trend detection
"""

import time
import json
import logging
import psutil
import threading
import statistics
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import queue
import concurrent.futures


@dataclass
class SystemSnapshot:
    """System resource snapshot at a point in time"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_sent_mb: float
    network_io_recv_mb: float
    active_threads: int
    open_files: int


@dataclass
class PerformanceMetric:
    """Individual performance metric measurement"""
    name: str
    value: float
    unit: str
    timestamp: float
    tags: Dict[str, str]
    context: Optional[Dict[str, Any]] = None


@dataclass
class StatisticalSummary:
    """Statistical summary of performance metrics"""
    count: int
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    percentile_95: float
    percentile_99: float


@dataclass
class PerformanceReport:
    """Comprehensive performance analysis report"""
    collection_period_seconds: float
    total_metrics_collected: int
    system_performance: Dict[str, StatisticalSummary]
    application_performance: Dict[str, StatisticalSummary]
    streaming_performance: Dict[str, Any]
    memory_safety_metrics: Dict[str, Any]
    concurrent_performance: Dict[str, Any]
    performance_trends: List[Dict[str, Any]]
    anomalies_detected: List[Dict[str, Any]]
    recommendations: List[str]


class PerformanceCollector:
    """Comprehensive performance metrics collection and analysis system"""
    
    def __init__(self, collection_interval: float = 0.5, max_history: int = 10000):
        """Initialize performance collector
        
        Args:
            collection_interval: Seconds between metric collections
            max_history: Maximum number of metrics to keep in memory
        """
        self.collection_interval = collection_interval
        self.max_history = max_history
        self.logger = logging.getLogger("PerformanceCollector")
        
        # Collection state
        self.collecting = False
        self.collection_thread = None
        self.metrics_queue = queue.Queue()
        
        # Metric storage
        self.system_snapshots = deque(maxlen=max_history)
        self.performance_metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.custom_metrics = defaultdict(lambda: deque(maxlen=max_history))
        
        # Analysis state
        self.anomaly_detectors = {}
        self.trend_analyzers = {}
        
        # Baseline tracking for regression detection
        self.baseline_metrics = {}
        
        self.logger.info("PerformanceCollector initialized")
    
    def start_collection(self) -> None:
        """Start continuous performance data collection"""
        if self.collecting:
            self.logger.warning("Collection already running")
            return
        
        self.collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        self.logger.info("Performance collection started")
    
    def stop_collection(self) -> None:
        """Stop performance data collection"""
        if not self.collecting:
            return
        
        self.collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        
        self.logger.info("Performance collection stopped")
    
    def _collection_loop(self) -> None:
        """Main collection loop running in background thread"""
        self.logger.debug("Collection loop started")
        
        # Initialize baseline measurements
        last_disk_io = psutil.disk_io_counters()
        last_network_io = psutil.net_io_counters()
        
        while self.collecting:
            try:
                start_time = time.time()
                
                # Collect system snapshot
                snapshot = self._collect_system_snapshot(last_disk_io, last_network_io)
                self.system_snapshots.append(snapshot)
                
                # Update baseline measurements
                last_disk_io = psutil.disk_io_counters()
                last_network_io = psutil.net_io_counters()
                
                # Process any queued custom metrics
                self._process_queued_metrics()
                
                # Sleep for remainder of interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.collection_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Collection loop error: {str(e)}")
                time.sleep(self.collection_interval)
    
    def _collect_system_snapshot(self, last_disk_io=None, last_network_io=None) -> SystemSnapshot:
        """Collect comprehensive system resource snapshot"""
        try:
            # Memory information
            memory = psutil.virtual_memory()
            
            # Disk I/O (calculate rates if baseline provided)
            disk_io = psutil.disk_io_counters()
            if last_disk_io:
                disk_read_mb = (disk_io.read_bytes - last_disk_io.read_bytes) / (1024*1024)
                disk_write_mb = (disk_io.write_bytes - last_disk_io.write_bytes) / (1024*1024)
            else:
                disk_read_mb = disk_write_mb = 0
            
            # Network I/O (calculate rates if baseline provided)
            network_io = psutil.net_io_counters()
            if last_network_io:
                network_sent_mb = (network_io.bytes_sent - last_network_io.bytes_sent) / (1024*1024)
                network_recv_mb = (network_io.bytes_recv - last_network_io.bytes_recv) / (1024*1024)
            else:
                network_sent_mb = network_recv_mb = 0
            
            # Process information
            current_process = psutil.Process()
            
            return SystemSnapshot(
                timestamp=time.time(),
                cpu_percent=psutil.cpu_percent(),
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024*1024),
                memory_available_mb=memory.available / (1024*1024),
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                network_io_sent_mb=network_sent_mb,
                network_io_recv_mb=network_recv_mb,
                active_threads=current_process.num_threads(),
                open_files=len(current_process.open_files())
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting system snapshot: {str(e)}")
            # Return minimal snapshot on error
            return SystemSnapshot(
                timestamp=time.time(),
                cpu_percent=0, memory_percent=0, memory_used_mb=0,
                memory_available_mb=0, disk_io_read_mb=0, disk_io_write_mb=0,
                network_io_sent_mb=0, network_io_recv_mb=0,
                active_threads=0, open_files=0
            )
    
    def _process_queued_metrics(self) -> None:
        """Process any custom metrics in the queue"""
        try:
            while not self.metrics_queue.empty():
                try:
                    metric = self.metrics_queue.get_nowait()
                    self.custom_metrics[metric.name].append(metric)
                except queue.Empty:
                    break
        except Exception as e:
            self.logger.error(f"Error processing queued metrics: {str(e)}")
    
    def record_metric(self, name: str, value: float, unit: str = "", 
                     tags: Dict[str, str] = None, context: Dict[str, Any] = None) -> None:
        """Record a custom performance metric
        
        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            tags: Optional tags for categorization
            context: Optional additional context data
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            tags=tags or {},
            context=context
        )
        
        # Queue metric for processing by collection thread
        try:
            self.metrics_queue.put_nowait(metric)
        except queue.Full:
            self.logger.warning("Metrics queue full, dropping metric")
    
    def record_query_performance(self, query: str, execution_time_ms: float, 
                               rows_processed: int, memory_used_mb: float,
                               streaming_activated: bool) -> None:
        """Record query performance metrics"""
        self.record_metric("query_execution_time_ms", execution_time_ms,
                          tags={"type": "query"}, 
                          context={"query": query[:100], "rows": rows_processed})
        
        self.record_metric("query_memory_usage_mb", memory_used_mb,
                          tags={"type": "query"})
        
        self.record_metric("query_rows_processed", rows_processed,
                          tags={"type": "query"})
        
        self.record_metric("streaming_activated", 1 if streaming_activated else 0,
                          tags={"type": "streaming"})
    
    def record_generation_performance(self, dataset_name: str, generation_time_s: float,
                                    dataset_size_mb: float, throughput_mb_s: float) -> None:
        """Record dataset generation performance metrics"""
        self.record_metric("generation_time_seconds", generation_time_s,
                          tags={"dataset": dataset_name, "type": "generation"})
        
        self.record_metric("dataset_size_mb", dataset_size_mb,
                          tags={"dataset": dataset_name, "type": "generation"})
        
        self.record_metric("generation_throughput_mb_s", throughput_mb_s,
                          tags={"dataset": dataset_name, "type": "generation"})
    
    def record_memory_safety_event(self, event_type: str, memory_mb: float,
                                  threshold_mb: float = None) -> None:
        """Record memory safety related events"""
        self.record_metric("memory_safety_event", memory_mb,
                          tags={"event": event_type, "type": "memory_safety"},
                          context={"threshold_mb": threshold_mb})
    
    def record_concurrent_operation(self, operation_id: str, duration_ms: float,
                                  thread_count: int, success: bool) -> None:
        """Record concurrent operation performance"""
        self.record_metric("concurrent_operation_duration_ms", duration_ms,
                          tags={"operation": operation_id, "type": "concurrency"},
                          context={"thread_count": thread_count, "success": success})
        
        self.record_metric("concurrent_thread_count", thread_count,
                          tags={"operation": operation_id, "type": "concurrency"})
        
        self.record_metric("concurrent_success", 1 if success else 0,
                          tags={"operation": operation_id, "type": "concurrency"})
    
    def analyze_performance(self, time_window_seconds: float = None) -> PerformanceReport:
        """Analyze collected performance data and generate comprehensive report"""
        self.logger.info("Analyzing performance data...")
        
        analysis_start_time = time.time()
        
        # Determine analysis time window
        if time_window_seconds:
            cutoff_time = time.time() - time_window_seconds
            system_data = [s for s in self.system_snapshots if s.timestamp >= cutoff_time]
        else:
            system_data = list(self.system_snapshots)
            time_window_seconds = system_data[-1].timestamp - system_data[0].timestamp if system_data else 0
        
        # Analyze system performance
        system_performance = self._analyze_system_metrics(system_data)
        
        # Analyze application performance  
        application_performance = self._analyze_application_metrics(time_window_seconds)
        
        # Analyze streaming performance
        streaming_performance = self._analyze_streaming_metrics()
        
        # Analyze memory safety
        memory_safety_metrics = self._analyze_memory_safety_metrics()
        
        # Analyze concurrent performance
        concurrent_performance = self._analyze_concurrent_metrics()
        
        # Detect trends and anomalies
        performance_trends = self._detect_performance_trends(system_data)
        anomalies_detected = self._detect_anomalies(system_data)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            system_performance, application_performance, streaming_performance
        )
        
        # Count total metrics
        total_metrics = len(system_data)
        for metrics_list in self.custom_metrics.values():
            total_metrics += len(metrics_list)
        
        report = PerformanceReport(
            collection_period_seconds=time_window_seconds,
            total_metrics_collected=total_metrics,
            system_performance=system_performance,
            application_performance=application_performance,
            streaming_performance=streaming_performance,
            memory_safety_metrics=memory_safety_metrics,
            concurrent_performance=concurrent_performance,
            performance_trends=performance_trends,
            anomalies_detected=anomalies_detected,
            recommendations=recommendations
        )
        
        analysis_duration = time.time() - analysis_start_time
        self.logger.info(f"Performance analysis completed in {analysis_duration:.2f} seconds")
        
        return report
    
    def _analyze_system_metrics(self, system_data: List[SystemSnapshot]) -> Dict[str, StatisticalSummary]:
        """Analyze system performance metrics"""
        if not system_data:
            return {}
        
        analysis = {}
        
        # CPU analysis
        cpu_values = [s.cpu_percent for s in system_data]
        analysis["cpu_percent"] = self._calculate_statistical_summary(cpu_values)
        
        # Memory analysis
        memory_values = [s.memory_percent for s in system_data]
        analysis["memory_percent"] = self._calculate_statistical_summary(memory_values)
        
        memory_used_values = [s.memory_used_mb for s in system_data]
        analysis["memory_used_mb"] = self._calculate_statistical_summary(memory_used_values)
        
        # Disk I/O analysis
        disk_read_values = [s.disk_io_read_mb for s in system_data]
        analysis["disk_read_mb_per_s"] = self._calculate_statistical_summary(disk_read_values)
        
        disk_write_values = [s.disk_io_write_mb for s in system_data]
        analysis["disk_write_mb_per_s"] = self._calculate_statistical_summary(disk_write_values)
        
        # Network I/O analysis
        network_sent_values = [s.network_io_sent_mb for s in system_data]
        analysis["network_sent_mb_per_s"] = self._calculate_statistical_summary(network_sent_values)
        
        # Thread analysis
        thread_values = [s.active_threads for s in system_data]
        analysis["active_threads"] = self._calculate_statistical_summary(thread_values)
        
        return analysis
    
    def _analyze_application_metrics(self, time_window_seconds: float) -> Dict[str, StatisticalSummary]:
        """Analyze application-level performance metrics"""
        analysis = {}
        
        # Analyze each custom metric type
        for metric_name, metrics_list in self.custom_metrics.items():
            if not metrics_list:
                continue
            
            # Filter by time window if specified
            if time_window_seconds:
                cutoff_time = time.time() - time_window_seconds
                filtered_metrics = [m for m in metrics_list if m.timestamp >= cutoff_time]
            else:
                filtered_metrics = list(metrics_list)
            
            if not filtered_metrics:
                continue
            
            values = [m.value for m in filtered_metrics]
            analysis[metric_name] = self._calculate_statistical_summary(values)
        
        return analysis
    
    def _analyze_streaming_metrics(self) -> Dict[str, Any]:
        """Analyze streaming architecture performance"""
        streaming_metrics = [m for m in self.custom_metrics.get("streaming_activated", [])]
        
        if not streaming_metrics:
            return {"status": "no_data"}
        
        # Calculate streaming activation rate
        activation_rate = sum(m.value for m in streaming_metrics) / len(streaming_metrics)
        
        # Analyze streaming memory efficiency
        query_memory_metrics = [m for m in self.custom_metrics.get("query_memory_usage_mb", [])]
        avg_memory_per_query = (sum(m.value for m in query_memory_metrics) / len(query_memory_metrics) 
                               if query_memory_metrics else 0)
        
        return {
            "activation_rate": activation_rate,
            "total_queries_analyzed": len(streaming_metrics),
            "avg_memory_per_query_mb": avg_memory_per_query,
            "streaming_efficiency_score": self._calculate_streaming_score(activation_rate, avg_memory_per_query)
        }
    
    def _analyze_memory_safety_metrics(self) -> Dict[str, Any]:
        """Analyze memory safety related metrics"""
        memory_events = [m for m in self.custom_metrics.get("memory_safety_event", [])]
        
        if not memory_events:
            return {"status": "no_events"}
        
        # Categorize memory events
        event_types = defaultdict(int)
        for metric in memory_events:
            event_type = metric.tags.get("event", "unknown")
            event_types[event_type] += 1
        
        # Calculate peak memory usage
        peak_memory = max(m.value for m in memory_events) if memory_events else 0
        
        return {
            "total_events": len(memory_events),
            "event_types": dict(event_types),
            "peak_memory_mb": peak_memory,
            "memory_safety_score": self._calculate_memory_safety_score(memory_events)
        }
    
    def _analyze_concurrent_metrics(self) -> Dict[str, Any]:
        """Analyze concurrent operation performance"""
        concurrent_metrics = [m for m in self.custom_metrics.get("concurrent_operation_duration_ms", [])]
        success_metrics = [m for m in self.custom_metrics.get("concurrent_success", [])]
        
        if not concurrent_metrics:
            return {"status": "no_data"}
        
        # Calculate success rate
        success_rate = (sum(m.value for m in success_metrics) / len(success_metrics) 
                       if success_metrics else 0)
        
        # Calculate average response time
        avg_response_time = sum(m.value for m in concurrent_metrics) / len(concurrent_metrics)
        
        # Analyze thread usage patterns
        thread_metrics = [m for m in self.custom_metrics.get("concurrent_thread_count", [])]
        max_threads = max(m.value for m in thread_metrics) if thread_metrics else 0
        
        return {
            "success_rate": success_rate,
            "avg_response_time_ms": avg_response_time,
            "max_concurrent_threads": max_threads,
            "total_operations": len(concurrent_metrics),
            "concurrency_efficiency_score": self._calculate_concurrency_score(success_rate, avg_response_time)
        }
    
    def _calculate_statistical_summary(self, values: List[float]) -> StatisticalSummary:
        """Calculate comprehensive statistical summary"""
        if not values:
            return StatisticalSummary(0, 0, 0, 0, 0, 0, 0, 0)
        
        return StatisticalSummary(
            count=len(values),
            mean=statistics.mean(values),
            median=statistics.median(values),
            std_dev=statistics.stdev(values) if len(values) > 1 else 0,
            min_value=min(values),
            max_value=max(values),
            percentile_95=np.percentile(values, 95),
            percentile_99=np.percentile(values, 99)
        )
    
    def _detect_performance_trends(self, system_data: List[SystemSnapshot]) -> List[Dict[str, Any]]:
        """Detect performance trends in the data"""
        trends = []
        
        if len(system_data) < 10:  # Need minimum data points for trend analysis
            return trends
        
        # Analyze CPU trend
        cpu_values = [s.cpu_percent for s in system_data[-50:]]  # Last 50 points
        if len(cpu_values) >= 10:
            cpu_trend = self._calculate_trend(cpu_values)
            if abs(cpu_trend) > 0.1:  # Significant trend
                trends.append({
                    "metric": "cpu_percent",
                    "trend": "increasing" if cpu_trend > 0 else "decreasing",
                    "slope": cpu_trend,
                    "significance": "high" if abs(cpu_trend) > 0.5 else "low"
                })
        
        # Analyze memory trend
        memory_values = [s.memory_used_mb for s in system_data[-50:]]
        if len(memory_values) >= 10:
            memory_trend = self._calculate_trend(memory_values)
            if abs(memory_trend) > 1.0:  # > 1MB trend
                trends.append({
                    "metric": "memory_used_mb",
                    "trend": "increasing" if memory_trend > 0 else "decreasing",
                    "slope": memory_trend,
                    "significance": "high" if abs(memory_trend) > 10 else "low"
                })
        
        return trends
    
    def _detect_anomalies(self, system_data: List[SystemSnapshot]) -> List[Dict[str, Any]]:
        """Detect performance anomalies"""
        anomalies = []
        
        if len(system_data) < 20:  # Need sufficient data for anomaly detection
            return anomalies
        
        # CPU spike detection
        cpu_values = [s.cpu_percent for s in system_data]
        cpu_mean = statistics.mean(cpu_values)
        cpu_std = statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
        
        for i, snapshot in enumerate(system_data):
            if cpu_std > 0 and snapshot.cpu_percent > cpu_mean + 3 * cpu_std:  # 3-sigma rule
                anomalies.append({
                    "type": "cpu_spike",
                    "timestamp": snapshot.timestamp,
                    "value": snapshot.cpu_percent,
                    "threshold": cpu_mean + 3 * cpu_std,
                    "severity": "high" if snapshot.cpu_percent > 90 else "medium"
                })
        
        # Memory spike detection
        memory_values = [s.memory_percent for s in system_data]
        memory_mean = statistics.mean(memory_values)
        memory_std = statistics.stdev(memory_values) if len(memory_values) > 1 else 0
        
        for snapshot in system_data:
            if memory_std > 0 and snapshot.memory_percent > memory_mean + 3 * memory_std:
                anomalies.append({
                    "type": "memory_spike", 
                    "timestamp": snapshot.timestamp,
                    "value": snapshot.memory_percent,
                    "threshold": memory_mean + 3 * memory_std,
                    "severity": "high" if snapshot.memory_percent > 90 else "medium"
                })
        
        return anomalies
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend slope using linear regression"""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression
        slope, _ = np.polyfit(x, y, 1)
        return slope
    
    def _calculate_streaming_score(self, activation_rate: float, avg_memory_mb: float) -> float:
        """Calculate streaming efficiency score"""
        # Score based on activation rate and memory efficiency
        activation_score = activation_rate * 60  # Max 60 points
        memory_score = max(0, 40 - (avg_memory_mb / 10))  # Max 40 points, penalty for high memory
        
        return min(100, activation_score + memory_score)
    
    def _calculate_memory_safety_score(self, memory_events: List[PerformanceMetric]) -> float:
        """Calculate memory safety score"""
        if not memory_events:
            return 100  # Perfect score if no events
        
        # Score based on event frequency and severity
        critical_events = sum(1 for m in memory_events if m.value > 8000)  # > 8GB events
        warning_events = len(memory_events) - critical_events
        
        score = 100 - (critical_events * 20) - (warning_events * 5)
        return max(0, score)
    
    def _calculate_concurrency_score(self, success_rate: float, avg_response_time_ms: float) -> float:
        """Calculate concurrency efficiency score"""
        success_score = success_rate * 60  # Max 60 points
        speed_score = max(0, 40 - (avg_response_time_ms / 50))  # Max 40 points, penalty for slow responses
        
        return min(100, success_score + speed_score)
    
    def _generate_recommendations(self, system_perf: Dict, app_perf: Dict,
                                streaming_perf: Dict) -> List[str]:
        """Generate performance recommendations based on analysis"""
        recommendations = []
        
        # CPU recommendations
        cpu_stats = system_perf.get("cpu_percent")
        if cpu_stats and cpu_stats.mean > 80:
            recommendations.append("High average CPU usage detected - consider optimization or scaling")
        
        # Memory recommendations
        memory_stats = system_perf.get("memory_percent")  
        if memory_stats and memory_stats.percentile_95 > 90:
            recommendations.append("High memory usage peaks detected - review memory management")
        
        # Query performance recommendations
        query_time_stats = app_perf.get("query_execution_time_ms")
        if query_time_stats and query_time_stats.mean > 1000:
            recommendations.append("Slow query performance - consider indexing and optimization")
        
        # Streaming recommendations
        if streaming_perf.get("activation_rate", 0) < 0.7:
            recommendations.append("Low streaming activation rate - review threshold configuration")
        
        # Disk I/O recommendations
        disk_read_stats = system_perf.get("disk_read_mb_per_s")
        if disk_read_stats and disk_read_stats.percentile_95 > 100:
            recommendations.append("High disk read activity - consider SSD upgrade or caching")
        
        return recommendations
    
    def export_metrics(self, output_file: Path, format: str = "json") -> None:
        """Export collected metrics to file"""
        if format.lower() == "json":
            self._export_json(output_file)
        elif format.lower() == "csv":
            self._export_csv(output_file)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, output_file: Path) -> None:
        """Export metrics as JSON"""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "collection_interval": self.collection_interval,
            "system_snapshots": [asdict(s) for s in self.system_snapshots],
            "custom_metrics": {
                name: [asdict(m) for m in metrics]
                for name, metrics in self.custom_metrics.items()
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Metrics exported to {output_file}")
    
    def _export_csv(self, output_file: Path) -> None:
        """Export system snapshots as CSV"""
        import csv
        
        with open(output_file, 'w', newline='') as f:
            if not self.system_snapshots:
                return
            
            writer = csv.DictWriter(f, fieldnames=asdict(self.system_snapshots[0]).keys())
            writer.writeheader()
            
            for snapshot in self.system_snapshots:
                writer.writerow(asdict(snapshot))
        
        self.logger.info(f"System metrics exported to {output_file}")