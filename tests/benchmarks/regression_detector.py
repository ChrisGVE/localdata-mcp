#!/usr/bin/env python3
"""
Regression Detector - Advanced Performance Regression Analysis

Detects and analyzes performance regressions in LocalData MCP:
- Statistical regression detection using multiple algorithms
- Trend analysis and anomaly detection
- Performance degradation pattern recognition
- Multi-dimensional regression analysis (memory, CPU, I/O)
- Confidence-based regression scoring
- Historical trend comparison
- Automated regression severity assessment
"""

import json
import logging
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')


class RegressionSeverity(Enum):
    """Severity levels for performance regressions"""
    CRITICAL = "critical"      # >30% degradation or system failure
    HIGH = "high"             # 15-30% degradation
    MEDIUM = "medium"         # 5-15% degradation  
    LOW = "low"               # 2-5% degradation
    NEGLIGIBLE = "negligible"  # <2% degradation


class RegressionType(Enum):
    """Types of performance regressions"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MEMORY_REGRESSION = "memory_regression"
    THROUGHPUT_REDUCTION = "throughput_reduction"
    LATENCY_INCREASE = "latency_increase"
    FAILURE_RATE_INCREASE = "failure_rate_increase"
    RESOURCE_UTILIZATION_SPIKE = "resource_utilization_spike"
    SCALABILITY_REGRESSION = "scalability_regression"


@dataclass
class RegressionAlert:
    """Individual regression detection alert"""
    metric_name: str
    regression_type: RegressionType
    severity: RegressionSeverity
    confidence: float  # 0.0 to 1.0
    current_value: float
    baseline_value: float
    percentage_change: float
    absolute_change: float
    detection_timestamp: str
    context: Dict[str, Any]
    recommendations: List[str]


@dataclass
class RegressionReport:
    """Comprehensive regression analysis report"""
    analysis_timestamp: str
    total_metrics_analyzed: int
    regressions_detected: List[RegressionAlert]
    regression_summary: Dict[RegressionSeverity, int]
    trend_analysis: Dict[str, Any]
    pattern_analysis: Dict[str, Any]
    confidence_distribution: Dict[str, float]
    historical_comparison: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    recommendations: List[str]


class RegressionDetector:
    """Advanced performance regression detection and analysis system"""
    
    def __init__(self, baseline_establisher, performance_collector = None):
        """Initialize regression detector
        
        Args:
            baseline_establisher: BaselineEstablisher instance for baseline data
            performance_collector: Optional PerformanceCollector for trend analysis
        """
        self.baseline_establisher = baseline_establisher
        self.performance_collector = performance_collector
        self.logger = logging.getLogger("RegressionDetector")
        
        # Detection configuration
        self.detection_config = {
            "critical_threshold": 0.30,    # 30% degradation = critical
            "high_threshold": 0.15,        # 15% degradation = high
            "medium_threshold": 0.05,      # 5% degradation = medium
            "low_threshold": 0.02,         # 2% degradation = low
            "confidence_threshold": 0.70,  # 70% confidence minimum
            "trend_window_days": 30,       # 30 days for trend analysis
            "min_samples_for_trend": 5     # Minimum samples for trend detection
        }
        
        # Detection algorithms
        self.detectors = {
            "statistical": self._detect_statistical_regression,
            "trend": self._detect_trend_regression,
            "anomaly": self._detect_anomaly_regression,
            "pattern": self._detect_pattern_regression
        }
        
        self.logger.info("RegressionDetector initialized")
    
    def detect_regressions(self, benchmark_result, baseline_version: str = None) -> RegressionReport:
        """Detect performance regressions in benchmark results"""
        self.logger.info("Starting regression detection analysis")
        
        analysis_start = datetime.now()
        
        # Get baseline for comparison
        baseline_comparison = self.baseline_establisher.compare_to_baseline(
            benchmark_result, baseline_version
        )
        
        detected_regressions = []
        
        # Run all detection algorithms
        for detector_name, detector_func in self.detectors.items():
            self.logger.debug(f"Running {detector_name} regression detection")
            
            try:
                regressions = detector_func(benchmark_result, baseline_comparison)
                detected_regressions.extend(regressions)
                
                self.logger.debug(f"{detector_name} detected {len(regressions)} potential regressions")
                
            except Exception as e:
                self.logger.error(f"Error in {detector_name} detection: {str(e)}")
        
        # Deduplicate and consolidate regressions
        consolidated_regressions = self._consolidate_regressions(detected_regressions)
        
        # Filter by confidence threshold
        high_confidence_regressions = [
            r for r in consolidated_regressions 
            if r.confidence >= self.detection_config["confidence_threshold"]
        ]
        
        # Analyze trends and patterns
        trend_analysis = self._analyze_trends(benchmark_result)
        pattern_analysis = self._analyze_patterns(high_confidence_regressions)
        
        # Calculate regression summary
        regression_summary = self._calculate_regression_summary(high_confidence_regressions)
        
        # Confidence distribution
        confidence_distribution = self._calculate_confidence_distribution(high_confidence_regressions)
        
        # Historical comparison
        historical_comparison = self._perform_historical_comparison(benchmark_result)
        
        # Risk assessment
        risk_assessment = self._assess_regression_risk(
            high_confidence_regressions, trend_analysis, pattern_analysis
        )
        
        # Generate recommendations
        recommendations = self._generate_regression_recommendations(
            high_confidence_regressions, risk_assessment
        )
        
        analysis_duration = datetime.now() - analysis_start
        
        report = RegressionReport(
            analysis_timestamp=analysis_start.isoformat(),
            total_metrics_analyzed=len(baseline_comparison.metric_comparisons),
            regressions_detected=high_confidence_regressions,
            regression_summary=regression_summary,
            trend_analysis=trend_analysis,
            pattern_analysis=pattern_analysis,
            confidence_distribution=confidence_distribution,
            historical_comparison=historical_comparison,
            risk_assessment=risk_assessment,
            recommendations=recommendations
        )
        
        self.logger.info(
            f"Regression analysis completed in {analysis_duration.total_seconds():.2f}s. "
            f"Found {len(high_confidence_regressions)} high-confidence regressions."
        )
        
        return report
    
    def _detect_statistical_regression(self, benchmark_result, baseline_comparison) -> List[RegressionAlert]:
        """Detect regressions using statistical analysis"""
        regressions = []
        
        for metric_name, comparison in baseline_comparison.metric_comparisons.items():
            if comparison["trend"] != "regression":
                continue
            
            # Calculate statistical significance
            pct_change = abs(comparison["percentage_change"])
            
            # Determine severity based on percentage change
            severity = self._calculate_severity_from_change(pct_change)
            
            # Calculate confidence based on significance and within confidence interval
            confidence = self._calculate_statistical_confidence(comparison)
            
            if confidence >= self.detection_config["confidence_threshold"]:
                # Determine regression type
                regression_type = self._determine_regression_type(metric_name, comparison)
                
                # Generate recommendations
                recommendations = self._generate_metric_recommendations(metric_name, comparison)
                
                regression = RegressionAlert(
                    metric_name=metric_name,
                    regression_type=regression_type,
                    severity=severity,
                    confidence=confidence,
                    current_value=comparison["current_value"],
                    baseline_value=comparison["baseline_mean"],
                    percentage_change=comparison["percentage_change"],
                    absolute_change=comparison["absolute_change"],
                    detection_timestamp=datetime.now().isoformat(),
                    context={
                        "detection_method": "statistical",
                        "significance": comparison["significance"],
                        "within_confidence_interval": comparison["within_confidence_interval"],
                        "unit": comparison["unit"]
                    },
                    recommendations=recommendations
                )
                
                regressions.append(regression)
        
        return regressions
    
    def _detect_trend_regression(self, benchmark_result, baseline_comparison) -> List[RegressionAlert]:
        """Detect regressions using trend analysis"""
        regressions = []
        
        if not self.performance_collector:
            return regressions
        
        # Analyze trends in collected performance data
        try:
            performance_report = self.performance_collector.analyze_performance()
            
            for trend in performance_report.performance_trends:
                if trend.get("trend") == "increasing" and trend.get("significance") == "high":
                    metric_name = trend["metric"]
                    
                    # Check if this metric is degrading performance
                    if self._is_performance_degrading_trend(metric_name, trend):
                        confidence = 0.8  # High confidence for trend-based detection
                        severity = self._calculate_severity_from_slope(trend.get("slope", 0))
                        
                        regression = RegressionAlert(
                            metric_name=f"trend_{metric_name}",
                            regression_type=RegressionType.PERFORMANCE_DEGRADATION,
                            severity=severity,
                            confidence=confidence,
                            current_value=trend.get("slope", 0),
                            baseline_value=0,  # Trends are relative to zero slope
                            percentage_change=0,  # Not applicable for trends
                            absolute_change=trend.get("slope", 0),
                            detection_timestamp=datetime.now().isoformat(),
                            context={
                                "detection_method": "trend",
                                "trend_data": trend,
                                "analysis_window": "recent_performance_data"
                            },
                            recommendations=[
                                f"Investigate increasing trend in {metric_name}",
                                "Monitor system resources for potential bottlenecks",
                                "Consider implementing performance optimizations"
                            ]
                        )
                        
                        regressions.append(regression)
        
        except Exception as e:
            self.logger.error(f"Error in trend regression detection: {str(e)}")
        
        return regressions
    
    def _detect_anomaly_regression(self, benchmark_result, baseline_comparison) -> List[RegressionAlert]:
        """Detect regressions using anomaly detection"""
        regressions = []
        
        if not self.performance_collector:
            return regressions
        
        try:
            performance_report = self.performance_collector.analyze_performance()
            
            for anomaly in performance_report.anomalies_detected:
                anomaly_type = anomaly.get("type")
                
                # Focus on performance-related anomalies
                if anomaly_type in ["cpu_spike", "memory_spike"]:
                    severity_level = anomaly.get("severity", "medium")
                    
                    # Map anomaly severity to regression severity
                    if severity_level == "high":
                        severity = RegressionSeverity.HIGH
                    elif severity_level == "medium":
                        severity = RegressionSeverity.MEDIUM
                    else:
                        severity = RegressionSeverity.LOW
                    
                    # Determine regression type
                    if "memory" in anomaly_type:
                        regression_type = RegressionType.MEMORY_REGRESSION
                    else:
                        regression_type = RegressionType.RESOURCE_UTILIZATION_SPIKE
                    
                    confidence = 0.75  # Moderate confidence for anomaly detection
                    
                    regression = RegressionAlert(
                        metric_name=f"anomaly_{anomaly_type}",
                        regression_type=regression_type,
                        severity=severity,
                        confidence=confidence,
                        current_value=anomaly.get("value", 0),
                        baseline_value=anomaly.get("threshold", 0),
                        percentage_change=0,  # Not applicable for anomalies
                        absolute_change=anomaly.get("value", 0) - anomaly.get("threshold", 0),
                        detection_timestamp=datetime.now().isoformat(),
                        context={
                            "detection_method": "anomaly",
                            "anomaly_data": anomaly,
                            "anomaly_timestamp": anomaly.get("timestamp")
                        },
                        recommendations=[
                            f"Investigate {anomaly_type} detected during benchmark",
                            "Review system resource allocation",
                            "Consider implementing resource monitoring alerts"
                        ]
                    )
                    
                    regressions.append(regression)
        
        except Exception as e:
            self.logger.error(f"Error in anomaly regression detection: {str(e)}")
        
        return regressions
    
    def _detect_pattern_regression(self, benchmark_result, baseline_comparison) -> List[RegressionAlert]:
        """Detect regressions using pattern analysis"""
        regressions = []
        
        # Analyze patterns in the benchmark results
        patterns = self._identify_performance_patterns(benchmark_result, baseline_comparison)
        
        for pattern in patterns:
            if pattern.get("regression_risk", 0) > 0.7:  # High regression risk
                confidence = pattern.get("confidence", 0.6)
                
                if confidence >= self.detection_config["confidence_threshold"]:
                    severity = self._calculate_severity_from_risk(pattern.get("regression_risk", 0))
                    
                    regression = RegressionAlert(
                        metric_name=f"pattern_{pattern['name']}",
                        regression_type=RegressionType.PERFORMANCE_DEGRADATION,
                        severity=severity,
                        confidence=confidence,
                        current_value=pattern.get("current_score", 0),
                        baseline_value=pattern.get("baseline_score", 0),
                        percentage_change=pattern.get("change_percentage", 0),
                        absolute_change=pattern.get("absolute_change", 0),
                        detection_timestamp=datetime.now().isoformat(),
                        context={
                            "detection_method": "pattern",
                            "pattern_data": pattern,
                            "pattern_description": pattern.get("description", "")
                        },
                        recommendations=pattern.get("recommendations", [])
                    )
                    
                    regressions.append(regression)
        
        return regressions
    
    def _consolidate_regressions(self, regressions: List[RegressionAlert]) -> List[RegressionAlert]:
        """Consolidate duplicate regressions and improve confidence scores"""
        # Group regressions by similar metrics
        regression_groups = defaultdict(list)
        
        for regression in regressions:
            # Create grouping key based on metric name and type
            base_metric = self._extract_base_metric_name(regression.metric_name)
            group_key = f"{base_metric}_{regression.regression_type.value}"
            regression_groups[group_key].append(regression)
        
        consolidated = []
        
        for group_key, group_regressions in regression_groups.items():
            if len(group_regressions) == 1:
                consolidated.append(group_regressions[0])
            else:
                # Consolidate multiple detections of same regression
                consolidated_regression = self._merge_regression_alerts(group_regressions)
                consolidated.append(consolidated_regression)
        
        return consolidated
    
    def _merge_regression_alerts(self, regressions: List[RegressionAlert]) -> RegressionAlert:
        """Merge multiple regression alerts for the same metric"""
        # Take the highest severity
        max_severity = max(regressions, key=lambda r: self._severity_to_numeric(r.severity))
        
        # Calculate weighted confidence (higher severity gets more weight)
        weights = [self._severity_to_numeric(r.severity) + 1 for r in regressions]
        weighted_confidence = sum(r.confidence * w for r, w in zip(regressions, weights)) / sum(weights)
        
        # Combine recommendations
        all_recommendations = []
        for regression in regressions:
            all_recommendations.extend(regression.recommendations)
        unique_recommendations = list(set(all_recommendations))
        
        # Combine contexts
        combined_context = {
            "detection_methods": [r.context.get("detection_method", "unknown") for r in regressions],
            "consolidated_from": len(regressions),
            "individual_confidences": [r.confidence for r in regressions]
        }
        
        # Use values from highest severity regression
        base_regression = max_severity
        
        return RegressionAlert(
            metric_name=base_regression.metric_name,
            regression_type=base_regression.regression_type,
            severity=base_regression.severity,
            confidence=min(weighted_confidence, 1.0),  # Cap at 1.0
            current_value=base_regression.current_value,
            baseline_value=base_regression.baseline_value,
            percentage_change=base_regression.percentage_change,
            absolute_change=base_regression.absolute_change,
            detection_timestamp=datetime.now().isoformat(),
            context=combined_context,
            recommendations=unique_recommendations
        )
    
    def _calculate_severity_from_change(self, pct_change: float) -> RegressionSeverity:
        """Calculate regression severity from percentage change"""
        if pct_change >= self.detection_config["critical_threshold"] * 100:
            return RegressionSeverity.CRITICAL
        elif pct_change >= self.detection_config["high_threshold"] * 100:
            return RegressionSeverity.HIGH
        elif pct_change >= self.detection_config["medium_threshold"] * 100:
            return RegressionSeverity.MEDIUM
        elif pct_change >= self.detection_config["low_threshold"] * 100:
            return RegressionSeverity.LOW
        else:
            return RegressionSeverity.NEGLIGIBLE
    
    def _calculate_severity_from_slope(self, slope: float) -> RegressionSeverity:
        """Calculate regression severity from trend slope"""
        abs_slope = abs(slope)
        
        if abs_slope > 50:  # Very steep trend
            return RegressionSeverity.HIGH
        elif abs_slope > 20:  # Moderate trend
            return RegressionSeverity.MEDIUM
        elif abs_slope > 5:   # Slight trend
            return RegressionSeverity.LOW
        else:
            return RegressionSeverity.NEGLIGIBLE
    
    def _calculate_severity_from_risk(self, risk_score: float) -> RegressionSeverity:
        """Calculate regression severity from pattern risk score"""
        if risk_score >= 0.9:
            return RegressionSeverity.CRITICAL
        elif risk_score >= 0.8:
            return RegressionSeverity.HIGH
        elif risk_score >= 0.7:
            return RegressionSeverity.MEDIUM
        else:
            return RegressionSeverity.LOW
    
    def _calculate_statistical_confidence(self, comparison: Dict) -> float:
        """Calculate confidence score for statistical regression"""
        base_confidence = 0.5
        
        # Boost confidence based on significance
        if comparison["significance"] == "high":
            base_confidence += 0.3
        elif comparison["significance"] == "medium":
            base_confidence += 0.2
        else:
            base_confidence += 0.1
        
        # Reduce confidence if within confidence interval (might be normal variance)
        if comparison["within_confidence_interval"]:
            base_confidence -= 0.1
        
        # Boost confidence for large percentage changes
        pct_change = abs(comparison["percentage_change"])
        if pct_change > 30:
            base_confidence += 0.2
        elif pct_change > 15:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _determine_regression_type(self, metric_name: str, comparison: Dict) -> RegressionType:
        """Determine the type of regression based on metric name and data"""
        metric_lower = metric_name.lower()
        
        if "memory" in metric_lower:
            return RegressionType.MEMORY_REGRESSION
        elif "throughput" in metric_lower:
            return RegressionType.THROUGHPUT_REDUCTION
        elif "time" in metric_lower or "latency" in metric_lower:
            return RegressionType.LATENCY_INCREASE
        elif "concurrent" in metric_lower:
            return RegressionType.SCALABILITY_REGRESSION
        elif "cpu" in metric_lower or "resource" in metric_lower:
            return RegressionType.RESOURCE_UTILIZATION_SPIKE
        else:
            return RegressionType.PERFORMANCE_DEGRADATION
    
    def _generate_metric_recommendations(self, metric_name: str, comparison: Dict) -> List[str]:
        """Generate specific recommendations based on metric and comparison"""
        recommendations = []
        metric_lower = metric_name.lower()
        
        if "memory" in metric_lower:
            recommendations.extend([
                "Review memory allocation patterns",
                "Check for memory leaks or inefficient data structures",
                "Consider memory optimization strategies"
            ])
        elif "throughput" in metric_lower:
            recommendations.extend([
                "Analyze throughput bottlenecks",
                "Review I/O operations and database queries",
                "Consider parallel processing optimizations"
            ])
        elif "time" in metric_lower:
            recommendations.extend([
                "Profile code execution to identify slow operations",
                "Review algorithm efficiency",
                "Consider caching strategies"
            ])
        elif "concurrent" in metric_lower:
            recommendations.extend([
                "Review thread safety and synchronization",
                "Analyze lock contention patterns",
                "Consider connection pooling optimization"
            ])
        else:
            recommendations.extend([
                f"Investigate performance degradation in {metric_name}",
                "Review recent code changes and optimizations",
                "Monitor system resources during operations"
            ])
        
        return recommendations
    
    def _is_performance_degrading_trend(self, metric_name: str, trend: Dict) -> bool:
        """Determine if a trend indicates performance degradation"""
        metric_lower = metric_name.lower()
        
        # Metrics where increasing values indicate degradation
        degrading_metrics = [
            "cpu_percent", "memory_percent", "memory_used", 
            "execution_time", "response_time", "latency",
            "error_rate", "failure_rate"
        ]
        
        for degrading_metric in degrading_metrics:
            if degrading_metric in metric_lower:
                return True
        
        return False
    
    def _identify_performance_patterns(self, benchmark_result, baseline_comparison) -> List[Dict]:
        """Identify performance patterns that might indicate regressions"""
        patterns = []
        
        # Pattern 1: Consistent degradation across multiple datasets
        dataset_degradations = 0
        total_datasets = 0
        
        for metric_name, comparison in baseline_comparison.metric_comparisons.items():
            if any(dataset in metric_name for dataset in ["ecommerce", "iot", "social", "financial"]):
                total_datasets += 1
                if comparison["trend"] == "regression":
                    dataset_degradations += 1
        
        if total_datasets > 0:
            degradation_rate = dataset_degradations / total_datasets
            
            if degradation_rate > 0.6:  # More than 60% of datasets showing degradation
                patterns.append({
                    "name": "widespread_dataset_degradation",
                    "description": f"{degradation_rate:.1%} of datasets showing performance degradation",
                    "regression_risk": degradation_rate,
                    "confidence": 0.8,
                    "current_score": degradation_rate,
                    "baseline_score": 0.0,
                    "change_percentage": degradation_rate * 100,
                    "absolute_change": degradation_rate,
                    "recommendations": [
                        "Investigate system-wide performance issues",
                        "Review recent infrastructure changes",
                        "Consider horizontal scaling solutions"
                    ]
                })
        
        # Pattern 2: Memory and throughput correlation
        memory_regression = any(
            "memory" in metric.lower() and comp["trend"] == "regression"
            for metric, comp in baseline_comparison.metric_comparisons.items()
        )
        
        throughput_regression = any(
            "throughput" in metric.lower() and comp["trend"] == "regression" 
            for metric, comp in baseline_comparison.metric_comparisons.items()
        )
        
        if memory_regression and throughput_regression:
            patterns.append({
                "name": "memory_throughput_correlation",
                "description": "Both memory usage and throughput showing degradation",
                "regression_risk": 0.85,
                "confidence": 0.9,
                "current_score": 0.0,
                "baseline_score": 1.0,
                "change_percentage": -100,
                "absolute_change": -1.0,
                "recommendations": [
                    "Memory pressure may be affecting throughput",
                    "Review memory allocation in high-throughput operations",
                    "Consider memory-efficient data structures"
                ]
            })
        
        return patterns
    
    def _extract_base_metric_name(self, metric_name: str) -> str:
        """Extract base metric name for grouping"""
        # Remove prefixes like "trend_", "anomaly_", "pattern_"
        prefixes = ["trend_", "anomaly_", "pattern_"]
        for prefix in prefixes:
            if metric_name.startswith(prefix):
                return metric_name[len(prefix):]
        
        # Remove dataset prefixes
        datasets = ["ecommerce_", "iot_", "social_media_", "financial_"]
        for dataset in datasets:
            if metric_name.startswith(dataset):
                return metric_name[len(dataset):]
        
        return metric_name
    
    def _severity_to_numeric(self, severity: RegressionSeverity) -> int:
        """Convert severity to numeric value for comparison"""
        severity_values = {
            RegressionSeverity.NEGLIGIBLE: 1,
            RegressionSeverity.LOW: 2,
            RegressionSeverity.MEDIUM: 3,
            RegressionSeverity.HIGH: 4,
            RegressionSeverity.CRITICAL: 5
        }
        return severity_values.get(severity, 1)
    
    def _analyze_trends(self, benchmark_result) -> Dict[str, Any]:
        """Analyze performance trends"""
        trend_analysis = {
            "overall_trend": "stable",
            "trend_confidence": 0.5,
            "trending_metrics": [],
            "trend_summary": "Insufficient data for trend analysis"
        }
        
        if self.performance_collector:
            try:
                performance_report = self.performance_collector.analyze_performance()
                
                if performance_report.performance_trends:
                    trend_analysis["trending_metrics"] = performance_report.performance_trends
                    trend_analysis["trend_summary"] = f"Analyzed {len(performance_report.performance_trends)} trending metrics"
                    
                    # Determine overall trend
                    increasing_trends = sum(1 for t in performance_report.performance_trends if t.get("trend") == "increasing")
                    total_trends = len(performance_report.performance_trends)
                    
                    if increasing_trends > total_trends * 0.6:
                        trend_analysis["overall_trend"] = "degrading"
                        trend_analysis["trend_confidence"] = 0.8
                    elif increasing_trends < total_trends * 0.3:
                        trend_analysis["overall_trend"] = "improving"
                        trend_analysis["trend_confidence"] = 0.7
                    else:
                        trend_analysis["overall_trend"] = "stable"
                        trend_analysis["trend_confidence"] = 0.6
                        
            except Exception as e:
                self.logger.error(f"Error in trend analysis: {str(e)}")
        
        return trend_analysis
    
    def _analyze_patterns(self, regressions: List[RegressionAlert]) -> Dict[str, Any]:
        """Analyze patterns in detected regressions"""
        if not regressions:
            return {
                "pattern_count": 0,
                "severity_distribution": {},
                "type_distribution": {},
                "common_patterns": []
            }
        
        # Severity distribution
        severity_counts = defaultdict(int)
        for regression in regressions:
            severity_counts[regression.severity.value] += 1
        
        # Type distribution
        type_counts = defaultdict(int)
        for regression in regressions:
            type_counts[regression.regression_type.value] += 1
        
        # Common patterns
        common_patterns = []
        
        # Pattern: Multiple memory-related regressions
        memory_regressions = [r for r in regressions if "memory" in r.metric_name.lower()]
        if len(memory_regressions) > 2:
            common_patterns.append({
                "pattern": "multiple_memory_regressions",
                "count": len(memory_regressions),
                "description": "Multiple memory-related performance regressions detected"
            })
        
        # Pattern: High severity concentration
        high_severity_count = len([r for r in regressions if r.severity in [RegressionSeverity.HIGH, RegressionSeverity.CRITICAL]])
        if high_severity_count > len(regressions) * 0.5:
            common_patterns.append({
                "pattern": "high_severity_concentration",
                "count": high_severity_count,
                "description": "High concentration of severe performance regressions"
            })
        
        return {
            "pattern_count": len(common_patterns),
            "severity_distribution": dict(severity_counts),
            "type_distribution": dict(type_counts),
            "common_patterns": common_patterns
        }
    
    def _calculate_regression_summary(self, regressions: List[RegressionAlert]) -> Dict[RegressionSeverity, int]:
        """Calculate summary of regressions by severity"""
        summary = {severity: 0 for severity in RegressionSeverity}
        
        for regression in regressions:
            summary[regression.severity] += 1
        
        return summary
    
    def _calculate_confidence_distribution(self, regressions: List[RegressionAlert]) -> Dict[str, float]:
        """Calculate confidence score distribution"""
        if not regressions:
            return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        
        confidences = [r.confidence for r in regressions]
        
        return {
            "mean": statistics.mean(confidences),
            "median": statistics.median(confidences),
            "min": min(confidences),
            "max": max(confidences)
        }
    
    def _perform_historical_comparison(self, benchmark_result) -> Dict[str, Any]:
        """Perform historical comparison analysis"""
        # This would compare against historical benchmark data
        # For now, return placeholder data
        return {
            "historical_data_available": False,
            "comparison_period": "N/A",
            "trend_analysis": "Insufficient historical data",
            "recommendations": ["Establish regular benchmarking to build historical data"]
        }
    
    def _assess_regression_risk(self, regressions: List[RegressionAlert], 
                               trend_analysis: Dict, pattern_analysis: Dict) -> Dict[str, Any]:
        """Assess overall regression risk"""
        if not regressions:
            return {
                "risk_level": "low",
                "risk_score": 0.1,
                "risk_factors": [],
                "mitigation_priority": "low"
            }
        
        # Calculate base risk from regressions
        critical_count = len([r for r in regressions if r.severity == RegressionSeverity.CRITICAL])
        high_count = len([r for r in regressions if r.severity == RegressionSeverity.HIGH])
        
        base_risk = (critical_count * 0.3 + high_count * 0.2) / max(len(regressions), 1)
        
        # Adjust for trends
        if trend_analysis["overall_trend"] == "degrading":
            base_risk += 0.2
        elif trend_analysis["overall_trend"] == "improving":
            base_risk -= 0.1
        
        # Adjust for patterns
        pattern_multiplier = 1.0 + (pattern_analysis["pattern_count"] * 0.1)
        final_risk = min(base_risk * pattern_multiplier, 1.0)
        
        # Determine risk level
        if final_risk >= 0.8:
            risk_level = "critical"
            priority = "critical"
        elif final_risk >= 0.6:
            risk_level = "high"
            priority = "high"
        elif final_risk >= 0.4:
            risk_level = "medium"
            priority = "medium"
        else:
            risk_level = "low"
            priority = "low"
        
        # Risk factors
        risk_factors = []
        if critical_count > 0:
            risk_factors.append(f"{critical_count} critical regressions detected")
        if high_count > 0:
            risk_factors.append(f"{high_count} high-severity regressions detected")
        if trend_analysis["overall_trend"] == "degrading":
            risk_factors.append("Overall performance trend is degrading")
        if pattern_analysis["pattern_count"] > 0:
            risk_factors.append(f"{pattern_analysis['pattern_count']} regression patterns identified")
        
        return {
            "risk_level": risk_level,
            "risk_score": final_risk,
            "risk_factors": risk_factors,
            "mitigation_priority": priority,
            "critical_regressions": critical_count,
            "high_severity_regressions": high_count
        }
    
    def _generate_regression_recommendations(self, regressions: List[RegressionAlert],
                                           risk_assessment: Dict) -> List[str]:
        """Generate recommendations based on detected regressions"""
        recommendations = []
        
        if not regressions:
            recommendations.append("No significant performance regressions detected - maintain current monitoring")
            return recommendations
        
        # High-priority recommendations based on risk
        if risk_assessment["risk_level"] == "critical":
            recommendations.append("🚨 CRITICAL: Immediate investigation required - significant performance regressions detected")
            recommendations.append("Stop deployment pipeline until regressions are resolved")
            recommendations.append("Engage performance engineering team immediately")
        
        elif risk_assessment["risk_level"] == "high":
            recommendations.append("⚠️  HIGH PRIORITY: Address performance regressions before production deployment")
            recommendations.append("Review recent code changes and rollback if necessary")
        
        # Specific recommendations from individual regressions
        all_regression_recommendations = []
        for regression in regressions:
            all_regression_recommendations.extend(regression.recommendations)
        
        # Deduplicate and add top recommendations
        unique_recommendations = list(set(all_regression_recommendations))
        recommendations.extend(unique_recommendations[:5])  # Top 5 specific recommendations
        
        # General recommendations
        recommendations.extend([
            "Establish baseline monitoring for early regression detection",
            "Implement automated performance testing in CI/CD pipeline",
            "Schedule regular performance review sessions"
        ])
        
        return recommendations
    
    def export_regression_report(self, report: RegressionReport, output_file: Path):
        """Export regression report to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        self.logger.info(f"Regression report exported to {output_file}")