"""Enhanced error logging with structured metadata and monitoring.

Provides ErrorMetrics and ErrorLogger classes.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .exceptions import (
    ErrorCategory,
    ErrorSeverity,
    LocalDataError,
)

# ============================================================================
# Enhanced Error Logging and Monitoring
# ============================================================================


@dataclass
class ErrorMetrics:
    """Metrics for error monitoring and analysis."""

    total_errors: int = 0
    errors_by_category: Dict[ErrorCategory, int] = field(default_factory=lambda: {})
    errors_by_severity: Dict[ErrorSeverity, int] = field(default_factory=lambda: {})
    errors_by_database: Dict[str, int] = field(default_factory=lambda: {})
    error_rate_per_minute: deque = field(default_factory=lambda: deque(maxlen=60))
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=100))
    first_error_time: Optional[float] = None
    last_error_time: Optional[float] = None


class ErrorLogger:
    """Enhanced error logging with structured metadata and monitoring."""

    def __init__(self, logger_name: str = "localdata.error_handler"):
        self.logger = logging.getLogger(logger_name)
        self.metrics = ErrorMetrics()
        self.lock = threading.RLock()

        # Setup structured logging format
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - [%(error_code)s] %(message)s",
                defaults={"error_code": "N/A"},
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def log_error(
        self, error: LocalDataError, extra_context: Optional[Dict[str, Any]] = None
    ):
        """Log an error with full context and metadata."""
        current_time = time.time()

        with self.lock:
            # Update metrics
            self.metrics.total_errors += 1
            self.metrics.errors_by_category[error.category] = (
                self.metrics.errors_by_category.get(error.category, 0) + 1
            )
            self.metrics.errors_by_severity[error.severity] = (
                self.metrics.errors_by_severity.get(error.severity, 0) + 1
            )

            if error.database_name:
                self.metrics.errors_by_database[error.database_name] = (
                    self.metrics.errors_by_database.get(error.database_name, 0) + 1
                )

            # Track timing
            if self.metrics.first_error_time is None:
                self.metrics.first_error_time = current_time
            self.metrics.last_error_time = current_time

            # Add to recent errors
            error_record = {
                "timestamp": current_time,
                "error_code": error.error_code,
                "category": error.category.value,
                "severity": error.severity.value,
                "message": error.message,
                "database_name": error.database_name,
                "query_snippet": error.query[:100] if error.query else None,
            }
            self.metrics.recent_errors.append(error_record)

            # Track error rate (errors per minute)
            minute_timestamp = int(current_time // 60)
            if (
                not self.metrics.error_rate_per_minute
                or self.metrics.error_rate_per_minute[-1][0] != minute_timestamp
            ):
                self.metrics.error_rate_per_minute.append([minute_timestamp, 1])
            else:
                self.metrics.error_rate_per_minute[-1][1] += 1

        # Create logging context
        log_context = {
            "error_code": error.error_code,
            "error_category": error.category.value,
            "error_severity": error.severity.value,
            "database_name": error.database_name,
            "error_metadata": error.metadata,
        }

        if extra_context:
            log_context.update(extra_context)

        # Choose log level based on severity
        if error.severity == ErrorSeverity.CRITICAL:
            log_level = logging.CRITICAL
        elif error.severity == ErrorSeverity.HIGH:
            log_level = logging.ERROR
        elif error.severity == ErrorSeverity.MEDIUM:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO

        # Log the error
        self.logger.log(log_level, error.message, extra=log_context)

        # Log recovery suggestions if available
        if error.recovery_suggestions:
            self.logger.info(
                f"Recovery suggestions for {error.error_code}: {'; '.join(error.recovery_suggestions)}",
                extra=log_context,
            )

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        with self.lock:
            current_time = time.time()

            # Calculate error rates
            recent_error_rate = 0.0
            if self.metrics.error_rate_per_minute:
                # Average errors per minute over last 5 minutes
                recent_minutes = [
                    rate
                    for timestamp, rate in self.metrics.error_rate_per_minute
                    if current_time - (timestamp * 60) <= 300  # 5 minutes
                ]
                if recent_minutes:
                    recent_error_rate = sum(recent_minutes) / len(recent_minutes)

            # Time-based statistics
            uptime = (
                current_time - self.metrics.first_error_time
                if self.metrics.first_error_time
                else 0
            )
            overall_error_rate = (
                self.metrics.total_errors / (uptime / 60) if uptime > 0 else 0
            )

            return {
                "total_errors": self.metrics.total_errors,
                "error_rate_per_minute": recent_error_rate,
                "overall_error_rate_per_minute": overall_error_rate,
                "uptime_minutes": uptime / 60,
                "errors_by_category": dict(self.metrics.errors_by_category),
                "errors_by_severity": dict(self.metrics.errors_by_severity),
                "errors_by_database": dict(self.metrics.errors_by_database),
                "first_error_time": self.metrics.first_error_time,
                "last_error_time": self.metrics.last_error_time,
                "recent_errors": list(self.metrics.recent_errors)[
                    -10:
                ],  # Last 10 errors
            }

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status based on error patterns."""
        stats = self.get_error_statistics()

        # Health scoring based on error rates and severity
        health_score = 100.0

        # Deduct points for error rate
        if stats["error_rate_per_minute"] > 10:
            health_score -= 30
        elif stats["error_rate_per_minute"] > 5:
            health_score -= 15
        elif stats["error_rate_per_minute"] > 1:
            health_score -= 5

        # Deduct points for critical/high severity errors
        critical_errors = stats["errors_by_severity"].get(
            ErrorSeverity.CRITICAL.value, 0
        )
        high_errors = stats["errors_by_severity"].get(ErrorSeverity.HIGH.value, 0)

        health_score -= critical_errors * 10
        health_score -= high_errors * 5

        health_score = max(0, health_score)

        # Determine health status
        if health_score >= 90:
            status = "healthy"
        elif health_score >= 70:
            status = "degraded"
        elif health_score >= 50:
            status = "unhealthy"
        else:
            status = "critical"

        return {
            "status": status,
            "health_score": health_score,
            "error_rate": stats["error_rate_per_minute"],
            "critical_errors": critical_errors,
            "high_errors": high_errors,
            "recommendations": self._get_health_recommendations(stats),
        }

    def _get_health_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate health recommendations based on error patterns."""
        recommendations = []

        if stats["error_rate_per_minute"] > 5:
            recommendations.append("High error rate detected - investigate root causes")

        if stats["errors_by_severity"].get(ErrorSeverity.CRITICAL.value, 0) > 0:
            recommendations.append(
                "Critical errors present - immediate attention required"
            )

        # Database-specific recommendations
        db_errors = stats["errors_by_database"]
        if db_errors:
            top_db = max(db_errors.items(), key=lambda x: x[1])
            if top_db[1] > stats["total_errors"] * 0.5:
                recommendations.append(
                    f"Database '{top_db[0]}' has high error rate - check connectivity"
                )

        # Category-specific recommendations
        category_errors = stats["errors_by_category"]
        if (
            category_errors.get(ErrorCategory.CONNECTION.value, 0)
            > stats["total_errors"] * 0.3
        ):
            recommendations.append(
                "Frequent connection errors - check network and database status"
            )

        if (
            category_errors.get(ErrorCategory.TIMEOUT.value, 0)
            > stats["total_errors"] * 0.3
        ):
            recommendations.append(
                "Frequent timeout errors - consider query optimization"
            )

        if not recommendations:
            recommendations.append("System operating within normal parameters")

        return recommendations
