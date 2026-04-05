"""Performance monitoring for benchmarks.

This module provides system-level performance monitoring using background
threads to sample memory and CPU usage during benchmark execution.
"""

import time
from typing import Any, Dict, List

import psutil

from ..logging_manager import get_logger

logger = get_logger(__name__)


class PerformanceMonitor:
    """Monitor system performance during benchmarks."""

    def __init__(self, sample_interval: float = 0.1):
        """Initialize performance monitor.

        Args:
            sample_interval: How often to sample performance metrics (seconds)
        """
        self.sample_interval = sample_interval
        self.memory_samples: List[float] = []
        self.cpu_samples: List[float] = []
        self.timestamps: List[float] = []
        self.monitoring = False
        self._monitor_thread = None

    def start_monitoring(self):
        """Start performance monitoring in background thread."""
        import threading

        if self.monitoring:
            return

        self.monitoring = True
        self.memory_samples.clear()
        self.cpu_samples.clear()
        self.timestamps.clear()

        def monitor():
            process = psutil.Process()
            start_time = time.time()

            while self.monitoring:
                try:
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)
                    cpu_percent = process.cpu_percent()

                    self.memory_samples.append(memory_mb)
                    self.cpu_samples.append(cpu_percent)
                    self.timestamps.append(time.time() - start_time)

                    time.sleep(self.sample_interval)
                except Exception as e:
                    logger.warning(f"Performance monitoring error: {e}")
                    break

        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return collected metrics."""
        self.monitoring = False

        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

        if not self.memory_samples:
            return {
                "peak_memory_mb": 0.0,
                "average_memory_mb": 0.0,
                "memory_samples": [],
                "timestamps": [],
            }

        return {
            "peak_memory_mb": max(self.memory_samples),
            "average_memory_mb": sum(self.memory_samples) / len(self.memory_samples),
            "memory_samples": self.memory_samples.copy(),
            "timestamps": self.timestamps.copy(),
        }
