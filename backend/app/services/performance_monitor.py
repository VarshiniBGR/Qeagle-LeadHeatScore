"""
Performance Monitoring Service
Tracks latency, error rates, and provides observability for RAG pipeline
"""

import time
import uuid
from typing import Dict, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager
from collections import defaultdict, deque
import threading
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single request"""
    trace_id: str
    start_time: float
    end_time: Optional[float] = None
    latency_ms: Optional[float] = None
    status_code: Optional[int] = None
    error_type: Optional[str] = None
    rag_steps: Dict[str, float] = None
    
    def __post_init__(self):
        if self.rag_steps is None:
            self.rag_steps = {}


class PerformanceMonitor:
    """Monitors performance metrics and error rates"""
    
    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.metrics_history = deque(maxlen=max_samples)
        self.error_counts = defaultdict(int)
        self.total_requests = 0
        self.lock = threading.Lock()
        
        # Performance thresholds
        self.latency_threshold_ms = 2500  # 2.5s
        self.error_rate_threshold = 0.005  # 0.5%
    
    def start_trace(self, operation: str = "rag_pipeline") -> str:
        """Start a new performance trace"""
        trace_id = str(uuid.uuid4())
        start_time = time.time()
        
        metrics = PerformanceMetrics(
            trace_id=trace_id,
            start_time=start_time
        )
        
        logger.info(
            "Performance trace started",
            trace_id=trace_id,
            operation=operation,
            timestamp=start_time
        )
        
        return trace_id
    
    def record_step(self, trace_id: str, step_name: str, duration_ms: float):
        """Record a step in the RAG pipeline"""
        logger.info(
            "RAG step completed",
            trace_id=trace_id,
            step=step_name,
            duration_ms=duration_ms
        )
    
    def finish_trace(self, trace_id: str, status_code: int = 200, error_type: str = None):
        """Finish a performance trace"""
        end_time = time.time()
        
        with self.lock:
            # Find the metrics for this trace
            metrics = None
            for m in self.metrics_history:
                if m.trace_id == trace_id:
                    metrics = m
                    break
            
            if metrics:
                metrics.end_time = end_time
                metrics.latency_ms = (end_time - metrics.start_time) * 1000
                metrics.status_code = status_code
                metrics.error_type = error_type
                
                # Update counters
                self.total_requests += 1
                if status_code != 200:
                    self.error_counts[error_type or f"http_{status_code}"] += 1
                
                # Log completion
                logger.info(
                    "Performance trace completed",
                    trace_id=trace_id,
                    latency_ms=metrics.latency_ms,
                    status_code=status_code,
                    error_type=error_type,
                    total_requests=self.total_requests
                )
                
                # Check thresholds
                self._check_thresholds(metrics)
    
    def _check_thresholds(self, metrics: PerformanceMetrics):
        """Check if performance thresholds are exceeded"""
        # Check latency threshold
        if metrics.latency_ms and metrics.latency_ms > self.latency_threshold_ms:
            logger.warning(
                "Latency threshold exceeded",
                trace_id=metrics.trace_id,
                latency_ms=metrics.latency_ms,
                threshold_ms=self.latency_threshold_ms
            )
        
        # Check error rate threshold
        if self.total_requests > 0:
            error_rate = sum(self.error_counts.values()) / self.total_requests
            if error_rate > self.error_rate_threshold:
                logger.warning(
                    "Error rate threshold exceeded",
                    error_rate=error_rate,
                    threshold=self.error_rate_threshold,
                    total_requests=self.total_requests,
                    error_counts=dict(self.error_counts)
                )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        with self.lock:
            if not self.metrics_history:
                return {
                    "total_requests": 0,
                    "error_rate": 0.0,
                    "avg_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "error_counts": {}
                }
            
            # Calculate metrics
            latencies = [m.latency_ms for m in self.metrics_history if m.latency_ms]
            error_count = sum(self.error_counts.values())
            error_rate = error_count / self.total_requests if self.total_requests > 0 else 0.0
            
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0
            
            return {
                "total_requests": self.total_requests,
                "error_rate": error_rate,
                "avg_latency_ms": avg_latency,
                "p95_latency_ms": p95_latency,
                "error_counts": dict(self.error_counts),
                "latency_threshold_ms": self.latency_threshold_ms,
                "error_rate_threshold": self.error_rate_threshold
            }
    
    @contextmanager
    def trace_rag_pipeline(self, operation: str = "rag_pipeline"):
        """Context manager for tracing RAG pipeline"""
        trace_id = self.start_trace(operation)
        start_time = time.time()
        
        try:
            yield trace_id
            self.finish_trace(trace_id, status_code=200)
        except Exception as e:
            self.finish_trace(trace_id, status_code=500, error_type=type(e).__name__)
            raise


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


