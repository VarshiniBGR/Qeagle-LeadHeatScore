"""
Performance Tracker for P95 Latency and Error Rate Monitoring
"""
import time
import asyncio
from typing import Dict, List, Optional
from collections import deque
import statistics
from app.utils.logging import get_logger

logger = get_logger(__name__)


class PerformanceTracker:
    """Track P95 latency and error rates for SLA monitoring."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)  # Rolling window
        self.error_count = 0
        self.total_requests = 0
        self.start_time = time.perf_counter()  # More precise timing for P95 measurement
        
    def record_request(self, latency: float, is_error: bool = False):
        """Record a request with its latency and error status."""
        self.latencies.append(latency)
        self.total_requests += 1
        if is_error:
            self.error_count += 1
            
        # Log if SLA is violated with more precise timing info
        if latency > 2.5:
            logger.warning(f"SLA violation: Request took {latency:.3f}s (>2.5s target) - P95 measurement")
    
    def get_p95_latency(self) -> float:
        """Calculate P95 latency from recent requests."""
        if not self.latencies:
            return 0.0
        
        sorted_latencies = sorted(self.latencies)
        p95_index = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[p95_index] if p95_index < len(sorted_latencies) else sorted_latencies[-1]
    
    def get_error_rate(self) -> float:
        """Calculate error rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.error_count / self.total_requests) * 100
    
    def get_stats(self) -> Dict[str, float]:
        """Get comprehensive performance statistics."""
        if not self.latencies:
            return {
                "p95_latency": 0.0,
                "avg_latency": 0.0,
                "error_rate": 0.0,
                "total_requests": 0,
                "sla_violations": 0
            }
        
        p95 = self.get_p95_latency()
        avg = statistics.mean(self.latencies)
        error_rate = self.get_error_rate()
        sla_violations = sum(1 for lat in self.latencies if lat > 2.5)
        
        return {
            "p95_latency": round(p95, 3),
            "avg_latency": round(avg, 3),
            "error_rate": round(error_rate, 3),
            "total_requests": self.total_requests,
            "sla_violations": sla_violations,
            "sla_compliance": round((1 - sla_violations / len(self.latencies)) * 100, 2)
        }
    
    def is_sla_compliant(self) -> bool:
        """Check if current performance meets SLA requirements."""
        stats = self.get_stats()
        p95_ok = stats["p95_latency"] <= 2.5
        error_rate_ok = stats["error_rate"] < 0.5
        return p95_ok and error_rate_ok
    
    def log_performance_summary(self):
        """Log current performance summary."""
        stats = self.get_stats()
        status = "✅ COMPLIANT" if self.is_sla_compliant() else "❌ SLA VIOLATION"
        
        logger.info(
            f"Performance Summary {status}: "
            f"P95={stats['p95_latency']}s, "
            f"Avg={stats['avg_latency']}s, "
            f"ErrorRate={stats['error_rate']}%, "
            f"Requests={stats['total_requests']}"
        )


# Global performance tracker instance
performance_tracker = PerformanceTracker()


def track_performance(func):
    """Decorator to track function performance."""
    from functools import wraps
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # High-precision timing for accurate P95
        is_error = False
        
        try:
            result = await func(*args, **kwargs)
            # Improved error detection logic
            if isinstance(result, dict):
                # Check for explicit error indicators
                if result.get("error") or result.get("success") is False:
                    is_error = True
            # For FastAPI responses, check if it's a valid response object
            elif hasattr(result, '__dict__') and hasattr(result, 'lead_id'):
                # This is likely a successful Recommendation object
                is_error = False
            elif result is None:
                is_error = True
            return result
        except Exception as e:
            is_error = True
            raise
        finally:
            latency = time.perf_counter() - start_time  # Precise P95 measurement
            performance_tracker.record_request(latency, is_error)
            
            # Log every 50 requests for better monitoring
            if performance_tracker.total_requests % 50 == 0:
                performance_tracker.log_performance_summary()
    
    return wrapper
