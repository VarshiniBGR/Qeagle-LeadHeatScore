import structlog
import logging
import sys
import uuid
import time
from typing import Dict, Any, Optional
from contextvars import ContextVar
from app.config import settings

# Context variables for request tracing
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
session_id_var: ContextVar[Optional[str]] = ContextVar('session_id', default=None)

def setup_logging():
    """Configure structured logging with enhanced tracing."""
    
    # Configure structlog with enhanced processors
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            add_tracing_context,  # Custom processor for tracing
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )
    
    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("motor").setLevel(logging.WARNING)
    logging.getLogger("pymongo").setLevel(logging.WARNING)
    
    # Add custom loggers for different components
    logging.getLogger("app.services.classifier").setLevel(logging.INFO)
    logging.getLogger("app.services.next_action_agent").setLevel(logging.INFO)
    logging.getLogger("app.services.email_service").setLevel(logging.INFO)
    logging.getLogger("app.utils.safety_filters").setLevel(logging.INFO)

def add_tracing_context(logger, method_name, event_dict):
    """Add tracing context to log events."""
    # Add request ID if available
    request_id = request_id_var.get()
    if request_id:
        event_dict['request_id'] = request_id
    
    # Add user ID if available
    user_id = user_id_var.get()
    if user_id:
        event_dict['user_id'] = user_id
    
    # Add session ID if available
    session_id = session_id_var.get()
    if session_id:
        event_dict['session_id'] = session_id
    
    # Add timestamp for performance tracking
    event_dict['timestamp'] = time.time()
    
    return event_dict

def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)

def set_request_context(request_id: str, user_id: Optional[str] = None, session_id: Optional[str] = None):
    """Set request context for tracing."""
    request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)
    if session_id:
        session_id_var.set(session_id)

def clear_request_context():
    """Clear request context."""
    request_id_var.set(None)
    user_id_var.set(None)
    session_id_var.set(None)

def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())

class PerformanceLogger:
    """Logger for performance metrics and timing."""
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = get_logger(logger_name)
        self.start_times: Dict[str, float] = {}
    
    def start_timer(self, operation: str) -> str:
        """Start timing an operation."""
        timer_id = f"{operation}_{uuid.uuid4().hex[:8]}"
        self.start_times[timer_id] = time.time()
        self.logger.info("Operation started", operation=operation, timer_id=timer_id)
        return timer_id
    
    def end_timer(self, timer_id: str, operation: str, **kwargs):
        """End timing an operation and log the duration."""
        if timer_id in self.start_times:
            duration = time.time() - self.start_times[timer_id]
            del self.start_times[timer_id]
            
            self.logger.info(
                "Operation completed",
                operation=operation,
                timer_id=timer_id,
                duration_ms=round(duration * 1000, 2),
                **kwargs
            )
            return duration
        return None
    
    def log_metric(self, metric_name: str, value: float, unit: str = "ms", **kwargs):
        """Log a performance metric."""
        self.logger.info(
            "Performance metric",
            metric_name=metric_name,
            value=value,
            unit=unit,
            **kwargs
        )

class BusinessLogger:
    """Logger for business events and metrics."""
    
    def __init__(self, logger_name: str = "business"):
        self.logger = get_logger(logger_name)
    
    def log_lead_classification(self, lead_id: str, heat_score: str, confidence: float, **kwargs):
        """Log lead classification event."""
        self.logger.info(
            "Lead classified",
            event_type="lead_classification",
            lead_id=lead_id,
            heat_score=heat_score,
            confidence=confidence,
            **kwargs
        )
    
    def log_recommendation_generated(self, lead_id: str, channel: str, priority: str, **kwargs):
        """Log recommendation generation event."""
        self.logger.info(
            "Recommendation generated",
            event_type="recommendation_generated",
            lead_id=lead_id,
            channel=channel,
            priority=priority,
            **kwargs
        )
    
    def log_email_sent(self, lead_id: str, to_email: str, template_type: str, **kwargs):
        """Log email sending event."""
        self.logger.info(
            "Email sent",
            event_type="email_sent",
            lead_id=lead_id,
            to_email=to_email,
            template_type=template_type,
            **kwargs
        )
    
    def log_csv_upload(self, filename: str, total_leads: int, processed_leads: int, **kwargs):
        """Log CSV upload event."""
        self.logger.info(
            "CSV uploaded",
            event_type="csv_upload",
            filename=filename,
            total_leads=total_leads,
            processed_leads=processed_leads,
            **kwargs
        )

class SecurityLogger:
    """Logger for security events and threats."""
    
    def __init__(self, logger_name: str = "security"):
        self.logger = get_logger(logger_name)
    
    def log_safety_threat(self, threat_type: str, confidence: float, content_preview: str, **kwargs):
        """Log detected safety threat."""
        self.logger.warning(
            "Safety threat detected",
            event_type="safety_threat",
            threat_type=threat_type,
            confidence=confidence,
            content_preview=content_preview[:100],  # Truncate for privacy
            **kwargs
        )
    
    def log_pii_detected(self, pii_type: str, count: int, **kwargs):
        """Log PII detection event."""
        self.logger.info(
            "PII detected and redacted",
            event_type="pii_detection",
            pii_type=pii_type,
            count=count,
            **kwargs
        )
    
    def log_api_access(self, endpoint: str, method: str, status_code: int, **kwargs):
        """Log API access event."""
        self.logger.info(
            "API access",
            event_type="api_access",
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            **kwargs
        )

# Global logger instances
performance_logger = PerformanceLogger()
business_logger = BusinessLogger()
security_logger = SecurityLogger()
