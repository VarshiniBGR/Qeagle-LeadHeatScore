import re
from typing import List, Dict, Any
from app.utils.logging import get_logger


logger = get_logger(__name__)


class SafetyFilter:
    """Safety filters for content validation."""
    
    def __init__(self):
        # PII patterns
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        }
        
        # Prompt injection patterns
        self.injection_patterns = [
            r'ignore\s+previous\s+instructions',
            r'forget\s+everything',
            r'you\s+are\s+now',
            r'pretend\s+to\s+be',
            r'act\s+as\s+if',
            r'system\s+prompt',
            r'jailbreak',
            r'override',
        ]
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect PII in text."""
        detected = {}
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected[pii_type] = matches
        
        return detected
    
    def redact_pii(self, text: str) -> str:
        """Redact PII from text."""
        redacted_text = text
        
        for pii_type, pattern in self.pii_patterns.items():
            redacted_text = re.sub(
                pattern, 
                f'[REDACTED_{pii_type.upper()}]', 
                redacted_text, 
                flags=re.IGNORECASE
            )
        
        return redacted_text
    
    def detect_prompt_injection(self, text: str) -> bool:
        """Detect potential prompt injection attempts."""
        text_lower = text.lower()
        
        for pattern in self.injection_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def validate_content(self, content: str) -> Dict[str, Any]:
        """Comprehensive content validation."""
        result = {
            'is_safe': True,
            'pii_detected': {},
            'injection_detected': False,
            'redacted_content': content,
            'warnings': []
        }
        
        # Check for PII
        pii_detected = self.detect_pii(content)
        if pii_detected:
            result['pii_detected'] = pii_detected
            result['redacted_content'] = self.redact_pii(content)
            result['warnings'].append("PII detected and redacted")
        
        # Check for prompt injection
        if self.detect_prompt_injection(content):
            result['injection_detected'] = True
            result['is_safe'] = False
            result['warnings'].append("Potential prompt injection detected")
        
        return result
    
    def sanitize_lead_data(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize lead data for processing."""
        sanitized = lead_data.copy()
        
        # Sanitize string fields
        string_fields = ['source', 'region', 'role', 'campaign', 'last_touch', 'prior_course_interest']
        
        for field in string_fields:
            if field in sanitized and isinstance(sanitized[field], str):
                validation = self.validate_content(sanitized[field])
                if not validation['is_safe']:
                    logger.warning(
                        "Unsafe content detected in lead data",
                        field=field,
                        warnings=validation['warnings']
                    )
                    # Use redacted content
                    sanitized[field] = validation['redacted_content']
        
        return sanitized


# Global safety filter instance
safety_filter = SafetyFilter()
