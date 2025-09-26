"""
Safety filters for prompt injection detection and content sanitization.
"""
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SafetyResult:
    """Result of safety filtering."""
    is_safe: bool
    filtered_content: str
    detected_threats: List[str]
    confidence: float

class PromptInjectionDetector:
    """Detects potential prompt injection attempts."""
    
    def __init__(self):
        # Common prompt injection patterns
        self.injection_patterns = [
            # Direct instruction attempts
            r'(?i)(ignore|forget|disregard).*(previous|above|instructions)',
            r'(?i)(you are|act as|pretend to be)',
            r'(?i)(system|admin|root).*(prompt|command)',
            
            # Role manipulation
            r'(?i)(new instructions|override|replace)',
            r'(?i)(jailbreak|escape|bypass)',
            
            # Data extraction attempts
            r'(?i)(show me|reveal|display).*(prompt|system|config)',
            r'(?i)(what are your|list your).*(instructions|rules)',
            
            # Code injection
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'data:text/html',
            
            # SQL injection patterns
            r'(?i)(union|select|drop|insert|update|delete).*(from|into|table)',
            r'(?i)(or|and).*(1=1|true)',
            
            # Command injection
            r'[;&|`$].*(rm|del|format|shutdown)',
            r'(?i)(exec|eval|system|shell)',
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern) for pattern in self.injection_patterns]
        
        # Threat severity weights
        self.threat_weights = {
            'instruction_override': 0.9,
            'role_manipulation': 0.8,
            'data_extraction': 0.7,
            'code_injection': 0.95,
            'sql_injection': 0.9,
            'command_injection': 0.95,
        }
    
    def detect_injection(self, text: str) -> SafetyResult:
        """
        Detect potential prompt injection in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            SafetyResult with detection results
        """
        if not text or not isinstance(text, str):
            return SafetyResult(
                is_safe=True,
                filtered_content=text or "",
                detected_threats=[],
                confidence=1.0
            )
        
        detected_threats = []
        threat_score = 0.0
        
        # Check for injection patterns
        for i, pattern in enumerate(self.compiled_patterns):
            matches = pattern.findall(text)
            if matches:
                threat_type = self._categorize_threat(i)
                detected_threats.append({
                    'type': threat_type,
                    'pattern': pattern.pattern,
                    'matches': matches,
                    'severity': self.threat_weights.get(threat_type, 0.5)
                })
                threat_score += self.threat_weights.get(threat_type, 0.5)
        
        # Additional heuristics
        heuristic_score = self._check_heuristics(text)
        threat_score += heuristic_score
        
        # Determine if content is safe
        is_safe = threat_score < 0.5
        confidence = min(threat_score, 1.0)
        
        # Filter content if unsafe
        filtered_content = self._filter_content(text) if not is_safe else text
        
        logger.info(f"Safety check: threat_score={threat_score:.2f}, is_safe={is_safe}")
        
        return SafetyResult(
            is_safe=is_safe,
            filtered_content=filtered_content,
            detected_threats=[t['type'] for t in detected_threats],
            confidence=confidence
        )
    
    def _categorize_threat(self, pattern_index: int) -> str:
        """Categorize threat based on pattern index."""
        if pattern_index < 3:
            return 'instruction_override'
        elif pattern_index < 6:
            return 'role_manipulation'
        elif pattern_index < 8:
            return 'data_extraction'
        elif pattern_index < 11:
            return 'code_injection'
        elif pattern_index < 13:
            return 'sql_injection'
        else:
            return 'command_injection'
    
    def _check_heuristics(self, text: str) -> float:
        """Check additional heuristics for suspicious content."""
        score = 0.0
        
        # Check for excessive special characters
        special_char_ratio = len(re.findall(r'[^\w\s]', text)) / len(text) if text else 0
        if special_char_ratio > 0.3:
            score += 0.3
        
        # Check for repeated patterns
        if len(set(text.split())) < len(text.split()) * 0.5:
            score += 0.2
        
        # Check for suspicious length patterns
        if len(text) > 1000:  # Very long inputs
            score += 0.1
        
        # Check for base64-like patterns
        if re.search(r'[A-Za-z0-9+/]{20,}={0,2}', text):
            score += 0.2
        
        return score
    
    def _filter_content(self, text: str) -> str:
        """Filter out potentially harmful content."""
        # Remove script tags
        text = re.sub(r'<script.*?>.*?</script>', '[FILTERED]', text, flags=re.IGNORECASE)
        
        # Remove javascript: protocols
        text = re.sub(r'javascript:', '[FILTERED]', text, flags=re.IGNORECASE)
        
        # Remove SQL keywords
        sql_keywords = ['union', 'select', 'drop', 'insert', 'update', 'delete', 'alter']
        for keyword in sql_keywords:
            text = re.sub(f'(?i)\\b{keyword}\\b', '[FILTERED]', text)
        
        # Remove command injection patterns
        text = re.sub(r'[;&|`$].*', '[FILTERED]', text)
        
        return text

class PIIDetector:
    """Detects and redacts personally identifiable information."""
    
    def __init__(self):
        # PII patterns
        self.pii_patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'),
            'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            'credit_card': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
        }
    
    def detect_and_redact(self, text: str) -> SafetyResult:
        """
        Detect and redact PII from text.
        
        Args:
            text: Input text to process
            
        Returns:
            SafetyResult with redacted content
        """
        if not text or not isinstance(text, str):
            return SafetyResult(
                is_safe=True,
                filtered_content=text or "",
                detected_threats=[],
                confidence=1.0
            )
        
        detected_threats = []
        filtered_content = text
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(filtered_content)
            if matches:
                detected_threats.append(pii_type)
                # Redact with appropriate mask
                mask = self._get_mask(pii_type)
                filtered_content = pattern.sub(mask, filtered_content)
        
        is_safe = len(detected_threats) == 0
        confidence = 1.0 - (len(detected_threats) * 0.2)
        
        logger.info(f"PII detection: found {len(detected_threats)} types, is_safe={is_safe}")
        
        return SafetyResult(
            is_safe=is_safe,
            filtered_content=filtered_content,
            detected_threats=detected_threats,
            confidence=confidence
        )
    
    def _get_mask(self, pii_type: str) -> str:
        """Get appropriate mask for PII type."""
        masks = {
            'email': '[EMAIL_REDACTED]',
            'phone': '[PHONE_REDACTED]',
            'ssn': '[SSN_REDACTED]',
            'credit_card': '[CARD_REDACTED]',
            'ip_address': '[IP_REDACTED]',
        }
        return masks.get(pii_type, '[REDACTED]')

class ContentSanitizer:
    """Main content sanitization class."""
    
    def __init__(self):
        self.injection_detector = PromptInjectionDetector()
        self.pii_detector = PIIDetector()
    
    def sanitize(self, text: str, check_injection: bool = True, check_pii: bool = True) -> SafetyResult:
        """
        Sanitize content by checking for threats and PII.
        
        Args:
            text: Content to sanitize
            check_injection: Whether to check for prompt injection
            check_pii: Whether to check for PII
            
        Returns:
            SafetyResult with sanitized content
        """
        if not text:
            return SafetyResult(
                is_safe=True,
                filtered_content="",
                detected_threats=[],
                confidence=1.0
            )
        
        all_threats = []
        confidence_scores = []
        current_content = text
        
        # Check for prompt injection
        if check_injection:
            injection_result = self.injection_detector.detect_injection(current_content)
            if not injection_result.is_safe:
                all_threats.extend(injection_result.detected_threats)
                current_content = injection_result.filtered_content
            confidence_scores.append(injection_result.confidence)
        
        # Check for PII
        if check_pii:
            pii_result = self.pii_detector.detect_and_redact(current_content)
            if not pii_result.is_safe:
                all_threats.extend(pii_result.detected_threats)
                current_content = pii_result.filtered_content
            confidence_scores.append(pii_result.confidence)
        
        # Overall safety assessment
        is_safe = len(all_threats) == 0
        overall_confidence = min(confidence_scores) if confidence_scores else 1.0
        
        logger.info(f"Content sanitization: threats={len(all_threats)}, is_safe={is_safe}")
        
        return SafetyResult(
            is_safe=is_safe,
            filtered_content=current_content,
            detected_threats=all_threats,
            confidence=overall_confidence
        )

# Global sanitizer instance
content_sanitizer = ContentSanitizer()

def sanitize_content(text: str, check_injection: bool = True, check_pii: bool = True) -> SafetyResult:
    """
    Convenience function for content sanitization.
    
    Args:
        text: Content to sanitize
        check_injection: Whether to check for prompt injection
        check_pii: Whether to check for PII
        
    Returns:
        SafetyResult with sanitized content
    """
    return content_sanitizer.sanitize(text, check_injection, check_pii)
