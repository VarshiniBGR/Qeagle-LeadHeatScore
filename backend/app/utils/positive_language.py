"""
Positive Language Utilities
Converts potentially demotivating terms to more positive, encouraging language.
"""

from typing import Dict, Any


class PositiveLanguageConverter:
    """Converts interest levels and other terms to more positive language."""
    
    INTEREST_MAPPING = {
        'low': 'exploring',
        'medium': 'showing interest in', 
        'high': 'strongly interested in'
    }
    
    ENGAGEMENT_MAPPING = {
        'low': 'beginning to explore',
        'medium': 'actively exploring',
        'high': 'deeply engaged with'
    }
    
    COMMITMENT_MAPPING = {
        'low': 'considering',
        'medium': 'evaluating',
        'high': 'ready to commit to'
    }
    
    @classmethod
    def convert_interest_level(cls, interest_level: str) -> str:
        """Convert interest level to positive language."""
        return cls.INTEREST_MAPPING.get(interest_level.lower(), 'interested in')
    
    @classmethod
    def convert_engagement_level(cls, engagement_level: str) -> str:
        """Convert engagement level to positive language."""
        return cls.ENGAGEMENT_MAPPING.get(engagement_level.lower(), 'engaging with')
    
    @classmethod
    def convert_commitment_level(cls, commitment_level: str) -> str:
        """Convert commitment level to positive language."""
        return cls.COMMITMENT_MAPPING.get(commitment_level.lower(), 'considering')
    
    @classmethod
    def get_positive_phrase(cls, category: str, level: str) -> str:
        """Get positive phrase based on category and level."""
        mappings = {
            'interest': cls.INTEREST_MAPPING,
            'engagement': cls.ENGAGEMENT_MAPPING,
            'commitment': cls.COMMITMENT_MAPPING
        }
        
        mapping = mappings.get(category.lower(), cls.INTEREST_MAPPING)
        return mapping.get(level.lower(), 'interested in')
    
    @classmethod
    def enhance_lead_data(cls, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance lead data with positive language versions."""
        enhanced_data = lead_data.copy()
        
        # Convert interest level
        if 'prior_course_interest' in enhanced_data:
            enhanced_data['positive_interest'] = cls.convert_interest_level(
                enhanced_data['prior_course_interest']
            )
            enhanced_data['positive_engagement'] = cls.convert_engagement_level(
                enhanced_data['prior_course_interest']
            )
            enhanced_data['positive_commitment'] = cls.convert_commitment_level(
                enhanced_data['prior_course_interest']
            )
        
        return enhanced_data


# Global instance for easy import
positive_language = PositiveLanguageConverter()










