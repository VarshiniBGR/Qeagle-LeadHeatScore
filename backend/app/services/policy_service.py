from typing import Dict, Any
from app.models.schemas import HeatScore
from app.utils.logging import get_logger

logger = get_logger(__name__)


class PolicyDocumentService:
    """Service for managing engagement policy documents."""
    
    def __init__(self):
        self.policy_doc = self._load_policy_document()
    
    def _load_policy_document(self) -> Dict[str, Any]:
        """Load the policy document with tone and CTA rules."""
        return {
            "tone_rules": {
                HeatScore.HOT: {
                    "tone": "urgent, personal, direct",
                    "urgency_level": "high",
                    "personalization": "high",
                    "formality": "casual-professional"
                },
                HeatScore.WARM: {
                    "tone": "nurturing, informative, helpful",
                    "urgency_level": "medium", 
                    "personalization": "medium",
                    "formality": "professional"
                },
                HeatScore.COLD: {
                    "tone": "educational, low-pressure, value-focused",
                    "urgency_level": "low",
                    "personalization": "low",
                    "formality": "formal"
                }
            },
            "cta_rules": {
                HeatScore.HOT: {
                    "primary_cta": "schedule_call",
                    "secondary_cta": "learn_more",
                    "urgency": "immediate",
                    "timeframe": "this_week"
                },
                HeatScore.WARM: {
                    "primary_cta": "learn_more",
                    "secondary_cta": "subscribe_newsletter",
                    "urgency": "moderate",
                    "timeframe": "next_week"
                },
                HeatScore.COLD: {
                    "primary_cta": "subscribe_newsletter",
                    "secondary_cta": "follow_social",
                    "urgency": "low",
                    "timeframe": "when_ready"
                }
            },
            "channel_rules": {
                HeatScore.HOT: {
                    "primary_channel": "email",  # All leads use email
                    "secondary_channel": "email",
                    "reasoning": "unified_channel_with_personalization"
                },
                HeatScore.WARM: {
                    "primary_channel": "email",  # All leads use email
                    "secondary_channel": "email", 
                    "reasoning": "unified_channel_with_personalization"
                },
                HeatScore.COLD: {
                    "primary_channel": "email",  # All leads use email
                    "secondary_channel": "email",
                    "reasoning": "unified_channel_with_personalization"
                }
            },
            "message_length_rules": {
                HeatScore.HOT: {
                    "email": "short_urgent"
                },
                HeatScore.WARM: {
                    "email": "medium_personalized"
                },
                HeatScore.COLD: {
                    "email": "detailed_educational"
                }
            }
        }
    
    def get_tone_rules(self, heat_score: HeatScore) -> Dict[str, Any]:
        """Get tone rules for a specific heat score."""
        return self.policy_doc["tone_rules"].get(heat_score, {})
    
    def get_cta_rules(self, heat_score: HeatScore) -> Dict[str, Any]:
        """Get CTA rules for a specific heat score."""
        return self.policy_doc["cta_rules"].get(heat_score, {})
    
    def get_channel_rules(self, heat_score: HeatScore) -> Dict[str, Any]:
        """Get channel rules for a specific heat score."""
        return self.policy_doc["channel_rules"].get(heat_score, {})
    
    def get_message_length_rules(self, heat_score: HeatScore) -> Dict[str, Any]:
        """Get message length rules for a specific heat score."""
        return self.policy_doc["message_length_rules"].get(heat_score, {})
    
    def get_optimal_channel(self, heat_score: HeatScore) -> str:
        """Get the optimal channel for a heat score."""
        channel_rules = self.get_channel_rules(heat_score)
        return channel_rules.get("primary_channel", "email")
    
    def get_optimal_cta(self, heat_score: HeatScore) -> str:
        """Get the optimal CTA for a heat score."""
        cta_rules = self.get_cta_rules(heat_score)
        return cta_rules.get("primary_cta", "learn_more")


# Global service instance
policy_service = PolicyDocumentService()
