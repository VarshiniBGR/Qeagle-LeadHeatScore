from typing import Dict, Any, List
from app.utils.logging import get_logger

logger = get_logger(__name__)


class PersonaSnippetsService:
    """Service for managing persona-specific content snippets."""
    
    def __init__(self):
        self.persona_snippets = self._load_persona_snippets()
    
    def _load_persona_snippets(self) -> Dict[str, Dict[str, Any]]:
        """Load persona-specific content snippets."""
        return {
            "software_engineer": {
                "pain_points": [
                    "Technical challenges and complex problem-solving",
                    "Career growth and skill development",
                    "Staying updated with latest technologies",
                    "Code quality and best practices"
                ],
                "interests": [
                    "New programming languages and frameworks",
                    "System architecture and design patterns",
                    "Performance optimization",
                    "Technical certifications and training"
                ],
                "messaging_focus": "Focus on technical solutions, career advancement, and skill development opportunities",
                "value_propositions": [
                    "Advanced technical training programs",
                    "Industry-recognized certifications",
                    "Hands-on project experience",
                    "Mentorship from senior engineers"
                ],
                "communication_style": "Technical but accessible, solution-oriented, growth-focused"
            },
            "marketing_manager": {
                "pain_points": [
                    "ROI measurement and campaign performance",
                    "Team management and productivity",
                    "Budget allocation and optimization",
                    "Staying ahead of marketing trends"
                ],
                "interests": [
                    "Digital marketing trends and strategies",
                    "Analytics and data-driven decisions",
                    "Team development and leadership",
                    "Marketing automation tools"
                ],
                "messaging_focus": "Focus on results, team effectiveness, and strategic marketing insights",
                "value_propositions": [
                    "Proven marketing strategies and frameworks",
                    "Team leadership and management training",
                    "Analytics and performance optimization",
                    "Industry best practices and case studies"
                ],
                "communication_style": "Results-oriented, strategic, team-focused"
            },
            "sales_manager": {
                "pain_points": [
                    "Team performance and quota achievement",
                    "Sales process optimization",
                    "Lead generation and conversion",
                    "Team training and development"
                ],
                "interests": [
                    "Sales methodologies and frameworks",
                    "Team performance metrics",
                    "Customer relationship management",
                    "Sales technology and tools"
                ],
                "messaging_focus": "Focus on revenue growth, team performance, and sales excellence",
                "value_propositions": [
                    "Proven sales methodologies",
                    "Team performance optimization",
                    "Advanced sales training programs",
                    "Revenue growth strategies"
                ],
                "communication_style": "Results-driven, performance-focused, team-oriented"
            },
            "product_manager": {
                "pain_points": [
                    "Product strategy and roadmap planning",
                    "Cross-functional team coordination",
                    "User research and market analysis",
                    "Product performance metrics"
                ],
                "interests": [
                    "Product strategy and vision",
                    "User experience and design",
                    "Market research and analytics",
                    "Agile methodologies and processes"
                ],
                "messaging_focus": "Focus on product strategy, user value, and market success",
                "value_propositions": [
                    "Product strategy frameworks",
                    "User research methodologies",
                    "Market analysis techniques",
                    "Cross-functional collaboration tools"
                ],
                "communication_style": "Strategic, user-focused, data-driven"
            },
            "data_scientist": {
                "pain_points": [
                    "Complex data analysis and modeling",
                    "Machine learning model deployment",
                    "Data quality and preprocessing",
                    "Business impact measurement"
                ],
                "interests": [
                    "Advanced machine learning techniques",
                    "Big data technologies",
                    "Statistical modeling",
                    "Data visualization and storytelling"
                ],
                "messaging_focus": "Focus on advanced analytics, machine learning, and data-driven insights",
                "value_propositions": [
                    "Advanced ML and AI training",
                    "Big data processing techniques",
                    "Statistical modeling methods",
                    "Data visualization and communication"
                ],
                "communication_style": "Analytical, data-driven, insight-focused"
            },
            "default": {
                "pain_points": [
                    "Professional development and growth",
                    "Industry knowledge and trends",
                    "Skill enhancement and training",
                    "Career advancement opportunities"
                ],
                "interests": [
                    "Professional development",
                    "Industry insights",
                    "Skill building",
                    "Career growth"
                ],
                "messaging_focus": "Focus on professional development, skill enhancement, and career growth",
                "value_propositions": [
                    "Comprehensive training programs",
                    "Industry expertise and insights",
                    "Professional development opportunities",
                    "Career advancement support"
                ],
                "communication_style": "Professional, growth-oriented, value-focused"
            }
        }
    
    def get_persona_content(self, role: str) -> Dict[str, Any]:
        """Get persona-specific content for a role."""
        # Normalize role name
        normalized_role = role.lower().replace(" ", "_").replace("-", "_")
        
        # Try exact match first
        if normalized_role in self.persona_snippets:
            return self.persona_snippets[normalized_role]
        
        # Try partial matches
        for persona_key, content in self.persona_snippets.items():
            if persona_key in normalized_role or normalized_role in persona_key:
                return content
        
        # Return default persona
        logger.info(f"No specific persona found for role '{role}', using default")
        return self.persona_snippets["default"]
    
    def get_pain_points(self, role: str) -> List[str]:
        """Get pain points for a specific role."""
        persona = self.get_persona_content(role)
        return persona.get("pain_points", [])
    
    def get_interests(self, role: str) -> List[str]:
        """Get interests for a specific role."""
        persona = self.get_persona_content(role)
        return persona.get("interests", [])
    
    def get_messaging_focus(self, role: str) -> str:
        """Get messaging focus for a specific role."""
        persona = self.get_persona_content(role)
        return persona.get("messaging_focus", "Focus on professional development and growth")
    
    def get_value_propositions(self, role: str) -> List[str]:
        """Get value propositions for a specific role."""
        persona = self.get_persona_content(role)
        return persona.get("value_propositions", [])
    
    def get_communication_style(self, role: str) -> str:
        """Get communication style for a specific role."""
        persona = self.get_persona_content(role)
        return persona.get("communication_style", "Professional and growth-oriented")


# Global service instance
persona_service = PersonaSnippetsService()
