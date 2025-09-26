from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from typing import Dict, Any, Optional, List
import time
from app.models.schemas import LeadInput, KnowledgeDocument
from app.services.retrieval import retrieval
from app.services.performance_monitor import performance_monitor
from app.services.circuit_breaker import openai_circuit_breaker
from app.utils.safety import safety_filter
from app.utils.logging import get_logger
from app.config import settings


logger = get_logger(__name__)


class RAGEmailService:
    """Service for generating RAG-powered personalized emails."""
    
    def __init__(self):
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM with fallback options."""
        try:
            if settings.openai_api_key:
                self.llm = OpenAI(
                    openai_api_key=settings.openai_api_key,
                    temperature=0.3,
                    max_tokens=300
                )
                logger.info("Initialized OpenAI LLM for RAG emails")
            else:
                logger.warning("No OpenAI API key, RAG emails will use fallback")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
    
    def _get_fallback_email(self, lead_data: LeadInput, lead_type: str) -> Dict[str, str]:
        """Generate fallback email when LLM is not available."""
        if lead_type.lower() == 'hot':
            subject = f"🔥 Perfect Match for {lead_data.role} Role"
            content = f"""Hi there! 👋

I noticed your strong interest in our {lead_data.campaign} program and your role as a {lead_data.role}. Based on your engagement level, I believe we have the perfect solution for you.

I'd love to discuss how our program can help you achieve your goals.

Are you available for a quick 15-minute call this week?

Best regards,
Lead HeatScore Team"""
        
        elif lead_type.lower() == 'warm':
            subject = f"🎓 Free Webinar Invitation - {lead_data.campaign} Masterclass"
            content = f"""Hi {lead_data.name}! 👋

I noticed your interest in our {lead_data.campaign} program. As a {lead_data.role}, this could be a great opportunity for your career development.

🎓 FREE WEBINAR INVITATION:
Join our FREE {lead_data.campaign} Masterclass this weekend and unlock your learning journey 🚀

✨ What You'll Learn:
• Industry-recognized certification
• Career advancement support
• Practical hands-on projects
• Expert mentorship
• Real-world case studies
• Professional portfolio building

💬 Want to learn more?

Reply 'WEBINAR' to join or 'INFO' for details!

Best regards,
LeadHeatScore Team"""
        
        else:  # cold
            subject = "🌟 Free Resources & Success Stories"
            content = f"""Hi {lead_data.name}! 👋

I hope you're doing well. I wanted to share some valuable resources that might be relevant for your role as a {lead_data.role}.

🌟 SUCCESS STORY:
Did you know 10,000+ students started with our free coding resources? Start your journey today ✨

Free Resources Available:
• Industry insights and trends
• Best practices and tips
• Educational content

Stay Updated:
📧 Subscribe to our newsletter
📱 Follow us for daily insights
💡 Access free resources

No pressure - just valuable content when you're ready!

Best regards,
Content Team

Reply with 'NEWS' to subscribe"""
        
        return {
            "subject": subject,
            "content": content,
            "type": "fallback"
        }
    
    async def generate_personalized_email(
        self, 
        lead_data: LeadInput, 
        lead_type: str,
        context_docs: Optional[List[KnowledgeDocument]] = None
    ) -> Dict[str, str]:
        """Generate RAG-powered personalized email."""
        trace_id = performance_monitor.start_trace("rag_email_generation")
        
        try:
            # If no LLM available, use fallback
            if not self.llm:
                performance_monitor.finish_trace(trace_id, status_code=200)
                return self._get_fallback_email(lead_data, lead_type)
            
            # Retrieve relevant context
            retrieval_start = time.time()
            if not context_docs:
                query = f"{lead_data.role} {lead_data.campaign} {lead_data.prior_course_interest}"
                search_results = await retrieval.hybrid_search(query, limit=3)
                context_docs = [result.document for result in search_results]
            
            retrieval_duration = (time.time() - retrieval_start) * 1000
            performance_monitor.record_step(trace_id, "retrieval", retrieval_duration)
            
            # Prepare context
            context_text = ""
            if context_docs:
                context_text = "\n\n".join([
                    f"Title: {doc.title}\nContent: {doc.content[:300]}..."
                    for doc in context_docs
                ])
            
            # Create email generation prompt
            prompt_template = PromptTemplate(
                input_variables=["lead_data", "lead_type", "context"],
                template="""
You are a sales expert writing personalized emails for educational platform leads.

Lead Information:
- Name: {lead_data[name]}
- Role: {lead_data[role]}
- Company: {lead_data[company]}
- Region: {lead_data[region]}
- Campaign: {lead_data[campaign]}
- Page Views: {lead_data[page_views]}
- Recency: {lead_data[recency_days]} days
- Last Touch: {lead_data[last_touch]}
- Prior Interest: {lead_data[prior_course_interest]}
- Lead Type: {lead_type}

Context from Knowledge Base:
{context}

Generate a personalized email for WARM leads in the following JSON format:
{{
    "subject": "Free Webinar Invitation - [Campaign] Masterclass",
    "content": "Personalized email content with proper formatting:\n\nHi [Name],\n\nI noticed your interest in our [Campaign] program. As a [Role], this could be a great opportunity for your career development.\n\n🎓 FREE WEBINAR INVITATION:\nJoin our FREE [Campaign] Masterclass this weekend and unlock your learning journey 🚀\n\n✨ What You'll Learn:\n• Industry-recognized certification\n• Career advancement support\n• Practical hands-on projects\n• Expert mentorship\n• Real-world case studies\n• Professional portfolio building\n\n💬 Want to learn more?\n\nReply 'WEBINAR' to join or 'INFO' for details!\n\nBest regards,\nGUVI Team"
}}

Guidelines for WARM leads:
- Focus on FREE WEBINAR invitation (low-commitment, high-value event)
- Build trust and move them closer to purchase
- Use nurturing tone, not pushy
- Emphasize learning journey and career development
- Include clear webinar CTA
- Reference their role and interests
- Keep it engaging but not urgent
- Use educational platform branding (GUVI)
- Format with proper line breaks and structure
- Use emojis effectively (2-3 max)
- Make it feel personal and encouraging
"""
            )
            
            # Generate email with circuit breaker protection
            generation_start = time.time()
            prompt = prompt_template.format(
                lead_data=lead_data.dict(),
                lead_type=lead_type,
                context=context_text
            )
            
            # Safety check for prompt injection
            if safety_filter.detect_prompt_injection(prompt):
                logger.warning("Potential prompt injection detected, using fallback")
                performance_monitor.finish_trace(trace_id, status_code=200)
                return self._get_fallback_email(lead_data, lead_type)
            
            # Use circuit breaker for LLM call
            response = openai_circuit_breaker.call(self.llm, prompt)
            
            generation_duration = (time.time() - generation_start) * 1000
            performance_monitor.record_step(trace_id, "generation", generation_duration)
            
            # Parse response
            try:
                import json
                import re
                
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    
                    # Safety check for content
                    subject = result.get('subject', '')
                    content = result.get('content', '')
                    
                    subject_safety = safety_filter.validate_content(subject)
                    content_safety = safety_filter.validate_content(content)
                    
                    if not subject_safety['is_safe'] or not content_safety['is_safe']:
                        logger.warning("Unsafe content detected, using fallback")
                        return self._get_fallback_email(lead_data, lead_type)
                    
                    logger.info("Generated RAG email successfully", trace_id=trace_id)
                    performance_monitor.finish_trace(trace_id, status_code=200)
                    
                    return {
                        "subject": subject,
                        "content": content,
                        "type": "rag"
                    }
                else:
                    raise ValueError("No JSON found in response")
                    
            except Exception as e:
                logger.error(f"Error parsing RAG email response: {e}")
                return self._get_fallback_email(lead_data, lead_type)
            
        except Exception as e:
            logger.error(
                "Error generating RAG email",
                trace_id=trace_id,
                error=str(e),
                error_type=type(e).__name__
            )
            performance_monitor.finish_trace(trace_id, status_code=500, error_type=type(e).__name__)
            return self._get_fallback_email(lead_data, lead_type)


# Global service instance
rag_email_service = RAGEmailService()
