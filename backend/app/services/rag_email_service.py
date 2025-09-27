from langchain_openai import OpenAI
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
        """Initialize OpenAI GPT-4o-mini for high-quality email generation."""
        try:
            if settings.openai_api_key:
                # Initialize OpenAI GPT-4o-mini
                self.llm = OpenAI(
                    openai_api_key=settings.openai_api_key,
                    model=settings.llm_model,  # GPT-4o-mini
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                    request_timeout=settings.rag_email_timeout,
                    max_retries=3,
                    retry_min_seconds=2,
                    retry_max_seconds=8
                )
                self.llm_type = settings.llm_model
                logger.info(f"Initialized OpenAI {settings.llm_model} for email generation")
            else:
                # Fallback to template-based approach
                self.llm = None
                self.llm_type = "template"
                logger.info("OpenAI API key not found - using template-based email generation")
                
        except Exception as e:
            logger.error(f"Error initializing OpenAI LLM: {e}")
            self.llm_type = "fallback"
    
    def switch_to_demo_mode(self):
        """Switch to GPT-4o for ultra-high-quality demo emails."""
        try:
            if settings.openai_api_key:
                self.llm = OpenAI(
                    openai_api_key=settings.openai_api_key,
                    model="gpt-4o",  # Ultra-high-quality model for demos
                    temperature=0.2,  # Lower temperature for consistency
                    max_tokens=1000,  # More tokens for comprehensive emails
                    request_timeout=90,  # Longer timeout for GPT-4o
                    max_retries=3,
                    retry_min_seconds=3,
                    retry_max_seconds=10
                )
                self.llm_type = "gpt-4o"
                logger.info("Switched to GPT-4o for demo mode")
            else:
                logger.warning("No OpenAI API key for demo mode")
        except Exception as e:
            logger.error(f"Error switching to demo mode: {e}")
    
    def switch_to_production_mode(self):
        """Switch to GPT-4o-mini for optimal production performance."""
        try:
            if settings.openai_api_key:
                self.llm = OpenAI(
                    openai_api_key=settings.openai_api_key,
                    model=settings.llm_model,  # GPT-4o-mini (optimal balance)
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                    request_timeout=settings.rag_email_timeout,
                    max_retries=3,
                    retry_min_seconds=2,
                    retry_max_seconds=8
                )
                self.llm_type = settings.llm_model
                logger.info(f"Switched to {settings.llm_model} for production mode")
            else:
                logger.warning("No OpenAI API key for production mode")
        except Exception as e:
            logger.error(f"Error switching to production mode: {e}")
    
    def _generate_template_email(self, lead_data: LeadInput, lead_type: str, context_text: str) -> str:
        """Generate personalized email using smart templates with RAG context."""
        try:
            # Extract key information from context
            context_keywords = []
            if context_text:
                # Simple keyword extraction from context
                keywords = ["certification", "career", "skills", "industry", "program", "course", "learning"]
                for keyword in keywords:
                    if keyword.lower() in context_text.lower():
                        context_keywords.append(keyword)
            
            # Base templates with RAG personalization
            templates = {
                "hot": {
                    "subject": f"🚀 Exclusive {lead_data.campaign} Opportunity - Limited Time!",
                    "content": f"""Hi {lead_data.name},

I noticed your strong interest in our {lead_data.campaign} program! As a {lead_data.role}, this could be a game-changer for your career.

🎯 EXCLUSIVE OFFER:
Get 50% OFF our {lead_data.campaign} certification program - Limited time only!

✨ What You'll Get:
• Industry-recognized certification
• Career advancement support
• Practical hands-on projects
• Expert mentorship
• Real-world case studies
{f"• Personalized {', '.join(context_keywords[:2])} focus" if context_keywords else ""}

💬 Ready to transform your career?

Reply 'YES' to claim your spot or 'INFO' for details!

Best regards,
GUVI Team"""
                },
                "warm": {
                    "subject": f"Free Webinar Invitation - {lead_data.campaign} Masterclass",
                    "content": f"""Hi {lead_data.name},

I noticed you're exploring our {lead_data.campaign} program. As a {lead_data.role}, this could be a great opportunity for your career development.

🎓 FREE WEBINAR INVITATION:
Join our FREE {lead_data.campaign} Masterclass this weekend and unlock your learning journey 🚀

✨ What You'll Learn:
• Industry-recognized certification
• Career advancement support
• Practical hands-on projects
• Expert mentorship
• Real-world case studies
• Professional portfolio building
{f"• Focus on {', '.join(context_keywords[:2])}" if context_keywords else ""}

💬 Want to learn more?

Reply 'WEBINAR' to join or 'INFO' for details!

Best regards,
GUVI Team"""
                },
                "cold": {
                    "subject": f"Discover {lead_data.campaign} - Free Learning Resources",
                    "content": f"""Hi {lead_data.name},

I hope you're doing well! I wanted to share some valuable resources about {lead_data.campaign} that might interest you as a {lead_data.role}.

📚 FREE LEARNING RESOURCES:
• Industry insights and trends
• Career guidance materials
• Free course previews
• Expert interviews
{f"• {lead_data.campaign} specific content" if context_keywords else ""}

🎯 Why This Matters:
The {lead_data.campaign} field is growing rapidly, and staying updated is crucial for career success.

💬 Interested in learning more?

Reply 'RESOURCES' to get access or 'INFO' for details!

Best regards,
GUVI Team"""
                }
            }
            
            template = templates.get(lead_type, templates["warm"])
            
            # Return as JSON string to match expected format
            import json
            return json.dumps(template)
            
        except Exception as e:
            logger.error(f"Error generating template email: {e}")
            return json.dumps(self._get_fallback_email(lead_data, lead_type))

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
        context_docs: Optional[List[KnowledgeDocument]] = None,
        force_template: bool = False
    ) -> Dict[str, str]:
        """Generate RAG-powered personalized email."""
        trace_id = performance_monitor.start_trace("rag_email_generation")
        
        try:
            # Use OpenAI LLM for premium email generation (unless force_template is True)
            if force_template:
                logger.info("Force template mode enabled - using template emails")
                performance_monitor.finish_trace(trace_id, status_code=200)
                return self._get_fallback_email(lead_data, lead_type)
            
            # If no LLM available, use fallback
            if not self.llm:
                logger.warning("No OpenAI LLM available, using fallback email")
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
- Engagement Level: {lead_data[prior_course_interest]}
- Lead Type: {lead_type}

Context from Knowledge Base:
{context}

Generate a personalized email for WARM leads in the following JSON format:
{{
    "subject": "Free Webinar Invitation - [Campaign] Masterclass",
    "content": "Personalized email content with proper formatting:\n\nHi [Name],\n\nI noticed you're exploring our [Campaign] program. As a [Role], this could be a great opportunity for your career development.\n\n🎓 FREE WEBINAR INVITATION:\nJoin our FREE [Campaign] Masterclass this weekend and unlock your learning journey 🚀\n\n✨ What You'll Learn:\n• Industry-recognized certification\n• Career advancement support\n• Practical hands-on projects\n• Expert mentorship\n• Real-world case studies\n• Professional portfolio building\n\n💬 Want to learn more?\n\nReply 'WEBINAR' to join or 'INFO' for details!\n\nBest regards,\nGUVI Team"
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
            
            # Use OpenAI LLM for premium email generation
            try:
                # Generate email using OpenAI with circuit breaker protection
                response = self.llm.invoke(prompt)
                logger.info(f"Generated premium email using {self.llm_type}")
                
            except Exception as e:
                logger.error(f"OpenAI email generation failed: {e}")
                performance_monitor.finish_trace(trace_id, status_code=200)
                return self._get_fallback_email(lead_data, lead_type)
            
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
