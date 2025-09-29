from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from typing import Dict, Any, Optional, List
import time
import hashlib
from app.models.schemas import LeadInput, KnowledgeDocument
from app.services.retrieval import retrieval
from app.services.performance_monitor import performance_monitor
from app.services.circuit_breaker import openai_circuit_breaker
from app.utils.safety import safety_filter
from app.utils.logging import get_logger
from app.config import settings


logger = get_logger(__name__)

# Aggressive in-memory cache for RAG responses (maximum speed)
_response_cache = {}
_cache_ttl = 7200  # 2 hours for maximum reuse
_cache_buffer_size = 1000  # Allow up to 1000 cached responses

def _get_cache_key(lead_data: Dict[str, Any], lead_type: str) -> str:
    """Generate cache key based on lead profile for aggressive caching."""
    # Create a hash of key lead attributes for caching (no time variation for maximum reuse)
    key_data = {
        'role': lead_data.get('role', ''),
        'campaign': lead_data.get('campaign', ''),
        'heat_score': lead_type
        # Removed time variation and simplified key for maximum cache hits
    }
    key_string = str(sorted(key_data.items()))
    return hashlib.md5(key_string.encode()).hexdigest()

def _get_cached_response(cache_key: str) -> Optional[Dict[str, str]]:
    """Get cached response if still valid."""
    if cache_key in _response_cache:
        cached_data = _response_cache[cache_key]
        if time.time() - cached_data['timestamp'] < _cache_ttl:
            logger.info(f"Cache hit for key: {cache_key[:8]}...")
            return cached_data['response']
        else:
            # Remove expired cache entry
            del _response_cache[cache_key]
    return None

def _cache_response(cache_key: str, response: Dict[str, str]) -> None:
    """Cache the response."""
    _response_cache[cache_key] = {
        'response': response,
        'timestamp': time.time()
    }
    logger.info(f"Cached response for key: {cache_key[:8]}...")


class RAGEmailService:
    """Service for generating RAG-powered personalized emails."""
    
    def __init__(self):
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize OpenAI GPT-4o-mini for email generation."""
        try:
            # Initialize OpenAI GPT-4o-mini
            if settings.openai_api_key:
                self.llm = OpenAI(
                    openai_api_key=settings.openai_api_key,
                    model=settings.llm_model,
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                    request_timeout=settings.rag_email_timeout,
                    max_retries=5  # More retries for better RAG success
                )
                self.llm_type = settings.llm_model
                logger.info(f"Initialized OpenAI {settings.llm_model} for email generation")
            else:
                self.llm = None
                self.llm_type = "template"
                logger.info("No OpenAI API key - using template-based email generation")
                
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            self.llm_type = "fallback"
    
    def switch_to_demo_mode(self):
        """Switch to GPT-4o for ultra-high-quality demo emails."""
        try:
            if settings.openai_api_key:
                self.llm = OpenAI(
                    openai_api_key=settings.openai_api_key,
                    model="gpt-4o",  # Ultra-high-quality model for demos
                    temperature=0.2,  # Lower temperature for consistency
                    max_tokens=200,  # Ultra-compact demo emails
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
                    "subject": f"* Exclusive {lead_data.campaign} Opportunity - Limited Time!",
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
            # Check cache first (unless force_template is True)
            if not force_template:
                cache_key = _get_cache_key(lead_data.__dict__, lead_type)
                cached_response = _get_cached_response(cache_key)
                if cached_response:
                    logger.info(f"Cache hit! RAG response returned instantly for {lead_type} lead")
                    performance_monitor.finish_trace(trace_id, status_code=200)
                    return cached_response
            
            # Force RAG mode unless explicitly disabled (worst case scenarios only)
            if force_template:
                logger.info("Force template mode enabled - using template emails only due to critical failure")
                performance_monitor.finish_trace(trace_id, status_code=200)
                return self._get_fallback_email(lead_data, lead_type)
            
            # RAG-FIRST: Always prioritize RAG generation over templates
            logger.info(f"RAG-FIRST mode: Generating RAG email for {lead_type} lead")
            
            # If no LLM available, use fallback
            if not self.llm:
                logger.warning("No OpenAI LLM available, using fallback email")
                performance_monitor.finish_trace(trace_id, status_code=200)
                return self._get_fallback_email(lead_data, lead_type)
            
            # Retrieve relevant context
            retrieval_start = time.time()
            if not context_docs:
                query = f"{lead_data.role} {lead_data.campaign}"
                search_results = await retrieval.fast_search(query, limit=1)  # Fastest possible search
                context_docs = [result.document for result in search_results]
            
            retrieval_duration = (time.time() - retrieval_start) * 1000
            performance_monitor.record_step(trace_id, "retrieval", retrieval_duration)
            
            # Ultra-minimal context for maximum token savings
            context_text = ""
            if context_docs:
                context_text = context_docs[0].title + ":" + context_docs[0].content[:50] + "..."
            
            # Create sales-focused email generation prompt based on lead type
            if lead_type == "hot":
                prompt_template = self._create_hot_lead_prompt_template()
            elif lead_type == "warm":
                prompt_template = self._create_warm_lead_prompt_template()
            else:  # cold
                prompt_template = self._create_cold_lead_prompt_template()
            
            # Generate email with circuit breaker protection
            generation_start = time.time()
            # Simplified formatting for token savings (no context or lead_type)
            prompt = prompt_template.format(
                lead_data=lead_data.dict()
            )
            
            # Safety check for prompt injection
            if safety_filter.detect_prompt_injection(prompt):
                logger.warning("Potential prompt injection detected, using fallback")
                performance_monitor.finish_trace(trace_id, status_code=200)
                return self._get_fallback_email(lead_data, lead_type)
            
            # Use LLM for email generation with optimistic settings for RAG priority
            try:
                import asyncio
                
                # Use moderate temperature for better RAG quality
                original_temp = self.llm.temperature
                self.llm.temperature = 0.2  # Slightly higher for better creativity
                

                response = await asyncio.wait_for(
                    asyncio.to_thread(self.llm.invoke, prompt),
                    timeout=8.0  # More time for RAG success
                )
                
                # Restore original temperature
                self.llm.temperature = original_temp
                
                logger.info(f"Generated email using {self.llm_type}")
                
            except Exception as e:
                logger.error(f"Email generation failed: {e}")
                performance_monitor.finish_trace(trace_id, status_code=200)
                return self._get_fallback_email(lead_data, lead_type)
            
            generation_duration = (time.time() - generation_start) * 1000
            performance_monitor.record_step(trace_id, "generation", generation_duration)
            
            # Parse response
            try:
                import json
                import re
                
                # Try to extract JSON from response (improved pattern)
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    result = json.loads(json_str)
                    
                    # Safety check for content
                    subject = result.get('subject', '')
                    content = result.get('content', '')
                    

                    # Relaxed validation for RAG priority (only reject very poor content)
                    if len(content) < 30:
                        logger.warning(f"Generated content too short ({len(content)} chars), using fallback")
                        return self._get_fallback_email(lead_data, lead_type)
                    
                    # Minimal content checks for RAG priority
                    if content.endswith('...') and len(content) < 50:
                        logger.warning("Generated content appears incomplete, using fallback")
                        return self._get_fallback_email(lead_data, lead_type)
                    
                    subject_safety = safety_filter.validate_content(subject)
                    content_safety = safety_filter.validate_content(content)
                    
                    if not subject_safety['is_safe'] or not content_safety['is_safe']:
                        logger.warning("Unsafe content detected, using fallback")
                        return self._get_fallback_email(lead_data, lead_type)
                    
                    logger.info("Generated RAG email successfully", trace_id=trace_id)
                    performance_monitor.finish_trace(trace_id, status_code=200)
                    
                    response_data = {
                        "subject": subject,
                        "content": content,
                        "type": "rag"
                    }
                    
                    # Cache the response
                    if not force_template:
                        cache_key = _get_cache_key(lead_data.__dict__, lead_type)
                        _cache_response(cache_key, response_data)
                    
                    return response_data
                else:
                    # If no JSON found, try to parse as plain text
                    logger.warning("No JSON found in response, attempting plain text parsing")
                    
                    # Extract subject and content from plain text
                    lines = response.strip().split('\n')
                    subject = ""
                    content = ""
                    
                    # Look for subject line
                    for line in lines:
                        if line.lower().startswith('subject:'):
                            subject = line.split(':', 1)[1].strip()
                            break
                    
                    # If no subject found, use first line
                    if not subject and lines:
                        subject = lines[0].strip()
                    
                    # Content is everything else
                    content = '\n'.join(lines[1:]) if len(lines) > 1 else response
                    
                    # Clean up content
                    content = content.strip()
                    
                    # If we only got a single line or very short content, use fallback
                    if not content or len(content) < 50:
                        logger.warning("Insufficient content generated, using fallback email")
                        return self._get_fallback_email(lead_data, lead_type)
                    
                    if subject and content:
                        logger.info("Parsed plain text email successfully", trace_id=trace_id)
                        performance_monitor.finish_trace(trace_id, status_code=200)
                        
                        response_data = {
                            "subject": subject,
                            "content": content,
                            "type": "rag"
                        }
                        
                        # Cache the response
                        if not force_template:
                            cache_key = _get_cache_key(lead_data.__dict__, lead_type)
                            _cache_response(cache_key, response_data)
                        
                        return response_data
                    else:
                        raise ValueError("Could not parse email from response")
                    
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


    def _create_hot_lead_prompt_template(self) -> PromptTemplate:
        """Create ultra-minimal HOT lead prompt."""
        return PromptTemplate(
            input_variables=["lead_data"],
            template="""HOT email for {lead_data[name]} ({lead_data[role]}) - {lead_data[page_views]} views.
JSON: {{"subject": "🔥 URGENT: Only 2 Spots Left", "content": "Hi {lead_data[name]}! Your {lead_data[page_views]} views show interest in {lead_data[campaign]}.\\n\\n🎯 48hr EXCLUSIVE:\\n• 30% OFF\\n• Only 2 spots\\n\\nReply YES!\\n\\nBest,\\nTeam"}}"""
        )
    
    def _create_warm_lead_prompt_template(self) -> PromptTemplate:
        """Create ultra-minimal WARM lead prompt."""
        return PromptTemplate(
            input_variables=["lead_data"],
            template="""WARM email for {lead_data[name]} ({lead_data[role]}) - {lead_data[page_views]} views.
JSON: {{"subject": "🎓 Free Webinar: {lead_data[role]} AI Success", "content": "Hi {lead_data[name]}! Your {lead_data[page_views]} views show interest in {lead_data[campaign]}.\\n\\n🎓 FREE WEBINAR:\\n• AI strategies for {lead_data[role]}s\\n• Free guide ($200 value)\\n• 14-day trial\\n\\nReply WEBINAR!\\n\\nBest,\\nTeam"}}"""
        )
    
    def _create_cold_lead_prompt_template(self) -> PromptTemplate:
        """Create ultra-minimal COLD lead prompt."""
        return PromptTemplate(
            input_variables=["lead_data"],
            template="""COLD email for {lead_data[name]} ({lead_data[role]}) - {lead_data[page_views]} views.
JSON: {{"subject": "📊 Free AI Assessment for {lead_data[role]}s", "content": "Hi {lead_data[name]}! Our AI Career Assessment is free.\\n\\nAs a {lead_data[role]}:\\n• Skill evaluation\\n• Career roadmap\\n• Industry insights\\n\\nNo strings attached. Used by 50k+ pros.\\n\\nBest,\\nTeam"}}"""
        )


# Global service instance
rag_email_service = RAGEmailService()
