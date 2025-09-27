from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from typing import List, Dict, Any, Optional
import json
import re
import time
import uuid
from app.models.schemas import LeadInput, LeadScore, Recommendation, Channel, KnowledgeDocument, HeatScore
from app.services.retrieval import retrieval
from app.services.performance_monitor import performance_monitor
from app.services.circuit_breaker import openai_circuit_breaker
from app.services.policy_service import policy_service
from app.services.persona_service import persona_service
from app.services.telegram_service import telegram_service
from app.utils.safety import safety_filter
from app.utils.logging import get_logger
from app.config import settings


logger = get_logger(__name__)


class RecommendationParser(BaseOutputParser):
    """Parse LLM output into structured recommendation."""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # Fallback: extract information using regex
            result = {
                'channel': 'email',
                'message': '',
                'rationale': '',
                'confidence': 0.8
            }
            
            # Extract channel
            channel_match = re.search(r'channel["\']?\s*:\s*["\']?(\w+)', text, re.IGNORECASE)
            if channel_match:
                result['channel'] = channel_match.group(1).lower()
            
            # Extract message
            message_match = re.search(r'message["\']?\s*:\s*["\']([^"\']+)', text, re.IGNORECASE)
            if message_match:
                result['message'] = message_match.group(1)
            
            # Extract rationale
            rationale_match = re.search(r'rationale["\']?\s*:\s*["\']([^"\']+)', text, re.IGNORECASE)
            if rationale_match:
                result['rationale'] = rationale_match.group(1)
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing recommendation: {e}")
            return {
                'channel': 'email',
                'message': 'Thank you for your interest. We will be in touch soon.',
                'rationale': 'Standard follow-up message',
                'confidence': 0.5
            }


class NextActionAgent:
    """AI agent for generating next action recommendations."""
    
    def __init__(self):
        self.llm = None
        self.parser = RecommendationParser()
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM."""
        try:
            if settings.openai_api_key:
                self.llm = OpenAI(
                    openai_api_key=settings.openai_api_key,
                    temperature=0.3,
                    max_tokens=500
                )
                logger.info("Initialized OpenAI LLM")
            else:
                logger.warning("OpenAI API key not provided, using fallback recommendations")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
    
    def _get_channel_recommendation(self, lead_data: LeadInput, heat_score: HeatScore) -> Channel:
        """Determine recommended channel based on heat score using policy document."""
        # Use policy service to get optimal channel
        optimal_channel = policy_service.get_optimal_channel(heat_score)
        
        # Map to Channel enum
        channel_mapping = {
            "telegram": Channel.TELEGRAM,
            "rag_email": Channel.EMAIL,  # RAG emails are still emails
            "newsletter": Channel.NEWSLETTER,
            "email": Channel.EMAIL
        }
        
        return channel_mapping.get(optimal_channel, Channel.EMAIL)
    
    def _get_fallback_recommendation(
        self, 
        lead_data: LeadInput, 
        heat_score: HeatScore
    ) -> Recommendation:
        """Generate fallback recommendation without LLM."""
        channel = self._get_channel_recommendation(lead_data, heat_score)
        
        # Generate standardized messages based on heat score
        if heat_score == HeatScore.HOT:
            # Hot leads: Telegram message
            message = f"""👋 Hi {lead_data.name}!

🎯 I noticed your interest in our {lead_data.campaign} program.

📊 Based on your profile as a {lead_data.role}, I believe this could be perfect for your career growth.

✨ What You Get:
• 🎓 Complete {lead_data.campaign} certification
• 👨‍💼 Career guidance & placement support  
• 💡 Practical hands-on projects
• 🤝 Expert mentorship program
• 📊 Real-world case studies
• 🏆 Industry-recognized certificate

📞 Contact Us:
• 📱 Phone: +1-800-LEADHEAT
• 📧 Email: support@leadheatscore.com
• 🌐 Website: www.leadheatscore.com

🚀 Ready to advance your career? 

Reply 'YES' to get started or 'INFO' for more details!

Best regards,
LeadHeatScore Team"""
            rationale = "Hot lead - Telegram outreach with immediate call-to-action"
        elif heat_score == HeatScore.WARM:
            # Warm leads: RAG Email message
            message = f"Hi {lead_data.name}! Thank you for your interest in our {lead_data.campaign} program. As a {lead_data.role}, I believe you'd find our resources particularly valuable. I'd like to share some additional information that might be relevant to your goals. Would you be interested in learning more? 📚"
            rationale = "Warm lead - RAG email for personalized nurturing"
        else:
            # Cold leads: Newsletter content
            message = f"Hi {lead_data.name}! I hope you're doing well. I wanted to share some valuable resources that might be relevant for your role as a {lead_data.role}. Stay updated with our newsletter for regular insights and valuable content."
            rationale = "Cold lead - Newsletter content for low-touch nurturing"
        
        return Recommendation(
            lead_id=str(hash(str(lead_data.dict()))),
            recommended_channel=channel,
            message_content=message,
            rationale=rationale,
            confidence=0.8
        )
    
    async def generate_recommendation(
        self, 
        lead_data: LeadInput, 
        lead_score: LeadScore,
        context_docs: Optional[List[KnowledgeDocument]] = None
    ) -> Recommendation:
        """Generate next action recommendation for a lead."""
        # Start performance trace
        trace_id = performance_monitor.start_trace("rag_recommendation")
        
        try:
            # Safety check
            sanitized_data = safety_filter.sanitize_lead_data(lead_data.dict())
            
            # If no LLM available, use fallback
            if not self.llm:
                performance_monitor.finish_trace(trace_id, status_code=200)
                return self._get_fallback_recommendation(lead_data, lead_score.heat_score)
            
            # Retrieve relevant context with timing and cross-encoder reranking
            retrieval_start = time.time()
            if not context_docs:
                query = f"{lead_data.role} {lead_data.campaign} {lead_data.prior_course_interest}"
                # Use hybrid search with cross-encoder reranking for better context
                search_results = await retrieval.hybrid_search(query, limit=5)  # Get more candidates for reranking
                
                # Apply cross-encoder reranking specifically for NextAction agent
                from app.services.cross_encoder_reranker import reranker
                if len(search_results) > 3:
                    reranked_results = await reranker.rerank_results(
                        query=query,
                        results=search_results,
                        top_k=3,  # Top 3 most relevant for context
                        alpha=0.4,  # Higher weight for cross-encoder in NextAction
                        use_async=True
                    )
                    context_docs = [result.document for result in reranked_results]
                    logger.info(f"NextAction agent: Reranked {len(search_results)} results to top 3 using cross-encoder")
                else:
                    context_docs = [result.document for result in search_results]
            
            retrieval_duration = (time.time() - retrieval_start) * 1000
            performance_monitor.record_step(trace_id, "retrieval", retrieval_duration)
            
            # Prepare context
            context_text = ""
            citations = []
            if context_docs:
                context_text = "\n\n".join([
                    f"Title: {doc.title}\nContent: {doc.content[:500]}..."
                    for doc in context_docs
                ])
                citations = [doc.title for doc in context_docs]
            
            # Create prompt
            prompt_template = PromptTemplate(
                input_variables=["lead_data", "heat_score", "context"],
                template="""
You are a sales expert providing next action recommendations for leads.

Lead Information:
- Source: {lead_data[source]}
- Role: {lead_data[role]}
- Region: {lead_data[region]}
- Campaign: {lead_data[campaign]}
- Page Views: {lead_data[page_views]}
- Recency: {lead_data[recency_days]} days
- Last Touch: {lead_data[last_touch]}
- Prior Interest: {lead_data[prior_course_interest]}
- Heat Score: {heat_score}

Context from Knowledge Base:
{context}

Based on this information, provide a recommendation in the following JSON format:
{{
    "channel": "email|phone|linkedin|sms|social",
    "message": "Personalized message content (2-3 sentences max)",
    "rationale": "Brief explanation of why this approach is recommended",
    "confidence": 0.8
}}

Guidelines:
- Choose the most appropriate channel based on heat score and lead characteristics
- Write a personalized, professional message
- Keep messages concise and actionable
- Consider the lead's role and interests
- Use the context to inform your recommendation
"""
            )
            
            # Generate recommendation with circuit breaker protection
            generation_start = time.time()
            prompt = prompt_template.format(
                lead_data=sanitized_data,
                heat_score=lead_score.heat_score.value,
                context=context_text
            )
            
            # Safety check for prompt injection
            if safety_filter.detect_prompt_injection(prompt):
                logger.warning("Potential prompt injection detected, using fallback")
                performance_monitor.finish_trace(trace_id, status_code=200)
                return self._get_fallback_recommendation(lead_data, lead_score.heat_score)
            
            # Use circuit breaker for LLM call
            response = openai_circuit_breaker.call(self.llm, prompt)
            
            generation_duration = (time.time() - generation_start) * 1000
            performance_monitor.record_step(trace_id, "generation", generation_duration)
            
            # Parse response
            parsed_result = self.parser.parse(response)
            
            # Validate channel
            try:
                channel = Channel(parsed_result['channel'])
            except ValueError:
                channel = self._get_channel_recommendation(lead_data, lead_score.heat_score)
            
            # Safety check for message content
            message = parsed_result.get('message', '')
            safety_check = safety_filter.validate_content(message)
            if not safety_check['is_safe']:
                message = safety_check['redacted_content']
                logger.warning("Unsafe content detected in message, redacted")
            
            recommendation = Recommendation(
                lead_id=str(hash(str(lead_data.dict()))),
                recommended_channel=channel,
                message_content=message,
                rationale=parsed_result.get('rationale', 'AI-generated recommendation'),
                citations=citations,
                confidence=parsed_result.get('confidence', 0.8)
            )
            
            logger.info(
                "Generated recommendation for lead",
                trace_id=trace_id,
                lead_id=recommendation.lead_id,
                channel=channel.value,
                confidence=recommendation.confidence
            )
            
            performance_monitor.finish_trace(trace_id, status_code=200)
            return recommendation
            
        except Exception as e:
            logger.error(
                "Error generating recommendation",
                trace_id=trace_id,
                error=str(e),
                error_type=type(e).__name__
            )
            performance_monitor.finish_trace(trace_id, status_code=500, error_type=type(e).__name__)
            return self._get_fallback_recommendation(lead_data, lead_score.heat_score)
    
    async def craft_first_message(
        self, 
        lead_data: LeadInput, 
        lead_score: LeadScore,
        context_docs: Optional[List[KnowledgeDocument]] = None
    ) -> Dict[str, Any]:
        """Craft the actual first message using policy and persona."""
        trace_id = performance_monitor.start_trace("message_crafting")
        
        try:
            # Get policy rules
            tone_rules = policy_service.get_tone_rules(lead_score.heat_score)
            cta_rules = policy_service.get_cta_rules(lead_score.heat_score)
            channel_rules = policy_service.get_channel_rules(lead_score.heat_score)
            
            # Get persona content
            persona = persona_service.get_persona_content(lead_data.role)
            
            # Determine optimal channel
            optimal_channel = policy_service.get_optimal_channel(lead_score.heat_score)
            
            # Craft message based on channel
            if optimal_channel == "telegram":
                message_content = telegram_service.craft_telegram_message(lead_data, lead_score.heat_score)
                channel = Channel.TELEGRAM
            elif optimal_channel == "rag_email":
                # Use RAG email service for detailed personalization
                from app.services.rag_email_service import rag_email_service
                rag_email = await rag_email_service.generate_personalized_email(lead_data, lead_score.heat_score.value, context_docs)
                message_content = rag_email["content"]
                channel = Channel.EMAIL
            else:  # newsletter
                message_content = self._craft_newsletter_message(lead_data, tone_rules, cta_rules, persona)
                channel = Channel.NEWSLETTER
            
            # Create rationale
            if optimal_channel == "telegram":
                if lead_score.heat_score.value == "hot":
                    rationale = f"Hot lead detected - they've shown strong buying intent (visited pricing page, high time spent, course actions). Best to strike while interest is high with limited-time discounts and direct course offers."
                elif lead_score.heat_score.value == "warm":
                    rationale = f"Warm lead identified - they're curious but not yet fully committed. A low-commitment, high-value event like a free webinar helps build trust and moves them closer to purchase."
                else:  # cold
                    rationale = f"Cold lead needs nurturing. Instead of pushing offers, showing social proof, testimonials, and free resources to spark interest and build awareness."
            else:
                rationale = f"Using {optimal_channel} channel based on {lead_score.heat_score.value} lead classification. "
                rationale += f"Tone: {tone_rules.get('tone', 'professional')}. "
                rationale += f"CTA: {cta_rules.get('primary_cta', 'learn_more')}. "
                rationale += f"Persona: {persona.get('messaging_focus', 'professional development')}"
            
            performance_monitor.finish_trace(trace_id, status_code=200)
            
            return {
                "channel": channel.value,
                "message_content": message_content,
                "rationale": rationale,
                "confidence": 0.9,
                "policy_applied": {
                    "tone_rules": tone_rules,
                    "cta_rules": cta_rules,
                    "channel_rules": channel_rules
                },
                "persona_used": {
                    "role": lead_data.role,
                    "messaging_focus": persona.get("messaging_focus"),
                    "value_propositions": persona.get("value_propositions", [])[:3]
                },
                "lead_id": str(uuid.uuid4()),
                "heat_score": lead_score.heat_score.value
            }
            
        except Exception as e:
            logger.error(f"Error crafting first message: {e}")
            performance_monitor.finish_trace(trace_id, status_code=500, error_type=type(e).__name__)
            
            # Fallback to simple recommendation
            return self._get_fallback_recommendation(lead_data, lead_score.heat_score)
    
    def _craft_newsletter_message(self, lead_data: LeadInput, tone_rules: Dict, cta_rules: Dict, persona: Dict) -> str:
        """Craft newsletter-style message for cold leads."""
        messaging_focus = persona.get("messaging_focus", "Focus on professional development and growth")
        value_props = persona.get("value_propositions", [])
        
        message = f"""Hi {lead_data.name}! 👋

I hope you're doing well. I wanted to share some valuable resources that might be relevant for your role as a {lead_data.role}.

{messaging_focus}

🌟 SUCCESS STORY:
Did you know 10,000+ students started with our free coding resources? Start your journey today ✨

Free Resources Available:
• {value_props[0] if value_props else 'Industry insights and trends'}
• {value_props[1] if len(value_props) > 1 else 'Best practices and tips'}
• {value_props[2] if len(value_props) > 2 else 'Educational content'}

Stay Updated:
📧 Subscribe to our newsletter
📱 Follow us for daily insights
💡 Access free resources

No pressure - just valuable content when you're ready!

Best regards,
Content Team

Reply with 'NEWS' to subscribe"""
        
        return message
    
    async def batch_generate_recommendations(
        self, 
        leads_data: List[Dict[str, Any]]
    ) -> List[Recommendation]:
        """Generate recommendations for multiple leads."""
        recommendations = []
        
        for lead_data in leads_data:
            try:
                # This would typically include lead_score from classification
                # For now, we'll use a simplified approach
                lead_input = LeadInput(**lead_data)
                
                # Generate recommendation
                recommendation = await self.generate_recommendation(lead_input, None)
                recommendations.append(recommendation)
                
            except Exception as e:
                logger.error(f"Error processing lead: {e}")
                continue
        
        return recommendations


# Global agent instance
next_action_agent = NextActionAgent()
