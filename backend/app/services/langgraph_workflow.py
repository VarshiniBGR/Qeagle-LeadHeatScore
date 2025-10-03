"""
LangGraph Workflow for Lead Processing
Implements a stateful workflow: Classify → Retrieve → Generate → Send
"""
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import asyncio
import time
import uuid
from datetime import datetime, timedelta
from app.models.schemas import LeadInput, LeadScore, KnowledgeDocument
from app.services.classifier import classifier
from app.services.retrieval import retrieval
from app.services.email_service import email_service
from app.utils.logging import get_logger
from app.config import settings

logger = get_logger(__name__)

# In-memory cache for generated emails (30-minute expiry)
email_cache = {}

def _clean_expired_emails():
    """Clean expired emails from cache."""
    current_time = datetime.now()
    expired_keys = [
        key for key, (_, timestamp) in email_cache.items()
        if current_time - timestamp > timedelta(minutes=30)
    ]
    for key in expired_keys:
        del email_cache[key]

def _get_lead_cache_key(lead_data: LeadInput) -> str:
    """Generate cache key for lead."""
    return f"{lead_data.email}_{lead_data.name}_{lead_data.role}_{lead_data.campaign}"

def store_generated_email(lead_data: LeadInput, email_content: Dict[str, Any]):
    """Store generated email in cache."""
    _clean_expired_emails()
    cache_key = _get_lead_cache_key(lead_data)
    email_cache[cache_key] = (email_content, datetime.now())
    logger.info(f"📧 Stored email in cache for key: {cache_key}")

def get_cached_email(lead_data: LeadInput) -> Optional[Dict[str, Any]]:
    """Get cached email if exists and not expired."""
    _clean_expired_emails()
    cache_key = _get_lead_cache_key(lead_data)
    if cache_key in email_cache:
        email_content, timestamp = email_cache[cache_key]
        logger.info(f"📧 Retrieved cached email for key: {cache_key}")
        return email_content
    return None

class WorkflowState(TypedDict):
    """State for the LangGraph workflow."""
    lead_data: LeadInput
    lead_score: Optional[LeadScore]
    context_docs: List[KnowledgeDocument]
    email_content: Optional[Dict[str, str]]
    email_sent: bool
    error: Optional[str]
    trace_id: str

class EmailOutput(BaseModel):
    """Output schema for email generation."""
    subject: str = Field(description="Email subject line")
    content: str = Field(description="Email body content")
    email_type: str = Field(description="Type of email (hot/warm/cold)")

class LeadProcessingWorkflow:
    """LangGraph workflow for processing leads."""
    
    def __init__(self):
        self.llm = None
        self.parser = JsonOutputParser(pydantic_object=EmailOutput)
        self._initialize_llm()
        self.graph = self._build_graph()
    
    def _initialize_llm(self):
        """Initialize the LLM for the workflow."""
        try:
            self.llm = ChatOpenAI(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=350,  # Increased to force longer detailed content while maintaining P95
                openai_api_key=settings.openai_api_key
            )
            logger.info(f"Initialized LangGraph LLM: {settings.llm_model}")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
    
    async def _initialize_models(self):
        """Pre-load all models for faster response."""
        try:
            logger.info("Pre-loading LangGraph workflow models...")
            
            # Ensure LLM is initialized
            if self.llm is None:
                self._initialize_llm()
            
            # Pre-load retrieval models if available
            try:
                if hasattr(retrieval, '_initialize_models'):
                    await retrieval._initialize_models()
                else:
                    logger.info("Initializing retrieval service...")
                    await retrieval.initialize()
            except Exception as e:
                logger.warning(f"Could not pre-load retrieval models: {e}")
            
            # Pre-load embedding models
            try:
                logger.info("Pre-loading embedding models...")
                from app.services.local_embedding_service import local_embedding_service
                if hasattr(local_embedding_service, 'initialize'):
                    await local_embedding_service.initialize()
            except Exception as e:
                logger.warning(f"Could not pre-load embedding models: {e}")
            
            logger.info("✅ All LangGraph models pre-loaded successfully")

        except Exception as e:
            logger.error(f"❌ Failed to pre-load LangGraph models: {e}")
            raise
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow - optimized for speed."""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes - skip email sending for faster response
        workflow.add_node("classify", self._classify_lead)
        workflow.add_node("retrieve", self._retrieve_context)
        workflow.add_node("generate", self._generate_email)
        workflow.add_node("error_handler", self._handle_error)
        
        # Set entry point
        workflow.set_entry_point("classify")
        
        # Simplified flow - no email sending step
        workflow.add_edge("classify", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        # Conditional edges for error handling
        workflow.add_conditional_edges(
            "classify",
            self._should_continue,
            {"continue": "retrieve", "error": "error_handler"}
        )
        workflow.add_conditional_edges(
            "retrieve",
            self._should_continue,
            {"continue": "generate", "error": "error_handler"}
        )
        workflow.add_conditional_edges(
            "generate",
            self._should_continue,
            {"continue": END, "error": "error_handler"}
        )
        workflow.add_edge("error_handler", END)
        
        return workflow.compile()
    
    async def _classify_lead(self, state: WorkflowState) -> Dict[str, Any]:
        """Classify the lead using the ML classifier."""
        try:
            logger.info(f"Classifying lead: {state['lead_data'].name}")
            lead_score = classifier.predict(state['lead_data'])
            logger.info(f"Lead classified as: {lead_score.heat_score} (confidence: {lead_score.confidence:.2f})")
            return {"lead_score": lead_score}
        except Exception as e:
            logger.error(f"Error in classification: {e}")
            return {"error": f"Classification failed: {str(e)}"}
    
    async def _retrieve_context(self, state: WorkflowState) -> Dict[str, Any]:
        """Retrieve relevant context using dynamic RAG retrieval."""
        try:
            logger.info("Retrieving dynamic context for email generation")
            lead_data = state['lead_data']
            
            # Create dynamic search query based on lead data
            search_query = self._create_dynamic_search_query(lead_data)
            logger.info(f"Dynamic search query: {search_query}")
            
            # Use actual RAG retrieval service with reduced limit for speed
            logger.info("🧠 Using REAL RAG retrieval with MongoDB Atlas vector search")
            context_docs = await retrieval.retrieve_documents(
                        query=search_query,
                        limit=3,  # Reduced from 5 to 3 for faster processing
                        filters={
                            "role": lead_data.role,
                            "campaign": lead_data.campaign,
                            "interest_level": lead_data.prior_course_interest
                        }
                    )
            logger.info(f"✅ REAL RAG: Retrieved {len(context_docs)} documents from MongoDB Atlas")
            return {"context_docs": context_docs}
            
        except Exception as e:
            logger.error(f"Error in dynamic retrieval: {e}")
            # Force retry with simpler query instead of fallback to cached docs
            try:
                # Retry with simpler query
                simple_query = f"{lead_data.role} {lead_data.campaign}"
                logger.info(f"Retrying with simpler query: {simple_query}")
                context_docs = await retrieval.retrieve_documents(
                    query=simple_query,
                    limit=3
                )
                logger.info(f"Retry successful: Retrieved {len(context_docs)} documents")
                return {"context_docs": context_docs}
            except Exception as retry_error:
                logger.error(f"Retry also failed: {retry_error}")
                # Only use cached docs as absolute last resort
                logger.warning("⚠️ Using cached docs as absolute last resort - RAG retrieval failed")
                context_docs = self._get_cached_course_docs(lead_data)
                logger.info(f"📋 Fallback: Using {len(context_docs)} cached documents")
                return {"context_docs": context_docs}
    
    def _create_dynamic_search_query(self, lead_data) -> str:
        """Create dynamic search query based on lead data."""
        query_parts = []
        
        # Add role (most important)
        if lead_data.role:
            query_parts.append(lead_data.role)
        
        # Add campaign
        if lead_data.campaign:
            query_parts.append(lead_data.campaign)
        
        # Add search keywords if available
        if hasattr(lead_data, 'search_keywords') and lead_data.search_keywords:
            # Take first 2 keywords to avoid overly complex queries
            keywords = lead_data.search_keywords.split(',')[:2]
            query_parts.extend([kw.strip() for kw in keywords if kw.strip()])
        
        # Ensure we have at least something to search for
        if not query_parts:
            query_parts = ["professional development", "career growth"]
        
        dynamic_query = " ".join(query_parts)
        logger.info(f"🔍 FINAL SEARCH QUERY: '{dynamic_query}' (from role='{lead_data.role}', campaign='{lead_data.campaign}', keywords='{getattr(lead_data, 'search_keywords', 'None')}')")
        return dynamic_query
    
    def _get_cached_course_docs(self, lead_data: LeadInput) -> List[KnowledgeDocument]:
        """Get dynamic course documents based on real lead data."""
        from app.models.schemas import KnowledgeDocument
        
        relevant_docs = []
        role = lead_data.role or "Professional"
        campaign = lead_data.campaign or "general"
        search_keywords = lead_data.search_keywords or ""
        prior_interest = lead_data.prior_course_interest or "medium"
        
        if "data" in role.lower() or "analyst" in role.lower():
            doc = KnowledgeDocument(
                id=f"data-analytics-{campaign}",
                title=f"Data Analytics & Business Intelligence Course",
                content=f"Comprehensive data analytics course designed for {role} professionals. Covers advanced Excel, Power BI, Tableau, SQL, statistical analysis, and dashboard creation.",
                category="data_analytics",
                tags=["data-analytics", "excel", "power-bi", "tableau", "sql", "statistics", role.lower(), campaign.lower()],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            relevant_docs.append(doc)
            
        elif "project" in role.lower() or "manager" in role.lower():
            doc = KnowledgeDocument(
                id=f"project-management-{campaign}",
                title=f"Project Management & Leadership Course",
                content=f"Advanced project management course tailored for {role} professionals. Covers Agile, Scrum, PMP methodologies, team leadership, stakeholder management, and project planning tools.",
                category="project_management",
                tags=["project-management", "agile", "scrum", "leadership", "pmp", role.lower(), campaign.lower()],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            relevant_docs.append(doc)
            
        elif "healthcare" in role.lower() or "medical" in role.lower():
            doc = KnowledgeDocument(
                id=f"healthcare-tech-{campaign}",
                title=f"Healthcare Technology & Digital Innovation Course",
                content=f"Specialized healthcare technology course for {role} professionals. Covers electronic health records (EHR), telemedicine, healthcare data analytics, patient management systems, and digital health innovations.",
                category="healthcare_technology",
                tags=["healthcare", "technology", "ehr", "telemedicine", "digital-health", role.lower(), campaign.lower()],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            relevant_docs.append(doc)
            
        else:
            doc = KnowledgeDocument(
                id=f"professional-development-{campaign}",
                title=f"Professional Development & Career Growth Course",
                content=f"Comprehensive professional development course designed for {role} professionals. Covers leadership skills, communication, strategic thinking, and career advancement strategies.",
                category="professional_development",
                tags=["professional-development", "leadership", "communication", "career-growth", role.lower(), campaign.lower()],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            relevant_docs.append(doc)
        
        # Add campaign-specific content based on search keywords
        if search_keywords:
            keywords_list = [kw.strip() for kw in search_keywords.split(',')]
            for keyword in keywords_list[:2]:
                if keyword.lower() not in [tag.lower() for tag in relevant_docs[0].tags]:
                    keyword_doc = KnowledgeDocument(
                        id=f"keyword-{keyword.replace(' ', '-')}-{campaign}",
                        title=f"{keyword.title()} Specialization Course",
                        content=f"Specialized course focusing on {keyword} for {role} professionals. Covers advanced concepts, practical applications, and industry best practices.",
                        category="specialization",
                        tags=[keyword.lower().replace(' ', '-'), "specialization", role.lower(), campaign.lower()],
                        created_at="2024-01-01T00:00:00Z",
                        updated_at="2024-01-01T00:00:00Z"
                    )
                    relevant_docs.append(keyword_doc)
        
        # Add interest-level specific content
        if prior_interest == "high":
            interest_doc = KnowledgeDocument(
                id=f"advanced-{campaign}",
                title=f"Advanced {campaign.replace('_', ' ').title()} Mastery Course",
                content=f"Advanced mastery course for highly engaged {role} professionals. Covers advanced techniques, expert-level strategies, and industry leadership skills.",
                category="advanced_mastery",
                tags=["advanced", "mastery", "expert-level", role.lower(), campaign.lower(), "high-interest"],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            relevant_docs.append(interest_doc)
        
        return relevant_docs[:3]
    
    async def _generate_email(self, state: WorkflowState) -> Dict[str, Any]:
        """Generate personalized email using RAG and LLM with structured logging."""
        import time
        start_time = time.time()
        trace_id = f"email_gen_{int(time.time() * 1000)}"
        
        try:
            logger.info(f"[{trace_id}] Starting email generation")
            lead_data = state['lead_data']
            lead_score = state['lead_score']
            context_docs = state['context_docs']
            
            # Log lead information (with PII redaction)
            logger.info(f"[{trace_id}] Lead: {self._redact_pii(lead_data.name)}, Type: {lead_score.heat_score}, Docs: {len(context_docs)}")
            
            # Prepare context text from retrieved documents with citations
            context_text = "\n\n".join([
                f"[SOURCE_{i+1}] Course: {doc.title}\nSyllabus: {doc.content}"
                for i, doc in enumerate(context_docs)
            ])
            
            # Create citation mapping for tracking
            citations = {
                f"SOURCE_{i+1}": {
                    "id": doc.id,
                    "title": doc.title,
                    "category": doc.category,
                    "tags": doc.tags
                }
                for i, doc in enumerate(context_docs)
            }
            
            # Determine lead type and let AI generate natural CTA
            lead_type = lead_score.heat_score.lower()
            
            # Create educational advisor prompt with citation instructions
            prompt = self._create_educational_prompt(lead_data, context_text, lead_type, citations)
            
            # Generate email using LLM
            llm_start = time.time()
            messages = [HumanMessage(content=prompt)]
            response = await self.llm.ainvoke(messages)
            llm_duration = time.time() - llm_start
            logger.info(f"[{trace_id}] LLM response received in {llm_duration:.2f}s")
            
            # Record token usage for monitoring
            if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
                token_usage = response.response_metadata['token_usage']
                prompt_tokens = token_usage.get('prompt_tokens', 0)
                completion_tokens = token_usage.get('completion_tokens', 0)
                logger.info(f"[{trace_id}] Tokens: {prompt_tokens} prompt + {completion_tokens} completion")
            
            # Parse JSON response
            parse_start = time.time()
            logger.info(f"[{trace_id}] Raw LLM response: {response.content[:200]}...")
            email_data = self._extract_json_from_response(response.content, lead_data)
            parse_duration = time.time() - parse_start
            
            # Replace [SOURCE_X] citations with actual document content
            email_data = self._replace_citations_with_content(email_data, context_docs)
            
            # Add citations to email data
            email_data["citations"] = citations
            email_data["source_count"] = len(context_docs)
            
            # Store generated email in cache for preview/send consistency
            store_generated_email(lead_data, email_data)
            
            total_duration = time.time() - start_time
            logger.info(f"[{trace_id}] Email generated successfully in {total_duration:.2f}s (LLM: {llm_duration:.2f}s, Parse: {parse_duration:.2f}s)")
            
            return {"email_content": email_data}
                
        except Exception as e:
            total_duration = time.time() - start_time
            logger.error(f"[{trace_id}] Email generation failed after {total_duration:.2f}s: {str(e)}")
            return {"error": f"Email generation failed: {str(e)}"}
    
    
    def _replace_citations_with_content(self, email_data: Dict[str, Any], context_docs: List[KnowledgeDocument]) -> Dict[str, Any]:
        """Replace [SOURCE_X] citations with actual document titles/content."""
        try:
            # Create replacement mapping
            replacements = {}
            for i, doc in enumerate(context_docs):
                source_tag = f"[SOURCE_{i+1}]"
                # Use document title as replacement (you can customize this)
                replacements[source_tag] = doc.title
            
            # Replace citations in subject and content
            if "subject" in email_data:
                for source_tag, replacement in replacements.items():
                    email_data["subject"] = email_data["subject"].replace(source_tag, replacement)
            
            if "content" in email_data:
                for source_tag, replacement in replacements.items():
                    email_data["content"] = email_data["content"].replace(source_tag, replacement)
            
            return email_data
            
        except Exception as e:
            logger.error(f"Error replacing citations: {e}")
            return email_data  # Return original if replacement fails
    
    def _get_required_emojis(self, lead_type: str) -> str:
        """Get required emojis based on lead type."""
        emoji_map = {
            "hot": "⏳🔥",
            "warm": "🎓✨",
            "cold": "📘🌱"
        }
        return emoji_map.get(lead_type.lower(), "📘🌱")
    
    def _create_educational_prompt(self, lead_data, context_text: str, lead_type: str, citations: dict = None) -> str:
        """Generate personalized email using enhanced dynamic prompt template with lead-type specific tones."""
        
        emojis = self._get_required_emojis(lead_type)
        return f"""You are an educational sales executive at Learn Sprouts. Create an engaging email for {lead_data.name}.

LEAD INFO: {lead_data.name}, {lead_data.role}, interested in {lead_data.campaign}
LEAD TYPE: {lead_type}

REQUIREMENTS:
- MINIMUM 80-100 words in the content section - count every word!
- Include emojis: {emojis}  
- Create unique subject line (not generic "Course Information")
- Use \\n\\n for line breaks between paragraphs
- Format course content as bullet points: • Item 1\\n• Item 2
- Write detailed course benefits, features, outcomes, and call-to-action

Context: {context_text[:400]}

FORMAT:
{{"subject": "[UNIQUE SUBJECT]", "content": "Hi {lead_data.name}!\\n\\n{emojis} [OPENING MESSAGE]\\n\\nOur course covers:\\n• [FEATURE 1]\\n• [FEATURE 2]\\n• [FEATURE 3]\\n\\n[CALL TO ACTION]\\n\\nBest regards,\\nLearn Sprouts", "email_type": "{lead_type}"}}

TONE BY TYPE:
- hot: "🔥 Limited spots! Enroll now - 50% OFF expires soon!"
- warm: "🎓 FREE trial + exclusive webinar access!"  
- cold: "📚 FREE resources + course preview!"

Write like a friendly sales executive - engaging, clear, benefit-focused!

CRITICAL: Your content must be at least 80-100 words. Use \\n\\n between paragraphs and bullet points for syllabus. Include specific course details, benefits, outcomes, and strong call-to-action. Do NOT write short messages!"""
    
    def _extract_json_from_response(self, response_text: str, lead_data=None) -> Dict[str, Any]:
        """Bulletproof JSON extraction that handles all LLM quirks."""
        import json
        import re
        
        try:
            raw_text = response_text.strip()
            logger.info(f"[JSON_PARSE] Processing {len(raw_text)} chars")
            logger.info(f"[JSON_PARSE] Raw: {raw_text[:150]}...")
            
            # Step 0: Aggressive cleaning first
            # Remove any text before first { and after last }
            start_idx = raw_text.find('{')
            end_idx = raw_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                raw_text = raw_text[start_idx:end_idx+1]
            
            # Clean up common issues that break JSON
            raw_text = raw_text.replace('\\"', '"')  # Fix escaped quotes
            # Don't remove quotes inside content - this was breaking emojis and formatting
            # Keep \\n for proper line breaks - don't replace with spaces
            
            # Step 0.5: Quick regex extraction for simple cases (more flexible)
            # Try multiple patterns to catch different formats - emoji-friendly
            patterns = [
                # More flexible patterns that handle \\n and emojis in content
                r'\{\s*"subject":\s*"([^"]*)",\s*"content":\s*"((?:[^"\\]|\\.)*)"\s*,\s*"email_type":\s*"([^"]*)"\s*\}',
                r'"subject":\s*"([^"]*)",\s*"content":\s*"((?:[^"\\]|\\.)*)",\s*"email_type":\s*"([^"]*)"',
                r'"subject"\s*:\s*"([^"]*)".*?"content"\s*:\s*"((?:[^"\\]|\\.)*)".*?"email_type"\s*:\s*"([^"]*)"'
            ]
            
            for pattern in patterns:
                simple_match = re.search(pattern, raw_text, re.DOTALL)
                if simple_match and len(simple_match.group(2)) > 10:  # Ensure content is not empty
                    logger.info("[JSON_PARSE] ✅ Regex extraction successful")
                    return {
                        "subject": simple_match.group(1).strip(),
                        "content": simple_match.group(2).strip().replace('\\n', '<br>'),
                        "email_type": simple_match.group(3).strip()
                    }
            
            # Step 1: Try direct JSON parse (best case)
            try:
                result = json.loads(raw_text)
                logger.info(f"[JSON_PARSE] ✅ Direct parse: {list(result.keys())}")
                # Convert \n to <br> for HTML display
                if 'content' in result:
                    result['content'] = result['content'].replace('\\n\\n', '<br><br>').replace('\\n', '<br>')
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"[JSON_PARSE] Direct parse failed: {e}")
                logger.warning(f"[JSON_PARSE] Failed text: {raw_text[:200]}...")
                pass
            
            # Step 1.5: Handle common LLM response format with <br> tags
            try:
                # Clean up <br> tags that break JSON parsing
                cleaned_text = raw_text
                # Replace <br><br> with \\n\\n and <br> with \\n in content field
                cleaned_text = re.sub(r'("content":\s*"[^"]*?)<br><br>([^"]*?")', r'\1\\n\\n\2', cleaned_text)
                cleaned_text = re.sub(r'("content":\s*"[^"]*?)<br>([^"]*?")', r'\1\\n\2', cleaned_text)
                
                result = json.loads(cleaned_text)
                if isinstance(result, dict) and 'subject' in result and 'content' in result:
                    # Convert back the \\n to <br> for HTML display
                    if 'content' in result:
                        result['content'] = result['content'].replace('\\n\\n', '<br><br>').replace('\\n', '<br>')
                    logger.info(f"[JSON_PARSE] ✅ Cleaned <br> parse: {list(result.keys())}")
                    return result
            except json.JSONDecodeError:
                pass
            
            # Step 2: Extract JSON from mixed content using regex
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_text, re.DOTALL)
            if json_match:
                json_candidate = json_match.group()
                logger.info(f"[JSON_PARSE] Extracted candidate: {json_candidate[:100]}...")
                
                # Step 3: Clean common LLM formatting issues
                cleaned_candidates = [
                    json_candidate,  # Try as-is first
                    # Fix smart quotes and unicode
                    json_candidate.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'").replace('🔥', '🔥'),
                    # Fix escaped newlines
                    json_candidate.replace('\\n', '\n').replace('\\"', '"'),
                    # Fix unescaped newlines in content
                    re.sub(r'("content":\s*"[^"]*)<br>([^"]*")', r'\1\\n\2', json_candidate),
                    # Remove trailing commas
                    re.sub(r',(\s*[}\]])', r'\1', json_candidate),
                    # Normalize whitespace but preserve line breaks
                    re.sub(r'[ \t]+', ' ', json_candidate),
                    # Fix quote escaping in content - more specific
                    re.sub(r'("content":\s*"[^"]*?)([^\\])"([^,:}\]]*[,:}\]])', r'\1\2\\"\3', json_candidate)
                ]
                
                for i, candidate in enumerate(cleaned_candidates):
                    try:
                        result = json.loads(candidate)
                        logger.info(f"[JSON_PARSE] ✅ Cleaned parse method {i+1}: {list(result.keys())}")
                        return result
                    except json.JSONDecodeError as e:
                        logger.debug(f"[JSON_PARSE] Method {i+1} failed: {e}")
                        continue
            
            # Step 4: Manual field extraction (very robust)
            logger.warning("[JSON_PARSE] Trying manual field extraction")
            
            def extract_field(field_name, text):
                """Extract a field value using multiple strategies."""
                patterns = [
                    rf'"{field_name}":\s*"([^"]*(?:\\"[^"]*)*)"',  # Standard
                    rf'"{field_name}":\s*"(.*?)"(?=\s*[,}}])',      # Greedy until comma/brace
                    rf'{field_name}.*?:\s*"([^"]+)"',               # Flexible key matching
                    rf'{field_name}.*?:\s*["\']([^"\']+)["\']'      # Single or double quotes
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                    if match:
                        value = match.group(1).replace('\\"', '"').replace('\\n', '\n').strip()
                        if len(value) > 0:
                            return value
                return None
            
            subject = extract_field("subject", raw_text)
            content = extract_field("content", raw_text)
            email_type = extract_field("email_type", raw_text) or (lead_data.prior_course_interest if hasattr(lead_data, 'prior_course_interest') else "warm")
            
            if subject and content and len(content) > 10:
                logger.info(f"[JSON_PARSE] ✅ Manual extraction successful")
                logger.info(f"[JSON_PARSE] Subject: {subject[:50]}...")
                logger.info(f"[JSON_PARSE] Content: {content[:100]}...")
                # Convert \n to <br> for HTML display
                content = content.replace('\\n\\n', '<br><br>').replace('\\n', '<br>')
                return {
                    "subject": subject,
                    "content": content,
                    "email_type": email_type
                }
            
            # Step 5: Last resort - use raw content if it looks meaningful
            if len(raw_text) > 50 and any(word in raw_text.lower() for word in ['course', 'program', 'learn', 'data', 'python', 'training']):
                logger.warning("[JSON_PARSE] Using raw content as fallback")
                
                # Try to extract subject from the raw content
                subject_match = re.search(r'"subject":\s*"([^"]+)"', raw_text)
                extracted_subject = subject_match.group(1) if subject_match else "Course Information"
                
                # Use more of the content - don't truncate so aggressively
                content_match = re.search(r'"content":\s*"([^"]+(?:\\.[^"]*)*)"', raw_text, re.DOTALL)
                if content_match:
                    extracted_content = content_match.group(1).replace('\\"', '"').replace('\\n', '\n').replace('<br>', '<br>')
                else:
                    # Try a more flexible content extraction
                    flexible_content_match = re.search(r'"content":\s*"([^"]*(?:[^"\\]|\\.)*)"', raw_text, re.DOTALL)
                    if flexible_content_match:
                        extracted_content = flexible_content_match.group(1).replace('\\"', '"').replace('\\n', '\n').replace('<br>', '<br>')
                    else:
                        # Generate a meaningful fallback instead of raw JSON
                        name = getattr(lead_data, 'name', 'there')
                        role = getattr(lead_data, 'role', 'professional')
                        campaign = getattr(lead_data, 'campaign', 'our program')
                        extracted_content = f"Hi {name},<br><br>Thank you for your interest in our {campaign} program. As a {role}, we believe this could be valuable for your career development.<br><br>Best regards,<br>Learn Sprouts"
                
                return {
                    "subject": extracted_subject,
                    "content": extracted_content,
                    "email_type": "warm"
                }
            
            # Absolute fallback
            logger.error("[JSON_PARSE] ❌ Complete parsing failure - using template")
            return {
                "subject": "Let's Connect",
                "content": "Hi there! I'd love to discuss our program with you.",
                "email_type": "warm"
            }
            
        except Exception as e:
            logger.error(f"[JSON_PARSE] Exception: {e}")
            return {
                "subject": "Let's Connect", 
                "content": "Hi there! I'd love to discuss our program with you.",
                "email_type": "warm"
            }
    
    def _detect_prompt_injection(self, text: str) -> bool:
        """Detect potential prompt injection attempts."""
        import re
        
        injection_patterns = [
            r'ignore\s+(previous|above|all)\s+instructions',
            r'forget\s+(everything|all)\s+(previous|above)',
            r'you\s+are\s+now\s+(a|an)\s+',
            r'pretend\s+to\s+be',
            r'act\s+as\s+(if\s+)?(you\s+are\s+)?(a|an)\s+',
            r'system\s*:\s*',
            r'<\|.*\|>',
            r'\[.*\]\s*:\s*'
        ]
        
        text_lower = text.lower()
        for pattern in injection_patterns:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def _create_personalization_elements(self, lead_data) -> str:
        """Create dynamic personalization elements for each lead."""
        elements = []
        
        # Role-specific insights
        if "data" in lead_data.role.lower() or "analyst" in lead_data.role.lower():
            elements.append("- Data professionals often benefit from hands-on analytics projects")
        elif "project" in lead_data.role.lower() or "manager" in lead_data.role.lower():
            elements.append("- Project managers typically need leadership and stakeholder management skills")
        elif "healthcare" in lead_data.role.lower() or "medical" in lead_data.role.lower():
            elements.append("- Healthcare professionals often seek technology integration skills")
        
        # Campaign-specific insights
        if "ai" in lead_data.campaign.lower():
            elements.append("- AI courses are in high demand across industries")
        elif "leadership" in lead_data.campaign.lower():
            elements.append("- Leadership skills are crucial for career advancement")
        
        # Interest level insights
        if lead_data.prior_course_interest == "high":
            elements.append("- High-interest learners typically seek advanced, practical content")
        elif lead_data.prior_course_interest == "low":
            elements.append("- New learners benefit from foundational concepts and clear explanations")
        
        # Search keywords insights
        if lead_data.search_keywords:
            keywords = [kw.strip() for kw in lead_data.search_keywords.split(',')]
            for keyword in keywords[:2]:
                elements.append(f"- {keyword} skills are valuable in today's market")
        
        return "\n".join(elements) if elements else "- Professional development is key to career growth"
    
    def _get_safe_fallback(self, lead_data) -> Dict[str, Any]:
        """Get safe fallback content with PII redaction."""
        if not lead_data:
            return {
                "subject": "Let's Connect",
                "content": "Hi there! I'd love to discuss our program with you.",
                "email_type": "warm"
            }
        
        # Redact sensitive information
        name = self._redact_pii(lead_data.name) if hasattr(lead_data, 'name') else 'there'
        role = self._redact_pii(lead_data.role) if hasattr(lead_data, 'role') else 'professional'
        campaign = self._redact_pii(lead_data.campaign) if hasattr(lead_data, 'campaign') else 'our program'
        
        return {
            "subject": f"Hi {name} - Let's Connect",
            "content": f"Hi {name}! I noticed your interest in our {campaign} program. As a {role}, this could be valuable for your career. Best regards, LearnSprout Team",
            "email_type": "warm"
        }
    
    def _redact_pii(self, text: str) -> str:
        """Redact PII from text."""
        import re
        
        if not text:
            return text
        
        # Redact email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Redact phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        
        # Redact SSN patterns
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        
        return text
    
    def _should_continue(self, state: WorkflowState) -> str:
        """Determine if workflow should continue or handle error."""
        return "continue" if not state.get("error") else "error"
    
    async def _handle_error(self, state: WorkflowState) -> Dict[str, Any]:
        """Handle workflow errors."""
        error_msg = state.get("error", "Unknown error")
        logger.error(f"Workflow error: {error_msg}")
        return {"error": error_msg}
    
    async def process_lead(self, lead_data: LeadInput, use_cached: bool = False) -> Dict[str, Any]:
        """Process a lead through the complete workflow."""
        try:
            # Check if we should use cached email (for send operations)
            if use_cached:
                cached_email = get_cached_email(lead_data)
                if cached_email:
                    logger.info("📧 Using cached email for consistent preview/send")
                    return {
                        "success": True,
                        "email_content": cached_email,
                        "lead_score": None,  # Not needed for cached emails
                        "trace_id": str(uuid.uuid4()),
                        "cached": True
                    }
                else:
                    logger.warning("📧 No cached email found, generating new one")
            
            # Create initial state
            state = {
                "lead_data": lead_data,
                "lead_score": None,
                "context_docs": [],
                "email_content": None,
                "email_sent": False,
                "error": None,
                "trace_id": str(uuid.uuid4())
            }
            
            # Run the workflow
            result = await self.graph.ainvoke(state)
            
            if result.get("error"):
                raise Exception(result["error"])
            
            return {
                "success": True,
                "email_content": result["email_content"],
                "lead_score": result["lead_score"],
                "trace_id": result["trace_id"],
                "cached": False
            }
            
        except Exception as e:
            logger.error(f"Error processing lead: {e}")
            return {
                "success": False,
                "error": str(e),
                "trace_id": str(uuid.uuid4())
            }

# Global workflow instance
lead_workflow = LeadProcessingWorkflow()