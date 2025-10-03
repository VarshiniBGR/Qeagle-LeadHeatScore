"""
RAG-Enabled Email Service with MongoDB Atlas
"""

import asyncio
import time
import json
from typing import Dict, Any, List
from datetime import datetime
import openai
from app.models.schemas import LeadInput, LeadScore, KnowledgeDocument
from app.services.classifier import classifier
from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

class RAGEmailService:
    """RAG-enabled email service with MongoDB Atlas."""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        if settings.openai_api_key:
            self.client = openai.OpenAI(api_key=settings.openai_api_key)
            logger.info("RAG email service initialized with OpenAI")
    
    async def _get_mongodb_connection(self):
        """Get MongoDB connection."""
        try:
            from app.db import get_database
            self.db = await get_database()
            self.collection = self.db[settings.mongo_collection]
            logger.info("MongoDB connection established")
            return True
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            return False
    
    async def _retrieve_documents(self, query: str, limit: int = 3) -> List[KnowledgeDocument]:
        """Retrieve documents using MongoDB Atlas vector search."""
        try:
            if not await self._get_mongodb_connection():
                return self._get_fallback_documents(query)
            
            # Generate embedding for query
            if self.client:
                response = self.client.embeddings.create(
                    input=query,
                    model=settings.embedding_model_name
                )
                query_embedding = response.data[0].embedding
                logger.info(f"Generated query embedding with {len(query_embedding)} dimensions")
            else:
                logger.warning("No OpenAI client, using dummy embedding")
                query_embedding = [0.0] * 1536
            
            # MongoDB Atlas vector search
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": settings.mongo_vector_index,
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 10,
                        "limit": limit
                    }
                },
                {
                    "$project": {
                        "title": 1,
                        "content": 1,
                        "category": 1,
                        "tags": 1,
                        "created_at": 1,
                        "updated_at": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            results = list(self.collection.aggregate(pipeline))
            logger.info(f"MongoDB Atlas vector search returned {len(results)} results")
            
            documents = []
            for i, doc in enumerate(results):
                try:
                    knowledge_doc = KnowledgeDocument(
                        id=str(doc['_id']),
                        title=doc.get('title', 'Untitled'),
                        content=doc.get('content', ''),
                        category=doc.get('category', 'general'),
                        tags=doc.get('tags', []),
                        created_at=doc.get('created_at', datetime.now().isoformat()),
                        updated_at=doc.get('updated_at', datetime.now().isoformat())
                    )
                    documents.append(knowledge_doc)
                except Exception as e:
                    logger.error(f"Error processing document {doc.get('_id', 'Unknown')}: {e}")
                    continue
            
            return documents
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return self._get_fallback_documents(query)
    
    def _get_fallback_documents(self, query: str) -> List[KnowledgeDocument]:
        """No fake fallback documents - return empty list."""
        logger.error(f"MongoDB Atlas not available for query: '{query[:50]}' - no fake fallbacks")
        return []
    
    async def generate_email(self, lead_data: LeadInput) -> Dict[str, Any]:
        """Generate RAG-personalized email with urgency."""
        start_time = time.time()
        
        try:
            logger.info(f"Generating RAG email for {lead_data.name} ({lead_data.role})")
            
            # 1. Classify lead
            lead_score = classifier.predict(lead_data)
            lead_type = lead_score.heat_score.lower()
            
            # 2. Create dynamic search query
            search_query = self._create_search_query(lead_data)
            logger.info(f"RAG search query: {search_query}")
            
            # 3. Retrieve relevant documents using RAG
            context_docs = await self._retrieve_documents(search_query, limit=3)
            logger.info(f"RAG retrieved {len(context_docs)} documents")
            
            # 4. Generate email with RAG context
            email_content = await self._generate_rag_email(lead_data, context_docs, lead_type)
            
            # 5. Create citations
            citations = {
                f"SOURCE_{i+1}": {
                    "id": doc.id,
                    "title": doc.title,
                    "category": doc.category,
                    "tags": doc.tags
                }
                for i, doc in enumerate(context_docs)
            }
            
            generation_time = time.time() - start_time
            logger.info(f"RAG email generated in {generation_time:.2f}s")
            
            return {
                "success": True,
                "email_content": email_content,
                "lead_score": lead_score,
                "context_docs": context_docs,
                "citations": citations,
                "source_count": len(context_docs),
                "generation_time": generation_time
            }
            
        except Exception as e:
            logger.error(f"RAG email generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "email_content": self._get_fallback_email(lead_data)
            }
    
    def _create_search_query(self, lead_data: LeadInput) -> str:
        """Create dynamic search query based on lead data."""
        query_parts = []
        
        if lead_data.role:
            query_parts.append(f"{lead_data.role} professional development")
        if lead_data.campaign:
            query_parts.append(f"{lead_data.campaign} course")
        if lead_data.search_keywords:
            keywords = [kw.strip() for kw in lead_data.search_keywords.split(',')]
            query_parts.extend(keywords[:3])
        if lead_data.prior_course_interest == "high":
            query_parts.append("advanced mastery")
        elif lead_data.prior_course_interest == "low":
            query_parts.append("beginner friendly")
        
        return " ".join(query_parts)
    
    async def _generate_rag_email(self, lead_data: LeadInput, context_docs: List[KnowledgeDocument], lead_type: str) -> Dict[str, str]:
        """Generate email with RAG context and urgency."""
        
        # Prepare context text with sources
        context_text = "\n\n".join([
            f"[SOURCE_{i+1}] {doc.title}\n{doc.content}"
            for i, doc in enumerate(context_docs)
        ])
        
        # Define urgency and tone
        urgency_map = {
            "hot": {
                "tone": "URGENT and action-driven",
                "phrases": "limited time, act now, secure your spot, don't miss out",
                "emojis": "🔥⏰",
                "cta": "Schedule your demo call today!"
            },
            "warm": {
                "tone": "Helpful and informative", 
                "phrases": "perfect for you, advance your career, great opportunity",
                "emojis": "📚💡",
                "cta": "Learn more about this program"
            },
            "cold": {
                "tone": "Friendly and educational",
                "phrases": "explore, learn more, discover, benefit from",
                "emojis": "📖🌱", 
                "cta": "Find out how this can help you"
            }
        }
        
        urgency = urgency_map.get(lead_type, urgency_map["warm"])
        
        # Create RAG prompt - COMMENTED OUT: Using new prompt in langgraph_workflow.py instead
        # prompt = f"""Create personalized email for {lead_data.name} ({lead_data.role}) about {lead_data.campaign}.
        #
        # Lead Details:
        # - Name: {lead_data.name}
        # - Role: {lead_data.role}
        # - Campaign: {lead_data.campaign}
        # - Interest Level: {lead_data.prior_course_interest}
        # - Keywords: {lead_data.search_keywords}
        # - Heat Score: {lead_type.upper()}
        #
        # Course Context (RAG Retrieved):
        # {context_text[:500]}...
        #
        # Instructions:
        # 1. Write a {urgency['tone']} email that feels personal and relevant to their role.
        # 2. Include specific skills/benefits from the course context above.
        # 3. Use {urgency['emojis']} emojis naturally in the content.
        # 4. Include urgency based on heat score: {urgency['phrases']}.
        # 5. Reference sources naturally like "Our course covers [specific skills] [SOURCE_1]".
        # 6. Keep it concise (100-120 words).
        # 7. Use <strong> tags for important parts and <br> for line breaks.
        #
        # Return ONLY valid JSON:
        # {{"subject": "Personalized subject line", "content": "Full email content with HTML formatting", "email_type": "{lead_type}"}}"""
        
        # FALLBACK: Use simplified prompt for this legacy service
        prompt = f"""Generate email for {lead_data.name} about {lead_data.campaign}. Return JSON: {{"subject": "...", "content": "...", "email_type": "{lead_type}"}}"""
        
        try:
            if self.client:
                response = self.client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,
                    temperature=0.3
                )
                
                content = response.choices[0].message.content
                
                # Parse JSON response
                try:
                    email_data = json.loads(content)
                except json.JSONDecodeError:
                    email_data = self._parse_fallback_response(content, lead_data, lead_type)
                
                return email_data
            else:
                return self._get_fallback_email(lead_data)
                
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return self._get_fallback_email(lead_data)
    
    def _parse_fallback_response(self, content: str, lead_data: LeadInput, lead_type: str) -> Dict[str, str]:
        """Parse response when JSON parsing fails."""
        subject = f"Hi {lead_data.name} - {lead_data.campaign} Opportunity"
        
        # Extract content between quotes
        email_content = content
        if '"content"' in content:
            try:
                start = content.find('"content": "') + 12
                end = content.rfind('"')
                email_content = content[start:end]
            except:
                pass
        
        return {
                    "subject": subject,
            "content": email_content,
            "email_type": lead_type
        }
    
    def _get_fallback_email(self, lead_data: LeadInput) -> Dict[str, str]:
        """Get fallback email when generation fails."""
        return {
            "subject": f"Hi {lead_data.name} - {lead_data.campaign} Program",
            "content": f"Hi {lead_data.name},<br><br>I noticed your interest in {lead_data.campaign}. As a {lead_data.role}, this program could be perfect for advancing your career.<br><br>Would you like to learn more?<br><br>Best regards,<br>LearnSprout Team",
            "email_type": "warm"
        }

# Global instance
rag_email_service = RAGEmailService()

