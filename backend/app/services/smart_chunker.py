#!/usr/bin/env python3
"""
Smart Chunking Service - Optimize token usage and costs
"""

import re
from typing import List, Dict, Any
from app.models.schemas import KnowledgeDocument, TextChunk
from app.utils.logging import get_logger

logger = get_logger(__name__)

class SmartChunker:
    """Smart chunking to minimize token usage and costs."""
    
    def __init__(self):
        self.max_chunk_tokens = 50   # Ultra-minimal chunks for max savings
        self.chunk_overlap_tokens = 5   # Minimal overlap
        self.min_chunk_tokens = 10   # Tiny minimum chunks
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 characters)."""
        return len(text) // 4
    
    def chunk_document(self, document: KnowledgeDocument) -> List[TextChunk]:
        """Chunk document with smart optimization."""
        try:
            content = document.content
            estimated_tokens = self.estimate_tokens(content)
            
            # If document is small enough, return as single chunk
            if estimated_tokens <= self.max_chunk_tokens:
                chunk = TextChunk(
                    chunk_id=f"{document.id}_chunk_0",
                    source_id=document.id,
                    chunk_index=0,
                    total_chunks=1,
                    text=content,
                    start_pos=0,
                    end_pos=len(content),
                    token_count=estimated_tokens,
                    metadata={
                        "title": document.title,
                        "category": document.category,
                        "tags": document.tags,
                        "created_at": document.created_at,
                        "updated_at": document.updated_at,
                        "optimization": "single_chunk"
                    }
                )
                return [chunk]
            
            # Smart chunking for larger documents
            chunks = self._smart_chunk_text(content, document)
            
            logger.info(f"Document chunked: {len(chunks)} chunks, avg {sum(c.token_count for c in chunks)//len(chunks)} tokens each")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            return []
    
    def _smart_chunk_text(self, text: str, document: KnowledgeDocument) -> List[TextChunk]:
        """Smart text chunking with semantic boundaries."""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        
        for para in paragraphs:
            para_tokens = self.estimate_tokens(para)
            
            # If adding this paragraph exceeds limit, create chunk
            if current_tokens + para_tokens > self.max_chunk_tokens and current_chunk:
                chunk = self._create_chunk(
                    current_chunk.strip(),
                    document,
                    chunk_index,
                    len(paragraphs)
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + "\n\n" + para
                current_tokens = self.estimate_tokens(current_chunk)
                chunk_index += 1
            else:
                current_chunk += "\n\n" + para if current_chunk else para
                current_tokens += para_tokens
        
        # Add final chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                current_chunk.strip(),
                document,
                chunk_index,
                len(paragraphs)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, text: str, document: KnowledgeDocument, index: int, total: int) -> TextChunk:
        """Create a TextChunk with optimized metadata."""
        return TextChunk(
            chunk_id=f"{document.id}_chunk_{index}",
            source_id=document.id,
            chunk_index=index,
            total_chunks=total,
            text=text,
            start_pos=0,  # Simplified for cost optimization
            end_pos=len(text),
            token_count=self.estimate_tokens(text),
            metadata={
                "title": document.title,
                "category": document.category,
                "tags": document.tags[:3],  # Limit tags to reduce metadata size
                "created_at": document.created_at,
                "updated_at": document.updated_at,
                "optimization": "smart_chunked",
                "cost_optimized": True
            }
        )
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text for continuity."""
        words = text.split()
        if len(words) <= self.chunk_overlap_tokens:
            return text
        
        # Take last few words for overlap
        overlap_words = words[-self.chunk_overlap_tokens:]
        return " ".join(overlap_words)

class CostOptimizer:
    """Optimize costs by reducing token usage."""
    
    @staticmethod
    def optimize_prompt(prompt: str) -> str:
        """Optimize prompt to reduce token usage."""
        # Remove unnecessary words
        optimizations = [
            (r'\bYou are a\b', 'You are'),
            (r'\bfor the following\b', 'for'),
            (r'\bthat might be relevant\b', 'relevant'),
            (r'\bI hope you are doing well\b', 'Hope you\'re well'),
            (r'\bI wanted to share\b', 'Sharing'),
            (r'\bI noticed that\b', 'Noticed'),
            (r'\bI believe that\b', 'Believe'),
            (r'\bI would like to\b', 'Would like to'),
            (r'\bI think that\b', 'Think'),
            (r'\bI feel that\b', 'Feel'),
        ]
        
        optimized = prompt
        for pattern, replacement in optimizations:
            optimized = re.sub(pattern, replacement, optimized, flags=re.IGNORECASE)
        
        return optimized
    
    @staticmethod
    def should_generate_email(lead_type: str, engagement_score: float) -> bool:
        """Decide if we should generate email or use template."""
        # Only generate for high-engagement leads to save costs
        if lead_type.lower() == 'hot' and engagement_score > 0.8:
            return True
        elif lead_type.lower() == 'warm' and engagement_score > 0.6:
            return True
        else:
            return False  # Use template for cold/low-engagement leads
    
    @staticmethod
    def get_template_for_lead(lead_type: str) -> Dict[str, str]:
        """Get optimized template for cost savings."""
        templates = {
            "hot": {
                "subject": "🔥 Exclusive AI Course Offer - Limited Time!",
                "content": """Hi {name},

Exclusive offer for {role}s interested in {campaign}!

🎯 SPECIAL DEAL: 50% OFF - Limited time only!

✨ What you get:
• Industry certification
• Career support
• Hands-on projects
• Expert mentorship

Reply 'YES' to claim your spot!

Best,
GUVI Team"""
            },
            "warm": {
                "subject": "🎓 Free Webinar - {campaign} Masterclass",
                "content": """Hi {name},

Free webinar for {role}s exploring {campaign}!

🎓 FREE MASTERCLASS this weekend

✨ Learn:
• Industry insights
• Career guidance
• Practical skills
• Expert tips

Reply 'WEBINAR' to join!

Best,
GUVI Team"""
            },
            "cold": {
                "subject": "📚 Free Resources - {campaign}",
                "content": """Hi {name},

Free resources for {role}s interested in {campaign}.

📚 FREE LEARNING:
• Industry insights
• Best practices
• Free previews

Reply 'RESOURCES' for access!

Best,
GUVI Team"""
            }
        }
        return templates.get(lead_type, templates["warm"])

# Global instances
smart_chunker = SmartChunker()
cost_optimizer = CostOptimizer()
