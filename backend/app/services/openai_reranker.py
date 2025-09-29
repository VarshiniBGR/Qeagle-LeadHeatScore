"""
OpenAI-based Reranker Service
Intelligent reranking using OpenAI's GPT models for relevance scoring
"""

import asyncio
import openai
from typing import List, Dict, Any, Optional, Tuple
from app.models.schemas import SearchResult, KnowledgeDocument
from app.utils.logging import get_logger
from app.config import settings

logger = get_logger(__name__)


class OpenAIReranker:
    """OpenAI-based reranker for search results using GPT models."""
    
    def __init__(self):
        self.client = None
        self.model_name = settings.llm_model
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            if settings.openai_api_key:
                self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
                logger.info(f"OpenAI reranker initialized with model: {self.model_name}")
            else:
                logger.warning("OpenAI API key not found - reranking disabled")
                self.client = None
        except Exception as e:
            logger.error(f"Error initializing OpenAI reranker: {e}")
            self.client = None
    
    async def _calculate_relevance_scores(
        self, 
        query: str, 
        results: List[SearchResult]
    ) -> List[float]:
        """Calculate relevance scores using OpenAI GPT model."""
        if not self.client:
            logger.warning("OpenAI client not available - skipping reranking")
            return [result.score for result in results]
        
        try:
            # Prepare content for scoring
            content_pairs = []
            for result in results:
                content_pairs.append({
                    "query": query,
                    "title": result.document.title,
                    "content": result.document.content[:500] + "..." if len(result.document.content) > 500 else result.document.content
                })
            
            # Create scoring prompt
            scoring_prompt = self._create_scoring_prompt(query, content_pairs)
            
            # Get relevance scores from OpenAI
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at evaluating search result relevance. Rate each result from 0.0 to 1.0 based on how well it answers the query."},
                    {"role": "user", "content": scoring_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse scores from response
            scores = self._parse_scores(response.choices[0].message.content, len(results))
            
            logger.info(f"Calculated {len(scores)} OpenAI relevance scores")
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating OpenAI relevance scores: {e}")
            # Fallback to original scores
            return [result.score for result in results]
    
    def _create_scoring_prompt(self, query: str, content_pairs: List[Dict]) -> str:
        """Create prompt for relevance scoring."""
        prompt = f"""Query: "{query}"

Rate each search result from 0.0 to 1.0 based on relevance to the query. Respond with only the scores separated by commas.

Results:
"""
        
        for i, pair in enumerate(content_pairs):
            prompt += f"{i+1}. Title: {pair['title']}\n   Content: {pair['content']}\n\n"
        
        prompt += "Scores (0.0-1.0, comma-separated):"
        return prompt
    
    def _parse_scores(self, response: str, expected_count: int) -> List[float]:
        """Parse scores from OpenAI response."""
        try:
            # Extract scores from response
            scores_text = response.strip()
            if "Scores:" in scores_text:
                scores_text = scores_text.split("Scores:")[-1].strip()
            
            # Parse comma-separated scores
            scores = []
            for score_str in scores_text.split(','):
                try:
                    score = float(score_str.strip())
                    scores.append(max(0.0, min(1.0, score)))  # Clamp to [0,1]
                except ValueError:
                    scores.append(0.5)  # Default score
            
            # Ensure we have the right number of scores
            while len(scores) < expected_count:
                scores.append(0.5)
            
            return scores[:expected_count]
            
        except Exception as e:
            logger.error(f"Error parsing scores: {e}")
            return [0.5] * expected_count
    
    def _combine_scores(
        self, 
        original_scores: List[float], 
        rerank_scores: List[float],
        alpha: float = 0.4
    ) -> List[float]:
        """
        Combine original retrieval scores with OpenAI rerank scores.
        
        Args:
            original_scores: Original BM25/vector scores
            rerank_scores: OpenAI relevance scores
            alpha: Weight for rerank scores (1-alpha for original scores)
            
        Returns:
            Combined scores
        """
        combined_scores = []
        
        for orig_score, rerank_score in zip(original_scores, rerank_scores):
            # Normalize original score to [0,1]
            normalized_orig = min(max(orig_score, 0), 1)
            
            # OpenAI score is already [0,1]
            normalized_rerank = min(max(rerank_score, 0), 1)
            
            # Weighted combination
            combined_score = (1 - alpha) * normalized_orig + alpha * normalized_rerank
            combined_scores.append(combined_score)
        
        return combined_scores
    
    async def rerank_results(
        self, 
        query: str, 
        results: List[SearchResult],
        top_k: int = 5,
        alpha: float = 0.4,
        use_async: bool = True
    ) -> List[SearchResult]:
        """
        Rerank search results using OpenAI GPT model.
        
        Args:
            query: Search query
            results: List of search results to rerank
            top_k: Number of top results to return
            alpha: Weight for rerank scores (higher = more emphasis on OpenAI)
            use_async: Whether to use async processing
            
        Returns:
            Reranked list of search results
        """
        try:
            if len(results) <= top_k:
                return results
            
            logger.info(f"OpenAI reranking {len(results)} results for query: '{query[:50]}...'")
            
            # Calculate OpenAI relevance scores
            rerank_scores = await self._calculate_relevance_scores(query, results)
            
            # Extract original scores
            original_scores = [result.score for result in results]
            
            # Combine scores
            combined_scores = self._combine_scores(
                original_scores, 
                rerank_scores, 
                alpha
            )
            
            # Create reranked results
            reranked_results = []
            for i, (result, new_score) in enumerate(zip(results, combined_scores)):
                reranked_result = SearchResult(
                    document=result.document,
                    score=new_score,
                    rank=i + 1  # Will be updated after sorting
                )
                reranked_results.append(reranked_result)
            
            # Sort by combined score
            reranked_results.sort(key=lambda x: x.score, reverse=True)
            
            # Update ranks
            for i, result in enumerate(reranked_results[:top_k]):
                result.rank = i + 1
            
            logger.info(f"OpenAI reranking completed. Top {top_k} results selected.")
            return reranked_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in OpenAI reranking: {e}")
            return results[:top_k]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model_name": self.model_name,
            "is_available": self.client is not None,
            "model_type": "openai-gpt",
            "supports_async": True,
            "reranking_method": "gpt-relevance-scoring"
        }


# Global reranker instance
reranker = OpenAIReranker()
