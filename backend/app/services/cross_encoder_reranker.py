"""
Cross-Encoder Reranker Service
Enhanced reranking using sentence-transformers cross-encoder models
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
from app.models.schemas import SearchResult, KnowledgeDocument
from app.utils.logging import get_logger
from app.config import settings

logger = get_logger(__name__)

# Try to import CrossEncoder, but make it optional
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    logger.warning("CrossEncoder not available - reranking disabled")
    CROSS_ENCODER_AVAILABLE = False
    CrossEncoder = None


class CrossEncoderReranker:
    """Advanced cross-encoder reranker for search results."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Cross-encoder model name. If None, uses config setting.
                Options:
                - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good quality)
                - cross-encoder/ms-marco-MiniLM-L-12-v2 (slower, better quality)
                - cross-encoder/nli-deberta-v3-base (best quality, slowest)
        """
        self.model_name = model_name or settings.cross_encoder_model
        self.cross_encoder = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the cross-encoder model."""
        if not CROSS_ENCODER_AVAILABLE:
            logger.warning("CrossEncoder not available - reranking disabled")
            self.cross_encoder = None
            return
            
        try:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self.cross_encoder = CrossEncoder(self.model_name)
            logger.info(f"Cross-encoder model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading cross-encoder model: {e}")
            self.cross_encoder = None
    
    def _prepare_query_document_pairs(
        self, 
        query: str, 
        results: List[SearchResult]
    ) -> List[Tuple[str, str]]:
        """Prepare query-document pairs for cross-encoder."""
        pairs = []
        
        for result in results:
            # Create document text from title and content
            doc_text = f"{result.document.title}\n{result.document.content}"
            
            # Truncate if too long (cross-encoders have token limits)
            if len(doc_text) > 512:
                doc_text = doc_text[:512] + "..."
            
            pairs.append((query, doc_text))
        
        return pairs
    
    def _calculate_rerank_scores(
        self, 
        query: str, 
        results: List[SearchResult]
    ) -> List[float]:
        """Calculate rerank scores using cross-encoder."""
        if not self.cross_encoder:
            logger.warning("Cross-encoder not available, using fallback scoring")
            return self._fallback_scoring(query, results)
        
        try:
            # Prepare query-document pairs
            pairs = self._prepare_query_document_pairs(query, results)
            
            # Get cross-encoder scores
            scores = self.cross_encoder.predict(pairs)
            
            # Convert to list if single score
            if isinstance(scores, np.ndarray):
                scores = scores.tolist()
            elif not isinstance(scores, list):
                scores = [scores]
            
            logger.info(f"Calculated {len(scores)} rerank scores")
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating rerank scores: {e}")
            return self._fallback_scoring(query, results)
    
    def _fallback_scoring(self, query: str, results: List[SearchResult]) -> List[float]:
        """Fallback scoring when cross-encoder is not available."""
        scores = []
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        for result in results:
            # Combine title and content for scoring
            doc_text = f"{result.document.title} {result.document.content}".lower()
            
            # Calculate term overlap score
            doc_terms = set(doc_text.split())
            overlap = len(query_terms.intersection(doc_terms))
            term_score = overlap / len(query_terms) if query_terms else 0
            
            # Calculate position-based score (title matches are more important)
            title_text = result.document.title.lower()
            title_overlap = len(query_terms.intersection(set(title_text.split())))
            title_score = title_overlap / len(query_terms) if query_terms else 0
            
            # Combine scores
            combined_score = (term_score * 0.6 + title_score * 0.4)
            scores.append(combined_score)
        
        return scores
    
    def _combine_scores(
        self, 
        original_scores: List[float], 
        rerank_scores: List[float],
        alpha: float = 0.3
    ) -> List[float]:
        """
        Combine original retrieval scores with rerank scores.
        
        Args:
            original_scores: Original retrieval scores
            rerank_scores: Cross-encoder rerank scores
            alpha: Weight for rerank scores (1-alpha for original scores)
        """
        combined_scores = []
        
        for orig_score, rerank_score in zip(original_scores, rerank_scores):
            # Normalize scores to [0, 1] range
            normalized_orig = min(max(orig_score, 0), 1)
            normalized_rerank = min(max(rerank_score, 0), 1)
            
            # Combine with weighted average
            combined_score = (1 - alpha) * normalized_orig + alpha * normalized_rerank
            combined_scores.append(combined_score)
        
        return combined_scores
    
    async def rerank_results(
        self, 
        query: str, 
        results: List[SearchResult],
        top_k: int = 5,
        alpha: float = 0.3,
        use_async: bool = True
    ) -> List[SearchResult]:
        """
        Rerank search results using cross-encoder.
        
        Args:
            query: Search query
            results: List of search results to rerank
            top_k: Number of top results to return
            alpha: Weight for rerank scores (higher = more emphasis on cross-encoder)
            use_async: Whether to use async processing
            
        Returns:
            Reranked list of search results
        """
        try:
            if len(results) <= top_k:
                return results
            
            logger.info(f"Reranking {len(results)} results for query: '{query[:50]}...'")
            
            if use_async and self.cross_encoder:
                # Run cross-encoder scoring in thread pool
                loop = asyncio.get_event_loop()
                rerank_scores = await loop.run_in_executor(
                    self.executor, 
                    self._calculate_rerank_scores, 
                    query, 
                    results
                )
            else:
                # Synchronous processing
                rerank_scores = self._calculate_rerank_scores(query, results)
            
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
            
            logger.info(f"Reranking completed. Top {top_k} results selected.")
            return reranked_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return results[:top_k]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model_name": self.model_name,
            "is_loaded": self.cross_encoder is not None,
            "model_type": "cross-encoder",
            "supports_async": True
        }
    
    def __del__(self):
        """Cleanup thread pool executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Global reranker instance
reranker = CrossEncoderReranker()
