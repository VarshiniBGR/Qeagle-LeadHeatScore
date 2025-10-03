"""
Cross-Encoder Reranker Service
Uses sentence-transformers cross-encoder for reranking search results
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from app.models.schemas import SearchResult, KnowledgeDocument
from app.utils.logging import get_logger
from app.config import settings

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers."""
    
    def __init__(self):
        self.model = None
        self.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Reverted - v3 had loading overhead
        self.is_loaded = False
        
        # Rerank result caching - TEMPORARILY DISABLED for testing
        self.rerank_cache = {}
        self.cache_ttl = 0      # TEMPORARILY DISABLED for testing real RAG
        self.max_cache_size = 2000  # Doubled cache size for better hit rate
        
        # Pre-computed embeddings cache for common queries
        self.query_embedding_cache = {}
        self.embedding_cache_ttl = 21600  # 6 hours for embeddings
        
        # Don't load immediately - will be loaded at server startup
        # self._load_model()
    
    def _load_model(self):
        """Load the cross-encoder model with optimization."""
        try:
            from sentence_transformers import CrossEncoder
            import torch
            
            logger.info(f"🚀 Loading cross-encoder model: {self.model_name}")
            
            # Optimize for inference speed
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            self.model = CrossEncoder(self.model_name, device=device)
            
            # Optimize model for inference
            if hasattr(self.model.model, 'eval'):
                self.model.model.eval()
            
            # Warm up the model with a dummy prediction
            dummy_pairs = [["test query", "test document"]]
            _ = self.model.predict(dummy_pairs)
            
            self.is_loaded = True
            logger.info("✅ Cross-encoder model loaded and warmed up successfully")
            
        except ImportError:
            logger.warning("sentence-transformers not available - cross-encoder reranking disabled")
            self.model = None
            self.is_loaded = False
        except Exception as e:
            logger.error(f"Error loading cross-encoder model: {e}")
            self.model = None
            self.is_loaded = False
    
    async def preload_model(self):
        """Preload model at server startup."""
        if not self.is_loaded:
            logger.info("🔄 Preloading cross-encoder model at startup...")
            await asyncio.to_thread(self._load_model)
            logger.info("✅ Cross-encoder model preloaded successfully")
    
    def is_available(self) -> bool:
        """Check if the cross-encoder model is available."""
        return self.is_loaded and self.model is not None
    
    async def rerank_results(
        self, 
        query: str, 
        results: List[SearchResult],
        top_k: int = 5,
        alpha: float = 0.3
    ) -> List[SearchResult]:
        """
        Rerank search results using cross-encoder with caching.
        
        Args:
            query: Search query
            results: List of search results to rerank
            top_k: Number of top results to return
            alpha: Weight for rerank scores (higher = more emphasis on cross-encoder)
            
        Returns:
            Reranked list of search results
        """
        try:
            if not self.is_available():
                logger.warning("Cross-encoder not available, returning original results")
                return results[:top_k]
            
            if len(results) <= top_k:
                logger.info(f"Results ({len(results)}) <= top_k ({top_k}), no reranking needed")
                return results
            
            # Enhanced rerank cache with better key generation
            import hashlib
            import time
            
            # Create more specific cache key including content hashes for better hit rate
            content_hash = hashlib.md5(''.join([r.document.content[:100] for r in results]).encode()).hexdigest()[:8]
            cache_key = hashlib.md5(f"{query}_{content_hash}_{top_k}_{alpha}".encode()).hexdigest()
            cached_result = self.rerank_cache.get(cache_key)
            
            if cached_result and time.time() - cached_result.get("timestamp", 0) < self.cache_ttl:
                logger.info(f"Rerank cache hit for query: '{query[:30]}' (saved ~300ms)")
                return cached_result["results"]
            
            logger.info(f"Cross-encoder reranking {len(results)} results for query: '{query[:50]}...'")
            
            # Prepare query-document pairs for cross-encoder
            # Aggressively limit content length for P95 optimization (512 -> 200 chars)
            query_doc_pairs = []
            for result in results:
                # Truncate content to first 200 characters for faster reranking (P95 optimization)
                content = result.document.content[:200] if len(result.document.content) > 200 else result.document.content
                # Also truncate title to 50 chars to reduce token processing
                title = result.document.title[:50] if len(result.document.title) > 50 else result.document.title
                doc_text = f"{title} {content}"
                query_doc_pairs.append([query, doc_text])
            
            # Calculate cross-encoder scores in batches
            batch_size = 8  # Optimized for P95 single email generation
            rerank_scores = []
            
            logger.info(f"Processing {len(query_doc_pairs)} query-document pairs in batches of {batch_size}")
            
            for i in range(0, len(query_doc_pairs), batch_size):
                batch = query_doc_pairs[i:i + batch_size]
                batch_num = i//batch_size + 1
                total_batches = (len(query_doc_pairs) + batch_size - 1) // batch_size
                
                try:
                    # Timeout optimized for single email P95
                    batch_scores = await asyncio.wait_for(
                        asyncio.to_thread(self.model.predict, batch),
                        timeout=5.0  # Optimized for P95 performance
                    )
                    logger.debug(f"Batch {batch_num}/{total_batches} completed successfully")
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Cross-encoder batch {batch_num}/{total_batches} timed out, using fallback scores")
                    # Use fallback scores based on original ranking
                    batch_scores = [0.5 - (j * 0.01) for j in range(len(batch))]
                except Exception as e:
                    logger.error(f"Cross-encoder batch {batch_num}/{total_batches} failed: {e}")
                    batch_scores = [0.5 - (j * 0.01) for j in range(len(batch))]
                
                rerank_scores.extend(batch_scores)
            
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
            
            final_results = reranked_results[:top_k]
            
            # Cache the reranked results
            self.rerank_cache[cache_key] = {
                "results": final_results,
                "timestamp": time.time()
            }
            
            # Clean cache periodically
            if len(self.rerank_cache) % 10 == 0:
                self._clean_rerank_cache()
            
            logger.info(f"Cross-encoder reranking completed. Top {top_k} results selected.")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {e}")
            return results[:top_k]
    
    def _combine_scores(
        self, 
        original_scores: List[float], 
        rerank_scores: List[float],
        alpha: float = 0.3
    ) -> List[float]:
        """
        Combine original retrieval scores with cross-encoder rerank scores.
        
        Args:
            original_scores: Original retrieval scores
            rerank_scores: Cross-encoder relevance scores
            alpha: Weight for rerank scores (1-alpha for original scores)
            
        Returns:
            Combined scores
        """
        combined_scores = []
        
        for orig_score, rerank_score in zip(original_scores, rerank_scores):
            # Normalize original score to 0-1 range
            normalized_orig = min(max(orig_score, 0), 1)
            
            # Normalize rerank score to 0-1 range
            normalized_rerank = min(max(rerank_score, 0), 1)
            
            # Combine scores
            combined_score = (1 - alpha) * normalized_orig + alpha * normalized_rerank
            
            combined_scores.append(combined_score)
        
        return combined_scores
    
    def _clean_rerank_cache(self):
        """Clean expired rerank cache entries."""
        import time
        current_time = time.time()
        
        # Remove expired entries
        expired_keys = [
            key for key, entry in self.rerank_cache.items()
            if current_time - entry.get("timestamp", 0) > self.cache_ttl
        ]
        for key in expired_keys:
            del self.rerank_cache[key]
        
        # Maintain size limits
        if len(self.rerank_cache) > self.max_cache_size:
            sorted_items = sorted(
                self.rerank_cache.items(),
                key=lambda x: x[1].get("timestamp", 0)
            )
            items_to_remove = len(self.rerank_cache) - self.max_cache_size
            for key, _ in sorted_items[:items_to_remove]:
                del self.rerank_cache[key]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model_name": self.model_name,
            "is_available": self.is_available(),
            "model_type": "cross-encoder",
            "supports_async": True,
            "reranking_method": "cross-encoder-relevance-scoring"
        }


# Global cross-encoder reranker instance
cross_encoder_reranker = CrossEncoderReranker()
