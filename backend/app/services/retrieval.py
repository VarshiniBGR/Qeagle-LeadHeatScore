import numpy as np
from typing import List, Dict, Any, Optional
import asyncio
import time
import openai
from motor.motor_asyncio import AsyncIOMotorCollection
from app.config import settings
from app.models.schemas import KnowledgeDocument, SearchResult
from app.utils.logging import get_logger
from app.db import get_database
from app.services.cross_encoder_reranker import reranker
from app.services.performance_monitor import performance_monitor
from app.services.circuit_breaker import mongodb_circuit_breaker


logger = get_logger(__name__)


class HybridRetrieval:
    """Hybrid retrieval system combining vector search and BM25."""
    
    def __init__(self):
        self.embedding_model = None
        self.collection: Optional[AsyncIOMotorCollection] = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the OpenAI client."""
        try:
            if settings.openai_api_key:
                # OpenAI client will be initialized per request
                logger.info("OpenAI API key configured for embeddings")
            else:
                logger.warning("OpenAI API key not found - running in lightweight mode")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            logger.info("Running in lightweight mode without ML models")
    
    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get MongoDB collection."""
        if self.collection is None:
            db = await get_database()
            self.collection = db[settings.mongo_collection]
        return self.collection
    
    def _compute_embedding(self, text: str) -> List[float]:
        """Compute embedding for text using OpenAI's latest embedding model."""
        try:
            if not settings.openai_api_key:
                logger.warning("OpenAI API key not available - using dummy embeddings")
                return [0.0] * 1536
            
            # Use OpenAI's latest embedding model
            client = openai.OpenAI(api_key=settings.openai_api_key)
            
            response = client.embeddings.create(
                input=text,
                model=settings.embedding_model_name  # text-embedding-3-small
            )
            
            embedding = response.data[0].embedding
            logger.info(f"Using OpenAI {settings.embedding_model_name} embedding - {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            logger.error(f"Error computing OpenAI embedding: {e}")
            # Fallback to dummy embedding
            logger.warning("Using dummy embedding as fallback")
            return [0.0] * 1536
    
    async def add_document(self, document: KnowledgeDocument) -> str:
        """Add a document to the knowledge base."""
        try:
            collection = await self._get_collection()
            
            # Compute embedding
            embedding = self._compute_embedding(document.content)
            document.embedding = embedding
            
            # Insert document
            result = await collection.insert_one(document.dict())
            document_id = str(result.inserted_id)
            
            logger.info(f"Added document to knowledge base: {document_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise
    
    async def vector_search(
        self, 
        query: str, 
        limit: int = 10,
        score_threshold: float = 0.0  # Lower threshold for testing
    ) -> List[SearchResult]:
        """Perform vector similarity search."""
        try:
            collection = await self._get_collection()
            
            # Compute query embedding
            query_embedding = self._compute_embedding(query)
            logger.info(f"Query embedding computed, dimensions: {len(query_embedding)}")
            
            # Simple cosine similarity search (no MongoDB vector index needed)
            logger.info("Using cosine similarity search")
            
            # Get all documents with embeddings
            cursor = collection.find({"embedding": {"$exists": True}})
            all_docs = await cursor.to_list(length=None)
            
            if not all_docs:
                logger.warning("No documents with embeddings found")
                return []
            
            # Calculate cosine similarity for each document
            similarities = []
            for doc in all_docs:
                doc_embedding = doc.get('embedding', [])
                
                # Skip if document embedding is all zeros
                if all(x == 0.0 for x in doc_embedding):
                    logger.warning(f"Skipping document {doc.get('title', 'Unknown')} - all zero embedding")
                    continue
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                logger.info(f"Document '{doc.get('title', 'Unknown')}' similarity: {similarity:.4f}")
                
                # Very low threshold to catch any reasonable matches
                if similarity >= max(score_threshold, 0.001):  # Even lower threshold!
                    similarities.append((doc, similarity))
            
            # Sort by similarity score
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Take top results
            results = [(doc, score) for doc, score in similarities[:limit]]
            
            logger.info(f"Cosine similarity search returned {len(results)} results (threshold: {max(score_threshold, 0.001)})")
            if results:
                logger.info(f"Top similarity score: {results[0][1]:.4f}")
            else:
                logger.warning("No vector results found - falling back to text search")
                # Fallback to text search
                return await self.bm25_search(query, limit)
            
            # Convert to SearchResult objects
            search_results = []
            for i, (doc, score) in enumerate(results):
                knowledge_doc = KnowledgeDocument(
                    id=str(doc['_id']),
                    title=doc['title'],
                    content=doc['content'],
                    category=doc['category'],
                    tags=doc.get('tags', []),
                    created_at=doc['created_at'],
                    updated_at=doc['updated_at']
                )
                
                search_results.append(SearchResult(
                    document=knowledge_doc,
                    score=score,
                    rank=i + 1
                ))
            
            logger.info(f"Vector search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            import numpy as np
            
            # Convert to numpy arrays
            a = np.array(vec1)
            b = np.array(vec2)
            
            # Calculate cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            similarity = dot_product / (norm_a * norm_b)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    async def bm25_search(
        self, 
        query: str, 
        limit: int = 10
    ) -> List[SearchResult]:
        """Perform BM25 text search (fallback to simple text matching)."""
        try:
            collection = await self._get_collection()
            
            # Simple text search without text index
            cursor = collection.find(
                {
                    "$or": [
                        {"title": {"$regex": query, "$options": "i"}},
                        {"content": {"$regex": query, "$options": "i"}},
                        {"category": {"$regex": query, "$options": "i"}}
                    ]
                }
            ).limit(limit)
            
            results = await cursor.to_list(length=limit)
            
            # Convert to SearchResult objects
            search_results = []
            for i, doc in enumerate(results):
                knowledge_doc = KnowledgeDocument(
                    id=str(doc['_id']),
                    title=doc['title'],
                    content=doc['content'],
                    category=doc['category'],
                    tags=doc.get('tags', []),
                    created_at=doc['created_at'],
                    updated_at=doc['updated_at']
                )
                
                search_results.append(SearchResult(
                    document=knowledge_doc,
                    score=doc['score'],
                    rank=i + 1
                ))
            
            logger.info(f"BM25 search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []
    
    async def hybrid_search(
        self, 
        query: str, 
        limit: int = 10,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ) -> List[SearchResult]:
        """Perform hybrid search combining vector and BM25."""
        # Start performance trace
        trace_id = performance_monitor.start_trace("hybrid_search")
        
        try:
            # Run both searches in parallel with timing
            search_start = time.time()
            vector_task = self.vector_search(query, limit * 2)
            bm25_task = self.bm25_search(query, limit * 2)
            
            vector_results, bm25_results = await asyncio.gather(
                vector_task, bm25_task
            )
            
            search_duration = (time.time() - search_start) * 1000
            performance_monitor.record_step(trace_id, "parallel_search", search_duration)
            
            # Combine results
            combine_start = time.time()
            combined_results = {}
            
            # Add vector results
            for result in vector_results:
                doc_id = result.document.id
                combined_results[doc_id] = {
                    'document': result.document,
                    'vector_score': result.score,
                    'bm25_score': 0.0,
                    'combined_score': result.score * vector_weight
                }
            
            # Add BM25 results
            for result in bm25_results:
                doc_id = result.document.id
                if doc_id in combined_results:
                    combined_results[doc_id]['bm25_score'] = result.score
                    combined_results[doc_id]['combined_score'] += result.score * bm25_weight
                else:
                    combined_results[doc_id] = {
                        'document': result.document,
                        'vector_score': 0.0,
                        'bm25_score': result.score,
                        'combined_score': result.score * bm25_weight
                    }
            
            # Sort by combined score and limit results
            sorted_results = sorted(
                combined_results.values(),
                key=lambda x: x['combined_score'],
                reverse=True
            )[:limit]
            
            combine_duration = (time.time() - combine_start) * 1000
            performance_monitor.record_step(trace_id, "combine_results", combine_duration)
            
            # Convert to SearchResult objects
            search_results = []
            for i, result in enumerate(sorted_results):
                search_results.append(SearchResult(
                    document=result['document'],
                    score=result['combined_score'],
                    rank=i + 1
                ))
            
            logger.info(
                "Hybrid search completed",
                trace_id=trace_id,
                query_length=len(query),
                results_count=len(search_results),
                vector_weight=vector_weight,
                bm25_weight=bm25_weight
            )
            
            # Apply enhanced cross-encoder reranking with timing
            if len(search_results) > limit and settings.enable_reranking:
                rerank_start = time.time()
                logger.info("Applying enhanced cross-encoder reranking")
                search_results = await self.rerank_results(
                    query=query,
                    results=search_results,
                    top_k=limit,
                    alpha=settings.rerank_alpha
                )
                rerank_duration = (time.time() - rerank_start) * 1000
                performance_monitor.record_step(trace_id, "reranking", rerank_duration)
            
            performance_monitor.finish_trace(trace_id, status_code=200)
            return search_results
            
        except Exception as e:
            logger.error(
                "Error in hybrid search",
                trace_id=trace_id,
                error=str(e),
                error_type=type(e).__name__
            )
            performance_monitor.finish_trace(trace_id, status_code=500, error_type=type(e).__name__)
            return []
    
    async def rerank_results(
        self, 
        query: str, 
        results: List[SearchResult],
        top_k: int = 5,
        alpha: float = 0.3
    ) -> List[SearchResult]:
        """Rerank results using enhanced cross-encoder."""
        try:
            if len(results) <= top_k:
                return results
            
            logger.info(f"Using enhanced cross-encoder reranker for {len(results)} results")
            
            # Use the enhanced cross-encoder reranker
            reranked_results = await reranker.rerank_results(
                query=query,
                results=results,
                top_k=top_k,
                alpha=alpha,
                use_async=True
            )
            
            logger.info(f"Enhanced reranking completed. Returned {len(reranked_results)} results")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error in enhanced reranking: {e}")
            # Fallback to basic reranking
            return await self._basic_rerank(query, results, top_k)
    
    async def _basic_rerank(
        self, 
        query: str, 
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """Fallback basic reranking when cross-encoder fails."""
        try:
            reranked_results = []
            
            for result in results:
                # Calculate relevance score based on content similarity
                content = result.document.content.lower()
                query_lower = query.lower()
                
                # Count query term matches
                query_terms = query_lower.split()
                matches = sum(1 for term in query_terms if term in content)
                relevance_score = matches / len(query_terms) if query_terms else 0
                
                # Combine with original score
                final_score = result.score * 0.7 + relevance_score * 0.3
                
                reranked_results.append(SearchResult(
                    document=result.document,
                    score=final_score,
                    rank=result.rank
                ))
            
            # Sort by final score
            reranked_results.sort(key=lambda x: x.score, reverse=True)
            
            # Update ranks
            for i, result in enumerate(reranked_results[:top_k]):
                result.rank = i + 1
            
            logger.info(f"Basic reranking completed for {len(results)} results")
            return reranked_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in basic reranking: {e}")
            return results[:top_k]


# Global retrieval instance
retrieval = HybridRetrieval()
