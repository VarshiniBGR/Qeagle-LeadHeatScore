import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import asyncio
import time
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
        """Initialize the embedding model."""
        try:
            self.embedding_model = SentenceTransformer(settings.embedding_model_name)
            logger.info(f"Initialized embedding model: {settings.embedding_model_name}")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            logger.info("Running in lightweight mode without ML models")
            self.embedding_model = None
    
    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get MongoDB collection."""
        if not self.collection:
            db = await get_database()
            self.collection = db[settings.mongo_collection]
        return self.collection
    
    def _compute_embedding(self, text: str) -> List[float]:
        """Compute embedding for text."""
        if self.embedding_model is None:
            # Return a dummy embedding for lightweight mode
            logger.warning("Using dummy embedding - ML model not available")
            return [0.0] * 384  # Standard embedding size
        
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error computing embedding: {e}")
            return [0.0] * 384  # Fallback to dummy embedding
    
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
        score_threshold: float = 0.7
    ) -> List[SearchResult]:
        """Perform vector similarity search."""
        try:
            collection = await self._get_collection()
            
            # Compute query embedding
            query_embedding = self._compute_embedding(query)
            
            # Vector search pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": settings.mongo_vector_index,
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 2,
                        "limit": limit,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "title": 1,
                        "content": 1,
                        "category": 1,
                        "tags": 1,
                        "created_at": 1,
                        "updated_at": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                },
                {
                    "$match": {
                        "score": {"$gte": score_threshold}
                    }
                }
            ]
            
            # Execute search
            cursor = collection.aggregate(pipeline)
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
            
            logger.info(f"Vector search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    async def bm25_search(
        self, 
        query: str, 
        limit: int = 10
    ) -> List[SearchResult]:
        """Perform BM25 text search."""
        try:
            collection = await self._get_collection()
            
            # MongoDB text search
            cursor = collection.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)
            
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
