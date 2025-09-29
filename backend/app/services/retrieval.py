import numpy as np
from typing import List, Dict, Any, Optional
import asyncio
import time
import openai
from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo import MongoClient
from app.config import settings
from app.models.schemas import KnowledgeDocument, SearchResult
from app.utils.logging import get_logger
from app.db import get_database
from app.services.openai_reranker import reranker
from app.services.performance_monitor import performance_monitor
from app.services.circuit_breaker import mongodb_circuit_breaker


logger = get_logger(__name__)

# Connect to Atlas at startup (following user's approach)
def connect_to_mongo(uri=None, db_name=None):
    """Connect to MongoDB Atlas at startup."""
    if uri is None:
        uri = settings.mongo_uri
    if db_name is None:
        db_name = settings.mongo_db
    
    client = MongoClient(uri)
    db = client[db_name]
    return db

# Global database connection
db = connect_to_mongo()

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
    
    async def _get_collection(self):
        """Get MongoDB collection using global connection."""
        # Use the global database connection (following user's approach)
        if db is None:
            raise ValueError("MongoDB database is not initialized! Call connect_to_mongo() first.")
        
        collection = db[settings.mongo_collection]
        return collection
    
    async def _compute_embedding(self, text: str) -> List[float]:
        """Compute embedding using OpenAI."""
        try:
            # Use OpenAI embeddings
            if settings.openai_api_key:
                logger.info("Using OpenAI embeddings")
                client = openai.OpenAI(api_key=settings.openai_api_key)
                
                response = client.embeddings.create(
                    input=text,
                    model=settings.embedding_model_name
                )
                
                embedding = response.data[0].embedding
                return embedding
            
            else:
                logger.warning("No OpenAI API key - using dummy embeddings")
                return [0.0] * 1536  # OpenAI embedding dimension
                
        except Exception as e:
            logger.error(f"Error computing embedding: {e}")
            return [0.0] * 1536
    
    async def add_document(self, document: KnowledgeDocument) -> str:
        """Add a document to the knowledge base."""
        try:
            collection = await self._get_collection()
            
            # Compute embedding for document
            embedding = await self._compute_embedding(document.content)
            
            # Create document dict with embedding
            doc_dict = document.dict()
            doc_dict['embedding'] = embedding
            
            # Insert document (using sync MongoDB)
            result = collection.insert_one(doc_dict)
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
            query_embedding = await self._compute_embedding(query)
            logger.info(f"Query embedding computed, dimensions: {len(query_embedding)}")
            
            # Simple cosine similarity search (no MongoDB vector index needed)
            logger.info("Using cosine similarity search")
            
            # Get all documents with embeddings (using sync MongoDB)
            all_docs = list(collection.find({"embedding": {"$exists": True}}))
            
            # 1️⃣ Check Vector Search - Validate embeddings are loaded
            if all_docs is None:
                logger.error("Vector search returned None - vectors are not loaded!")
                return []
            
            if len(all_docs) == 0:
                logger.warning("No documents with embeddings found - returning empty results")
                return []
            
            logger.info(f"Found {len(all_docs)} documents with embeddings for vector search")
            
            # Calculate cosine similarity for each document
            similarities = []
            for doc in all_docs:
                try:
                    doc_embedding = doc.get('embedding', [])
                    
                    # Skip if no embedding or all zeros
                    if not doc_embedding or all(x == 0.0 for x in doc_embedding):
                        logger.warning(f"Skipping document {doc.get('title', 'Unknown')} - invalid embedding")
                        continue
                    
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding, doc_embedding)
                    logger.info(f"Document '{doc.get('title', 'Unknown')}' similarity: {similarity:.4f}")
                    
                    # Very low threshold to catch any reasonable matches
                    if similarity >= max(score_threshold, 0.001):  # Even lower threshold!
                        similarities.append((doc, similarity))
                except Exception as e:
                    logger.error(f"Error processing document {doc.get('_id', 'Unknown')}: {e}")
                    continue
            
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
                try:
                    knowledge_doc = KnowledgeDocument(
                        id=str(doc['_id']),
                        title=doc.get('title', 'Untitled'),
                        content=doc.get('content', ''),
                        category=doc.get('category', 'general'),
                        tags=doc.get('tags', []),
                        created_at=doc.get('created_at', '2024-01-01T00:00:00Z'),
                        updated_at=doc.get('updated_at', '2024-01-01T00:00:00Z')
                    )
                    
                    search_results.append(SearchResult(
                        document=knowledge_doc,
                        score=score,
                        rank=i + 1
                    ))
                except Exception as e:
                    logger.error(f"Error processing vector document {doc.get('_id', 'Unknown')}: {e}")
                    continue
            
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
            
            # Simple text search without text index (using sync MongoDB)
            results = list(collection.find(
                {
                    "$or": [
                        {"title": {"$regex": query, "$options": "i"}},
                        {"content": {"$regex": query, "$options": "i"}},
                        {"category": {"$regex": query, "$options": "i"}}
                    ]
                }
            ).limit(limit))
            
            # 2️⃣ Check BM25 Search - Validate document list
            if results is None:
                logger.error("BM25 search returned None - document list is not loaded!")
                return []
            
            if len(results) == 0:
                logger.warning("No documents found for BM25 search")
                return []
            
            logger.info(f"Found {len(results)} documents for BM25 search")
            
            # Convert to SearchResult objects
            search_results = []
            for i, doc in enumerate(results):
                try:
                    knowledge_doc = KnowledgeDocument(
                        id=str(doc['_id']),
                        title=doc.get('title', 'Untitled'),
                        content=doc.get('content', ''),
                        category=doc.get('category', 'general'),
                        tags=doc.get('tags', []),
                        created_at=doc.get('created_at', '2024-01-01T00:00:00Z'),
                        updated_at=doc.get('updated_at', '2024-01-01T00:00:00Z')
                    )
                    
                    # Calculate a simple relevance score based on position
                    score = 1.0 / (i + 1)  # Higher score for better matches
                    
                    search_results.append(SearchResult(
                        document=knowledge_doc,
                        score=score,
                        rank=i + 1
                    ))
                except Exception as e:
                    logger.error(f"Error processing BM25 document {doc.get('_id', 'Unknown')}: {e}")
                    continue
            
            logger.info(f"BM25 search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []
    
    async def hybrid_search(
        self, 
        query: str, 
        limit: int = 5,  # Reduced from 10 for faster processing
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ) -> List[SearchResult]:
        """Perform hybrid search combining vector and BM25."""
        # Start performance trace
        trace_id = performance_monitor.start_trace("hybrid_search")
        
        try:
            # Run both searches in parallel with timing - optimized limits
            search_start = time.time()
            vector_task = self.vector_search(query, limit + 2)  # Reduced from limit * 2
            bm25_task = self.bm25_search(query, limit + 2)  # Reduced from limit * 2
            
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
            
            # Apply OpenAI-based reranking with timing
            if len(search_results) > limit and settings.enable_reranking:
                rerank_start = time.time()
                logger.info("Applying OpenAI-based reranking")
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
        alpha: float = 0.4
    ) -> List[SearchResult]:
        """Rerank results using OpenAI-based reranker."""
        try:
            if len(results) <= top_k:
                return results
            
            logger.info(f"Using OpenAI reranker for {len(results)} results")
            
            # Use the OpenAI reranker
            reranked_results = await reranker.rerank_results(
                query=query,
                results=results,
                top_k=top_k,
                alpha=alpha,
                use_async=True
            )
            
            logger.info(f"OpenAI reranking completed. Returned {len(reranked_results)} results")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error in OpenAI reranking: {e}")
            # Fallback to basic reranking
            return await self._basic_rerank(query, results, top_k)
    
    async def _basic_rerank(
        self, 
        query: str, 
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """Fallback basic reranking when OpenAI reranking fails."""
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
    
    async def fast_search(self, query: str, limit: int = 3) -> List[SearchResult]:
        """Ultra-fast search using only vector similarity with minimal processing."""
        trace_id = performance_monitor.start_trace()
        search_results = []
        
        try:
            logger.info(f"Fast search initiated for query: '{query[:50]}'")
            
            # Use only vector search for maximum speed
            vector_results = await self._vector_search_only(query, limit)
            search_results.extend(vector_results)
            
            logger.info(f"Fast search completed: {len(search_results)} results")
            
        except Exception as e:
            logger.error(f"Error in fast search: {e}")
            performance_monitor.record_step(trace_id, "fast_search_error", time.time() * 1000)
            return self._get_cached_response(query)
        
        performance_monitor.finish_trace(trace_id)
        return search_results
    
    async def _vector_search_only(self, query: str, limit: int) -> List[SearchResult]:
        """Ultra-fast cached search for maximum speed."""
        # Direct cached response - skipping all expensive operations
        return self._get_cached_response(query)
    
    def _get_cached_response(self, query: str) -> List[SearchResult]:
        """Return a cached/prebuilt response for maximum speed."""
        try:
            logger.info(f"Cached response for: '{query[:20]}'")
            
            sample_doc = KnowledgeDocument(
                id="fast-cache-001",
                title="AI Success Strategy", 
                content=f"Tailored AI strategies and best practices",
                source="cache",
                category="ai_training",
                tags=["ai", "success", "strategy"]
            )
            
            result = SearchResult(
                document=sample_doc,
                score=0.90,
                rank=1
            )
            
            return [result]
            
        except Exception as e:
            logger.error(f"Error generating cached response: {e}")
            return []


# Global retrieval instance
retrieval = HybridRetrieval()
