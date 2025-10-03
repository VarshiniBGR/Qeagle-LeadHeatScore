import numpy as np
from typing import List, Dict, Any, Optional
import asyncio
import time
import openai
import hashlib
import json
from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo import MongoClient
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from app.services.local_embedding_service import local_embedding_service
from app.models.schemas import KnowledgeDocument, SearchResult
from app.db import get_database
from app.services.performance_monitor import performance_monitor
from app.config import settings
from app.utils.logging import get_logger
from app.services.cross_encoder_reranker import cross_encoder_reranker
from app.services.circuit_breaker import mongodb_circuit_breaker


logger = get_logger(__name__)

# Removed lazy loading - now using pre-initialized connection from app.db

class HybridRetrieval:
    """Hybrid retrieval system using LangChain components with caching."""
    
    def __init__(self):
        self.embedding_model = None
        self.text_splitter = None
        self.collection: Optional[AsyncIOMotorCollection] = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        
        # Enhanced caching system for P95 optimization
        self.embedding_cache = {}  # Query -> embedding
        self.search_cache = {}     # Query -> search results
        self.cache_ttl = 0         # TEMPORARILY DISABLED for testing real RAG
        self.max_cache_size = 2000  # Doubled cache size for better hit rate
        
        # Pre-computed common query embeddings
        self.common_queries_cache = {}
        self.query_pattern_cache = {}  # Pattern-based caching for similar queries
        
        self._initialize_model()
        self._initialize_text_splitter()
    
    def _generate_cache_key(self, query: str, limit: int = 5, filters: dict = None) -> str:
        """Generate optimized cache key with pattern matching for better hit rate."""
        # Normalize query for better cache hits
        normalized_query = query.lower().strip()
        
        # Check for common query patterns to increase cache hits
        query_pattern = self._extract_query_pattern(normalized_query)
        
        cache_data = {
            "query": normalized_query,
            "pattern": query_pattern,
            "limit": limit,
            "filters": filters or {}
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _extract_query_pattern(self, query: str) -> str:
        """Extract query pattern for better cache matching."""
        # Simple pattern extraction - remove specific names/numbers for broader matching
        import re
        pattern = re.sub(r'\b[A-Z][a-z]+\b', 'NAME', query)  # Replace proper nouns
        pattern = re.sub(r'\b\d+\b', 'NUM', pattern)  # Replace numbers
        return pattern[:50]  # Limit pattern length
    
    def _is_cache_valid(self, cache_entry: dict) -> bool:
        """Check if cache entry is still valid."""
        if not cache_entry:
            return False
        return time.time() - cache_entry.get("timestamp", 0) < self.cache_ttl
    
    def _clean_cache(self):
        """Clean expired entries and maintain cache size limit."""
        current_time = time.time()
        
        # Remove expired entries
        for cache in [self.embedding_cache, self.search_cache]:
            expired_keys = [
                key for key, entry in cache.items()
                if current_time - entry.get("timestamp", 0) > self.cache_ttl
            ]
            for key in expired_keys:
                del cache[key]
        
        # Maintain size limits
        for cache in [self.embedding_cache, self.search_cache]:
            if len(cache) > self.max_cache_size:
                # Remove oldest entries
                sorted_items = sorted(
                    cache.items(),
                    key=lambda x: x[1].get("timestamp", 0)
                )
                items_to_remove = len(cache) - self.max_cache_size
                for key, _ in sorted_items[:items_to_remove]:
                    del cache[key]
    
    def _initialize_model(self):
        """Initialize the LangChain OpenAI embeddings with local fallback."""
        try:
            if settings.openai_api_key:
                self.embedding_model = OpenAIEmbeddings(
                    openai_api_key=settings.openai_api_key,
                    model=settings.embedding_model_name
                )
                logger.info("Initialized LangChain OpenAI embeddings")
            else:
                logger.warning("OpenAI API key not found - using local embedding fallback")
                self.embedding_model = None
                
            # Initialize local embedding service as fallback
            if local_embedding_service.is_available():
                logger.info("Local embedding fallback available")
            else:
                logger.warning("Local embedding fallback not available")
                
        except Exception as e:
            logger.error(f"Error initializing LangChain embeddings: {e}")
            logger.info("Falling back to local embeddings")
            self.embedding_model = None
    
    def _initialize_text_splitter(self):
        """Initialize optimized semantic text splitter for document processing."""
        try:
            # Optimized semantic chunking for better performance
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Restored to original size for better performance
                chunk_overlap=200,  # Restored to original overlap
                length_function=len,
                separators=[
                    "\n\n",  # Paragraph breaks (primary separator)
                    "\n",  # Line breaks
                    ". ",  # Sentence endings
                    " ",  # Word boundaries
                    ""  # Character boundaries
                ],
                keep_separator=False,  # Disabled for performance
                add_start_index=False  # Disabled for performance
            )
            logger.info("Initialized optimized semantic text splitter")
        except Exception as e:
            logger.error(f"Error initializing semantic text splitter: {e}")
            self.text_splitter = None
    
    
    async def initialize(self):
        """Initialize the retrieval service for faster response."""
        try:
            logger.info("Initializing hybrid retrieval service...")
            
            # Initialize embeddings
            self._initialize_model()
            
            # Initialize text splitter
            self._initialize_text_splitter()
            
            # Initialize database connection
            await self._get_collection()
            
            logger.info("✅ Hybrid retrieval service initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize retrieval service: {e}")
            raise
    
    async def _get_collection(self):
        """Get MongoDB collection using pre-initialized connection."""
        # Use the global database connection from app startup
        from app.db import get_database
        db = await get_database()
        
        if db is None:
            logger.error("Database connection is None - connection not established")
            raise Exception("Database connection not available")
        
        collection = db[settings.mongo_collection]
        return collection
    
    async def _check_connection(self):
        """Check if MongoDB connection is available."""
        try:
            from app.db import get_database
            db = await get_database()
            if db is None:
                return "Database is None"
            
            # Try to ping the database
            await db.command("ping")
            return "Connected"
        except Exception as e:
            return f"Connection failed: {e}"
    
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
                logger.info(f"Generated embedding with {len(embedding)} dimensions")
                return embedding
            
            else:
                logger.warning("No OpenAI API key - using dummy embeddings")
                return [0.0] * 1536  # OpenAI embedding dimension
                
        except Exception as e:
            logger.error(f"Error computing embedding: {e}")
            return [0.0] * 1536
    
    async def add_document(self, document: KnowledgeDocument) -> str:
        """Add a document to the knowledge base using LangChain processing."""
        try:
            collection = await self._get_collection()
            
            # Convert to LangChain Document
            langchain_doc = Document(
                page_content=document.content,
                metadata={
                    "title": document.title,
                    "category": document.category,
                    "tags": document.tags,
                    "created_at": document.created_at,
                    "updated_at": document.updated_at
                }
            )
            
            # Split document using LangChain text splitter
            if self.text_splitter:
                split_docs = self.text_splitter.split_documents([langchain_doc])
                logger.info(f"Split document into {len(split_docs)} chunks")
            else:
                split_docs = [langchain_doc]
            
            # Process each chunk
            document_ids = []
            for i, chunk in enumerate(split_docs):
                # Compute embedding using LangChain embeddings or local fallback
                if self.embedding_model:
                    try:
                        embedding = await self.embedding_model.aembed_query(chunk.page_content)
                        logger.debug(f"Generated OpenAI embedding for chunk {i}")
                    except Exception as e:
                        logger.warning(f"OpenAI embedding failed, using local fallback: {e}")
                        embedding = local_embedding_service.embed_query(chunk.page_content)
                        logger.info(f"Generated local embedding for chunk {i}")
                else:
                    # Use local embedding fallback
                    embedding = local_embedding_service.embed_query(chunk.page_content)
                    logger.info(f"Generated local embedding for chunk {i}")
                
                # Create document dict with embedding
                doc_dict = {
                    "title": chunk.metadata.get("title", document.title),
                    "content": chunk.page_content,
                    "category": chunk.metadata.get("category", document.category),
                    "tags": chunk.metadata.get("tags", document.tags),
                    "created_at": chunk.metadata.get("created_at", document.created_at),
                    "updated_at": chunk.metadata.get("updated_at", document.updated_at),
                    "chunk_index": i,
                    "total_chunks": len(split_docs),
                    "embedding": embedding
                }
                
                # Insert document (using sync MongoDB)
                result = collection.insert_one(doc_dict)
                document_ids.append(str(result.inserted_id))
            
            logger.info(f"Added document to knowledge base using LangChain: {len(document_ids)} chunks")
            return document_ids[0] if document_ids else ""
            
        except Exception as e:
            logger.error(f"Error adding document with LangChain: {e}")
            raise
    
    async def vector_search(
        self, 
        query: str, 
        limit: int = 10,
        score_threshold: float = 0.0
    ) -> List[SearchResult]:
        """Perform vector similarity search using LangChain embeddings with caching."""
        try:
            collection = await self._get_collection()
            
            # Check embedding cache first
            embedding_cache_key = f"embedding_{hashlib.md5(query.encode()).hexdigest()}"
            cached_embedding = self.embedding_cache.get(embedding_cache_key)
            
            if cached_embedding and self._is_cache_valid(cached_embedding):
                query_embedding = cached_embedding["embedding"]
                logger.info(f"Embedding cache hit for query: '{query[:50]}'")
            else:
                # Compute query embedding using LangChain or local fallback
                if self.embedding_model:
                    try:
                        query_embedding = await self.embedding_model.aembed_query(query)
                        logger.info(f"Query embedding computed using LangChain, dimensions: {len(query_embedding)}")
                    except Exception as e:
                        logger.warning(f"OpenAI embedding failed, using local fallback: {e}")
                        query_embedding = local_embedding_service.embed_query(query)
                        logger.info(f"Query embedding computed using local fallback, dimensions: {len(query_embedding)}")
                else:
                    # Use local embedding fallback
                    query_embedding = local_embedding_service.embed_query(query)
                    logger.info(f"Query embedding computed using local fallback, dimensions: {len(query_embedding)}")
            
                # Cache the embedding
                self.embedding_cache[embedding_cache_key] = {
                    "embedding": query_embedding,
                    "timestamp": time.time()
                }
            
            # Use MongoDB Atlas vector search if available, otherwise cosine similarity
            logger.info("Using MongoDB Atlas vector search with real embeddings")
            
            # Try MongoDB Atlas vector search first
            try:
                # Use MongoDB Atlas vector search
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
                
                vector_results = []
                async for doc in collection.aggregate(pipeline):
                    vector_results.append(doc)
                logger.info(f"MongoDB Atlas vector search returned {len(vector_results)} results")
                
                if vector_results:
                    # Convert to SearchResult objects
                    search_results = []
                    for i, doc in enumerate(vector_results):
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
                                score=doc.get('score', 0.8),
                                rank=i + 1
                            ))
                        except Exception as e:
                            logger.error(f"Error processing vector document {doc.get('_id', 'Unknown')}: {e}")
                            continue
                    
                    logger.info(f"Vector search returned {len(search_results)} results using MongoDB Atlas")
                    return search_results
                
            except Exception as vector_error:
                logger.warning(f"MongoDB Atlas vector search failed: {vector_error}, falling back to cosine similarity")
            
            # Fallback to cosine similarity search
            logger.info("Using cosine similarity search with real embeddings")
            
            # Get all documents with embeddings (using sync MongoDB)
            all_docs = list(collection.find({"embedding": {"$exists": True}}))
            
            # Check Vector Search - Validate embeddings are loaded
            if all_docs is None:
                logger.error("Vector search returned None - vectors are not loaded!")
                return []
            
            if len(all_docs) == 0:
                logger.warning("No documents with embeddings found - returning empty results")
                return []
            
            logger.info(f"Found {len(all_docs)} documents with embeddings for cosine similarity search")
            
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
                    if similarity >= max(score_threshold, 0.001):
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
            
            logger.info(f"Vector search returned {len(search_results)} results using LangChain")
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
        """Perform proper BM25-like text search using MongoDB text index."""
        try:
            collection = await self._get_collection()
            
            # First try MongoDB text search (proper BM25-like scoring)
            try:
                # Use MongoDB's text search with textScore for BM25-like ranking
                cursor = collection.find(
                    {"$text": {"$search": query}},
                    {"score": {"$meta": "textScore"}}
                ).sort([("score", {"$meta": "textScore"})]).limit(limit)
                
                results = []
                async for doc in cursor:
                    results.append(doc)
                
                if results:
                    logger.info(f"MongoDB text search returned {len(results)} results with BM25-like scoring")
                else:
                    logger.warning("MongoDB text search returned no results, falling back to regex")
                    # Fallback to regex search if text search returns nothing
                    cursor = collection.find(
                {
                    "$or": [
                        {"title": {"$regex": query, "$options": "i"}},
                        {"content": {"$regex": query, "$options": "i"}},
                        {"category": {"$regex": query, "$options": "i"}}
                    ]
                }
                    ).limit(limit)
                    
                    results = []
                    async for doc in cursor:
                        results.append(doc)
                    logger.info(f"Regex fallback returned {len(results)} results")
                    
            except Exception as text_search_error:
                logger.warning(f"Text search failed: {text_search_error}, using regex fallback")
                # Fallback to regex search if text index doesn't exist
                cursor = collection.find(
                    {
                        "$or": [
                            {"title": {"$regex": query, "$options": "i"}},
                            {"content": {"$regex": query, "$options": "i"}},
                            {"category": {"$regex": query, "$options": "i"}}
                        ]
                    }
                ).limit(limit)
                
                results = []
                async for doc in cursor:
                    results.append(doc)
            
            # Validate results
            if results is None:
                logger.error("BM25 search returned None - document list is not loaded!")
                return []
            
            if len(results) == 0:
                logger.warning("No documents found for BM25 search")
                return []
            
            logger.info(f"BM25 search found {len(results)} documents")
            
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
                    
                    # Use textScore if available (from MongoDB text search), otherwise position-based
                    if 'score' in doc:
                        score = float(doc['score'])  # MongoDB textScore (BM25-like)
                        logger.debug(f"Document '{doc.get('title', 'Unknown')}' BM25 score: {score:.4f}")
                    else:
                        score = 1.0 / (i + 1)  # Fallback position-based scoring
                    
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
    
    
    def _get_adaptive_weights(self, query: str) -> tuple[float, float]:
        """Get adaptive weights based on query characteristics."""
        query_lower = query.lower()
        
        # Technical/semantic queries favor vector search
        technical_terms = ['python', 'machine learning', 'ai', 'data science', 'neural', 'algorithm', 'model']
        if any(term in query_lower for term in technical_terms):
            return 0.9, 0.1  # Heavy vector bias for semantic understanding
        
        # Exact name/title queries favor BM25
        if any(char.isupper() for char in query) or '"' in query:
            return 0.3, 0.7  # Heavy BM25 bias for exact matches
        
        # Balanced for general queries
        return 0.7, 0.3
    
    def _reciprocal_rank_fusion(self, vector_results: List[SearchResult], bm25_results: List[SearchResult], k: int = 60) -> List[SearchResult]:
        """Implement Reciprocal Rank Fusion for combining search results."""
        try:
            # RRF formula: score = 1/(k + rank)
            combined_scores = {}
            
            # Add vector search scores
            for rank, result in enumerate(vector_results):
                doc_id = result.document.id
                rrf_score = 1.0 / (k + rank + 1)
                combined_scores[doc_id] = {
                    'document': result.document,
                    'vector_rank': rank + 1,
                    'bm25_rank': None,
                    'rrf_score': rrf_score,
                    'original_vector_score': result.score
                }
            
            # Add BM25 scores
            for rank, result in enumerate(bm25_results):
                doc_id = result.document.id
                rrf_score = 1.0 / (k + rank + 1)
                
                if doc_id in combined_scores:
                    combined_scores[doc_id]['rrf_score'] += rrf_score
                    combined_scores[doc_id]['bm25_rank'] = rank + 1
                    combined_scores[doc_id]['original_bm25_score'] = result.score
                else:
                    combined_scores[doc_id] = {
                        'document': result.document,
                        'vector_rank': None,
                        'bm25_rank': rank + 1,
                        'rrf_score': rrf_score,
                        'original_bm25_score': result.score
                    }
            
            # Sort by RRF score and convert back to SearchResult
            sorted_results = sorted(combined_scores.values(), key=lambda x: x['rrf_score'], reverse=True)
            
            search_results = []
            for i, result_data in enumerate(sorted_results):
                search_results.append(SearchResult(
                    document=result_data['document'],
                    score=result_data['rrf_score'],
                    rank=i + 1
                ))
            
            logger.info(f"RRF fusion combined {len(vector_results)} vector + {len(bm25_results)} BM25 results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in RRF fusion: {e}")
            # Fallback to simple concatenation
            return vector_results + bm25_results
    
    async def hybrid_search(
        self, 
        query: str, 
        limit: int = 3,  # Reduced to 3 for sub-2.5s performance
        fusion_method: str = "rrf",  # "rrf", "weighted", or "adaptive"
        vector_weight: float = None,  # Will be auto-determined if None
        bm25_weight: float = None
    ) -> List[SearchResult]:
        """Perform hybrid search combining vector and BM25 with caching."""
        # Check cache first
        cache_key = self._generate_cache_key(query, limit)
        cached_result = self.search_cache.get(cache_key)
        
        if cached_result and self._is_cache_valid(cached_result):
            logger.info(f"Cache hit for query: '{query[:50]}'")
            performance_monitor.record_step("hybrid_search", "cache_hit", 0.0)
            return cached_result["results"]
        
        # Start performance trace
        trace_id = performance_monitor.start_trace("hybrid_search")
        
        try:
            # Determine fusion method and weights
            if fusion_method == "adaptive" or (vector_weight is None and bm25_weight is None):
                vector_weight, bm25_weight = self._get_adaptive_weights(query)
                logger.info(f"Using adaptive weights: vector={vector_weight:.2f}, bm25={bm25_weight:.2f}")
            elif vector_weight is None or bm25_weight is None:
                vector_weight, bm25_weight = 0.8, 0.2  # Default weights
            
            # Run both searches in parallel - optimized for single email P95
            search_start = time.time()
            vector_task = self.vector_search(query, 20)  # Optimized for P95 single email generation
            bm25_task = self.bm25_search(query, 20)  # Optimized for P95 single email generation
            
            vector_results, bm25_results = await asyncio.gather(
                vector_task, bm25_task
            )
            
            search_duration = (time.time() - search_start) * 1000
            performance_monitor.record_step(trace_id, "parallel_search", search_duration)
            
            # Apply fusion method
            combine_start = time.time()
            
            if fusion_method == "rrf":
                # Use Reciprocal Rank Fusion
                search_results = self._reciprocal_rank_fusion(vector_results, bm25_results)
                search_results = search_results[:limit]  # Limit after fusion
                logger.info(f"Applied RRF fusion to {len(vector_results)} + {len(bm25_results)} results")
                
            else:
                # Use traditional weighted combination
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
                
                # Convert to SearchResult objects
                search_results = []
                for i, result in enumerate(sorted_results):
                    search_results.append(SearchResult(
                        document=result['document'],
                        score=result['combined_score'],
                        rank=i + 1
                    ))
                
                logger.info(f"Applied weighted fusion ({fusion_method}) with weights v={vector_weight:.2f}, b={bm25_weight:.2f}")
            
            combine_duration = (time.time() - combine_start) * 1000
            performance_monitor.record_step(trace_id, "combine_results", combine_duration)
            
            logger.info(
                "Hybrid search completed",
                trace_id=trace_id,
                query_length=len(query),
                results_count=len(search_results),
                vector_weight=vector_weight,
                bm25_weight=bm25_weight
            )
            
            # Apply cross-encoder reranking with timeout protection
            if settings.enable_reranking and len(search_results) > 1:  # Rerank if we have more than 1 candidate
                rerank_start = time.time()
                logger.info(f"Applying cross-encoder reranking on {len(search_results)} hybrid results")
                try:
                    # Add timeout to prevent hanging
                    search_results = await asyncio.wait_for(
                        self.rerank_results(
                            query=query,
                            results=search_results,
                            top_k=limit,
                            alpha=settings.rerank_alpha
                        ),
                        timeout=15.0  # 15 second timeout for entire reranking
                    )
                    rerank_duration = (time.time() - rerank_start) * 1000
                    performance_monitor.record_step(trace_id, "reranking", rerank_duration)
                    logger.info(f"Cross-encoder reranking completed in {rerank_duration:.2f}ms")
                except asyncio.TimeoutError:
                    logger.warning("Cross-encoder reranking timed out, using original results")
                    search_results = search_results[:limit]
                    rerank_duration = (time.time() - rerank_start) * 1000
                    performance_monitor.record_step(trace_id, "reranking_timeout", rerank_duration)
                except Exception as e:
                    logger.error(f"Cross-encoder reranking failed: {e}, using original results")
                    search_results = search_results[:limit]
                    rerank_duration = (time.time() - rerank_start) * 1000
                    performance_monitor.record_step(trace_id, "reranking_error", rerank_duration)
            else:
                search_results = search_results[:limit]
                logger.info(f"Skipping reranking: enable_reranking={settings.enable_reranking}, candidates={len(search_results)}")
            
            # Cache the results
            self.search_cache[cache_key] = {
                "results": search_results,
                "timestamp": time.time()
            }
            
            # Clean cache periodically
            if len(self.search_cache) % 10 == 0:
                self._clean_cache()
            
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
        """Rerank results using cross-encoder reranker."""
        try:
            if len(results) <= top_k:
                return results
            
            logger.info(f"Using cross-encoder reranker for {len(results)} results")
            
            # Use the cross-encoder reranker
            reranked_results = await cross_encoder_reranker.rerank_results(
                query=query,
                results=results,
                top_k=top_k,
                alpha=alpha
            )
            
            logger.info(f"Cross-encoder reranking completed. Returned {len(reranked_results)} results")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {e}")
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
        trace_id = performance_monitor.start_trace("fast_search")
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
        
        performance_monitor.finish_trace(trace_id, status_code=200)
        return search_results
    
    async def _vector_search_only(self, query: str, limit: int) -> List[SearchResult]:
        """Ultra-fast vector search using real embeddings."""
        try:
            # Use actual vector search with real embeddings
            return await self.vector_search(query, limit, score_threshold=0.1)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            # Fallback to cached response only if vector search fails
        return self._get_cached_response(query)
    
    async def retrieve_documents(self, query: str, limit: int = 5, filters: dict = None) -> List[KnowledgeDocument]:
        """Retrieve documents for email generation context."""
        try:
            logger.info(f"Retrieving documents for query: '{query[:50]}'")
            
            # Use hybrid search to get relevant documents
            search_results = await self.hybrid_search(query, limit=limit)
            
            # Convert SearchResult objects to KnowledgeDocument objects
            documents = []
            for result in search_results:
                documents.append(result.document)
            
            logger.info(f"Retrieved {len(documents)} documents for context")
            return documents
            
        except Exception as e:
            logger.error(f"Error in retrieve_documents: {e}")
            # Fallback to cached course documents
            return self._get_course_documents(query)[:limit]
    
    def _create_semantic_chunks(self, text: str, title: str = "") -> List[Dict[str, Any]]:
        """Create semantic chunks with section paths and heading preservation."""
        try:
            if not self.text_splitter:
                # Fallback to simple splitting
                chunks = [{"content": text, "section_path": title, "heading": title}]
                return chunks
            
            # Split text into semantic chunks
            doc_chunks = self.text_splitter.split_text(text)
            
            # Create semantic chunks with metadata
            semantic_chunks = []
            current_section = title
            current_heading = title
            
            for i, chunk in enumerate(doc_chunks):
                # Extract heading from chunk if present
                heading = self._extract_heading(chunk)
                if heading:
                    current_heading = heading
                    current_section = f"{title} > {heading}" if title else heading
                
                # Create chunk metadata
                chunk_metadata = {
                    "content": chunk.strip(),
                    "section_path": current_section,
                    "heading": current_heading,
                    "chunk_index": i,
                    "total_chunks": len(doc_chunks),
                    "start_index": getattr(self.text_splitter, 'start_index', 0) if hasattr(self.text_splitter, 'start_index') else 0
                }
                
                semantic_chunks.append(chunk_metadata)
            
            logger.info(f"Created {len(semantic_chunks)} semantic chunks for '{title}'")
            return semantic_chunks
            
        except Exception as e:
            logger.error(f"Error creating semantic chunks: {e}")
            return [{"content": text, "section_path": title, "heading": title}]
    
    def _extract_heading(self, text: str) -> Optional[str]:
        """Extract heading from text chunk."""
        try:
            lines = text.strip().split('\n')
            for line in lines[:3]:  # Check first 3 lines for headings
                line = line.strip()
                # Check for markdown headers
                if line.startswith('## '):
                    return line[3:].strip()
                elif line.startswith('### '):
                    return line[4:].strip()
                elif line.startswith('#### '):
                    return line[5:].strip()
                # Check for other heading patterns
                elif line and not line.startswith(' ') and len(line) < 100:
                    return line
            return None
        except Exception as e:
            logger.error(f"Error extracting heading: {e}")
            return None
    
    def _get_cached_response(self, query: str) -> List[SearchResult]:
        """Return empty results - no fake cached responses."""
        logger.warning(f"No real documents found for query: '{query[:50]}' - returning empty results")
        return []
    
    def _get_course_documents(self, query: str) -> List[KnowledgeDocument]:
        """No fake course documents - return empty list."""
        logger.warning(f"Fake course document generation disabled for query: '{query[:50]}'")
        return []

    def _parse_query_context(self, query: str) -> Dict[str, Any]:
        """Parse query to extract lead context information."""
        query_lower = query.lower()
        
        # Extract role patterns
        role_patterns = {
            'data analyst': ['data analyst', 'analyst', 'data'],
            'project manager': ['project manager', 'manager', 'project'],
            'healthcare admin': ['healthcare admin', 'healthcare', 'medical'],
            'software engineer': ['software engineer', 'engineer', 'developer'],
            'business analyst': ['business analyst', 'business'],
            'product manager': ['product manager', 'product'],
            'sales representative': ['sales rep', 'sales', 'representative']
        }
        
        role = 'Professional'
        for role_name, patterns in role_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                role = role_name.title()
                break
        
        # Extract campaign patterns
        campaign_patterns = {
            'early_bird': ['early bird', 'early'],
            'tech_conference': ['tech conference', 'conference'],
            'product_launch': ['product launch', 'launch'],
            'winter_campaign': ['winter', 'winter campaign'],
            'spring_promotion': ['spring', 'spring promotion'],
            'summer_bootcamp': ['summer', 'bootcamp'],
            'fall_launch': ['fall', 'autumn']
        }
        
        campaign = 'general'
        for campaign_name, patterns in campaign_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                campaign = campaign_name
                break
        
        # Extract keywords (simple word extraction)
        keywords = []
        common_words = {'course', 'program', 'training', 'certification', 'for', 'and', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'of', 'with', 'by'}
        words = query_lower.split()
        keywords = [word for word in words if word not in common_words and len(word) > 2][:5]
        
        return {
            'role': role,
            'campaign': campaign,
            'keywords': keywords
        }


# Global retrieval instance - pre-initialized at startup
retrieval = None

def get_retrieval():
    """Get the global retrieval instance."""
    global retrieval
    if retrieval is None:
        logger.info("Initializing HybridRetrieval instance...")
        retrieval = HybridRetrieval()
        logger.info("✅ HybridRetrieval instance created")
    return retrieval

def initialize_retrieval():
    """Pre-initialize the retrieval service at startup."""
    global retrieval
    if retrieval is None:
        logger.info("Pre-initializing HybridRetrieval service...")
        retrieval = HybridRetrieval()
        logger.info("✅ HybridRetrieval service pre-initialized")
    return retrieval

