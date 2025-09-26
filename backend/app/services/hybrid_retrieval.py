"""
Hybrid retrieval system combining BM25 keyword search with vector search.
"""
import math
import re
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from app.utils.logging import get_logger, performance_logger

logger = get_logger(__name__)

class BM25Retriever:
    """BM25 keyword-based retrieval system."""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[str] = []
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        
    def add_documents(self, documents: List[str]):
        """Add documents to the BM25 index."""
        self.documents = documents
        self.doc_len = [len(doc.split()) for doc in documents]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
        
        # Calculate document frequencies
        self.doc_freqs = []
        df = defaultdict(int)
        
        for doc in documents:
            words = self._tokenize(doc)
            word_count = Counter(words)
            self.doc_freqs.append(word_count)
            
            for word in word_count:
                df[word] += 1
        
        # Calculate IDF scores
        N = len(documents)
        self.idf = {word: math.log(N / freq) for word, freq in df.items()}
        
        logger.info(f"BM25 index built with {N} documents, {len(self.idf)} unique terms")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Convert to lowercase and split on whitespace and punctuation
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search for documents using BM25 scoring."""
        if not self.documents:
            return []
        
        query_words = self._tokenize(query)
        scores = []
        
        for i, doc in enumerate(self.documents):
            score = 0
            doc_len = self.doc_len[i]
            
            for word in query_words:
                if word in self.doc_freqs[i]:
                    tf = self.doc_freqs[i][word]
                    idf = self.idf.get(word, 0)
                    
                    # BM25 formula
                    score += idf * (tf * (self.k1 + 1)) / (
                        tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                    )
            
            scores.append((i, score))
        
        # Sort by score and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

class HybridRetrieval:
    """Hybrid retrieval combining BM25 and vector search."""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.bm25_retriever = BM25Retriever()
        self.documents: List[str] = []
        self.document_embeddings: Optional[np.ndarray] = None
        self.alpha = 0.7  # Weight for BM25 (1-alpha for vector)
        
    def add_documents(self, documents: List[str]):
        """Add documents to both BM25 and vector indices."""
        self.documents = documents
        
        # Add to BM25
        self.bm25_retriever.add_documents(documents)
        
        # Generate embeddings for vector search
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        self.document_embeddings = self.embedding_model.encode(documents)
        
        logger.info(f"Hybrid retrieval index built with {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 10, alpha: Optional[float] = None) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Perform hybrid search combining BM25 and vector search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for BM25 (1-alpha for vector). If None, uses instance default.
            
        Returns:
            List of (doc_id, score, metadata) tuples
        """
        if alpha is None:
            alpha = self.alpha
        
        timer_id = performance_logger.start_timer("hybrid_search")
        
        try:
            # BM25 search
            bm25_results = self.bm25_retriever.search(query, top_k * 2)
            bm25_scores = {doc_id: score for doc_id, score in bm25_results}
            
            # Vector search
            query_embedding = self.embedding_model.encode([query])
            if self.document_embeddings is not None:
                # Calculate cosine similarities
                similarities = np.dot(self.document_embeddings, query_embedding.T).flatten()
                vector_scores = {i: float(sim) for i, sim in enumerate(similarities)}
            else:
                vector_scores = {}
            
            # Normalize scores
            bm25_max = max(bm25_scores.values()) if bm25_scores else 1.0
            vector_max = max(vector_scores.values()) if vector_scores else 1.0
            
            # Combine scores
            combined_scores = {}
            all_doc_ids = set(bm25_scores.keys()) | set(vector_scores.keys())
            
            for doc_id in all_doc_ids:
                bm25_score = bm25_scores.get(doc_id, 0) / bm25_max if bm25_max > 0 else 0
                vector_score = vector_scores.get(doc_id, 0) / vector_max if vector_max > 0 else 0
                
                combined_score = alpha * bm25_score + (1 - alpha) * vector_score
                combined_scores[doc_id] = combined_score
            
            # Sort and return top results
            results = []
            for doc_id, score in sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
                metadata = {
                    "bm25_score": bm25_scores.get(doc_id, 0),
                    "vector_score": vector_scores.get(doc_id, 0),
                    "combined_score": score,
                    "alpha": alpha
                }
                results.append((doc_id, score, metadata))
            
            performance_logger.end_timer(
                timer_id, 
                "hybrid_search",
                query_length=len(query),
                results_count=len(results),
                alpha=alpha
            )
            
            return results
            
        except Exception as e:
            performance_logger.end_timer(timer_id, "hybrid_search", error=str(e))
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def search_bm25_only(self, query: str, top_k: int = 10) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Search using only BM25."""
        timer_id = performance_logger.start_timer("bm25_search")
        
        try:
            bm25_results = self.bm25_retriever.search(query, top_k)
            results = []
            
            for doc_id, score in bm25_results:
                metadata = {
                    "bm25_score": score,
                    "vector_score": 0,
                    "combined_score": score,
                    "alpha": 1.0
                }
                results.append((doc_id, score, metadata))
            
            performance_logger.end_timer(timer_id, "bm25_search", results_count=len(results))
            return results
            
        except Exception as e:
            performance_logger.end_timer(timer_id, "bm25_search", error=str(e))
            logger.error(f"Error in BM25 search: {e}")
            return []
    
    def search_vector_only(self, query: str, top_k: int = 10) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Search using only vector similarity."""
        timer_id = performance_logger.start_timer("vector_search")
        
        try:
            if self.document_embeddings is None:
                return []
            
            query_embedding = self.embedding_model.encode([query])
            similarities = np.dot(self.document_embeddings, query_embedding.T).flatten()
            
            # Get top_k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for doc_id in top_indices:
                score = float(similarities[doc_id])
                metadata = {
                    "bm25_score": 0,
                    "vector_score": score,
                    "combined_score": score,
                    "alpha": 0.0
                }
                results.append((int(doc_id), score, metadata))
            
            performance_logger.end_timer(timer_id, "vector_search", results_count=len(results))
            return results
            
        except Exception as e:
            performance_logger.end_timer(timer_id, "vector_search", error=str(e))
            logger.error(f"Error in vector search: {e}")
            return []
    
    def get_document(self, doc_id: int) -> Optional[str]:
        """Get document by ID."""
        if 0 <= doc_id < len(self.documents):
            return self.documents[doc_id]
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval system statistics."""
        return {
            "total_documents": len(self.documents),
            "embedding_dimension": self.document_embeddings.shape[1] if self.document_embeddings is not None else 0,
            "bm25_terms": len(self.bm25_retriever.idf),
            "alpha": self.alpha,
            "avg_document_length": self.bm25_retriever.avgdl
        }

# Global hybrid retrieval instance
hybrid_retrieval = HybridRetrieval()
