#!/usr/bin/env python3
"""
Local Embedding Service
Fallback embedding model using sentence-transformers/all-MiniLM-L6-v2
"""

import numpy as np
from typing import List, Optional
from app.utils.logging import get_logger
from app.config import settings

logger = get_logger(__name__)

class LocalEmbeddingService:
    """Local embedding service using sentence-transformers as fallback."""
    
    def __init__(self):
        self.model = None
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_dimension = 384  # all-MiniLM-L6-v2 dimension
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the local sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Initializing local embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"✅ Local embedding model loaded successfully")
            logger.info(f"   Model: {self.model_name}")
            logger.info(f"   Dimension: {self.embedding_dimension}")
            
        except ImportError:
            logger.warning("sentence-transformers not available - local embeddings disabled")
            self.model = None
        except Exception as e:
            logger.error(f"Error initializing local embedding model: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if local embedding model is available."""
        return self.model is not None
    
    async def initialize(self):
        """Initialize the local embedding model for faster response."""
        try:
            logger.info("Pre-loading local embedding model...")
            if not self.is_available():
                self._initialize_model()
            logger.info("✅ Local embedding model pre-loaded")
        except Exception as e:
            logger.error(f"❌ Failed to pre-load local embedding model: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text."""
        try:
            if not self.is_available():
                logger.warning("Local embedding model not available")
                return [0.0] * self.embedding_dimension
            
            # Generate embedding
            embedding = self.model.encode(text, convert_to_tensor=False)
            
            # Convert to list of floats
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            logger.debug(f"Generated local embedding for query: {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating local embedding: {e}")
            return [0.0] * self.embedding_dimension
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents."""
        try:
            if not self.is_available():
                logger.warning("Local embedding model not available")
                return [[0.0] * self.embedding_dimension] * len(texts)
            
            # Generate embeddings for all texts
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            
            # Convert to list of lists
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            logger.debug(f"Generated local embeddings for {len(texts)} documents")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating local document embeddings: {e}")
            return [[0.0] * self.embedding_dimension] * len(texts)
    
    def get_model_info(self) -> dict:
        """Get information about the local embedding model."""
        return {
            "model_name": self.model_name,
            "dimension": self.embedding_dimension,
            "available": self.is_available(),
            "type": "local_fallback"
        }

# Global service instance
local_embedding_service = LocalEmbeddingService()







