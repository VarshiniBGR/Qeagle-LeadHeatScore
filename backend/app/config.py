from pydantic_settings import BaseSettings
from typing import Dict, List
import os


class Settings(BaseSettings):
    # MongoDB Configuration
    mongo_uri: str = "mongodb://localhost:27017"  # Default local MongoDB
    mongo_db: str = "leadheat"
    mongo_collection: str = "vectors"
    mongo_vector_index: str = "vector_index"
    
    # ML Configuration
    embedding_model_name: str = "text-embedding-3-small"
    model_dir: str = "./backend/models"
    # Note: Classification thresholds are hardcoded in classifier.py (optimized for F1-macro)
    
    # AI/LLM Configuration - OpenAI Only (Optimized)
    openai_api_key: str = ""  # Set via environment variable OPENAI_API_KEY
    vector_backend: str = "mongo"
    
    # OpenAI Model Configuration (Primary)
    llm_model: str = "gpt-3.5-turbo"  # Fast, cost-effective for email generation
    llm_temperature: float = 0.3
    llm_max_tokens: int = 300  # Increased for complete email content
    
    # RAG Email Configuration - OpenAI Mode
    enable_rag_emails: bool = True
    rag_email_fallback: bool = False  # Disabled for premium quality
    rag_email_timeout: int = 15  # Increased timeout for reliability
    
    # Reranker Configuration
    rerank_alpha: float = 0.3  # Weight for rerank scores vs original scores
    enable_reranking: bool = True
    
    # Email Configuration
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    from_email: str = ""
    from_name: str = "Lead HeatScore Team"
    
    # Logging
    log_level: str = "info"
    
    # Cache Configuration
    cache_ttl: int = 7200  # 2 hours for maximum reuse
    cache_buffer_size: int = 1000  # Allow up to 1000 cached responses
    
    # API Configuration
    api_title: str = "Lead HeatScore API"  # Project name
    api_version: str = "1.0.0"
    debug: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        protected_namespaces = ('settings_',)
        extra = "ignore"
    
    def get_thresholds(self) -> Dict[str, float]:
        """Parse class thresholds from string format."""
        thresholds = {}
        for pair in self.class_thresholds.split(','):
            if ':' in pair:
                class_name, threshold = pair.split(':')
                thresholds[class_name.strip()] = float(threshold.strip())
        return thresholds
    
    def get_model_path(self, model_name: str) -> str:
        """Get full path to model file."""
        return os.path.join(self.model_dir, f"{model_name}.joblib")


settings = Settings()
