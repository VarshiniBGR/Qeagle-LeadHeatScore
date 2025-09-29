from pydantic_settings import BaseSettings
from typing import Dict, List
import os


class Settings(BaseSettings):
    # MongoDB Configuration
    mongo_uri: str = "mongodb+srv://padmavarshinib_db_user:Padma123@cluster0.ptelg28.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    mongo_db: str = "leadheat"  # Fixed to match actual database
    mongo_collection: str = "vectors"
    mongo_vector_index: str = "vector_index"
    
    # ML Configuration
    embedding_model_name: str = "text-embedding-3-small"  # Latest OpenAI embedding model (80% cheaper)
    model_dir: str = "./backend/models"
    class_thresholds: str = "hot:0.8,warm:0.5"
    
    # AI/LLM Configuration - OpenAI Only (Optimized)
    openai_api_key: str = "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"
    vector_backend: str = "mongo"
    
    # OpenAI Model Configuration (Primary)
    llm_model: str = "gpt-4o-mini"  # Cost-effective, high-quality
    llm_temperature: float = 0.3
    llm_max_tokens: int = 100  # Ultra-fast RAG generation
    
    # RAG Email Configuration - OpenAI Mode
    enable_rag_emails: bool = True
    rag_email_fallback: bool = False  # Disabled for premium quality
    rag_email_timeout: int = 15  # Reduced timeout for faster responses
    
    # Telegram Bot Configuration
    telegram_bot_token: str = ""
    enable_telegram_messages: bool = True
    telegram_message_timeout: int = 30  # seconds
    
    # Cross-Encoder Reranker Configuration
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
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
    
    # API Configuration
    api_title: str = "Lead HeatScore API"
    api_version: str = "1.0.0"
    debug: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        protected_namespaces = ('settings_',)
    
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
