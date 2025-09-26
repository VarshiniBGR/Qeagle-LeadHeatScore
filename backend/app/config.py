from pydantic_settings import BaseSettings
from typing import Dict, List
import os


class Settings(BaseSettings):
    # MongoDB Configuration
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "leadheat_fresh"
    mongo_collection: str = "vectors"
    mongo_vector_index: str = "vector_index"
    
    # ML Configuration
    embedding_model_name: str = "all-MiniLM-L6-v2"
    model_dir: str = "./models"
    class_thresholds: str = "hot:0.8,warm:0.5"
    
    # AI/LLM Configuration
    openai_api_key: str = ""
    vector_backend: str = "mongo"
    
    # LLM Model Configuration
    llm_provider: str = "auto"  # auto, openai, ollama, huggingface, fallback
    ollama_model: str = "llama2:7b"
    huggingface_model: str = "microsoft/DialoGPT-medium"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 500
    
    # RAG Email Configuration
    enable_rag_emails: bool = True
    rag_email_fallback: bool = True
    rag_email_timeout: int = 30  # seconds
    
    # Telegram Bot Configuration
    telegram_bot_token: str = ""
    enable_telegram_messages: bool = True
    telegram_message_timeout: int = 30  # seconds
    
    # WhatsApp Business API Configuration
    whatsapp_business_token: str = ""
    whatsapp_phone_number_id: str = ""
    enable_whatsapp_messages: bool = True
    whatsapp_message_timeout: int = 30  # seconds
    
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
