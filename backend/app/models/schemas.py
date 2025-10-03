from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class HeatScore(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


class Channel(str, Enum):
    EMAIL = "email"
    NEWSLETTER = "newsletter"


class LeadInput(BaseModel):
    """Input schema for lead data."""
    name: Optional[str] = Field(None, description="Lead name")
    email: Optional[str] = Field(None, description="Lead email")
    phone: Optional[str] = Field(None, description="Lead phone number")
    source: str = Field(..., description="Lead source (website, linkedin, referral, etc.)")
    recency_days: int = Field(..., ge=0, description="Days since last interaction")
    region: str = Field(..., description="Geographic region")
    role: str = Field(..., description="Job role/title")
    campaign: str = Field(..., description="Marketing campaign")
    page_views: int = Field(..., ge=0, description="Number of page views")
    last_touch: str = Field(..., description="Last touchpoint")
    prior_course_interest: str = Field(..., description="Previous course interest level")
    search_keywords: Optional[str] = Field(None, description="Search keywords used")
    time_spent: Optional[int] = Field(None, ge=0, description="Time spent on pages (seconds)")
    course_actions: Optional[str] = Field(None, description="Actions taken (download, demo_request, etc.)")
    
    @validator('prior_course_interest')
    def validate_interest(cls, v):
        valid_levels = ['high', 'medium', 'low', 'none']
        if v.lower() not in valid_levels:
            raise ValueError(f'prior_course_interest must be one of {valid_levels}')
        return v.lower()


class LeadScore(BaseModel):
    """Lead classification result."""
    lead_id: str
    heat_score: HeatScore
    confidence: float = Field(..., ge=0, le=1)
    probabilities: Dict[str, float] = Field(..., description="Probability for each class")
    features_importance: Dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Recommendation(BaseModel):
    """Next action recommendation."""
    lead_id: str
    recommended_channel: Channel
    message_content: str
    rationale: str
    citations: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0, le=1)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class LeadResult(BaseModel):
    """Complete lead analysis result."""
    lead_id: str
    lead_data: LeadInput
    score: LeadScore
    recommendation: Optional[Recommendation] = None


class BatchResult(BaseModel):
    """Batch processing result."""
    total_leads: int
    processed_leads: int
    failed_leads: int
    results: List[LeadResult]
    processing_time: float
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UploadResponse(BaseModel):
    """File upload response."""
    filename: str
    total_rows: int
    valid_rows: int
    invalid_rows: int
    batch_id: str
    message: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str
    database_status: str
    ml_model_status: str


class MetricsResponse(BaseModel):
    """Model performance metrics."""
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    confusion_matrix: List[List[int]]
    calibration_score: float
    last_updated: datetime


class TextChunk(BaseModel):
    """Schema for text chunks."""
    chunk_id: str
    source_id: str
    chunk_index: int
    total_chunks: int
    text: str
    start_pos: int
    end_pos: int
    token_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeDocument(BaseModel):
    """Knowledge base document."""
    id: str
    title: str
    content: str
    category: str
    tags: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SearchResult(BaseModel):
    """Vector search result."""
    document: KnowledgeDocument
    score: float
    rank: int


class ABTestRequest(BaseModel):
    """Request schema for A/B testing."""
    lead_data: LeadInput
    test_name: Optional[str] = Field("template_vs_rag", description="Name of the A/B test")
    include_metrics: bool = Field(True, description="Include performance metrics")


class ABTestResult(BaseModel):
    """A/B test comparison result."""
    test_name: str
    lead_id: str
    template_email: Dict[str, Any] = Field(..., description="Template-based email result")
    rag_email: Dict[str, Any] = Field(..., description="RAG-generated email result")
    comparison_metrics: Dict[str, Any] = Field(default_factory=dict)
    recommendation: str = Field(..., description="Which approach performed better")
    timestamp: datetime = Field(default_factory=datetime.now)


class ABTestMetrics(BaseModel):
    """Metrics for A/B testing comparison."""
    template_score: float = Field(..., description="Template email quality score (1-5)")
    rag_score: float = Field(..., description="RAG email quality score (1-5)")
    template_latency_ms: float = Field(..., description="Template generation latency")
    rag_latency_ms: float = Field(..., description="RAG generation latency")
    template_tokens: int = Field(..., description="Template email token count")
    rag_tokens: int = Field(..., description="RAG email token count")
    template_cost: float = Field(..., description="Template generation cost")
    rag_cost: float = Field(..., description="RAG generation cost")
    winner: str = Field(..., description="Which approach won (template/rag/tie)")


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
