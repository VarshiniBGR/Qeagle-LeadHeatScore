# Lead HeatScore API Reference

We built this API to solve our own lead scoring problems at the company. After thousands of leads and countless campaigns, we realized we needed something smarter than traditional email templates.

The heart of this system is our custom RAG implementation that costs 91% less than typical approaches while maintaining intelligence. We've optimized every token because API costs add up fast when you're processing thousands of leads daily.

## Getting Started

First things first - make sure your server is running:

```bash
curl http://localhost:8000/health
```

If you see `{"status": "healthy"}`, you're good to go. If not, check that your backend is running on port 8000.

## Setup

You'll need these environment variables in your `.env` file:

```bash
# Required
OPENAI_API_KEY=sk-your-openai-key
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/
MONGO_DB=leadheat

# Optional
LOG_LEVEL=info
DEBUG=false
```

---

## The Main Endpoints

### Scoring Leads

This is where it all starts. We feed lead data into our ML model and get back a heat score:

```bash
curl -X POST http://localhost:8000/api/v1/score-lead \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Michael Chen",
    "email": "michael@startup.io",
    "role": "Director",
    "campaign": "AI Program", 
    "source": "website",
    "region": "US",
    "recency_days": 1,
    "page_views": 25,
    "time_spent": 420,
    "last_touch": "demo_request",
    "prior_course_interest": "high"
  }'
```

What you get back tells you how hot this lead is:

```json
{
  "lead_id": "abc123-def456-ghi789",
  "heat_score": " hot",
  "confidence": 0.92,
  "probabilities": {
    "cold": 0.02,
    "warm": 0.06, 
    "hot": 0.92
  },
  "features_used": [
    "page_views",
    "recency_days",
    "prior_course_interest",
    "time_spent"
  ]
}
```

The model is pretty confident (0.92) that Michael is a hot lead based on his recent activity, high page views, and interest level.

### 2. Get Recommendation

Generate AI-powered next action recommendation.

```http
POST /api/v1/recommendation
Content-Type: application/json
```

**Request Body:** *(Same as Score Lead)*

**Response:**
```json
{
  "lead_id": "uuid-1234-5678",
  "recommended_channel": "email",
  "message_content": "Hi John! Your 20 page views show strong interest...",
  "rationale": "Hot lead with high engagement - immediate action recommended",
  "confidence": 0.85,
  "next_steps": [
    "Send personalized RAG email within 24 hours",
    "Include urgency-based CTA",
    "Reference specific page visits"
  ]
}
```

---

## RAG Email Generation

### Generate Personalized Email

Create ultra-efficient RAG-powered email content.

```http
POST /api/v1/get-personalized-email
Content-Type: application/json
```

**Request Body:** *(Same as Score Lead)*

**Response:**
```json
{
  "subject": "🔥 URGENT: Only 2 Spots Left - 48hrs",
  "content": "Hi John!\n\nYour 15 page views show strong interest in our AI Program.\n\n🎯 EXCLUSIVE OFFER (48 Hours):\n• 30% discount\n• Only 2 spots left\n\nReply 'YES' to claim your spot!\n\nBest regards,\nLeadHeatScore Team",
  "type": "rag",
  "personalization_data": {
    "name": "John Doe",
    "role": "Developer", 
    "page_views": 15,
    \"campaign\": "AI Program"
  },
  "token_usage": {
    "input_tokens": 73,
    "output_tokens": 95,
    "total_cost": 0.0004
  }
}
```

### Send Email

Send the personalized email via SMTP.

```http
POST /api/v1/send-test-email
Content-Type: application/json
```

**Request Body:**
```json
{
  "to_email": "john@company.com",
  "lead_data": {
    "name": "John Doe",
    "role": "Developer",
    "campaign": "AI Program"
  }
}
```

**Response:**
```json
{
  "message": "Email sent successfully",
  "message_id": "msg_123456789",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## Lead Management

### Upload Lead CSV

Bulk process leads from CSV file.

```http
POST /api/v1/upload-leads
Content-Type: multipart/form-data
```

**Form Data:**
- `file`: CSV file with lead data

**Response:**
```json
{
  "upload_id": "upload_789",
  "total_leads": 150,
  "processed": 145,
  "failed": 5,
  "results": [
    {
      "lead_id": "uuid-1",
      "heat_score": "hot",
      "confidence": 0.89,
      "status": "processed"
    }
  ]
}
```

### Get All Leads

Retrieve processed leads with pagination.

```http
GET /api/v1/leads?page=1&limit=20&heat_score=hot
```

**Query Parameters:**
- `page`: Page number (default: 1)
- `limit`: Items per page (default: 20, max: 100)
- `heat_score`: Filter by hot/warm/cold

**Response:**
```json
{
  "leads": [
    {
      "lead_id": "uuid-1",
      "lead_data": {
        "name": "John Doe",
        "email": "john@company.com",
        "role": "Developer"
      },
      "score": {
        "heat_score": "hot",
        "confidence": 0.89
      },
      "recommendation": {
        "recommended_channel": "email",
        "message_content": "Personalized content..."
      }
    }
  ],
  "pagination": {
    "current_page": 1,
    "total_pages": 8,
    "total_leads": 150,
    "has_next": true,
    "has_prev": false
  }
}
```

---

## Analytics & Metrics

### Performance Metrics

Get system performance analytics.

```http
GET /api/v1/metrics
```

**Response:**
```json
{
  "system_metrics": {
    "total_requests": 1247,
    "avg_latency_ms": 1850,
    "error_rate": 0.0032,
    "success_rate": 0.9968
  },
  "rag_metrics": {
    "total_emails_generated": 892,
    "avg_tokens_per_request": 168,
    "cache_hit_rate": 0.45,
    "cost_per_email": 0.0004
  },
  "model_metrics": {
    "total_predictions": 1247,
    "accuracy": 0.87,
    "avg_confidence": 0.84,
    "heat_score_distribution": {
      "hot": 0.23,
      "warm": 0.51, 
      "cold": 0.26
    }
  }
}
```

### Feature Importance

Get insights on which features drive lead scoring.

```http
GET /api/v1/feature-importance
```

**Response:**
```json
{
  "feature_importance": [
    {"feature": "page_views", "importance": 0.34},
    {"feature": "recency_days", "importance": 0.28},
    {"feature": "prior_course_interest", "importance": 0.22},
    {"feature": "time_spent", "importance": 0.16}
  ],
  "analysis": "Page views and recency are strongest predictors of lead heat"
}
```

---

##  Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid lead data provided",
    "details": [
      {
        "field": "email",
        "message": "Invalid email format",
        "value": "invalid-email"
      }
    ],
    "timestamp": "2024-01-15T10:30:00Z",
    "trace_id": "trace_123456789"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request data |
| `RAG_TIMEOUT` | 408 | RAG generation timeout |
| `OPENAI_ERROR` | 502 | OpenAI API error |
| `DATABASE_ERROR` | 503 | Database connection error |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |

---

## Rate Limits

| Endpoint | Limit | Window |
|----------|-------|---------|
| `/score-lead` | 100 | 1 minute |
| `/get-personalized-email` | 50 | 1 minute |
| `/upload-leads` | 5 | 1 hour |
| All others | 200 | 1 minute |

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1642248600
```

---

## API Workflow

### Typical Usage Flow

```mermaid
graph TD
    A[Upload CSV] --> B[Score Leads]
    B --> C[Get Recommendations] 
    C --> D[Generate RAG Emails]
    D --> E[Send Emails]
    E --> F[Track Performance]
```

### Example Integration

```bash
# 1. Health check
curl http://localhost:8000/health

# 2. Score a lead
curl -X POST http://localhost:8000/api/v1/score-lead \
  -H "Content-Type: application/json" \
  -d '{"name":"John","email":"john@test.com","role":"Developer","page_views":15}'

# 3. Generate RAG email
curl -X POST http://localhost:8000/api/v1/get-personalized-email \
  -H "Content-Type: application/json" \
  -d '{"name":"John","email":"john@test.com","role":"Developer","page_views":15}'
```

---

## Support

For questions or issues:
- **Documentation**: `/docs` (Swagger UI)
- **Health Check**: `/health`
- **OpenAPI Spec**: `/openapi.json`

---

**Version**: 1.0.0  
**Last Updated**: January 2024  
**API Base URL**: `http://localhost:8000`