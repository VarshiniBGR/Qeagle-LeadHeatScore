# 📚 Lead HeatScore API Documentation

## Base URL
```
http://localhost:8000
```

## Authentication
Currently no authentication required. In production, implement API key or JWT authentication.

---

## 🎯 Core Endpoints

### 1. Lead Classification

**Endpoint:** `POST /api/v1/score`

**Description:** Classify a lead as Hot, Warm, or Cold based on engagement patterns.

**Request Body:**
```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "source": "Web",
  "recency_days": 3,
  "region": "North America",
  "role": "Manager",
  "campaign": "AI Course",
  "page_views": 15,
  "last_touch": "Email Open",
  "prior_course_interest": "high",
  "cta": "Request Demo"
}
```

**Response:**
```json
{
  "lead_id": "abc123-def456-ghi789",
  "heat_score": "hot",
  "confidence": 0.87,
  "probabilities": {
    "cold": 0.05,
    "warm": 0.08,
    "hot": 0.87
  },
  "features_importance": {
    "page_views": 0.35,
    "recency_days": 0.28,
    "prior_course_interest": 0.20,
    "source": 0.12,
    "role": 0.05
  },
  "model_version": "1.0.0",
  "timestamp": "2025-09-24T10:30:00Z"
}
```

**Status Codes:**
- `200` - Success
- `400` - Invalid input data
- `500` - Internal server error

---

### 2. Next Action Recommendation

**Endpoint:** `POST /api/v1/recommend`

**Description:** Get personalized next action recommendation using RAG.

**Request Body:**
```json
{
  "lead_id": "abc123-def456-ghi789",
  "context": "Recent webinar attendee, high engagement"
}
```

**Response:**
```json
{
  "lead_id": "abc123-def456-ghi789",
  "recommendation": {
    "channel": "email",
    "message": "Hi John, I noticed you attended our AI Course webinar. Based on your interest in AI Course, I'd love to schedule a personalized demo...",
    "rationale": "High engagement + recent webinar attendance suggests immediate follow-up",
    "references": [
      "webinar_attendance_pattern",
      "ai_course_engagement"
    ],
    "priority": "high",
    "suggested_timing": "within_24_hours"
  },
  "confidence": 0.85,
  "model_used": "gpt-3.5-turbo",
  "timestamp": "2025-09-24T10:30:00Z"
}
```

---

### 3. CSV Upload & Batch Processing

**Endpoint:** `POST /api/v1/upload`

**Description:** Upload CSV file with multiple leads for batch processing.

**Request:** `multipart/form-data`
- `file`: CSV file with lead data

**CSV Format:**
```csv
lead_id,name,email,source,recency_days,region,role,campaign,page_views,last_touch,prior_interest,cta
L001,John Doe,john@example.com,Web,2,North America,Manager,AI Course,15,Email Open,high,Request Demo
L002,Jane Smith,jane@example.com,Social Media,5,Europe,Student,Data Science,8,Webinar,medium,Sign-up
```

**Response:**
```json
{
  "message": "CSV processed successfully",
  "total_leads": 10,
  "processed_leads": 10,
  "failed_leads": 0,
  "results": [
    {
      "lead_id": "L001",
      "heat_score": "hot",
      "confidence": 0.87
    },
    {
      "lead_id": "L002", 
      "heat_score": "warm",
      "confidence": 0.72
    }
  ],
  "processing_time": "2.3s",
  "timestamp": "2025-09-24T10:30:00Z"
}
```

---

### 4. Retrieve Leads

**Endpoint:** `GET /api/v1/leads`

**Description:** Retrieve scored leads with pagination and filtering.

**Query Parameters:**
- `limit` (optional): Number of leads to return (default: 50, max: 100)
- `offset` (optional): Number of leads to skip (default: 0)
- `heat_score` (optional): Filter by heat score (hot, warm, cold)
- `sort_by` (optional): Sort field (confidence, timestamp, heat_score)
- `sort_order` (optional): Sort order (asc, desc)

**Example:**
```
GET /api/v1/leads?limit=20&offset=0&heat_score=hot&sort_by=confidence&sort_order=desc
```

**Response:**
```json
{
  "leads": [
    {
      "lead_id": "abc123-def456-ghi789",
      "name": "John Doe",
      "email": "john@example.com",
      "heat_score": "hot",
      "confidence": 0.87,
      "probabilities": {
        "cold": 0.05,
        "warm": 0.08,
        "hot": 0.87
      },
      "created_at": "2025-09-24T10:30:00Z",
      "last_updated": "2025-09-24T10:30:00Z"
    }
  ],
  "pagination": {
    "total": 150,
    "limit": 20,
    "offset": 0,
    "has_next": true,
    "has_prev": false
  },
  "filters_applied": {
    "heat_score": "hot",
    "sort_by": "confidence",
    "sort_order": "desc"
  }
}
```

---

### 5. Send Email

**Endpoint:** `POST /api/v1/send-email`

**Description:** Send personalized email to a lead.

**Request Body:**
```json
{
  "lead_id": "abc123-def456-ghi789",
  "to_email": "john@example.com",
  "template_type": "hot_lead_followup"
}
```

**Response:**
```json
{
  "message": "Email sent successfully",
  "lead_id": "abc123-def456-ghi789",
  "to_email": "john@example.com",
  "lead_type": "hot",
  "subject": "Exclusive AI Course Demo - Limited Time",
  "template_used": "hot_lead_followup",
  "sent_at": "2025-09-24T10:30:00Z"
}
```

---

## 🔧 Utility Endpoints

### Health Check

**Endpoint:** `GET /health`

**Description:** Check system health and status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-09-24T10:30:00Z",
  "version": "1.0.0",
  "services": {
    "database": "connected",
    "ml_model": "loaded",
    "email_service": "configured",
    "vector_search": "active"
  },
  "uptime": "2h 15m 30s",
  "memory_usage": "45.2%",
  "cpu_usage": "12.8%"
}
```

---

### Model Information

**Endpoint:** `GET /api/v1/model/info`

**Description:** Get information about the loaded ML model.

**Response:**
```json
{
  "model_name": "lead_classifier_v1",
  "model_type": "XGBoost",
  "version": "1.0.0",
  "training_date": "2025-09-20T00:00:00Z",
  "performance_metrics": {
    "f1_macro": 0.823,
    "accuracy": 0.847,
    "brier_score": 0.156
  },
  "feature_count": 12,
  "class_labels": ["cold", "warm", "hot"],
  "model_size": "2.3MB"
}
```

---

## 📊 Data Models

### Lead Input Schema

```json
{
  "name": "string (required)",
  "email": "string (required, valid email)",
  "source": "string (required)",
  "recency_days": "integer (required, 0-365)",
  "region": "string (required)",
  "role": "string (required)",
  "campaign": "string (required)",
  "page_views": "integer (required, 0-1000)",
  "last_touch": "string (required)",
  "prior_course_interest": "string (required, low|medium|high)",
  "cta": "string (optional)"
}
```

### Heat Score Enum

```json
{
  "heat_score": "hot | warm | cold"
}
```

### Probability Distribution

```json
{
  "probabilities": {
    "cold": "float (0.0-1.0)",
    "warm": "float (0.0-1.0)", 
    "hot": "float (0.0-1.0)"
  }
}
```

---

## ⚠️ Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "email",
      "issue": "Invalid email format"
    },
    "timestamp": "2025-09-24T10:30:00Z",
    "request_id": "req_abc123"
  }
}
```

### Common Error Codes

| Code | Status | Description |
|------|--------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid input data |
| `LEAD_NOT_FOUND` | 404 | Lead ID not found |
| `EMAIL_SEND_FAILED` | 500 | Email service error |
| `MODEL_ERROR` | 500 | ML model prediction error |
| `DATABASE_ERROR` | 500 | Database connection error |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |

---

## 🔒 Security Considerations

### Input Validation
- All inputs validated using Pydantic schemas
- Email format validation
- Numeric range validation
- String length limits

### Rate Limiting
- 100 requests per minute per IP
- 1000 requests per hour per API key

### Data Privacy
- Email addresses masked in logs
- PII redaction in responses
- Configurable data retention

---

## 📈 Performance Metrics

### Response Times (p95)
- Lead Classification: ~200ms
- Recommendations: ~800ms
- CSV Upload: ~2.3s (10 leads)
- Email Sending: ~1.2s

### Throughput
- Concurrent requests: 50
- Peak load: 100 requests/second
- Database connections: 20

---

## 🧪 Testing

### Test Endpoints

**Test Classification:**
```bash
curl -X POST http://localhost:8000/api/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Lead",
    "email": "test@example.com",
    "source": "Web",
    "recency_days": 1,
    "region": "North America",
    "role": "Manager",
    "campaign": "AI Course",
    "page_views": 20,
    "last_touch": "Email Open",
    "prior_course_interest": "high"
  }'
```

**Test Health:**
```bash
curl http://localhost:8000/health
```

### Sample Test Data

```json
{
  "hot_lead": {
    "name": "Sarah Johnson",
    "email": "sarah@techcorp.com",
    "source": "Web",
    "recency_days": 1,
    "region": "North America",
    "role": "Director",
    "campaign": "AI Course",
    "page_views": 25,
    "last_touch": "Demo Request",
    "prior_course_interest": "high"
  },
  "warm_lead": {
    "name": "Mike Chen",
    "email": "mike@startup.io",
    "source": "Social Media",
    "recency_days": 5,
    "region": "Europe",
    "role": "Manager",
    "campaign": "Data Science",
    "page_views": 12,
    "last_touch": "Webinar",
    "prior_course_interest": "medium"
  },
  "cold_lead": {
    "name": "Lisa Wang",
    "email": "lisa@company.com",
    "source": "Referral",
    "recency_days": 30,
    "region": "Asia",
    "role": "Student",
    "campaign": "Business Analytics",
    "page_views": 3,
    "last_touch": "Brochure Download",
    "prior_course_interest": "low"
  }
}
```

---

## 📝 Changelog

### Version 1.0.0 (2025-09-24)
- Initial API release
- Lead classification endpoint
- RAG-powered recommendations
- CSV upload functionality
- Email automation
- Comprehensive error handling

---

## 🤝 Support

- **API Issues**: Create GitHub issue
- **Documentation**: Check this file
- **Examples**: See `/examples` directory
- **Contact**: your-email@example.com
