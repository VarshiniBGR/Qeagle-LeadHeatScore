# 🏗️ Lead HeatScore System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Lead HeatScore System                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend│    │   FastAPI       │    │   MongoDB        │
│   (Port 3000)   │◄──►│   Backend       │◄──►│   Atlas          │
│                 │    │   (Port 8000)   │    │   • Vector DB    │
│ • Leads Table   │    │                 │    │   • Lead Storage │
│ • Upload CSV    │    │ • Classification│    │   • Search Index │
│ • Email Panel   │    │ • RAG Agent     │    │                 │
│ • Analytics     │    │ • Email Service │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                         │
         │                         ▼
         │                ┌─────────────────┐
         │                │   ML Pipeline   │
         │                │                 │
         │                │ • XGBoost Model │
         │                │ • Embeddings    │
         │                │ • Feature Eng.  │
         │                │ • Calibration   │
         │                └─────────────────┘
         │
         ▼
┌─────────────────┐
│   External      │
│   Services      │
│                 │
│ • OpenAI GPT    │
│ • Gmail SMTP    │
│ • Email Templates│
└─────────────────┘
```

## Data Flow Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   CSV       │    │   Feature   │    │   ML        │    │   RAG       │
│   Upload    │───►│   Engineering│───►│   Model     │───►│   Agent     │
│             │    │             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Lead      │    │   Vector    │    │   Heat      │    │   Next      │
│   Storage   │    │   Search    │    │   Score     │    │   Action    │
│             │    │             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   MongoDB   │    │   MongoDB   │    │   Frontend  │    │   Email     │
│   Collection│    │   Vector   │    │   Display   │    │   Service   │
│             │    │   Index    │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## Component Architecture

### Backend Services
```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                         │
├─────────────────────────────────────────────────────────────┤
│  API Layer                                                 │
│  ├── /api/v1/score      - Lead classification             │
│  ├── /api/v1/recommend  - Next action recommendations     │
│  ├── /api/v1/upload     - CSV upload & processing         │
│  ├── /api/v1/leads      - Lead retrieval                  │
│  └── /api/v1/send-email - Email automation                │
├─────────────────────────────────────────────────────────────┤
│  Service Layer                                             │
│  ├── LeadClassifier     - XGBoost ML model                │
│  ├── NextActionAgent    - RAG-powered recommendations     │
│  ├── EmailService       - SMTP email handling              │
│  ├── HybridRetrieval    - Vector + keyword search          │
│  └── SafetyFilters      - Prompt injection & PII detection│
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                               │
│  ├── MongoDB Atlas      - Vector database                 │
│  ├── Model Storage      - Trained ML models               │
│  └── Configuration     - Environment & settings         │
└─────────────────────────────────────────────────────────────┘
```

### Frontend Components
```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend                          │
├─────────────────────────────────────────────────────────────┤
│  Pages                                                    │
│  ├── Leads.jsx          - Main leads dashboard           │
│  ├── Upload.jsx         - CSV upload interface           │
│  ├── Detail.jsx         - Individual lead details         │
│  └── Overview.jsx       - Analytics & metrics           │
├─────────────────────────────────────────────────────────────┤
│  Components                                               │
│  ├── LeadsTable.jsx     - Lead data table                │
│  ├── ProbabilityBar.jsx - Visual probability display     │
│  ├── RecommendationPanel.jsx - Next action suggestions  │
│  ├── CsvUpload.jsx      - File upload component          │
│  └── Navbar.jsx         - Navigation                     │
├─────────────────────────────────────────────────────────────┤
│  Services                                                 │
│  ├── api.js             - HTTP client & API calls        │
│  └── utils.js           - Helper functions               │
└─────────────────────────────────────────────────────────────┘
```

## Technology Stack

### Backend Stack
- **FastAPI** - Modern Python web framework
- **XGBoost** - Gradient boosting classifier
- **LangChain** - LLM orchestration & RAG
- **MongoDB Atlas** - Vector database & search
- **sentence-transformers** - Embedding generation
- **Pydantic** - Data validation

### Frontend Stack
- **React 18** - UI framework
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Axios** - HTTP client

### ML/AI Stack
- **OpenAI GPT** - Large language model
- **all-MiniLM-L6-v2** - Embedding model
- **scikit-learn** - ML utilities
- **joblib** - Model persistence

## Security Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Security Layers                         │
├─────────────────────────────────────────────────────────────┤
│  Input Validation                                         │
│  ├── Pydantic Schemas    - Data type validation           │
│  ├── Sanitization        - XSS prevention                 │
│  └── Rate Limiting       - Request throttling             │
├─────────────────────────────────────────────────────────────┤
│  Content Security                                         │
│  ├── Prompt Injection    - Pattern detection              │
│  ├── PII Redaction       - Email/phone masking           │
│  └── Output Filtering    - Response sanitization          │
├─────────────────────────────────────────────────────────────┤
│  Infrastructure Security                                  │
│  ├── Environment Variables - Secret management           │
│  ├── HTTPS/TLS           - Encrypted communication       │
│  └── Access Control      - Authentication & authorization│
└─────────────────────────────────────────────────────────────┘
```

## Performance Architecture

### Latency Optimization
- **Model Caching** - Pre-loaded ML models
- **Vector Indexing** - MongoDB Atlas vector search
- **Connection Pooling** - Database connection reuse
- **Async Processing** - Non-blocking operations

### Scalability Design
- **Stateless Backend** - Horizontal scaling ready
- **Database Sharding** - MongoDB Atlas auto-scaling
- **CDN Ready** - Static asset optimization
- **Load Balancing** - Multiple instance support

## Monitoring & Observability

```
┌─────────────────────────────────────────────────────────────┐
│                Monitoring Stack                            │
├─────────────────────────────────────────────────────────────┤
│  Application Metrics                                       │
│  ├── Request Tracing      - End-to-end request tracking    │
│  ├── Performance Metrics  - Latency, throughput            │
│  ├── Error Tracking       - Exception monitoring          │
│  └── Business Metrics     - Lead conversion rates         │
├─────────────────────────────────────────────────────────────┤
│  Infrastructure Monitoring                                 │
│  ├── Health Checks        - Service availability          │
│  ├── Resource Usage       - CPU, memory, disk             │
│  ├── Database Metrics     - Query performance             │
│  └── External Services    - API response times           │
└─────────────────────────────────────────────────────────────┘
```
