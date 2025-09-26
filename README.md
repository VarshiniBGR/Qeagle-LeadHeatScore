# 🔥 Lead HeatScore: AI-Powered Lead Classification & Next-Action Agent

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-blue.svg)](https://reactjs.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green.svg)](https://mongodb.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A comprehensive RAG-powered system for intelligent lead scoring and personalized outreach recommendations**

## 🎯 Project Overview

Lead HeatScore is an advanced AI system that combines machine learning classification with retrieval-augmented generation (RAG) to:

- **Classify leads** as Hot, Warm, or Cold based on engagement patterns
- **Generate personalized recommendations** for next actions
- **Provide intelligent email templates** tailored to lead characteristics
- **Track performance metrics** with comprehensive evaluation

### 🏆 Key Features

- ✅ **XGBoost Classification** with 82.3% F1 Macro Score
- ✅ **RAG-Powered Recommendations** using LangChain + OpenAI
- ✅ **Hybrid Retrieval** (Vector + Keyword search)
- ✅ **Real-time Lead Scoring** with probability calibration
- ✅ **Email Automation** with personalized templates
- ✅ **Comprehensive Evaluation** with F1, ROC, and Brier scores
- ✅ **Modern UI** with React + Tailwind CSS
- ✅ **Production Ready** with FastAPI + MongoDB Atlas

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **MongoDB Atlas** account
- **OpenAI API Key** (optional, has fallback)

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd lead-heatscore
```

### 2. Backend Setup

```bash
# Install dependencies
cd backend
pip install -r requirements.txt

# Set up environment
cp env.example .env
# Edit .env with your MongoDB and OpenAI credentials

# Train the model (first time only)
python simple_train.py

# Start the server
python -m uvicorn app.main:app --reload --port 8000
```

### 3. Frontend Setup

```bash
# In a new terminal
cd frontend
npm install
npm run dev
```

### 4. Access the Application

- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## 📊 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │   FastAPI       │    │   MongoDB       │
│   (Port 3000)   │◄──►│   Backend       │◄──►│   Atlas         │
│                 │    │   (Port 8000)   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                         │
         │                         ▼
         │                ┌─────────────────┐
         │                │   ML Pipeline   │
         │                │   • XGBoost     │
         │                │   • Embeddings  │
         │                │   • RAG Agent   │
         │                └─────────────────┘
         │
         ▼
┌─────────────────┐
│   Email Service │
│   (SMTP/Gmail)  │
└─────────────────┘
```

---

## 🔧 Technical Stack

### Backend
- **FastAPI** - Modern Python web framework
- **XGBoost** - Gradient boosting for classification
- **LangChain** - LLM orchestration and RAG
- **MongoDB Atlas** - Vector database with search
- **sentence-transformers** - Embedding generation
- **Pydantic** - Data validation and serialization

### Frontend
- **React 18** - Modern UI framework
- **Vite** - Fast build tool
- **Tailwind CSS** - Utility-first styling
- **Axios** - HTTP client

### ML/AI
- **OpenAI GPT** - LLM for recommendations (with fallback)
- **all-MiniLM-L6-v2** - Embedding model
- **scikit-learn** - ML utilities and evaluation

---

## 📈 Model Performance

### Classification Metrics
| Metric | Value | Status |
|--------|-------|--------|
| **F1 Macro** | 82.3% | ✅ Meets 80% threshold |
| **Accuracy** | 84.7% | ✅ Excellent |
| **Brier Score** | 0.156 | ✅ Good calibration |

### Per-Class Performance
| Class | F1 Score | ROC AUC | Performance |
|-------|----------|---------|-------------|
| **Hot** | 86.8% | 92.3% | 🟢 Excellent |
| **Warm** | 81.2% | 85.6% | 🟡 Good |
| **Cold** | 78.9% | 89.1% | 🟡 Good |

### Feature Importance
1. **Page Views** (35%) - Engagement indicator
2. **Recency Days** (28%) - Temporal signal
3. **Prior Interest** (20%) - Historical engagement
4. **Source** (12%) - Lead quality
5. **Role** (5%) - Demographic factor

---

## 🔌 API Endpoints

### Core Endpoints

```http
POST /api/v1/score
Content-Type: application/json

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
  "prior_course_interest": "high"
}
```

**Response:**
```json
{
  "lead_id": "abc123",
  "heat_score": "hot",
  "confidence": 0.87,
  "probabilities": {
    "cold": 0.05,
    "warm": 0.08,
    "hot": 0.87
  },
  "features_importance": {...}
}
```

### Additional Endpoints

- `POST /api/v1/recommend` - Get next action recommendation
- `POST /api/v1/upload` - Upload CSV leads
- `GET /api/v1/leads` - Retrieve scored leads
- `POST /api/v1/send-email` - Send personalized emails
- `GET /health` - System health check

---

## 📁 Project Structure

```
lead-heatscore/
├── backend/
│   ├── app/
│   │   ├── api/           # API routes
│   │   ├── models/        # Pydantic schemas
│   │   ├── services/      # Business logic
│   │   └── utils/         # Utilities
│   ├── models/            # Trained ML models
│   ├── scripts/           # Training scripts
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── pages/        # Page components
│   │   └── lib/          # API client
│   └── package.json
├── evaluation_results/    # Model evaluation
├── docs/                 # Documentation
└── README.md
```

---

## 🛠️ Configuration

### Environment Variables

Create `backend/.env`:

```env
# MongoDB Atlas
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/
MONGO_DB=leadheat
MONGO_COLLECTION=vectors

# ML Configuration
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
MODEL_DIR=./models
CLASS_THRESHOLDS=hot:0.7,warm:0.4

# AI/LLM (Optional)
OPENAI_API_KEY=your_openai_key_here

# Email Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
FROM_EMAIL=your_email@gmail.com
```

### Email Setup

For Gmail SMTP, you need an **App Password**:
1. Enable 2-Factor Authentication
2. Generate App Password: Google Account → Security → App passwords
3. Use the 16-character password in `SMTP_PASSWORD`

---

## 🧪 Evaluation & Testing

### Run Evaluation

```bash
# Generate evaluation metrics
python evaluation_system.py

# View results
cat evaluation_results/metrics_summary.csv
```

### Test API

```bash
# Health check
curl http://localhost:8000/health

# Score a lead
curl -X POST http://localhost:8000/api/v1/score \
  -H "Content-Type: application/json" \
  -d '{"name":"Test Lead","email":"test@example.com",...}'
```

---

## 📊 Performance Metrics

### Latency Requirements
- **p95 Latency**: ≤ 2.5s ✅ (Achieved: ~1.8s)
- **Error Rate**: < 0.5% ✅ (Achieved: ~0.1%)

### Cost Analysis
- **OpenAI API**: ~$0.02 per 1000 leads (with fallback)
- **MongoDB Atlas**: ~$0.01 per 1000 queries
- **Compute**: Minimal (local inference)

---

## 🔒 Security & Safety

### Implemented Safeguards
- ✅ **Prompt Injection Detection** - Basic pattern matching
- ✅ **PII Redaction** - Email/phone number masking
- ✅ **Input Validation** - Pydantic schemas
- ✅ **Error Handling** - Graceful degradation
- ✅ **Rate Limiting** - Basic request throttling

### Privacy Considerations
- No sensitive data stored permanently
- Email addresses masked in logs
- Configurable data retention policies

---

## 🚀 Deployment

### Docker Deployment

```bash
# Build and run
docker-compose up -d

# Check status
docker-compose ps
```

### Production Considerations
- Use environment variables for secrets
- Set up proper logging and monitoring
- Configure reverse proxy (nginx)
- Enable HTTPS/TLS
- Set up database backups

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **LangChain** for RAG orchestration
- **FastAPI** for the excellent web framework
- **MongoDB Atlas** for vector search capabilities
- **OpenAI** for LLM capabilities
- **React** and **Tailwind** for the modern UI

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Documentation**: [Wiki](https://github.com/your-repo/wiki)
- **Email**: your-email@example.com

---

**Built with ❤️ for intelligent lead management**