# Lead HeatScore

> A machine learning system for intelligent lead classification and personalized outreach

## 🎯 What it does

Lead HeatScore analyzes lead engagement data to classify prospects as **Hot**, **Warm**, or **Cold**. The system combines XGBoost classification with multi-channel outreach including personalized emails, Telegram messages, and SMS for comprehensive lead management.

### Key Features
- **Lead Classification** - ML-powered scoring with 87.3% accuracy
- **Multi-Channel Outreach** - Email, Telegram, and SMS messaging
- **Personalized Content** - AI-generated recommendations using RAG
- **Performance Analytics** - Comprehensive metrics and evaluation
- **Modern API** - RESTful endpoints for easy integration

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │   FastAPI       │    │   MongoDB       │
│   (Port 3000)   │◄──►│   Backend       │◄──►│   Atlas         │
│                 │    │   (Port 8000)   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                         │
         │                         ▼
         │                ┌─────────────────┐
         │                │   ML Pipeline    │
         │                │   • XGBoost     │
         │                │   • OpenAI     │
         │                │   • Embeddings  │
         │                └─────────────────┘
         │
         ▼
┌─────────────────┐
│   Multi-Channel  │
│   Messaging      │
└─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- MongoDB Atlas account
- OpenAI API key

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
cp env.example .env
# Edit .env with your credentials
python -m uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

**Access:** http://localhost:3000 | **API Docs:** http://localhost:8000/docs

## 🛠️ Technology Stack

**Backend:** FastAPI, XGBoost, MongoDB Atlas, LangChain, OpenAI  
**Frontend:** React 18, Tailwind CSS, Vite

## 📊 Model Performance

| Metric | Value | Status |
|--------|-------|--------|
| **Accuracy** | 87.3% | ✅ Excellent |
| **F1 Score (Macro)** | 84.2% | ✅ Meets target |
| **Latency** | 45.2ms | ✅ Fast inference |

### Per-Class Results
- **Cold**: F1 = 81.5% ✅
- **Warm**: F1 = 85.6% ✅  
- **Hot**: F1 = 85.5% ✅

## 🔌 API Usage

### Score a Lead

```bash
POST /api/v1/score
```

**Request:**
```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "source": "website",
  "recency_days": 5,
  "region": "North America",
  "role": "Manager",
  "campaign": "AI Course",
  "page_views": 15,
  "last_touch": "email",
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
  "recommendations": "Schedule demo call within 24 hours"
}
```

### Additional Endpoints
- `POST /api/v1/recommend` - Get personalized recommendations
- `POST /api/v1/send-telegram-message` - Send Telegram messages
- `POST /api/v1/send-telegram-to-phone` - Send Telegram via phone number
- `GET /health` - System health check

## ⚙️ Configuration

Create `backend/.env`:

```env
# Database
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/
MONGO_DB=leadheat

# AI Configuration
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL_NAME=text-embedding-3-small
LLM_MODEL=gpt-4o-mini

# Telegram Bot (Optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Email (Optional)
SMTP_SERVER=smtp.gmail.com
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
```

## 📁 Project Structure

```
lead-heatscore/
├── backend/
│   ├── app/           # API endpoints, models, services
│   ├── data/          # Training datasets
│   └── models/        # Trained ML models
├── frontend/
│   └── src/           # React components and pages
├── notebooks/
│   └── metrics_report.ipynb  # Performance analysis
└── docs/
    └── ARCHITECTURE.md
```

## 📈 Training Data

The model is trained on real lead data with features including:
- **Engagement**: Page views, time spent, course actions
- **Temporal**: Recency days, last touchpoint
- **Demographic**: Region, role, campaign source
- **Behavioral**: Prior course interest, search keywords

**Data Sources:** `backend/data/leads_train.csv` and `backend/data/leads_test.csv`

## 🔬 Evaluation & Metrics

Comprehensive evaluation including confusion matrix, ROC curves, calibration plots, and A/B testing results. View detailed metrics in `notebooks/metrics_report.ipynb`



---

**Built for intelligent lead management** 🎯