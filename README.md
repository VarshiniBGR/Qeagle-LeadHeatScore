# Lead HeatScore

AI-powered lead classification and personalized outreach platform using machine learning and RAG technology.

## Features

- **Lead Classification**: Lightweight Logistic Regression baseline (74.2% accuracy) with XGBoost comparison
- **Personalized Recommendations**: RAG-powered next-action suggestions
- **Hybrid Search**: BM25 + Vector similarity
- **Real-time Processing**: Sub-2-second response times
- **Enterprise Security**: Prompt injection detection and PII protection

## Tech Stack

**Backend**
- FastAPI (Python 3.10+)
- Logistic Regression (Lightweight Baseline)
- XGBoost (ML Classification Comparison)
- LangChain (LLM Orchestration)
- MongoDB Atlas (Vector Database)
- OpenAI GPT-4o-mini

**Frontend**
- React 18
- Tailwind CSS
- Vite

## Project Structure

```
lead-heatscore/
├── backend/
│   ├── app/
│   │   ├── api/           # API endpoints
│   │   ├── models/        # Data schemas
│   │   ├── services/      # Business logic
│   │   └── utils/         # Utilities
│   ├── data/              # Training datasets
│   ├── models/            # Trained ML models
│   ├── metrics/           # Performance metrics
│   ├── scripts/           # Utility scripts
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── pages/         # Page components
│   │   ├── contexts/      # React contexts
│   │   ├── lib/           # API client
│   │   └── utils/         # Frontend utilities
│   └── package.json
├── docs/                  # Documentation
├── notebooks/             # Jupyter notebooks
└── metrics/               # Performance analysis
```

## Setup Instructions

### Prerequisites
- Python 3.10+
- Node.js 18+
- MongoDB Atlas account
- OpenAI API key (required for premium AI features)

### Backend Setup

```bash
# Clone repository
git clone <repository-url>
cd lead-heatscore

# Install dependencies
cd backend
pip install -r requirements.txt

# Configure environment
cp env.example .env
# Edit .env with your credentials

# Start server
python -m uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

**Access:** http://localhost:3000 | **API Docs:** http://localhost:8000/docs

## AI Models (OpenAI Premium)

This project uses **OpenAI's premium models** for superior performance:

- **Embeddings**: `text-embedding-3-small` (1536 dimensions)
- **LLM Generation**: `gpt-4o-mini` (advanced conversational AI)
- **Classification**: Logistic Regression + XGBoost comparison

**Premium AI** - Superior accuracy and performance!

## MongoDB Atlas Setup

1. Create MongoDB Atlas account at [mongodb.com/atlas](https://mongodb.com/atlas)
2. Create new cluster (M0 free tier available)
3. Create database user with read/write permissions
4. Whitelist your IP address
5. Get connection string and add to `.env` file
6. Create vector search index for embeddings





