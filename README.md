# Lead HeatScore

A lead scoring system that classifies leads as Hot, Warm, or Cold using machine learning.

## What it does

- Scores leads based on engagement data
- Suggests next actions for each lead
- Generates personalized email templates
- Tracks performance metrics

## Setup

### Backend

```bash
cd backend
pip install -r requirements.txt
cp env.example .env
# Edit .env with your MongoDB and OpenAI credentials
python -m uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Access at http://localhost:3000

## Tech Stack

**Backend:**
- FastAPI
- XGBoost (ML model)
- MongoDB Atlas
- LangChain

**Frontend:**
- React
- Tailwind CSS

## Model Performance

- Accuracy: 84.7%
- F1 Score: 82.3%

## API

Main endpoint: `POST /api/v1/score`

Send lead data, get back heat score (hot/warm/cold) with confidence.

## Configuration

Create `backend/.env`:

```env
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/
MONGO_DB=leadheat
OPENAI_API_KEY=your_key_here
```

## License
