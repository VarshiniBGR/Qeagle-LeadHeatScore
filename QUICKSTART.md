# 🚀 Lead HeatScore - Quick Start Guide

Get up and running with the Lead HeatScore system in minutes!

## Prerequisites

- **Python 3.11+** - [Download here](https://python.org/downloads/)
- **Node.js 18+** - [Download here](https://nodejs.org/)
- **MongoDB Atlas Account** - [Sign up here](https://cloud.mongodb.com/)

## ⚡ Quick Start (5 minutes)

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd qeagle

# Install Python dependencies
pip install -r backend/requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..
```

### 2. Configure Environment

```bash
# Copy environment template
cp backend/env.example backend/.env

# Edit backend/.env with your MongoDB Atlas connection:
# MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/
```

### 3. Generate Sample Data & Train Model

```bash
# Generate 1000 sample leads
python backend/scripts/generate_synth_data.py --n 1000 --out backend/data/leads.csv

# Train the ML model
python backend/simple_train.py --csv backend/data/leads.csv --out backend/models
```

### 4. Start the System

**Terminal 1 - Backend:**
```bash
python -m uvicorn backend.app.main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend && npm run dev
```

### 5. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 🎯 What You Can Do

### Upload Lead Data
1. Go to **Upload** page
2. Upload CSV with lead data
3. System automatically processes and classifies leads

### View Results
1. Go to **Leads** page
2. See AI-powered heat scores (Hot/Warm/Cold)
3. Get personalized next-action recommendations
4. Export results for your CRM

### Dashboard Analytics
1. Go to **Overview** page
2. View lead distribution charts
3. Monitor model performance
4. Track processing metrics

## 📊 Sample Data Format

Create a CSV file with these columns:

```csv
source,recency_days,region,role,campaign,page_views,last_touch,prior_course_interest
website,5,US,manager,summer_sale,15,email,high
linkedin,12,EU,engineer,tech_conference,8,social,medium
referral,2,APAC,director,partner_program,25,phone,high
```

## 🔧 Configuration Options

### Model Thresholds
Edit `backend/.env`:
```env
CLASS_THRESHOLDS=hot:0.7,warm:0.4
```

### OpenAI Integration (Optional)
```env
OPENAI_API_KEY=your_openai_key_here
```

## 🚨 Troubleshooting

### Backend Issues
```bash
# Check if all dependencies are installed
pip list | grep -E "(fastapi|pandas|scikit-learn)"

# Regenerate sample data
python backend/scripts/generate_synth_data.py --n 100 --out backend/data/leads.csv

# Retrain model
python backend/simple_train.py --csv backend/data/leads.csv --out backend/models
```

### Frontend Issues
```bash
# Clear npm cache
npm cache clean --force

# Reinstall dependencies
rm -rf frontend/node_modules
cd frontend && npm install
```

### MongoDB Connection
- Ensure your MongoDB Atlas cluster is running
- Check your connection string in `backend/.env`
- Verify network access settings in MongoDB Atlas

## 📈 Next Steps

1. **Customize Models**: Adjust classification thresholds based on your data
2. **Add Knowledge Base**: Upload sales playbooks and best practices
3. **Integrate CRM**: Connect to Salesforce, HubSpot, or other systems
4. **Scale Up**: Deploy to cloud for production use

## 🆘 Need Help?

- Check the full [README.md](README.md) for detailed documentation
- Review API documentation at http://localhost:8000/docs
- Create an issue in the GitHub repository

---

**🎉 You're all set! Start uploading leads and see the AI magic happen!**
