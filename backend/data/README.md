# 📊 Lead HeatScore Dataset

**Production-ready dataset for lead classification system**

## 📋 Dataset Summary

- **Total Records:** 600 leads
- **Split:** Train (480) / Validation (60) / Test (60)
- **Target:** `heat_score` (Cold/Warm/Hot)
- **Features:** 15 variables across demographics, engagement, and behavior

## 📁 Data Files

| File | Description | Records | Purpose |
|------|-------------|---------|---------|
| `leads_train.csv` | Training dataset | 480 | Model training |
| `leads_valid.csv` | Validation dataset | 60 | Hyperparameter tuning |
| `leads_test.csv` | Test dataset | 60 | Final evaluation |
| `dictionary.md` | Feature documentation | - | Data schema |

## 🎯 Quick Start

```python
import pandas as pd

# Load training data
train_df = pd.read_csv('backend/data/leads_train.csv')

# Basic info
print(f"Training samples: {len(train_df)}")
print(f"Features: {train_df.shape[1]-2}")  # -2 for target and ID
print(f"Classes: {train_df['heat_score'].value_counts().to_dict()}")
```

## 📊 Target Distribution

- **Cold:** 20% (Low engagement, older interactions)
- **Warm:** 60% (Medium engagement, recent activity)  
- **Hot:** 20% (High engagement, recent high-value actions)

## 🚀 Usage in Project

The data is used by:
- `backend/app/services/classifier.py` - Model training
- `backend/app/services/rag_email_service.py` - Email personalization
- `metrics_analysis.ipynb` - Performance evaluation

## 📋 Data Quality

- **Missing Values:** < 2%
- **Outliers:** Handled via capping thresholds
- **Validation:** Schema validation in preprocessing
- **Privacy:** PII anonymized/hashed

**Status:** Production Ready ✅

