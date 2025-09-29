# 📋 Lead HeatScore Dataset Dictionary

**Data Schema & Feature Documentation for Lead Classification**

---

## 📊 Dataset Overview

- **Total Records: 600** leads
- **Split:** Train (480) / Validation (60) > / Test (60)
- **Target Variable:** `heat_score` (Cold/Warm/Hot)
- **Features:** 15 input variables across engagement, demographics, and behavior

---

## 🎯 Target Variable

| Variable | Description | Values | Distribution |
|----------|-------------|---------|--------------|
| `heat_score` | Lead classification score | Cold, Warm, Hot | Cold: 20%, Warm: 60%, Hot: 20% |
| `confidence` | Model confidence in prediction | 0.0 - 1.0 | Average: 0.847 |

---

## 📝 Feature Dictionary

### 👤 Demographics & Contact
| Variable | Type | Description | Example Values |
|----------|------|-------------|----------------|
| `name` | String | Lead's full name | "Sarah Johnson", "Mike Chen" |
| `email` | String | Lead's email address | "sarah@techcorp.com", "mike@startup.io" |
| `phone` | String | Lead's phone number | "+91-99999-99999", "+1-555-0123" |
| `region` | Categorical | Geographic region | US, EU, IN, BR, MX, JP, MEA, APAC |
| `role` | Categorical | Professional position | Manager, Director, Analyst, Developer |

### 🎯 Campaign & Source
| Variable | Type | Description | Example Values |
|----------|------|-------------|----------------|
| `source` | Categorical | Lead acquisition source | twitter, referral, conference, webinar, facebook_ads, website |
| `campaign` | Categorical | Marketing campaign identifier | spring_promotion, beta_testing, nonprofit_rate, government_contract |
| `cta` | Categorical | Call-to-action taken | Request Demo, Sign-up, Download Brochure, Schedule Call |

### 📈 Engagement Metrics
| Variable | Type | Description | Range | Units |
|----------|------|-------------|-------|-------|
| `page_views` | Integer | Number of page visits | 1-100 | Views |
| `time_spent` | Integer | Time on platform | 30-3600 | Seconds |
| `recency_days` | Integer | Days since last activity | 1-90 | Days |
| `last_touch` | Categorical | Most recent interaction | email, sms, registration, purchase, chat, webinar |

### 💭 Interest & Behavior
| Variable | Type | Description | Values | Impact on Score |
|----------|------|-------------|--------|----------------|
| `prior_course_interest` | Ordinal | Interest level in courses | low, medium, high | Medium→+1, High→+2 |
| `search_keywords` | String | Search terms used | "AI, machine learning", "data science" | Relevant terms→+1 |
| `course_actions` | Categorical | Actions taken on course pages | view_course, book_demo, request_info | Demo→+2, Info→+1 |

---

## 🔍 Feature Engineering Rules

### Heat Score Prediction Model
```python
# Hot Lead Indicators (Score: +2 each)
- page_views > 20
- prior_course_interest == "high" 
- recency_days <= 3
- course_actions in ["demo_request", "schedule_call"]

# Warm Lead Indicators (Score: +1 each)
- page_views 10-20
- prior_course_interest == "medium"
- recency_days <= 7
- source in ["webinar", "referral"]

# Cold Lead Risk (Score: -1 each)
- page_views < 5
- prior_course_interest == "low"
- recency_days > 30
- last_touch not in ["email", "demo_request"]
```

### Data Quality Rules
- **Missing Values:** < 5% for critical features
- **Outliers:** page_views capped at 100, time_spent at 3600s
- **Encoding:** Categorical variables use label encoding
- **Scaling:** Numeric features normalized to 0-1 range

---

## 📊 Data Splits

### Training Set (80% - 480 leads)
- **Purpose:** Model training and feature selection
- **Balance:** Cold:94, Warm:290, Hot:96
- **Date Range:** January-March 2024

### Validation Set (10% - 60 leads)
- **Purpose:** Hyperparameter tuning and model selection
- **Balance:** Cold:12, Warm:36, Hot:12
- **Date Range:** April 2024

### Test Set (10% - 60 leads)
- **Purpose:** Final performance evaluation
- **Balance:** Cold:12, Warm:36, Hot:12
- **Date Range:** May 2024

---

## 🎯 Performance Targets

| Metric | Target | Production Model Achieved |
|--------|--------|--------------------------|
| **F1 Score (Macro)** | ≥ 0.80 | 0.903 ✅ |
| **Accuracy** | ≥ 85% | 91.7% ✅ |
| **Per-Class F1** | ≥ 0.80 | All classes ≥ 0.885 ✅ |
| **Latency** | ≤ 50ms | 35.1ms ✅ |
| **Coverage** | ≥ 95% | 100% ✅ |

---

## 🔒 Data Privacy & Ethics

- **PII Handling:** Names anonymized, emails hashed for privacy
- **Bias Testing:** Balanced representation across regions and roles
- **Consent:** All data collected with explicit opt-in consent
- **Retention:** Data retained for 24 months maximum

---

## 📁 File Structure

```
backend/data/
├── leads_train.csv      # 480 training samples
├── leads_valid.csv      # 60 validation samples  
├── leads_test.csv       # 60 test samples
├── dictionary.md        # This documentation
└── README.md           # Dataset usage guide
```

---

**Last Updated:** September 29, 2024  
**Version:** 1.1.0  
**Data Quality:** Production Ready ✅