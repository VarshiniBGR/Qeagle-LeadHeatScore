# Lead HeatScore Dataset Dictionary

## Overview
This dataset contains synthetic lead data for training and evaluating the Lead HeatScore classification system. The dataset includes 2,001 leads with features for lead scoring and next-action recommendation.

## Dataset Structure
- **Total Records**: 2,001 leads
- **Train Split**: 1,400 rows (70%)
- **Validation Split**: 300 rows (15%)
- **Test Split**: 301 rows (15%)
- **Random Seed**: 42 (for reproducibility)

## Feature Descriptions

### Basic Information
| Field | Type | Description | Example Values |
|-------|------|-------------|----------------|
| `name` | String | Lead's full name | "John Smith", "Sarah Johnson" |
| `email` | String | Lead's email address | "john@company.com" |
| `phone` | String | Lead's phone number | "+91-12345-67890" |

### Lead Source & Attribution
| Field | Type | Description | Example Values |
|-------|------|-------------|----------------|
| `source` | Categorical | Lead acquisition channel | "google_ads", "linkedin", "website", "cold_call", "referral", "webinar", "email", "conference" |
| `campaign` | Categorical | Marketing campaign identifier | "enterprise_package", "loyalty_program", "newsletter", "government_contract", "winter_campaign", "spring_promotion" |
| `region` | Categorical | Geographic region | "US", "EU", "APAC", "LATAM", "FR", "DE", "UK", "IN", "AUS" |

### Engagement Metrics
| Field | Type | Description | Example Values |
|-------|------|-------------|----------------|
| `recency_days` | Integer | Days since last interaction | 1-30 |
| `page_views` | Integer | Number of page views | 1-100 |
| `time_spent` | Integer | Time spent on site (seconds) | 100-2000 |
| `last_touch` | Categorical | Last interaction type | "email", "phone", "website", "social", "trial", "purchase", "video_call", "sms" |

### Lead Profile
| Field | Type | Description | Example Values |
|-------|------|-------------|----------------|
| `role` | Categorical | Job title/role | "Manager", "Director", "Analyst", "Coordinator", "Administrator", "Researcher" |
| `prior_course_interest` | Categorical | Interest level in courses | "low", "medium", "high" |

### Behavioral Data
| Field | Type | Description | Example Values |
|-------|------|-------------|----------------|
| `search_keywords` | String | Search terms used | "machine learning", "data science", "project management" |
| `course_actions` | String | Actions taken on courses | "view_course", "download_brochure", "book_demo", "register_interest" |

## Target Variable
The target variable for classification is derived from `prior_course_interest`:
- **Low**: 0 (low interest)
- **Medium**: 1 (medium interest)  
- **High**: 2 (high interest)

## Data Quality Notes
- All data is synthetically generated for demonstration purposes
- No real personal information is included
- Phone numbers follow Indian format (+91-XXXXX-XXXXX)
- Email addresses use example domains
- Names are randomly generated

## Usage Guidelines
- **Training**: Use `leads_train.csv` for model training
- **Validation**: Use `leads_valid.csv` for hyperparameter tuning
- **Testing**: Use `leads_test.csv` for final evaluation
- **Reproducibility**: All splits use `random_state=42`

## Feature Engineering
The following features are used for lead scoring:
- Categorical features are one-hot encoded
- Numerical features are standardized
- Text features (search_keywords) are processed separately
- Target variable is stratified across splits

## Data Preprocessing
1. **Missing Values**: Handled with default values
2. **Categorical Encoding**: One-hot encoding for categorical variables
3. **Feature Scaling**: StandardScaler for numerical features
4. **Text Processing**: Separate pipeline for search keywords

## Model Performance Targets
- **F1 Score (Macro)**: ≥ 0.80
- **Precision**: ≥ 0.75
- **Recall**: ≥ 0.75
- **Calibration**: Brier score < 0.25

## File Structure
```
backend/data/
├── leads.csv           # Original dataset (2,001 rows)
├── leads_train.csv     # Training split (1,400 rows)
├── leads_valid.csv     # Validation split (300 rows)
├── leads_test.csv      # Test split (301 rows)
└── dictionary.md       # This file
```

## Version History
- **v1.0**: Initial dataset with 2,001 synthetic leads
- **v1.1**: Added train/valid/test splits with stratification
- **v1.2**: Added comprehensive feature documentation

## Contact
For questions about this dataset, refer to the project documentation or contact the development team.

