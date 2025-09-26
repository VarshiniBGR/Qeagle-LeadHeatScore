import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb
from typing import Dict, List, Tuple, Any
import os
from app.config import settings
from app.models.schemas import LeadInput, HeatScore, LeadScore
from app.utils.logging import get_logger


logger = get_logger(__name__)


class LeadClassifier:
    """Lead heat score classification service."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        self.class_names = ['cold', 'warm', 'hot']
        self.thresholds = settings.get_thresholds()
        self.is_trained = False
        
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for training/prediction."""
        # Create feature matrix
        features = []
        
        # Categorical features
        categorical_features = ['source', 'region', 'role', 'campaign', 'last_touch', 'prior_course_interest']
        
        for feature in categorical_features:
            if feature in df.columns:
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                    encoded = self.label_encoders[feature].fit_transform(df[feature])
                else:
                    # Handle unseen categories
                    try:
                        encoded = self.label_encoders[feature].transform(df[feature])
                    except ValueError:
                        # Map unseen categories to most frequent category
                        encoded = np.zeros(len(df))
                        logger.warning(f"Unseen categories in {feature}, mapping to default")
                
                features.append(encoded)
        
        # Numerical features
        numerical_features = ['recency_days', 'page_views']
        for feature in numerical_features:
            if feature in df.columns:
                features.append(df[feature].values)
        
        # Combine features
        feature_matrix = np.column_stack(features)
        
        # Store feature names
        self.feature_names = categorical_features + numerical_features
        
        return feature_matrix
    
    def _create_target(self, df: pd.DataFrame) -> np.ndarray:
        """Create target variable based on business rules."""
        targets = []
        
        for _, row in df.iterrows():
            # Business rules for heat scoring
            score = 0
            
            # Recency (more recent = hotter)
            if row['recency_days'] <= 3:
                score += 3
            elif row['recency_days'] <= 7:
                score += 2
            elif row['recency_days'] <= 14:
                score += 1
            
            # Page views (more engagement = hotter)
            if row['page_views'] >= 20:
                score += 3
            elif row['page_views'] >= 10:
                score += 2
            elif row['page_views'] >= 5:
                score += 1
            
            # Prior interest
            interest_map = {'high': 3, 'medium': 2, 'low': 1, 'none': 0}
            score += interest_map.get(row['prior_course_interest'], 0)
            
            # Source quality
            source_map = {'referral': 3, 'website': 2, 'linkedin': 2, 'social': 1, 'other': 1}
            score += source_map.get(row['source'], 1)
            
            # Convert score to class
            if score >= 8:
                targets.append(2)  # hot
            elif score >= 5:
                targets.append(1)  # warm
            else:
                targets.append(0)  # cold
        
        return np.array(targets)
    
    def train(self, csv_path: str) -> Dict[str, Any]:
        """Train the classification model."""
        try:
            # Load data
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded training data: {len(df)} samples")
            
            # Prepare features and targets
            X = self._prepare_features(df)
            y = self._create_target(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Logistic Regression with calibration
            base_model = LogisticRegression(random_state=42, max_iter=1000)
            self.model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            y_proba = self.model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            report = classification_report(y_test, y_pred, target_names=self.class_names, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            
            # Calculate calibration score
            calibration_score = self._calculate_calibration_score(y_test, y_proba)
            
            metrics = {
                'accuracy': report['accuracy'],
                'precision': {k: v['precision'] for k, v in report.items() if k in self.class_names},
                'recall': {k: v['recall'] for k, v in report.items() if k in self.class_names},
                'f1_score': {k: v['f1-score'] for k, v in report.items() if k in self.class_names},
                'confusion_matrix': cm.tolist(),
                'calibration_score': calibration_score
            }
            
            self.is_trained = True
            
            # Save model
            self.save_model()
            
            logger.info("Model training completed", metrics=metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def _calculate_calibration_score(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Calculate calibration score (Brier score)."""
        from sklearn.metrics import brier_score_loss
        
        # Convert to binary for each class
        scores = []
        for i in range(len(self.class_names)):
            binary_true = (y_true == i).astype(int)
            binary_proba = y_proba[:, i]
            score = brier_score_loss(binary_true, binary_proba)
            scores.append(score)
        
        return 1 - np.mean(scores)  # Higher is better
    
    def predict(self, lead_data: LeadInput) -> LeadScore:
        """Predict heat score for a single lead."""
        if not self.is_trained:
            raise ValueError("Model not trained. Please train the model first.")
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([lead_data.dict()])
            
            # Prepare features
            X = self._prepare_features(df)
            X_scaled = self.scaler.transform(X)
            
            # Predict probabilities
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Get feature importance
            feature_importance = self._get_feature_importance(X[0])
            
            # Determine heat score based on highest probability
            hot_prob = probabilities[2]
            warm_prob = probabilities[1]
            cold_prob = probabilities[0]
            
            # Find the class with highest probability
            max_prob = max(hot_prob, warm_prob, cold_prob)
            
            if max_prob == hot_prob:
                heat_score = HeatScore.HOT
                confidence = hot_prob
            elif max_prob == warm_prob:
                heat_score = HeatScore.WARM
                confidence = warm_prob
            else:
                heat_score = HeatScore.COLD
                confidence = cold_prob
            
            return LeadScore(
                lead_id=str(hash(str(lead_data.dict()))),
                heat_score=heat_score,
                confidence=confidence,
                probabilities={
                    'cold': cold_prob,
                    'warm': warm_prob,
                    'hot': hot_prob
                },
                features_importance=feature_importance
            )
            
        except Exception as e:
            logger.error(f"Error predicting lead score: {e}")
            raise
    
    def _get_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """Get feature importance for a prediction."""
        # Try different ways to access coefficients based on model type
        coef = None
        if hasattr(self.model, 'base_estimator') and hasattr(self.model.base_estimator, 'coef_'):
            coef = self.model.base_estimator.coef_[0]
        elif hasattr(self.model, 'coef_'):
            coef = self.model.coef_[0]
        elif hasattr(self.model, 'estimator') and hasattr(self.model.estimator, 'coef_'):
            coef = self.model.estimator.coef_[0]
        
        if coef is not None:
            importance = {}
            for i, feature_name in enumerate(self.feature_names):
                if i < len(coef):
                    importance[feature_name] = float(coef[i])
            return importance
        return {}
    
    def save_model(self):
        """Save trained model and components."""
        os.makedirs(settings.model_dir, exist_ok=True)
        
        model_path = settings.get_model_path('lead_clf')
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'thresholds': self.thresholds,
            'is_trained': self.is_trained
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self):
        """Load trained model and components."""
        model_path = settings.get_model_path('lead_clf')
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return False
        
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_names = model_data['feature_names']
            self.class_names = model_data['class_names']
            self.thresholds = model_data['thresholds']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


# Global classifier instance
classifier = LeadClassifier()
 
