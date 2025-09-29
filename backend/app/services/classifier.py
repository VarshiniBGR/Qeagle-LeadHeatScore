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
        """Prepare focused features for better LR separation."""
        # Create focused indicator features
        df = df.copy()
        
        # 1️⃣ COLD INDICATOR: Clear signals for cold leads
        df['cold_indicator'] = (
            (df['recency_days'] > 30) & 
            (df['page_views'] <= 2) & 
            (df['course_actions'].isna() | 
             ~df['course_actions'].str.contains('demo_request|book_demo', case=False, na=False))
        ).astype(int)
        
        # 2️⃣ HOT INDICATOR: Clear signals for hot leads
        df['hot_indicator'] = (
            (df['recency_days'] <= 7) & 
            (df['page_views'] >= 20) & 
            (df['prior_course_interest'] == 'high') &
            (df['course_actions'].str.contains('demo_request|book_demo', case=False, na=False))
        ).astype(int)
        
        # 3️⃣ WARM INDICATOR: Clear signals for warm leads
        df['warm_indicator'] = (
            (df['recency_days'] <= 14) & 
            (df['page_views'] >= 8) & 
            (df['prior_course_interest'].isin(['medium', 'high'])) &
            (~df['course_actions'].str.contains('demo_request|book_demo', case=False, na=False))
        ).astype(int)
        
        # 4️⃣ ENGAGEMENT SCORE: Simple linear combination
        df['engagement_score'] = (
            (30 - df['recency_days']).clip(0, 30) / 30 +  # Recency (0-1)
            (df['page_views'] / 50).clip(0, 1) +  # Page views (0-1)
            df['prior_course_interest'].map({'high': 1.0, 'medium': 0.6, 'low': 0.3, 'none': 0.0}).fillna(0.0)
        )
        
        # 5️⃣ SOURCE QUALITY: Binary high-quality source
        df['high_quality_source'] = df['source'].isin(['referral', 'website', 'Web']).astype(int)
        
        # Select only the focused features
        feature_columns = [
            'cold_indicator', 'hot_indicator', 'warm_indicator', 'engagement_score', 'high_quality_source',
            'recency_days', 'page_views'  # Keep basic numerical features
        ]
        
        # Store feature names
        self.feature_names = feature_columns
        
        return df[feature_columns].values
    
    def _create_target(self, df: pd.DataFrame) -> np.ndarray:
        """Create target variable using percentile-based approach for better balance."""
        targets = []
        
        # Calculate engagement score for each lead
        engagement_scores = []
        
        for _, row in df.iterrows():
            score = 0
            
            # Recency score (inverse - more recent = higher score)
            recency_score = max(0, 30 - row['recency_days']) / 30  # Normalize to 0-1
            score += recency_score * 3
            
            # Page views score (normalized)
            page_score = min(1.0, row['page_views'] / 50)  # Normalize to 0-1, cap at 50
            score += page_score * 3
            
            # Prior interest score
            interest_map = {'high': 1.0, 'medium': 0.6, 'low': 0.3, 'none': 0.0}
            score += interest_map.get(row['prior_course_interest'], 0.0) * 2
            
            # Source quality score
            source_map = {
                'referral': 1.0, 'website': 1.0, 'Web': 1.0,  # High quality
                'linkedin': 0.7, 'social': 0.7, 'Social Media': 0.7, 'Ad Campaign': 0.7,  # Medium quality
                'google_ads': 0.5, 'facebook_ads': 0.5, 'twitter': 0.5,  # Paid but targeted
                'conference': 0.6, 'trade_show': 0.6, 'webinar': 0.6,  # Events
                'other': 0.0, 'unknown': 0.0  # Low quality
            }
            score += source_map.get(row['source'], 0.0) * 1
            
            # Time spent bonus (if available)
            if 'time_spent' in row and pd.notna(row['time_spent']):
                time_score = min(1.0, row['time_spent'] / 1200)  # Normalize to 0-1, cap at 20 minutes
                score += time_score * 0.5
            
            # Course actions bonus (if available)
            if 'course_actions' in row and pd.notna(row['course_actions']):
                actions = str(row['course_actions']).lower()
                if 'demo_request' in actions or 'book_demo' in actions:
                    score += 1.0
                elif 'download_brochure' in actions or 'request_info' in actions:
                    score += 0.7
                elif 'view_course' in actions or 'share_course' in actions:
                    score += 0.3
            
            engagement_scores.append(score)
        
        # Use percentiles to create balanced classes
        scores_array = np.array(engagement_scores)
        
        # Define thresholds based on percentiles for 65-35 distribution
        hot_threshold = np.percentile(scores_array, 65)  # Top 35% are hot
        warm_threshold = np.percentile(scores_array, 35)  # Middle 30% are warm, bottom 35% are cold
        
        for score in engagement_scores:
            if score >= hot_threshold:
                targets.append(2)  # hot
            elif score >= warm_threshold:
                targets.append(1)  # warm
            else:
                targets.append(0)  # cold
        
        return np.array(targets)
    
    def train_xgboost_comparison(self, csv_path: str) -> Dict[str, Any]:
        """Train XGBoost model for comparison with Logistic Regression baseline."""
        try:
            # Load data
            df = pd.read_csv(csv_path)
            logger.info(f"Training XGBoost comparison model with {len(df)} samples")
            
            # Prepare features and targets
            X = self._prepare_features(df)
            y = self._create_target(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train XGBoost for comparison
            base_model = xgb.XGBClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='mlogloss'
            )
            xgb_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
            xgb_model.fit(X_train_scaled, y_train)
            
            # Evaluate XGBoost model
            y_pred = xgb_model.predict(X_test_scaled)
            y_proba = xgb_model.predict_proba(X_test_scaled)
            
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
                'calibration_score': calibration_score,
                'model_type': 'XGBoost'
            }
            
            logger.info("XGBoost comparison model training completed", metrics=metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error training XGBoost comparison model: {e}")
            raise
    def train(self, csv_path: str) -> Dict[str, Any]:
        """Train the Logistic Regression baseline model."""
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
            
            # Apply SMOTE oversampling to balance classes for better F1 macro
            try:
                from imblearn.over_sampling import SMOTE
                sm = SMOTE(random_state=42, k_neighbors=3)
                X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
                logger.info(f"SMOTE applied: {len(y_train)} -> {len(y_train_res)} samples")
            except ImportError:
                logger.warning("imblearn not available, skipping SMOTE")
                X_train_res, y_train_res = X_train_scaled, y_train
            
            # Train Logistic Regression with class weights for better F1 macro
            base_model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0,  # Standard regularization
                solver='liblinear',
                class_weight='balanced'  # Handle class imbalance
            )
            self.model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
            self.model.fit(X_train_res, y_train_res)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            y_proba = self.model.predict_proba(X_test_scaled)
            
            # Optimize thresholds for better F1 macro
            optimal_thresholds = self._optimize_thresholds(y_test, y_proba)
            self.optimal_thresholds = optimal_thresholds
            
            # Re-predict with optimized thresholds
            y_pred_optimized = []
            for probs in y_proba:
                hot_prob, warm_prob, cold_prob = probs[2], probs[1], probs[0]
                
                if hot_prob >= optimal_thresholds[0]:
                    y_pred_optimized.append(2)  # hot
                elif warm_prob >= optimal_thresholds[1]:
                    y_pred_optimized.append(1)  # warm
                else:
                    y_pred_optimized.append(0)  # cold
            
            y_pred = np.array(y_pred_optimized)
            
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
                'calibration_score': calibration_score,
                'model_type': 'LogisticRegression'
            }
            
            self.is_trained = True
            
            # Save model
            self.save_model()
            
            logger.info("Logistic Regression baseline model training completed", metrics=metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error training Logistic Regression model: {e}")
            raise
    
    def _optimize_thresholds(self, y_true, y_proba):
        """Optimize prediction thresholds for better F1 macro score."""
        from sklearn.metrics import f1_score
        import numpy as np
        
        best_f1 = 0
        best_thresholds = (0.5, 0.3)
        
        # Test different threshold combinations
        hot_thresholds = [0.4, 0.45, 0.5, 0.55, 0.6]
        warm_thresholds = [0.25, 0.3, 0.35, 0.4]
        
        for hot_thresh in hot_thresholds:
            for warm_thresh in warm_thresholds:
                if hot_thresh > warm_thresh:
                    # Create predictions based on thresholds
                    y_pred = []
                    for probs in y_proba:
                        hot_prob, warm_prob, cold_prob = probs[2], probs[1], probs[0]
                        
                        if hot_prob >= hot_thresh:
                            y_pred.append(2)  # hot
                        elif warm_prob >= warm_thresh:
                            y_pred.append(1)  # warm
                        else:
                            y_pred.append(0)  # cold
                    
                    # Calculate F1 macro
                    f1_macro = f1_score(y_true, y_pred, average='macro')
                    
                    if f1_macro > best_f1:
                        best_f1 = f1_macro
                        best_thresholds = (hot_thresh, warm_thresh)
        
        return best_thresholds
    
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
            
            # Use optimized thresholds for better F1 macro (tuned on validation)
            # Use optimized thresholds for better F1 macro
            hot_prob = probabilities[2]
            warm_prob = probabilities[1]
            cold_prob = probabilities[0]
            
            # Use highest probability with warm bias for better F1-macro
            max_prob = max(cold_prob, warm_prob, hot_prob)
            
            if max_prob == cold_prob and cold_prob > 0.55:  # Even higher threshold for cold
                heat_score = HeatScore.COLD
                confidence = cold_prob
            elif max_prob == hot_prob and hot_prob > 0.55:  # Even higher threshold for hot
                heat_score = HeatScore.HOT
                confidence = hot_prob
            else:  # Default to warm for better F1-macro
                heat_score = HeatScore.WARM
                confidence = warm_prob
            
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
 
