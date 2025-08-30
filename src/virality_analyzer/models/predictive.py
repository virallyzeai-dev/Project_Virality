"""Predictive models for virality analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import pickle
import joblib
from pathlib import Path

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier, MLPRegressor

# XGBoost and CatBoost
import xgboost as xgb
import catboost as cb

from ..core.models import (
    ContentData, FeatureVector, PredictionResult, 
    ModelPerformance, TrainingResult
)
from ..core.config import ModelConfig
from ..features import (
    TextFeatureExtractor, VisualFeatureExtractor,
    EngagementFeatureExtractor, PlatformFeatureExtractor
)


class FeatureProcessor:
    """Process and combine features from different extractors."""
    
    def __init__(self):
        """Initialize feature processor."""
        self.text_extractor = TextFeatureExtractor()
        self.visual_extractor = VisualFeatureExtractor()
        self.engagement_extractor = EngagementFeatureExtractor()
        self.platform_extractor = PlatformFeatureExtractor()
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
    
    def extract_features(self, content_data: ContentData) -> FeatureVector:
        """Extract all features from content data."""
        feature_vector = FeatureVector()
        
        # Text features
        if content_data.text:
            feature_vector.text_features = self.text_extractor.extract_all_features(content_data.text)
            feature_vector.embeddings = self.text_extractor.get_text_embeddings(content_data.text)
        
        # Visual features
        if content_data.image_url:
            feature_vector.visual_features = self.visual_extractor.extract_all_features(content_data.image_url)
        elif content_data.video_url:
            feature_vector.visual_features = self.visual_extractor.extract_video_features(content_data.video_url)
        
        # Engagement features
        feature_vector.engagement_features = self.engagement_extractor.extract_all_features(
            content_data.metrics, content_data.posting_time
        )
        
        # Platform features
        feature_vector.platform_features = self.platform_extractor.extract_all_features(content_data)
        
        return feature_vector
    
    def features_to_array(self, feature_vector: FeatureVector) -> np.ndarray:
        """Convert feature vector to numpy array."""
        features = []
        
        # Combine all feature dictionaries
        all_features = {}
        all_features.update(feature_vector.text_features)
        all_features.update(feature_vector.visual_features)
        all_features.update(feature_vector.engagement_features)
        all_features.update(feature_vector.platform_features)
        
        # If we haven't fitted yet, store feature names
        if not self.is_fitted:
            self.feature_names = sorted(all_features.keys())
        
        # Create feature array in consistent order
        feature_array = []
        for feature_name in self.feature_names:
            value = all_features.get(feature_name, 0.0)
            feature_array.append(float(value))
        
        return np.array(feature_array)
    
    def fit_scaler(self, feature_vectors: List[FeatureVector]) -> None:
        """Fit the feature scaler."""
        feature_arrays = [self.features_to_array(fv) for fv in feature_vectors]
        X = np.array(feature_arrays)
        self.scaler.fit(X)
        self.is_fitted = True
    
    def transform_features(self, feature_vectors: List[FeatureVector]) -> np.ndarray:
        """Transform feature vectors to scaled numpy arrays."""
        feature_arrays = [self.features_to_array(fv) for fv in feature_vectors]
        X = np.array(feature_arrays)
        
        if self.is_fitted:
            return self.scaler.transform(X)
        else:
            return X


class ViralityPredictor:
    """Main predictor class for content virality."""
    
    def __init__(self, model_type: str = "xgboost", virality_threshold: float = 2.0):
        """Initialize the virality predictor."""
        self.model_type = model_type
        self.virality_threshold = virality_threshold
        self.model = None
        self.feature_processor = FeatureProcessor()
        self.is_trained = False
        
        # Initialize model based on type
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the ML model based on model_type."""
        if self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(**ModelConfig.XGBOOST_PARAMS)
        elif self.model_type == "catboost":
            self.model = cb.CatBoostClassifier(**ModelConfig.CATBOOST_PARAMS)
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == "neural_net":
            params = ModelConfig.NEURAL_NET_PARAMS
            self.model = MLPClassifier(
                hidden_layer_sizes=params["hidden_layers"],
                learning_rate_init=params["learning_rate"],
                max_iter=params["epochs"],
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def prepare_training_data(
        self, 
        content_data_list: List[ContentData]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from content data list."""
        # Extract features
        feature_vectors = []
        labels = []
        
        for content_data in content_data_list:
            feature_vector = self.feature_processor.extract_features(content_data)
            feature_vectors.append(feature_vector)
            
            # Create binary label based on virality coefficient
            virality_coeff = self.feature_processor.engagement_extractor.metrics_calculator.calculate_virality_coefficient(
                content_data.metrics.shares,
                content_data.metrics.unique_users_reached or content_data.metrics.impressions
            )
            is_viral = 1 if virality_coeff >= self.virality_threshold else 0
            labels.append(is_viral)
        
        # Fit scaler and transform features
        self.feature_processor.fit_scaler(feature_vectors)
        X = self.feature_processor.transform_features(feature_vectors)
        y = np.array(labels)
        
        return X, y
    
    def train(
        self, 
        content_data_list: List[ContentData],
        validation_split: float = 0.2,
        perform_cv: bool = True
    ) -> TrainingResult:
        """Train the virality prediction model."""
        # Prepare data
        X, y = self.prepare_training_data(content_data_list)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Validate model
        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        performance = ModelPerformance(
            accuracy=accuracy_score(y_val, y_pred),
            precision=precision_score(y_val, y_pred),
            recall=recall_score(y_val, y_pred),
            f1_score=f1_score(y_val, y_pred),
            auc_roc=roc_auc_score(y_val, y_pred_proba),
            confusion_matrix=confusion_matrix(y_val, y_pred).tolist()
        )
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(
                self.feature_processor.feature_names,
                self.model.feature_importances_
            ))
            performance.feature_importance = importance_dict
        
        # Cross-validation
        if perform_cv:
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='f1')
            print(f"Cross-validation F1 scores: {cv_scores}")
            print(f"Mean CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Create training result
        training_result = TrainingResult(
            model_type=self.model_type,
            performance=performance,
            model_path="",  # Will be set when saving
            training_data_size=len(X_train),
            validation_data_size=len(X_val),
            hyperparameters=self._get_model_params()
        )
        
        return training_result
    
    def predict(self, content_data: ContentData) -> PredictionResult:
        """Predict virality for a single piece of content."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        feature_vector = self.feature_processor.extract_features(content_data)
        X = self.feature_processor.transform_features([feature_vector])
        
        # Make prediction
        prediction_proba = self.model.predict_proba(X)[0]
        virality_score = prediction_proba[1]  # Probability of being viral
        is_viral_prediction = virality_score >= 0.5
        
        # Feature importance for this prediction
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(
                self.feature_processor.feature_names,
                self.model.feature_importances_
            ))
            # Get top 10 most important features
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            feature_importance = dict(sorted_features[:10])
        
        return PredictionResult(
            content_id=content_data.content_id,
            virality_score=float(virality_score),
            is_viral_prediction=is_viral_prediction,
            confidence=float(max(prediction_proba)),
            feature_importance=feature_importance
        )
    
    def predict_batch(self, content_data_list: List[ContentData]) -> List[PredictionResult]:
        """Predict virality for multiple pieces of content."""
        return [self.predict(content_data) for content_data in content_data_list]
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model and feature processor."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'feature_processor': self.feature_processor,
            'model_type': self.model_type,
            'virality_threshold': self.virality_threshold,
            'feature_names': self.feature_processor.feature_names
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model and feature processor."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_processor = model_data['feature_processor']
        self.model_type = model_data['model_type']
        self.virality_threshold = model_data['virality_threshold']
        self.is_trained = True
    
    def _get_model_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        else:
            return {}
    
    def hyperparameter_tuning(
        self, 
        content_data_list: List[ContentData],
        param_grid: Optional[Dict[str, List]] = None
    ) -> Dict[str, Any]:
        """Perform hyperparameter tuning using GridSearchCV."""
        X, y = self.prepare_training_data(content_data_list)
        
        if param_grid is None:
            # Default parameter grids for different models
            if self.model_type == "xgboost":
                param_grid = {
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [50, 100, 200]
                }
            elif self.model_type == "random_forest":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                }
            else:
                param_grid = {}
        
        if param_grid:
            grid_search = GridSearchCV(
                self.model, param_grid, cv=5, scoring='f1', n_jobs=-1
            )
            grid_search.fit(X, y)
            
            # Update model with best parameters
            self.model = grid_search.best_estimator_
            self.is_trained = True
            
            return {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
        
        return {}
