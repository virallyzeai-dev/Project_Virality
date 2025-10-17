"""Explainable AI models for understanding predictions."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import warnings

warnings.filterwarnings("ignore")

# SHAP for model explanations
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Model explanations will be limited.")

# LIME for local explanations
try:
    from lime import lime_tabular

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME not available. Local explanations will be limited.")

# OpenAI for natural language explanations
try:
    import openai
    from ..core.config import settings

    OPENAI_AVAILABLE = True
    if settings.openai_api_key:
        openai.api_key = settings.openai_api_key
except ImportError:
    OPENAI_AVAILABLE = False

from ..core.models import ContentData, ExplanationResult, PredictionResult
from .predictive import ViralityPredictor


class ModelExplainer:
    """Explain model predictions using various XAI techniques."""

    def __init__(self, predictor: ViralityPredictor):
        """Initialize the model explainer."""
        self.predictor = predictor
        self.shap_explainer = None
        self.lime_explainer = None
        self.background_data = None

    def setup_shap_explainer(self, background_data: np.ndarray) -> None:
        """Setup SHAP explainer with background data."""
        if not SHAP_AVAILABLE:
            return

        self.background_data = background_data

        # Choose appropriate SHAP explainer based on model type
        if self.predictor.model_type in ["xgboost", "catboost", "random_forest"]:
            # Tree-based explainer
            self.shap_explainer = shap.TreeExplainer(self.predictor.model)
        else:
            # For neural networks and other models
            def model_predict(X):
                return self.predictor.model.predict_proba(X)[:, 1]

            self.shap_explainer = shap.KernelExplainer(
                model_predict, background_data[:100]  # Use subset for efficiency
            )

    def setup_lime_explainer(self, training_data: np.ndarray) -> None:
        """Setup LIME explainer with training data."""
        if not LIME_AVAILABLE:
            return

        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=self.predictor.feature_processor.feature_names,
            class_names=["Not Viral", "Viral"],
            mode="classification",
        )

    def explain_with_shap(
        self, content_data: ContentData, max_features: int = 10
    ) -> Dict[str, float]:
        """Explain prediction using SHAP values."""
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return {}

        # Get features for the content
        feature_vector = self.predictor.feature_processor.extract_features(content_data)
        X = self.predictor.feature_processor.transform_features([feature_vector])

        # Calculate SHAP values
        try:
            shap_values = self.shap_explainer.shap_values(X)

            # For binary classification, get values for positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class

            # Create feature importance dictionary
            feature_names = self.predictor.feature_processor.feature_names
            importance_dict = dict(zip(feature_names, shap_values[0]))

            # Sort by absolute importance and return top features
            sorted_features = sorted(
                importance_dict.items(), key=lambda x: abs(x[1]), reverse=True
            )

            return dict(sorted_features[:max_features])

        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            return {}

    def explain_with_lime(
        self, content_data: ContentData, max_features: int = 10
    ) -> Dict[str, float]:
        """Explain prediction using LIME."""
        if not LIME_AVAILABLE or self.lime_explainer is None:
            return {}

        # Get features for the content
        feature_vector = self.predictor.feature_processor.extract_features(content_data)
        X = self.predictor.feature_processor.transform_features([feature_vector])

        # Define prediction function for LIME
        def predict_fn(X):
            return self.predictor.model.predict_proba(X)

        try:
            # Generate explanation
            explanation = self.lime_explainer.explain_instance(
                X[0], predict_fn, num_features=max_features
            )

            # Extract feature importances
            importance_dict = {}
            for feature_idx, importance in explanation.as_list():
                feature_name = self.predictor.feature_processor.feature_names[
                    feature_idx
                ]
                importance_dict[feature_name] = importance

            return importance_dict

        except Exception as e:
            print(f"LIME explanation failed: {e}")
            return {}

    def generate_text_explanation(
        self,
        content_data: ContentData,
        prediction_result: PredictionResult,
        feature_importance: Dict[str, float],
    ) -> str:
        """Generate human-readable explanation of the prediction."""

        # Basic rule-based explanation
        explanation_parts = []

        # Overall prediction
        if prediction_result.is_viral_prediction:
            explanation_parts.append(
                f"This content is predicted to be VIRAL with a {prediction_result.virality_score:.1%} probability."
            )
        else:
            explanation_parts.append(
                f"This content is predicted to be NON-VIRAL with a {(1-prediction_result.virality_score):.1%} probability."
            )

        # Top contributing factors
        if feature_importance:
            positive_factors = {k: v for k, v in feature_importance.items() if v > 0}
            negative_factors = {k: v for k, v in feature_importance.items() if v < 0}

            if positive_factors:
                top_positive = sorted(
                    positive_factors.items(), key=lambda x: x[1], reverse=True
                )[:3]
                factor_names = [
                    self._humanize_feature_name(name) for name, _ in top_positive
                ]
                explanation_parts.append(
                    f"Key factors supporting virality: {', '.join(factor_names)}"
                )

            if negative_factors:
                top_negative = sorted(
                    negative_factors.items(), key=lambda x: abs(x[1]), reverse=True
                )[:3]
                factor_names = [
                    self._humanize_feature_name(name) for name, _ in top_negative
                ]
                explanation_parts.append(
                    f"Key factors hindering virality: {', '.join(factor_names)}"
                )

        return " ".join(explanation_parts)

    def generate_llm_explanation(
        self,
        content_data: ContentData,
        prediction_result: PredictionResult,
        feature_importance: Dict[str, float],
    ) -> Optional[str]:
        """Generate explanation using LLM (OpenAI GPT)."""
        if not OPENAI_AVAILABLE or not settings.openai_api_key:
            return None

        try:
            # Prepare context for LLM
            context = self._prepare_llm_context(
                content_data, prediction_result, feature_importance
            )

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in social media content analysis. Explain why content will or won't go viral based on the provided data and model predictions.",
                    },
                    {"role": "user", "content": context},
                ],
                max_tokens=300,
                temperature=0.7,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"LLM explanation failed: {e}")
            return None

    def _prepare_llm_context(
        self,
        content_data: ContentData,
        prediction_result: PredictionResult,
        feature_importance: Dict[str, float],
    ) -> str:
        """Prepare context for LLM explanation."""
        context_parts = []

        # Content information
        if content_data.text:
            context_parts.append(
                f"Content text: '{content_data.text[:200]}...' if longer"
            )

        context_parts.append(f"Platform: {content_data.platform.value}")
        context_parts.append(f"Content type: {content_data.content_type.value}")

        # Prediction
        context_parts.append(
            f"Virality prediction: {prediction_result.virality_score:.1%} probability"
        )
        context_parts.append(
            f"Predicted as: {'VIRAL' if prediction_result.is_viral_prediction else 'NON-VIRAL'}"
        )

        # Key features
        if feature_importance:
            top_features = sorted(
                feature_importance.items(), key=lambda x: abs(x[1]), reverse=True
            )[:5]
            feature_list = [
                f"{self._humanize_feature_name(name)}: {value:.3f}"
                for name, value in top_features
            ]
            context_parts.append(f"Key influencing factors: {'; '.join(feature_list)}")

        context_parts.append(
            "Please explain in simple terms why this content is predicted to be viral or not viral, and what could be improved."
        )

        return "\n".join(context_parts)

    def _humanize_feature_name(self, feature_name: str) -> str:
        """Convert technical feature names to human-readable descriptions."""
        humanized_names = {
            "sentiment_polarity": "sentiment positivity",
            "emotional_intensity": "emotional intensity",
            "readability_score": "readability",
            "hashtag_count": "number of hashtags",
            "face_count": "number of faces",
            "brightness": "image brightness",
            "engagement_rate": "engagement rate",
            "virality_coefficient": "viral potential",
            "posting_hour": "posting time",
            "text_length": "text length",
            "word_count": "word count",
            "caps_ratio": "use of capital letters",
            "emoji_count": "number of emojis",
            "exclamation_count": "use of exclamation marks",
            "aesthetic_score": "visual appeal",
            "completion_rate": "video completion rate",
        }

        return humanized_names.get(feature_name, feature_name.replace("_", " "))

    def generate_recommendations(
        self,
        content_data: ContentData,
        prediction_result: PredictionResult,
        feature_importance: Dict[str, float],
    ) -> List[str]:
        """Generate actionable recommendations for improving content virality."""
        recommendations = []

        if not prediction_result.is_viral_prediction:
            # Content is predicted to be non-viral, suggest improvements

            # Analyze negative contributing factors
            negative_factors = {k: v for k, v in feature_importance.items() if v < 0}
            top_negative = sorted(
                negative_factors.items(), key=lambda x: abs(x[1]), reverse=True
            )[:3]

            for factor, impact in top_negative:
                recommendations.extend(self._get_factor_recommendations(factor, impact))

        # General recommendations based on content type and platform
        recommendations.extend(self._get_platform_recommendations(content_data))

        # Remove duplicates and limit to top recommendations
        unique_recommendations = list(dict.fromkeys(recommendations))
        return unique_recommendations[:5]

    def _get_factor_recommendations(self, factor: str, impact: float) -> List[str]:
        """Get recommendations for specific factors."""
        recommendations = []

        factor_recommendations = {
            "sentiment_polarity": [
                "Consider using more positive language",
                "Add emotional appeal to your content",
            ],
            "emotional_intensity": [
                "Increase emotional engagement",
                "Use more compelling storytelling",
            ],
            "readability_score": [
                "Simplify your language for broader appeal",
                "Use shorter sentences",
            ],
            "hashtag_count": [
                "Optimize your hashtag usage",
                "Research trending hashtags",
            ],
            "posting_hour": [
                "Post during peak engagement hours",
                "Consider your audience's time zone",
            ],
            "text_length": [
                "Adjust content length for your platform",
                "Consider breaking up long content",
            ],
            "face_count": [
                "Include people in your visuals",
                "Use authentic human faces",
            ],
            "brightness": [
                "Improve image lighting and contrast",
                "Use more vibrant visuals",
            ],
            "aesthetic_score": [
                "Enhance visual quality",
                "Use professional photography/editing",
            ],
        }

        return factor_recommendations.get(factor, [])

    def _get_platform_recommendations(self, content_data: ContentData) -> List[str]:
        """Get platform-specific recommendations."""
        platform_recs = {
            "twitter": [
                "Keep text under 280 characters",
                "Use 1-2 relevant hashtags",
                "Include visuals",
            ],
            "instagram": [
                "Use high-quality visuals",
                "Write engaging captions",
                "Use 5-11 hashtags",
            ],
            "tiktok": [
                "Create engaging first 3 seconds",
                "Use trending sounds",
                "Keep videos under 60 seconds",
            ],
            "youtube": [
                "Create compelling thumbnails",
                "Write descriptive titles",
                "Optimize for search",
            ],
            "linkedin": [
                "Share professional insights",
                "Use industry-relevant hashtags",
                "Engage in discussions",
            ],
        }

        return platform_recs.get(
            content_data.platform.value,
            ["Focus on high-quality content", "Engage with your audience"],
        )

    def explain_prediction(
        self,
        content_data: ContentData,
        use_shap: bool = True,
        use_lime: bool = False,
        use_llm: bool = True,
    ) -> ExplanationResult:
        """Generate comprehensive explanation for a prediction."""

        # Get prediction
        prediction_result = self.predictor.predict(content_data)

        # Get feature importance
        feature_importance = {}

        if use_shap and SHAP_AVAILABLE:
            shap_importance = self.explain_with_shap(content_data)
            feature_importance.update(shap_importance)

        if use_lime and LIME_AVAILABLE and not feature_importance:
            lime_importance = self.explain_with_lime(content_data)
            feature_importance.update(lime_importance)

        # Use model's built-in feature importance as fallback
        if not feature_importance:
            feature_importance = prediction_result.feature_importance

        # Generate text explanation
        if use_llm:
            explanation_text = self.generate_llm_explanation(
                content_data, prediction_result, feature_importance
            )

        if not explanation_text or not use_llm:
            explanation_text = self.generate_text_explanation(
                content_data, prediction_result, feature_importance
            )

        # Extract key factors
        key_factors = list(feature_importance.keys())[:5]

        # Generate recommendations
        recommendations = self.generate_recommendations(
            content_data, prediction_result, feature_importance
        )

        return ExplanationResult(
            content_id=content_data.content_id,
            explanation_text=explanation_text,
            key_factors=key_factors,
            recommendations=recommendations,
        )
