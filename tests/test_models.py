"""Tests for model components."""

import numpy as np
import pytest

from virality_analyzer.core.models import (
    ContentData,
    ContentMetrics,
    ContentType,
    Platform,
)
from virality_analyzer.models import CausalAnalyzer, FeatureProcessor, ViralityPredictor


class TestFeatureProcessor:
    """Test the feature processor."""

    def test_feature_extraction(self):
        processor = FeatureProcessor()

        content_data = ContentData(
            content_id="test_001",
            text="Test content with #hashtags",
            content_type=ContentType.TEXT,
            platform=Platform.TWITTER,
            metrics=ContentMetrics(likes=10, shares=5, comments=2, impressions=100),
        )

        feature_vector = processor.extract_features(content_data)

        assert hasattr(feature_vector, "text_features")
        assert hasattr(feature_vector, "visual_features")
        assert hasattr(feature_vector, "engagement_features")
        assert hasattr(feature_vector, "platform_features")

        # Check that features are extracted
        assert len(feature_vector.text_features) > 0
        assert len(feature_vector.engagement_features) > 0
        assert len(feature_vector.platform_features) > 0

    def test_features_to_array(self):
        processor = FeatureProcessor()

        content_data = ContentData(
            content_id="test_001",
            text="Test content",
            content_type=ContentType.TEXT,
            platform=Platform.TWITTER,
            metrics=ContentMetrics(likes=10, shares=5, comments=2, impressions=100),
        )

        feature_vector = processor.extract_features(content_data)
        feature_array = processor.features_to_array(feature_vector)

        assert isinstance(feature_array, np.ndarray)
        assert len(feature_array) > 0
        assert all(isinstance(x, (int, float, np.number)) for x in feature_array)

    def test_feature_scaling(self):
        processor = FeatureProcessor()

        # Create multiple feature vectors
        content_data_list = []
        for i in range(5):
            content_data = ContentData(
                content_id=f"test_{i:03d}",
                text=f"Test content {i}",
                content_type=ContentType.TEXT,
                platform=Platform.TWITTER,
                metrics=ContentMetrics(
                    likes=10 * (i + 1),
                    shares=5 * (i + 1),
                    comments=2 * (i + 1),
                    impressions=100 * (i + 1),
                ),
            )
            content_data_list.append(content_data)

        feature_vectors = [processor.extract_features(cd) for cd in content_data_list]

        # Fit scaler
        processor.fit_scaler(feature_vectors)

        # Transform features
        transformed = processor.transform_features(feature_vectors)

        assert isinstance(transformed, np.ndarray)
        assert transformed.shape[0] == len(feature_vectors)
        assert processor.is_fitted


class TestViralityPredictor:
    """Test the virality predictor."""

    def test_initialization(self):
        predictor = ViralityPredictor(model_type="xgboost", virality_threshold=2.0)

        assert predictor.model_type == "xgboost"
        assert predictor.virality_threshold == 2.0
        assert predictor.model is not None
        assert not predictor.is_trained

    def test_model_types(self):
        # Test different model types
        for model_type in ["xgboost", "catboost", "random_forest", "neural_net"]:
            predictor = ViralityPredictor(model_type=model_type)
            assert predictor.model is not None

    def test_invalid_model_type(self):
        with pytest.raises(ValueError):
            ViralityPredictor(model_type="invalid_model")

    def test_prepare_training_data(self):
        predictor = ViralityPredictor()

        # Create sample training data
        content_data_list = []
        for i in range(10):
            # Vary the shares to create different virality levels
            shares = i * 50  # Some will be viral, some won't
            content_data = ContentData(
                content_id=f"train_{i:03d}",
                text=f"Training content {i}",
                content_type=ContentType.TEXT,
                platform=Platform.TWITTER,
                metrics=ContentMetrics(
                    likes=10 * (i + 1),
                    shares=shares,
                    comments=2 * (i + 1),
                    impressions=100 * (i + 1),
                    unique_users_reached=50 * (i + 1),
                ),
            )
            content_data_list.append(content_data)

        X, y = predictor.prepare_training_data(content_data_list)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == len(content_data_list)
        assert y.shape[0] == len(content_data_list)
        assert all(label in [0, 1] for label in y)


class TestCausalAnalyzer:
    """Test the causal analyzer."""

    def test_initialization(self):
        analyzer = CausalAnalyzer(virality_threshold=2.0)

        assert analyzer.virality_threshold == 2.0
        assert analyzer.feature_processor is not None
        assert analyzer.causal_model is None

    def test_prepare_causal_data(self):
        analyzer = CausalAnalyzer()

        # Create sample data
        content_data_list = []
        for i in range(5):
            content_data = ContentData(
                content_id=f"causal_{i:03d}",
                text=f"Causal analysis content {i}",
                content_type=ContentType.TEXT,
                platform=Platform.TWITTER,
                metrics=ContentMetrics(
                    likes=10 * (i + 1),
                    shares=5 * (i + 1),
                    comments=2 * (i + 1),
                    impressions=100 * (i + 1),
                ),
            )
            content_data_list.append(content_data)

        data_df = analyzer.prepare_causal_data(content_data_list)

        assert len(data_df) == len(content_data_list)
        assert "virality_coefficient" in data_df.columns
        assert "is_viral" in data_df.columns
        assert "content_id" in data_df.columns

    def test_identify_confounders(self):
        analyzer = CausalAnalyzer()

        # Create simple test data
        data = {
            "treatment": [1, 0, 1, 0, 1, 0],
            "outcome": [3, 1, 4, 1, 3, 2],
            "confounder1": [2, 1, 3, 1, 2, 1],  # Correlated with both
            "confounder2": [0.5, 0.1, 0.6, 0.2, 0.4, 0.1],  # Correlated with both
            "irrelevant": [10, 20, 30, 40, 50, 60],  # Not correlated
        }

        import pandas as pd

        df = pd.DataFrame(data)

        confounders = analyzer.identify_confounders(df, "treatment", "outcome")

        # Should identify the correlated variables
        assert isinstance(confounders, list)

    def test_treatment_effect_estimation(self):
        analyzer = CausalAnalyzer()

        # Create test data with clear treatment effect
        data = {
            "treatment": [1, 1, 1, 0, 0, 0],
            "outcome": [5, 6, 4, 2, 3, 1],  # Higher outcome for treatment group
            "confounder": [1, 1, 1, 0, 0, 0],
        }

        import pandas as pd

        df = pd.DataFrame(data)

        effect_results = analyzer.estimate_treatment_effect_simple(
            df, "treatment", "outcome", ["confounder"]
        )

        assert "simple_effect" in effect_results
        assert "adjusted_effect" in effect_results
        assert "treatment_group_size" in effect_results
        assert "control_group_size" in effect_results

        # Treatment group should have higher outcome
        assert effect_results["simple_effect"] > 0


class TestModelIntegration:
    """Test integration between model components."""

    def test_predictor_training_flow(self):
        """Test the complete training flow."""
        predictor = ViralityPredictor(model_type="random_forest")  # Faster for testing

        # Create larger training dataset
        content_data_list = []
        for i in range(20):
            # Create varied data for better training
            shares = np.random.randint(0, 500)
            unique_users = np.random.randint(100, 1000)

            content_data = ContentData(
                content_id=f"integration_{i:03d}",
                text=f"Integration test content {i} with some text",
                content_type=ContentType.TEXT,
                platform=Platform.TWITTER,
                metrics=ContentMetrics(
                    likes=np.random.randint(10, 200),
                    shares=shares,
                    comments=np.random.randint(5, 50),
                    impressions=np.random.randint(500, 5000),
                    unique_users_reached=unique_users,
                ),
            )
            content_data_list.append(content_data)

        # Train the model
        training_result = predictor.train(
            content_data_list, validation_split=0.3, perform_cv=False
        )

        assert predictor.is_trained
        assert training_result is not None
        assert hasattr(training_result, "performance")
        assert hasattr(training_result.performance, "accuracy")

        # Test prediction
        test_content = content_data_list[0]
        prediction = predictor.predict(test_content)

        assert hasattr(prediction, "virality_score")
        assert hasattr(prediction, "is_viral_prediction")
        assert hasattr(prediction, "confidence")
        assert 0 <= prediction.virality_score <= 1
        assert isinstance(prediction.is_viral_prediction, bool)


if __name__ == "__main__":
    pytest.main([__file__])
