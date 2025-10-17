"""Basic tests for the virality analyzer."""

from datetime import datetime

import pytest

from virality_analyzer.core.metrics import MetricsCalculator, ViralityMetrics
from virality_analyzer.core.models import (
    ContentData,
    ContentMetrics,
    ContentType,
    Platform,
)
from virality_analyzer.features.text_features import TextFeatureExtractor


class TestMetricsCalculator:
    """Test the metrics calculator."""

    def test_engagement_rate_calculation(self):
        calculator = MetricsCalculator()

        # Test normal case
        rate = calculator.calculate_engagement_rate(100, 50, 25, 1000)
        assert rate == 0.175  # (100 + 50 + 25) / 1000

        # Test zero impressions
        rate = calculator.calculate_engagement_rate(100, 50, 25, 0)
        assert rate == 0.0

    def test_virality_coefficient(self):
        calculator = MetricsCalculator()

        # Test normal case
        coeff = calculator.calculate_virality_coefficient(200, 1000)
        assert coeff == 0.2

        # Test zero users
        coeff = calculator.calculate_virality_coefficient(200, 0)
        assert coeff == 0.0

    def test_virality_metrics_calculation(self):
        calculator = MetricsCalculator(virality_threshold=2.0)

        content_data = {
            "likes": 100,
            "shares": 50,
            "comments": 25,
            "impressions": 1000,
            "unique_users_reached": 500,
        }

        metrics = calculator.calculate_all_metrics(content_data)

        assert isinstance(metrics, ViralityMetrics)
        assert metrics.engagement_rate == 0.175
        assert metrics.virality_coefficient == 0.1  # 50/500
        assert not metrics.is_viral  # 0.1 < 2.0


class TestTextFeatureExtractor:
    """Test the text feature extractor."""

    def test_sentiment_extraction(self):
        extractor = TextFeatureExtractor()

        # Test positive text
        features = extractor.extract_sentiment_features("I love this amazing product!")
        assert "sentiment_polarity" in features
        assert features["sentiment_polarity"] > 0

        # Test negative text
        features = extractor.extract_sentiment_features("This is terrible and awful!")
        assert features["sentiment_polarity"] < 0

        # Test empty text
        features = extractor.extract_sentiment_features("")
        assert features["sentiment_polarity"] == 0.0

    def test_structural_features(self):
        extractor = TextFeatureExtractor()

        text = "This is a test sentence. It has multiple words and sentences!"
        features = extractor.extract_structural_features(text)

        assert features["word_count"] > 0
        assert features["sentence_count"] == 2
        assert features["text_length"] == len(text)
        assert features["avg_word_length"] > 0

    def test_social_features(self):
        extractor = TextFeatureExtractor()

        text = "Check out this #amazing #content with @user mentions! 🚀"
        features = extractor.extract_social_features(text)

        assert features["hashtag_count"] == 2
        assert features["mention_count"] == 1
        assert features["emoji_count"] >= 1
        assert features["exclamation_count"] == 1


class TestContentData:
    """Test the ContentData model."""

    def test_content_data_creation(self):
        metrics = ContentMetrics(likes=100, shares=50, comments=25, impressions=1000)

        content = ContentData(
            content_id="test_001",
            text="Test content",
            content_type=ContentType.TEXT,
            platform=Platform.TWITTER,
            metrics=metrics,
            posting_time=datetime.now(),
        )

        assert content.content_id == "test_001"
        assert content.text == "Test content"
        assert content.content_type == ContentType.TEXT
        assert content.platform == Platform.TWITTER
        assert content.metrics.likes == 100


def test_package_imports():
    """Test that main package imports work correctly."""
    from virality_analyzer import ViralityAnalyzer
    from virality_analyzer.features import TextFeatureExtractor

    # Test basic instantiation
    analyzer = ViralityAnalyzer()
    assert analyzer is not None

    extractor = TextFeatureExtractor()
    assert extractor is not None


if __name__ == "__main__":
    pytest.main([__file__])
