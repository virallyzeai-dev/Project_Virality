"""Tests for feature extraction components."""

from datetime import datetime

import numpy as np
import pytest

from virality_analyzer.core.models import (
    ContentData,
    ContentMetrics,
    ContentType,
    Platform,
)
from virality_analyzer.features import (
    EngagementFeatureExtractor,
    PlatformFeatureExtractor,
    TextFeatureExtractor,
    VisualFeatureExtractor,
)


class TestTextFeatureExtractor:
    """Test text feature extraction."""

    def test_readability_features(self):
        extractor = TextFeatureExtractor()

        # Simple text
        simple_text = "This is easy to read. Short sentences work well."
        features = extractor.extract_readability_features(simple_text)

        assert "readability_score" in features
        assert "avg_sentence_length" in features
        assert "complexity_score" in features

        # Complex text
        complex_text = "The implementation of sophisticated algorithms necessitates comprehensive understanding of multifaceted computational paradigms."
        complex_features = extractor.extract_readability_features(complex_text)

        # Simple text should be more readable
        assert features["readability_score"] >= complex_features["readability_score"]

    def test_emotional_features(self):
        extractor = TextFeatureExtractor()

        # Emotional text
        emotional_text = "OMG I'm SO EXCITED!!! This is AMAZING!!! 😍🎉"
        features = extractor.extract_emotional_intensity(emotional_text)

        assert features["emotional_intensity"] > 0.5
        assert features["caps_ratio"] > 0.2
        assert features["exclamation_count"] > 0

    def test_text_embeddings(self):
        extractor = TextFeatureExtractor()

        text = "This is a test sentence for embeddings."
        embeddings = extractor.get_text_embeddings(text)

        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings) > 0


class TestVisualFeatureExtractor:
    """Test visual feature extraction."""

    def test_placeholder_features(self):
        extractor = VisualFeatureExtractor()

        # Test with placeholder URL
        features = extractor.extract_all_features("https://example.com/image.jpg")

        assert "face_count" in features
        assert "brightness" in features
        assert "contrast" in features
        assert "aesthetic_score" in features

        # All should be default values for placeholder
        assert features["face_count"] == 0
        assert 0 <= features["brightness"] <= 1
        assert 0 <= features["aesthetic_score"] <= 1


class TestEngagementFeatureExtractor:
    """Test engagement feature extraction."""

    def test_metrics_features(self):
        extractor = EngagementFeatureExtractor()

        metrics = ContentMetrics(
            likes=100, shares=50, comments=25, impressions=1000, saves=30
        )

        features = extractor.extract_basic_engagement_features(metrics)

        assert features["engagement_rate"] == 0.175  # (100+50+25)/1000
        assert features["like_rate"] == 0.1  # 100/1000
        assert features["share_rate"] == 0.05  # 50/1000
        assert features["save_rate"] == 0.03  # 30/1000

    def test_temporal_features(self):
        extractor = EngagementFeatureExtractor()

        metrics = ContentMetrics(likes=100, shares=50, comments=25, impressions=1000)

        # Peak hour (6 PM)
        peak_time = datetime(2024, 1, 15, 18, 0)
        features = extractor.extract_temporal_features(metrics, peak_time)

        assert features["posting_hour"] == 18
        assert features["is_peak_hour"] == 1
        assert features["day_of_week"] == peak_time.weekday()
        assert features["is_weekend"] in [0, 1]

    def test_virality_features(self):
        extractor = EngagementFeatureExtractor()

        metrics = ContentMetrics(
            likes=100,
            shares=200,  # High share rate
            comments=50,
            impressions=1000,
            unique_users_reached=500,
        )

        features = extractor.extract_virality_features(metrics)

        assert features["virality_coefficient"] == 0.4  # 200/500
        assert features["share_to_like_ratio"] == 2.0  # 200/100
        assert features["comment_to_like_ratio"] == 0.5  # 50/100


class TestPlatformFeatureExtractor:
    """Test platform-specific feature extraction."""

    def test_platform_encoding(self):
        extractor = PlatformFeatureExtractor()

        content_data = ContentData(
            content_id="test",
            text="Test content",
            content_type=ContentType.TEXT,
            platform=Platform.TWITTER,
            metrics=ContentMetrics(likes=10, shares=5, comments=2, impressions=100),
        )

        features = extractor.extract_all_features(content_data)

        # Should have one-hot encoding for platforms
        assert "platform_twitter" in features
        assert features["platform_twitter"] == 1
        assert features.get("platform_instagram", 0) == 0

    def test_content_type_encoding(self):
        extractor = PlatformFeatureExtractor()

        content_data = ContentData(
            content_id="test",
            text="Test content",
            content_type=ContentType.VIDEO,
            platform=Platform.YOUTUBE,
            metrics=ContentMetrics(likes=10, shares=5, comments=2, impressions=100),
        )

        features = extractor.extract_all_features(content_data)

        # Should have one-hot encoding for content types
        assert "content_type_video" in features
        assert features["content_type_video"] == 1
        assert features.get("content_type_text", 0) == 0

    def test_metadata_features(self):
        extractor = PlatformFeatureExtractor()

        content_data = ContentData(
            content_id="test",
            text="Test content",
            content_type=ContentType.TEXT,
            platform=Platform.TWITTER,
            metrics=ContentMetrics(likes=10, shares=5, comments=2, impressions=100),
            metadata={
                "follower_count": 5000,
                "account_age_days": 365,
                "previous_viral_count": 3,
            },
        )

        features = extractor.extract_all_features(content_data)

        assert features["follower_count_log"] > 0  # log(5000)
        assert features["account_age_days"] == 365
        assert features["previous_viral_count"] == 3


class TestFeatureIntegration:
    """Test integration between different feature extractors."""

    def test_complete_feature_extraction(self):
        """Test that all feature extractors work together."""
        text_extractor = TextFeatureExtractor()
        visual_extractor = VisualFeatureExtractor()
        engagement_extractor = EngagementFeatureExtractor()
        platform_extractor = PlatformFeatureExtractor()

        # Create test content
        content_data = ContentData(
            content_id="integration_test",
            text="Amazing content with #hashtags and @mentions! 🚀",
            image_url="https://example.com/image.jpg",
            content_type=ContentType.IMAGE,
            platform=Platform.INSTAGRAM,
            metrics=ContentMetrics(
                likes=150, shares=30, comments=20, impressions=2000, saves=25
            ),
            posting_time=datetime(2024, 1, 15, 14, 30),
            hashtags=["#hashtags"],
            mentions=["@mentions"],
        )

        # Extract all features
        text_features = text_extractor.extract_all_features(content_data.text or "")
        visual_features = visual_extractor.extract_all_features(
            content_data.image_url or ""
        )
        engagement_features = engagement_extractor.extract_all_features(
            content_data.metrics, content_data.posting_time
        )
        platform_features = platform_extractor.extract_all_features(content_data)

        # Verify we got features from each extractor
        assert len(text_features) > 0
        assert len(visual_features) > 0
        assert len(engagement_features) > 0
        assert len(platform_features) > 0

        # Verify no overlapping feature names
        all_feature_names = set()
        for feature_dict in [
            text_features,
            visual_features,
            engagement_features,
            platform_features,
        ]:
            for name in feature_dict.keys():
                assert name not in all_feature_names, f"Duplicate feature name: {name}"
                all_feature_names.add(name)

        # Verify all values are numeric
        for feature_dict in [
            text_features,
            visual_features,
            engagement_features,
            platform_features,
        ]:
            for name, value in feature_dict.items():
                assert isinstance(value, (int, float, np.number)), (
                    f"Non-numeric feature {name}: {value}"
                )


if __name__ == "__main__":
    pytest.main([__file__])
