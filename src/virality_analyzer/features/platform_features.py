"""Platform-specific feature extraction."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

import numpy as np

from ..core.models import ContentData, Platform


class PlatformFeatureExtractor:
    """Extract platform-specific features that affect content virality."""

    def __init__(self):
        """Initialize the platform feature extractor."""
        self.platform_peak_hours = {
            Platform.TWITTER: [9, 12, 15, 18, 21],
            Platform.INSTAGRAM: [11, 13, 17, 19, 21],
            Platform.YOUTUBE: [14, 16, 20, 22],
            Platform.TIKTOK: [6, 10, 19, 20, 21, 22],
            Platform.FACEBOOK: [9, 13, 15],
            Platform.LINKEDIN: [8, 9, 10, 11, 12, 17, 18],
            Platform.GENERIC: [9, 12, 15, 18, 21],
        }

        # Trending hashtags data (this would typically come from APIs)
        self.trending_hashtags = set()
        self.hashtag_popularity_scores = {}

    def extract_timing_features(
        self, posting_time: datetime, platform: Platform
    ) -> Dict[str, float]:
        """Extract timing-related features specific to platform."""
        posting_hour = posting_time.hour
        posting_day = posting_time.weekday()  # 0 = Monday

        # Platform-specific peak hours
        peak_hours = self.platform_peak_hours.get(platform, [])
        is_peak_hour = 1.0 if posting_hour in peak_hours else 0.0

        # Distance to nearest peak hour
        if peak_hours:
            distances = [
                min(abs(posting_hour - ph), 24 - abs(posting_hour - ph))
                for ph in peak_hours
            ]
            min_distance_to_peak = min(distances)
        else:
            min_distance_to_peak = 12.0  # Maximum distance

        # Weekend vs weekday
        is_weekend = 1.0 if posting_day >= 5 else 0.0

        # Time zone considerations (assuming UTC for now)
        # In practice, you'd want to consider the target audience's time zone

        return {
            "posting_hour": float(posting_hour),
            "posting_day": float(posting_day),
            "is_peak_hour": is_peak_hour,
            "distance_to_peak_hour": min_distance_to_peak,
            "is_weekend": is_weekend,
            "hour_sin": np.sin(2 * np.pi * posting_hour / 24),  # Cyclical encoding
            "hour_cos": np.cos(2 * np.pi * posting_hour / 24),
            "day_sin": np.sin(2 * np.pi * posting_day / 7),
            "day_cos": np.cos(2 * np.pi * posting_day / 7),
        }

    def extract_hashtag_features(
        self, hashtags: List[str], platform: Platform
    ) -> Dict[str, float]:
        """Extract hashtag-related features."""
        if not hashtags:
            return {
                "hashtag_count": 0.0,
                "avg_hashtag_popularity": 0.0,
                "trending_hashtag_count": 0.0,
                "hashtag_diversity": 0.0,
                "optimal_hashtag_count": 0.0,
            }

        hashtag_count = len(hashtags)

        # Hashtag popularity scores
        popularity_scores = []
        trending_count = 0

        for hashtag in hashtags:
            # Check if hashtag is trending
            if hashtag.lower() in self.trending_hashtags:
                trending_count += 1

            # Get popularity score (default to 1.0 if not found)
            popularity = self.hashtag_popularity_scores.get(hashtag.lower(), 1.0)
            popularity_scores.append(popularity)

        avg_popularity = float(np.mean(popularity_scores))

        # Hashtag diversity (uniqueness)
        unique_hashtags = len(set(h.lower() for h in hashtags))
        hashtag_diversity = unique_hashtags / hashtag_count

        # Optimal hashtag count for platform
        optimal_counts = {
            Platform.TWITTER: 2,
            Platform.INSTAGRAM: 11,
            Platform.TIKTOK: 5,
            Platform.LINKEDIN: 3,
            Platform.FACEBOOK: 2,
            Platform.YOUTUBE: 3,
            Platform.GENERIC: 3,
        }

        optimal_count = optimal_counts.get(platform, 3)
        optimal_hashtag_count = (
            1.0
            if hashtag_count == optimal_count
            else max(0.0, 1.0 - abs(hashtag_count - optimal_count) / optimal_count)
        )

        return {
            "hashtag_count": float(hashtag_count),
            "avg_hashtag_popularity": avg_popularity,
            "trending_hashtag_count": float(trending_count),
            "hashtag_diversity": hashtag_diversity,
            "optimal_hashtag_count": optimal_hashtag_count,
        }

    def extract_account_features(self, content_data: ContentData) -> Dict[str, float]:
        """Extract account-related features from metadata."""
        metadata = content_data.metadata

        # Extract account metrics from metadata
        follower_count = metadata.get("follower_count", 0)
        following_count = metadata.get("following_count", 0)
        account_age_days = metadata.get("account_age_days", 365)
        previous_posts_count = metadata.get("previous_posts_count", 0)
        previous_viral_count = metadata.get("previous_viral_count", 0)
        verified_account = metadata.get("verified", False)

        # Calculate derived features
        follower_following_ratio = follower_count / max(following_count, 1)
        posts_per_day = previous_posts_count / max(account_age_days, 1)
        viral_rate = previous_viral_count / max(previous_posts_count, 1)

        # Account authority score (combination of metrics)
        authority_score = min(
            1.0,
            (
                np.log10(max(follower_count, 1)) / 6 * 0.4  # Follower count (log scale)
                + min(follower_following_ratio / 10, 1.0) * 0.3  # Follower ratio
                + min(viral_rate * 10, 1.0) * 0.3  # Historical viral success
            ),
        )

        return {
            "follower_count_log": np.log10(max(follower_count, 1)),
            "follower_following_ratio": follower_following_ratio,
            "account_age_log": np.log10(max(account_age_days, 1)),
            "posts_per_day": posts_per_day,
            "viral_rate": viral_rate,
            "verified_account": 1.0 if verified_account else 0.0,
            "authority_score": authority_score,
        }

    def extract_algorithm_features(
        self, content_data: ContentData, platform: Platform
    ) -> Dict[str, float]:
        """Extract features related to platform algorithms."""
        metadata = content_data.metadata

        # Features that might affect algorithmic boost
        has_link = 1.0 if content_data.text and ("http" in content_data.text) else 0.0
        has_mentions = 1.0 if content_data.mentions else 0.0
        has_media = 1.0 if (content_data.image_url or content_data.video_url) else 0.0

        # Content length optimization for platform
        content_length_optimal = 0.0
        if content_data.text:
            text_length = len(content_data.text)

            # Platform-specific optimal lengths
            optimal_lengths = {
                Platform.TWITTER: 280,
                Platform.INSTAGRAM: 2200,
                Platform.TIKTOK: 150,
                Platform.LINKEDIN: 1300,
                Platform.FACEBOOK: 400,
                Platform.YOUTUBE: 125,  # For video descriptions
                Platform.GENERIC: 500,
            }

            optimal_length = optimal_lengths.get(platform, 500)
            content_length_optimal = max(
                0.0, 1.0 - abs(text_length - optimal_length) / optimal_length
            )

        # Engagement recency (how recent was the last engagement)
        last_engagement_hours = metadata.get("last_engagement_hours_ago", 24.0)
        engagement_recency = max(0.0, 1.0 - last_engagement_hours / 24.0)

        # Cross-platform posting (might reduce reach on some platforms)
        cross_posted = metadata.get("cross_posted", False)

        return {
            "has_external_link": has_link,
            "has_mentions": has_mentions,
            "has_media": has_media,
            "content_length_optimal": content_length_optimal,
            "engagement_recency": engagement_recency,
            "is_cross_posted": 1.0 if cross_posted else 0.0,
        }

    def extract_competition_features(
        self, posting_time: datetime, platform: Platform
    ) -> Dict[str, float]:
        """Extract features related to content competition at posting time."""
        # This would typically integrate with platform APIs to get real-time data
        # For now, we'll use estimated patterns

        posting_hour = posting_time.hour
        posting_day = posting_time.weekday()

        # Estimated competition level based on posting time
        # Higher competition during peak hours
        peak_hours = self.platform_peak_hours.get(platform, [])
        competition_level = 0.8 if posting_hour in peak_hours else 0.3

        # Weekend competition (generally lower)
        if posting_day >= 5:
            competition_level *= 0.7

        # Trending topic alignment (would come from real API data)
        trending_alignment = 0.5  # Default neutral value

        return {
            "competition_level": competition_level,
            "trending_topic_alignment": trending_alignment,
            "content_saturation": competition_level
            * 0.8,  # Proxy for content saturation
        }

    def update_trending_data(
        self, trending_hashtags: Set[str], hashtag_scores: Dict[str, float]
    ) -> None:
        """Update trending hashtags and popularity scores."""
        self.trending_hashtags = trending_hashtags
        self.hashtag_popularity_scores.update(hashtag_scores)

    def extract_all_features(self, content_data: ContentData) -> Dict[str, float]:
        """Extract all platform-specific features."""
        features = {}

        # Timing features
        if content_data.posting_time:
            features.update(
                self.extract_timing_features(
                    content_data.posting_time, content_data.platform
                )
            )

        # Hashtag features
        if content_data.hashtags:
            features.update(
                self.extract_hashtag_features(
                    content_data.hashtags, content_data.platform
                )
            )

        # Account features
        features.update(self.extract_account_features(content_data))

        # Algorithm features
        features.update(
            self.extract_algorithm_features(content_data, content_data.platform)
        )

        # Competition features
        if content_data.posting_time:
            features.update(
                self.extract_competition_features(
                    content_data.posting_time, content_data.platform
                )
            )

        return features
