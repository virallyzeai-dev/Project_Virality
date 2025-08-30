"""Engagement feature extraction and calculation."""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from ..core.models import ContentMetrics
from ..core.metrics import MetricsCalculator


class EngagementFeatureExtractor:
    """Extract engagement-related features from content metrics."""
    
    def __init__(self, virality_threshold: float = 2.0):
        """Initialize the engagement feature extractor."""
        self.metrics_calculator = MetricsCalculator(virality_threshold)
        self.virality_threshold = virality_threshold
    
    def extract_basic_engagement_features(self, metrics: ContentMetrics) -> Dict[str, float]:
        """Extract basic engagement rate features."""
        if metrics.impressions == 0:
            return {
                "engagement_rate": 0.0,
                "like_rate": 0.0,
                "share_rate": 0.0,
                "comment_rate": 0.0,
                "click_rate": 0.0,
                "save_rate": 0.0
            }
        
        total_engagement = metrics.likes + metrics.shares + metrics.comments
        engagement_rate = total_engagement / metrics.impressions
        like_rate = metrics.likes / metrics.impressions
        share_rate = metrics.shares / metrics.impressions
        comment_rate = metrics.comments / metrics.impressions
        
        # Optional metrics
        click_rate = (metrics.clicks / metrics.impressions) if metrics.clicks else 0.0
        save_rate = (metrics.saves / metrics.impressions) if metrics.saves else 0.0
        
        return {
            "engagement_rate": engagement_rate,
            "like_rate": like_rate,
            "share_rate": share_rate,
            "comment_rate": comment_rate,
            "click_rate": click_rate,
            "save_rate": save_rate
        }
    
    def extract_virality_features(self, metrics: ContentMetrics) -> Dict[str, float]:
        """Extract virality-specific features."""
        unique_users = metrics.unique_users_reached or metrics.impressions
        
        # Virality coefficient
        virality_coefficient = self.metrics_calculator.calculate_virality_coefficient(
            metrics.shares, unique_users
        )
        
        # Reach velocity (if timeline data available)
        reach_velocity = 0.0
        if metrics.impressions_timeline and metrics.time_intervals:
            reach_velocity = self.metrics_calculator.calculate_reach_velocity(
                metrics.impressions_timeline, metrics.time_intervals
            )
        
        # Amplification rate (shares per engagement)
        total_engagement = metrics.likes + metrics.shares + metrics.comments
        amplification_rate = (metrics.shares / total_engagement) if total_engagement > 0 else 0.0
        
        # Conversation rate (comments per engagement)
        conversation_rate = (metrics.comments / total_engagement) if total_engagement > 0 else 0.0
        
        return {
            "virality_coefficient": virality_coefficient,
            "reach_velocity": reach_velocity,
            "amplification_rate": amplification_rate,
            "conversation_rate": conversation_rate,
            "is_viral": 1.0 if virality_coefficient >= self.virality_threshold else 0.0
        }
    
    def extract_video_engagement_features(self, metrics: ContentMetrics) -> Dict[str, float]:
        """Extract video-specific engagement features."""
        features = {}
        
        if metrics.total_views and metrics.completed_views:
            completion_rate = self.metrics_calculator.calculate_completion_rate(
                metrics.completed_views, metrics.total_views
            )
            features["completion_rate"] = completion_rate
            
            # View-through rate (completed views / impressions)
            view_through_rate = (metrics.completed_views / metrics.impressions) if metrics.impressions > 0 else 0.0
            features["view_through_rate"] = view_through_rate
        else:
            features["completion_rate"] = 0.0
            features["view_through_rate"] = 0.0
        
        # Video engagement rate (using total views instead of impressions)
        if metrics.total_views and metrics.total_views > 0:
            video_engagement = metrics.likes + metrics.shares + metrics.comments
            video_engagement_rate = video_engagement / metrics.total_views
            features["video_engagement_rate"] = video_engagement_rate
        else:
            features["video_engagement_rate"] = 0.0
        
        return features
    
    def extract_temporal_features(
        self, 
        metrics: ContentMetrics, 
        posting_time: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Extract time-based engagement features."""
        features = {}
        
        # Early engagement indicators (if timeline data available)
        if metrics.impressions_timeline and len(metrics.impressions_timeline) >= 2:
            # Growth rate in first hour/period
            initial_growth = metrics.impressions_timeline[1] - metrics.impressions_timeline[0]
            features["early_growth_rate"] = initial_growth
            
            # Engagement acceleration
            if len(metrics.impressions_timeline) >= 3:
                acceleration = (
                    (metrics.impressions_timeline[2] - metrics.impressions_timeline[1]) - 
                    (metrics.impressions_timeline[1] - metrics.impressions_timeline[0])
                )
                features["engagement_acceleration"] = acceleration
            else:
                features["engagement_acceleration"] = 0.0
        else:
            features["early_growth_rate"] = 0.0
            features["engagement_acceleration"] = 0.0
        
        # Peak engagement time (if we have hourly data)
        if posting_time:
            posting_hour = posting_time.hour
            posting_day = posting_time.weekday()  # 0 = Monday
            
            features["posting_hour"] = float(posting_hour)
            features["posting_day"] = float(posting_day)
            
            # Peak hours indicator (based on general social media patterns)
            peak_hours = [9, 10, 11, 15, 16, 17, 19, 20, 21]  # General peak hours
            features["posted_in_peak_hours"] = 1.0 if posting_hour in peak_hours else 0.0
            
            # Weekend indicator
            features["posted_on_weekend"] = 1.0 if posting_day >= 5 else 0.0
        else:
            features["posting_hour"] = 0.0
            features["posting_day"] = 0.0
            features["posted_in_peak_hours"] = 0.0
            features["posted_on_weekend"] = 0.0
        
        return features
    
    def extract_engagement_distribution_features(self, metrics: ContentMetrics) -> Dict[str, float]:
        """Extract features related to engagement distribution."""
        total_engagement = metrics.likes + metrics.shares + metrics.comments
        
        if total_engagement == 0:
            return {
                "like_share_ratio": 0.0,
                "like_comment_ratio": 0.0,
                "share_comment_ratio": 0.0,
                "engagement_diversity": 0.0
            }
        
        # Ratios between different engagement types
        like_share_ratio = metrics.likes / max(metrics.shares, 1)
        like_comment_ratio = metrics.likes / max(metrics.comments, 1)
        share_comment_ratio = metrics.shares / max(metrics.comments, 1)
        
        # Engagement diversity (how evenly distributed are the engagement types)
        engagement_values = [metrics.likes, metrics.shares, metrics.comments]
        engagement_props = [val / total_engagement for val in engagement_values]
        
        # Calculate entropy as diversity measure
        engagement_diversity = -sum(
            p * np.log(p + 1e-10) for p in engagement_props if p > 0
        ) / np.log(3)  # Normalize by max entropy
        
        return {
            "like_share_ratio": like_share_ratio,
            "like_comment_ratio": like_comment_ratio,
            "share_comment_ratio": share_comment_ratio,
            "engagement_diversity": engagement_diversity
        }
    
    def extract_audience_behavior_features(self, metrics: ContentMetrics) -> Dict[str, float]:
        """Extract features related to audience behavior patterns."""
        features = {}
        
        # Impression-to-engagement conversion
        total_engagement = metrics.likes + metrics.shares + metrics.comments
        if metrics.impressions > 0:
            impression_conversion = total_engagement / metrics.impressions
            features["impression_conversion"] = impression_conversion
        else:
            features["impression_conversion"] = 0.0
        
        # Share velocity (indicates viral potential)
        if metrics.shares > 0 and metrics.impressions > 0:
            share_velocity = metrics.shares / metrics.impressions
            features["share_velocity"] = share_velocity
        else:
            features["share_velocity"] = 0.0
        
        # Engagement intensity (total engagement per unique user)
        if metrics.unique_users_reached and metrics.unique_users_reached > 0:
            engagement_intensity = total_engagement / metrics.unique_users_reached
            features["engagement_intensity"] = engagement_intensity
        else:
            features["engagement_intensity"] = 0.0
        
        return features
    
    def extract_all_features(
        self, 
        metrics: ContentMetrics, 
        posting_time: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Extract all engagement features."""
        features = {}
        
        # Combine all feature types
        features.update(self.extract_basic_engagement_features(metrics))
        features.update(self.extract_virality_features(metrics))
        features.update(self.extract_video_engagement_features(metrics))
        features.update(self.extract_temporal_features(metrics, posting_time))
        features.update(self.extract_engagement_distribution_features(metrics))
        features.update(self.extract_audience_behavior_features(metrics))
        
        return features
    
    def calculate_engagement_score(self, metrics: ContentMetrics) -> float:
        """Calculate a composite engagement score."""
        features = self.extract_all_features(metrics)
        
        # Weighted combination of key metrics
        weights = {
            "engagement_rate": 0.25,
            "virality_coefficient": 0.30,
            "completion_rate": 0.15,
            "share_velocity": 0.20,
            "engagement_diversity": 0.10
        }
        
        score = 0.0
        total_weight = 0.0
        
        for feature, weight in weights.items():
            if feature in features:
                score += features[feature] * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
