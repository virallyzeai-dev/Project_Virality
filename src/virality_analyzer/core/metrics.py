"""Core metrics calculations for virality analysis."""

# import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ViralityMetrics:
    """Container for calculated virality metrics."""

    engagement_rate: float
    virality_coefficient: float
    reach_velocity: float
    completion_rate: Optional[float] = None
    save_rate: Optional[float] = None
    is_viral: bool = False


class MetricsCalculator:
    """Calculate various virality and engagement metrics."""

    def __init__(self, virality_threshold: float = 2.0):
        self.virality_threshold = virality_threshold

    def calculate_engagement_rate(
        self, likes: int, shares: int, comments: int, impressions: int
    ) -> float:
        """Calculate engagement rate: (likes + shares + comments) / impressions."""
        if impressions == 0:
            return 0.0
        return (likes + shares + comments) / impressions

    def calculate_virality_coefficient(
        self, shares: int, unique_users_reached: int
    ) -> float:
        """Calculate virality coefficient: average number of re-shares per user."""
        if unique_users_reached == 0:
            return 0.0
        return shares / unique_users_reached

    def calculate_reach_velocity(
        self, impressions_over_time: List[int], time_intervals: List[float]
    ) -> float:
        """Calculate reach velocity: impressions gained per unit time."""
        if len(impressions_over_time) < 2 or len(time_intervals) == 0:
            return 0.0

        total_time = sum(time_intervals)
        total_impressions = impressions_over_time[-1] - impressions_over_time[0]

        if total_time == 0:
            return 0.0
        return total_impressions / total_time

    def calculate_completion_rate(
        self, completed_views: int, total_views: int
    ) -> float:
        """Calculate completion rate for video content."""
        if total_views == 0:
            return 0.0
        return completed_views / total_views

    def calculate_save_rate(self, saves: int, impressions: int) -> float:
        """Calculate save rate: saves / impressions."""
        if impressions == 0:
            return 0.0
        return saves / impressions

    def calculate_all_metrics(self, content_data: Dict[str, Any]) -> ViralityMetrics:
        """Calculate all available metrics from content data."""
        # Extract basic engagement metrics
        likes = content_data.get("likes", 0)
        shares = content_data.get("shares", 0)
        comments = content_data.get("comments", 0)
        impressions = content_data.get("impressions", 1)  # Avoid division by zero

        # Calculate core metrics
        engagement_rate = self.calculate_engagement_rate(
            likes, shares, comments, impressions
        )

        # For virality coefficient, use shares and impressions as proxy if unique users not available
        unique_users = content_data.get("unique_users_reached", impressions)
        virality_coefficient = self.calculate_virality_coefficient(shares, unique_users)

        # Calculate reach velocity if time data available
        impressions_timeline = content_data.get(
            "impressions_timeline", [0, impressions]
        )
        time_intervals = content_data.get("time_intervals", [1.0])  # Default 1 hour
        reach_velocity = self.calculate_reach_velocity(
            impressions_timeline, time_intervals
        )

        # Optional metrics
        completion_rate = None
        if "completed_views" in content_data and "total_views" in content_data:
            completion_rate = self.calculate_completion_rate(
                content_data["completed_views"], content_data["total_views"]
            )

        save_rate = None
        if "saves" in content_data:
            save_rate = self.calculate_save_rate(content_data["saves"], impressions)

        # Determine if content is viral
        is_viral = virality_coefficient >= self.virality_threshold

        return ViralityMetrics(
            engagement_rate=engagement_rate,
            virality_coefficient=virality_coefficient,
            reach_velocity=reach_velocity,
            completion_rate=completion_rate,
            save_rate=save_rate,
            is_viral=is_viral,
        )


class BenchmarkMetrics:
    """Store and compare against benchmark metrics for viral content."""

    def __init__(self):
        # Default benchmarks based on industry standards
        self.viral_benchmarks = {
            "engagement_rate": 0.06,  # 6% is considered high
            "virality_coefficient": 2.0,  # 2+ shares per user
            "reach_velocity": 1000,  # 1000 impressions per hour
            "completion_rate": 0.8,  # 80% completion rate
            "save_rate": 0.02,  # 2% save rate
        }

    def compare_to_benchmark(
        self, metrics: ViralityMetrics
    ) -> Dict[str, Dict[str, float]]:
        """Compare calculated metrics against viral benchmarks."""
        comparison = {}

        # Engagement rate comparison
        comparison["engagement_rate"] = {
            "value": metrics.engagement_rate,
            "benchmark": self.viral_benchmarks["engagement_rate"],
            "ratio": metrics.engagement_rate / self.viral_benchmarks["engagement_rate"],
        }

        # Virality coefficient comparison
        comparison["virality_coefficient"] = {
            "value": metrics.virality_coefficient,
            "benchmark": self.viral_benchmarks["virality_coefficient"],
            "ratio": metrics.virality_coefficient
            / self.viral_benchmarks["virality_coefficient"],
        }

        # Reach velocity comparison
        comparison["reach_velocity"] = {
            "value": metrics.reach_velocity,
            "benchmark": self.viral_benchmarks["reach_velocity"],
            "ratio": metrics.reach_velocity / self.viral_benchmarks["reach_velocity"]
            if self.viral_benchmarks["reach_velocity"] > 0
            else 0,
        }

        # Optional metrics
        if metrics.completion_rate is not None:
            comparison["completion_rate"] = {
                "value": metrics.completion_rate,
                "benchmark": self.viral_benchmarks["completion_rate"],
                "ratio": metrics.completion_rate
                / self.viral_benchmarks["completion_rate"],
            }

        if metrics.save_rate is not None:
            comparison["save_rate"] = {
                "value": metrics.save_rate,
                "benchmark": self.viral_benchmarks["save_rate"],
                "ratio": metrics.save_rate / self.viral_benchmarks["save_rate"],
            }

        return comparison

    def update_benchmarks(self, new_benchmarks: Dict[str, float]) -> None:
        """Update benchmark values with new data."""
        self.viral_benchmarks.update(new_benchmarks)

    def get_underperforming_metrics(self, metrics: ViralityMetrics) -> List[str]:
        """Get list of metrics that are underperforming compared to benchmarks."""
        comparison = self.compare_to_benchmark(metrics)
        underperforming = []

        for metric_name, metric_data in comparison.items():
            if metric_data["ratio"] < 1.0:  # Below benchmark
                underperforming.append(metric_name)

        return underperforming
