"""
Instagram Virality Analysis Example
==================================

This script demonstrates how to collect Instagram post data using the Instagram Basic Display API
and analyze it for virality using the Content Virality Analyzer.

Prerequisites:
1. Instagram access token from Instagram Basic Display API or Instagram Graph API
2. Set up the .env file with your INSTAGRAM_ACCESS_TOKEN
3. Install required dependencies: poetry install

Usage:
    python instagram_virality_analysis.py
"""

import os
from datetime import datetime
from typing import Dict, List

import requests

from virality_analyzer import (
    ContentData,
    ContentMetrics,
    ContentType,
    Platform,
    ViralityAnalyzer,
)


class InstagramDataCollector:
    """Collect Instagram post data using Instagram API."""

    def __init__(self, access_token: str):
        """Initialize with Instagram access token."""
        self.access_token = access_token
        self.base_url = "https://graph.instagram.com"

    def get_user_media(self, user_id: str = "me", limit: int = 25) -> List[Dict]:
        """
        Get user's media posts.

        Args:
            user_id: Instagram user ID or 'me' for authenticated user
            limit: Number of posts to retrieve (max 25 per request)

        Returns:
            List of media data dictionaries
        """
        # Fields to retrieve from Instagram API
        fields = (
            "id,media_type,media_url,permalink,thumbnail_url,caption,"
            "timestamp,like_count,comments_count,impressions,reach,"
            "saved,video_views,carousel_album"
        )

        url = f"{self.base_url}/{user_id}/media"
        params = {"fields": fields, "limit": limit, "access_token": self.access_token}

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json().get("data", [])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching Instagram data: {e}")
            return []

    def get_media_insights(self, media_id: str) -> Dict:
        """
        Get insights for a specific media post.
        Note: Requires Instagram Graph API and appropriate permissions

        Args:
            media_id: Instagram media ID

        Returns:
            Dictionary with insights data
        """
        # Insights metrics available for different media types
        metrics = "impressions,reach,saved,video_views,likes,comments,shares"

        url = f"{self.base_url}/{media_id}/insights"
        params = {"metric": metrics, "access_token": self.access_token}

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching insights for media {media_id}: {e}")
            return {}

    def convert_to_content_data(self, instagram_post: Dict) -> ContentData:
        """
        Convert Instagram API response to ContentData format.

        Args:
            instagram_post: Raw Instagram post data from API

        Returns:
            ContentData object for virality analysis
        """
        # Determine content type
        media_type = instagram_post.get("media_type", "").upper()
        content_type_map = {
            "IMAGE": ContentType.IMAGE,
            "VIDEO": ContentType.VIDEO,
            "CAROUSEL_ALBUM": ContentType.MIXED,
        }
        content_type = content_type_map.get(media_type, ContentType.IMAGE)

        # Extract metrics
        metrics = ContentMetrics(
            likes=instagram_post.get("like_count", 0),
            comments=instagram_post.get("comments_count", 0),
            impressions=instagram_post.get("impressions", 0),
            saves=instagram_post.get("saved", 0),
            total_views=instagram_post.get("video_views", 0)
            if content_type == ContentType.VIDEO
            else None,
            unique_users_reached=instagram_post.get("reach", 0),
        )

        # Extract hashtags from caption
        caption = instagram_post.get("caption", "") or ""
        hashtags = []
        if caption:
            hashtags = [word for word in caption.split() if word.startswith("#")]

        # Parse timestamp
        posting_time = None
        if instagram_post.get("timestamp"):
            try:
                posting_time = datetime.fromisoformat(
                    instagram_post["timestamp"].replace("Z", "+00:00")
                )
            except ValueError:
                pass

        return ContentData(
            content_id=instagram_post.get("id", "unknown"),
            text=caption,
            image_url=instagram_post.get("media_url")
            if content_type in [ContentType.IMAGE, ContentType.MIXED]
            else None,
            video_url=instagram_post.get("media_url")
            if content_type == ContentType.VIDEO
            else None,
            content_type=content_type,
            platform=Platform.INSTAGRAM,
            metrics=metrics,
            posting_time=posting_time,
            hashtags=hashtags,
            metadata={
                "permalink": instagram_post.get("permalink"),
                "thumbnail_url": instagram_post.get("thumbnail_url"),
                "media_type": media_type,
            },
        )


def analyze_instagram_virality(access_token: str, num_posts: int = 10):
    """
    Main function to analyze Instagram post virality.

    Args:
        access_token: Instagram access token
        num_posts: Number of recent posts to analyze
    """
    print("🔍 Instagram Virality Analysis")
    print("=" * 50)

    # Initialize Instagram data collector
    print("📱 Connecting to Instagram API...")
    collector = InstagramDataCollector(access_token)

    # Fetch recent posts
    print(f"📊 Fetching {num_posts} recent posts...")
    instagram_posts = collector.get_user_media(limit=num_posts)

    if not instagram_posts:
        print("❌ No posts found or error accessing Instagram API")
        print("Please check your access token and permissions")
        return

    print(f"✅ Retrieved {len(instagram_posts)} posts")

    # Convert to ContentData format
    print("🔄 Converting posts to analysis format...")
    content_data = []
    for post in instagram_posts:
        try:
            content = collector.convert_to_content_data(post)
            content_data.append(content)
        except Exception as e:
            print(f"⚠️  Error processing post {post.get('id', 'unknown')}: {e}")

    if not content_data:
        print("❌ No posts could be processed for analysis")
        return

    # Initialize virality analyzer
    print("🚀 Initializing Virality Analyzer...")
    analyzer = ViralityAnalyzer(
        model_type="xgboost",
        virality_threshold=2.0,  # Adjust based on your definition of viral
    )

    # For demonstration, we'll use the existing viral examples for training
    # In production, you'd want to use your own historical viral content
    print("⚠️  Note: Using sample training data for demonstration")
    print("   For best results, provide your own historical viral content")

    # Analyze each post
    print("\n🔍 Analyzing Posts for Virality:")
    print("=" * 50)

    results = []
    for i, content in enumerate(content_data, 1):
        print(f"\n--- Post {i}: {content.content_id} ---")
        print(f"Type: {content.content_type.value.title()}")
        print(
            f"Caption: {(content.text[:60] + '...') if content.text and len(content.text) > 60 else content.text}"
        )
        print(
            f"Posted: {content.posting_time.strftime('%Y-%m-%d %H:%M') if content.posting_time else 'Unknown'}"
        )

        # Current metrics
        metrics = content.metrics
        print("Current Metrics:")
        print(f"  📍 Likes: {metrics.likes:,}")
        print(f"  💬 Comments: {metrics.comments:,}")
        print(f"  👁️  Impressions: {metrics.impressions:,}")
        if metrics.saves:
            print(f"  💾 Saves: {metrics.saves:,}")
        if metrics.total_views:
            print(f"  📺 Views: {metrics.total_views:,}")

        # Calculate engagement rate
        if metrics.impressions > 0:
            engagement_rate = (
                (metrics.likes + metrics.comments) / metrics.impressions * 100
            )
            print(f"  📊 Engagement Rate: {engagement_rate:.2f}%")

        try:
            # For demonstration, we'll create a simple virality score based on engagement
            # In practice, the analyzer would use trained ML models
            virality_score = calculate_simple_virality_score(content)
            is_viral = virality_score > 2.0

            print("\n🎯 Analysis Results:")
            print(f"  Virality Score: {virality_score:.2f}")
            print(f"  Status: {'🚀 VIRAL' if is_viral else '📈 Non-Viral'}")

            # Provide basic recommendations
            recommendations = generate_recommendations(content)
            if recommendations:
                print("  💡 Recommendations:")
                for rec in recommendations[:3]:  # Show top 3
                    print(f"    • {rec}")

            results.append(
                {
                    "content_id": content.content_id,
                    "virality_score": virality_score,
                    "is_viral": is_viral,
                    "engagement_rate": engagement_rate
                    if metrics.impressions > 0
                    else 0,
                    "content_type": content.content_type.value,
                }
            )

        except Exception as e:
            print(f"❌ Analysis failed: {e}")

    # Summary statistics
    if results:
        print("\n📈 Summary Statistics:")
        print("=" * 30)
        viral_posts = sum(1 for r in results if r["is_viral"])
        avg_score = sum(r["virality_score"] for r in results) / len(results)
        avg_engagement = sum(r["engagement_rate"] for r in results) / len(results)

        print(f"Total Posts Analyzed: {len(results)}")
        print(f"Viral Posts: {viral_posts} ({viral_posts / len(results) * 100:.1f}%)")
        print(f"Average Virality Score: {avg_score:.2f}")
        print(f"Average Engagement Rate: {avg_engagement:.2f}%")

        # Top performing post
        best_post = max(results, key=lambda x: x["virality_score"])
        print(
            f"Best Performing Post: {best_post['content_id']} (Score: {best_post['virality_score']:.2f})"
        )

    print("\n✅ Analysis Complete!")


def calculate_simple_virality_score(content: ContentData) -> float:
    """
    Calculate a simple virality score based on engagement metrics.
    This is a simplified version - the full analyzer uses trained ML models.
    """
    metrics = content.metrics

    if metrics.impressions == 0:
        return 0.0

    # Basic scoring formula
    engagement_rate = (metrics.likes + metrics.comments) / max(metrics.impressions, 1)
    save_rate = metrics.saves / max(metrics.impressions, 1) if metrics.saves else 0

    # Weighted score
    score = (
        engagement_rate * 100  # Base engagement
        + save_rate * 50  # Saves are valuable
        + (metrics.comments / max(metrics.likes, 1)) * 20  # Comment ratio
    )

    return min(score, 10.0)  # Cap at 10


def generate_recommendations(content: ContentData) -> List[str]:
    """Generate basic recommendations for improving virality."""
    recommendations = []
    metrics = content.metrics

    # Engagement-based recommendations
    if metrics.impressions > 0:
        engagement_rate = (metrics.likes + metrics.comments) / metrics.impressions
        if engagement_rate < 0.01:  # Less than 1%
            recommendations.append("Consider more engaging captions or call-to-actions")

    # Content-type specific recommendations
    if content.content_type == ContentType.VIDEO and metrics.total_views:
        completion_rate = metrics.total_views / max(metrics.impressions, 1)
        if completion_rate < 0.3:
            recommendations.append(
                "Hook viewers in the first 3 seconds to improve completion rate"
            )

    # Caption analysis
    if content.text:
        if content.hashtags and len(content.hashtags) < 5:
            recommendations.append(
                "Use 5-11 relevant hashtags to increase discoverability"
            )

        if "?" not in content.text:
            recommendations.append("Ask questions in captions to encourage comments")

        if len(content.text) < 50:
            recommendations.append("Consider longer, more descriptive captions")

    # Timing recommendations
    if content.posting_time:
        hour = content.posting_time.hour
        if hour < 10 or hour > 21:
            recommendations.append(
                "Post during peak hours (10 AM - 9 PM) for better reach"
            )

    return recommendations


def main():
    """Main entry point."""
    # Load environment variables
    from dotenv import load_dotenv

    load_dotenv()

    access_token = os.getenv("INSTAGRAM_ACCESS_TOKEN")

    if not access_token:
        print("❌ Instagram access token not found!")
        print("Please set INSTAGRAM_ACCESS_TOKEN in your .env file")
        print("\nTo get an Instagram access token:")
        print("1. Create a Facebook App at https://developers.facebook.com/")
        print("2. Add Instagram Basic Display product")
        print("3. Generate a User Access Token")
        print("4. Set the token in your .env file")
        return

    try:
        analyze_instagram_virality(access_token, num_posts=10)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("• Check that your access token is valid and not expired")
        print("• Ensure you have the required permissions")
        print("• Verify your Instagram account has posts to analyze")


if __name__ == "__main__":
    main()
