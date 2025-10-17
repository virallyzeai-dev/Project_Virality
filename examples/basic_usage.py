"""Example usage of the Content Virality Analyzer."""

from datetime import datetime
from virality_analyzer import (
    ViralityAnalyzer,
    ContentData,
    ContentMetrics,
    ContentType,
    Platform,
)


def create_sample_content():
    """Create sample content data for demonstration."""

    # Sample content 1: High engagement text post
    content1 = ContentData(
        content_id="post_001",
        text="🚀 Just launched our new AI product! This revolutionary technology will change how we think about automation. What do you think? #AI #Innovation #TechLaunch",
        content_type=ContentType.TEXT,
        platform=Platform.TWITTER,
        metrics=ContentMetrics(
            likes=1500, shares=300, comments=89, impressions=25000, clicks=450
        ),
        posting_time=datetime(2024, 1, 15, 14, 30),  # Posted at 2:30 PM
        hashtags=["#AI", "#Innovation", "#TechLaunch"],
        mentions=[],
    )

    # Sample content 2: Video content with lower engagement
    content2 = ContentData(
        content_id="video_001",
        text="Check out our latest tutorial on machine learning basics",
        video_url="https://example.com/tutorial_video.mp4",
        content_type=ContentType.VIDEO,
        platform=Platform.YOUTUBE,
        metrics=ContentMetrics(
            likes=45,
            shares=8,
            comments=12,
            impressions=2000,
            total_views=1800,
            completed_views=450,
        ),
        posting_time=datetime(2024, 1, 10, 9, 0),  # Posted at 9 AM
        hashtags=["#MachineLearning", "#Tutorial"],
        mentions=[],
    )

    # Sample content 3: Image post with moderate engagement
    content3 = ContentData(
        content_id="image_001",
        text="Beautiful sunset from our office! 🌅 #WorkLife #Inspiration",
        image_url="https://example.com/sunset.jpg",
        content_type=ContentType.IMAGE,
        platform=Platform.INSTAGRAM,
        metrics=ContentMetrics(
            likes=234, shares=45, comments=23, impressions=5600, saves=67
        ),
        posting_time=datetime(2024, 1, 12, 18, 45),  # Posted at 6:45 PM
        hashtags=["#WorkLife", "#Inspiration"],
        mentions=[],
    )

    return [content1, content2, content3]


def create_viral_examples():
    """Create examples of viral content for benchmarking."""

    viral1 = ContentData(
        content_id="viral_001",
        text="BREAKING: This simple trick will change your life forever! 🤯 Thread below 👇 #LifeHack #Viral #MustRead",
        content_type=ContentType.TEXT,
        platform=Platform.TWITTER,
        metrics=ContentMetrics(
            likes=50000, shares=25000, comments=3500, impressions=500000
        ),
        hashtags=["#LifeHack", "#Viral", "#MustRead"],
    )

    viral2 = ContentData(
        content_id="viral_002",
        text="This AI can predict the future with 99% accuracy! See for yourself:",
        video_url="https://example.com/ai_demo.mp4",
        content_type=ContentType.VIDEO,
        platform=Platform.TIKTOK,
        metrics=ContentMetrics(
            likes=120000,
            shares=45000,
            comments=8900,
            impressions=800000,
            total_views=750000,
            completed_views=600000,
        ),
        hashtags=["#AI", "#Future", "#Prediction"],
    )

    return [viral1, viral2]


def main():
    """Main example demonstrating the virality analyzer."""

    print("🚀 Content Virality Analyzer - Example Usage\n")

    # Create sample data
    print("📊 Creating sample content...")
    training_data = create_sample_content()
    viral_examples = create_viral_examples()

    # Initialize and train the analyzer
    print("🎯 Initializing and training the analyzer...")
    analyzer = ViralityAnalyzer(virality_threshold=2.0, model_type="xgboost")

    # Note: In a real scenario, you'd have much more training data
    print("⚠️  Note: Using minimal sample data for demonstration")
    print(
        "   In practice, you need hundreds/thousands of examples for good performance\n"
    )

    try:
        analyzer.train(training_data, viral_examples)
        print("✅ Training completed!\n")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print(
            "This is expected with minimal sample data. In practice, use real historical data.\n"
        )
        return

    # Analyze individual content
    print("🔍 Analyzing individual content pieces...\n")

    for i, content in enumerate(training_data, 1):
        print(f"--- Content {i}: {content.content_id} ---")

        try:
            # Get prediction
            prediction = analyzer.predict(content)
            print(f"Virality Score: {prediction.virality_score:.2%}")
            print(
                f"Prediction: {'🚀 VIRAL' if prediction.is_viral_prediction else '📉 Non-Viral'}"
            )
            print(f"Confidence: {prediction.confidence:.2%}")

            # Get detailed analysis
            report = analyzer.analyze(content)
            if report.explanation:
                print(f"Explanation: {report.explanation.explanation_text}")
                if report.explanation.recommendations:
                    print("Top Recommendation:", report.explanation.recommendations[0])

            # Benchmark against viral content
            if analyzer.diagnostic_engine.benchmark.viral_content_db:
                benchmark = analyzer.benchmark(content)
                print(f"Similarity to viral content: {benchmark.similarity_score:.2%}")

        except Exception as e:
            print(f"❌ Analysis failed: {e}")

        print()

    # Batch analysis
    print("📈 Performing batch analysis...\n")

    try:
        batch_results = analyzer.analyze_batch(training_data)

        # Summary statistics
        summary = batch_results.get("summary_stats", {})
        print("📊 Batch Analysis Summary:")
        print(f"Total Content Analyzed: {summary.get('total_content', 0)}")
        print(f"Predicted Viral Rate: {summary.get('predicted_viral_rate', 0):.2%}")
        print(f"Average Virality Score: {summary.get('avg_virality_score', 0):.2%}")

        # Recommendations
        recommendations = batch_results.get("recommendations", [])
        if recommendations:
            print("\n💡 Key Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"{i}. {rec}")

    except Exception as e:
        print(f"❌ Batch analysis failed: {e}")

    print("\n" + "=" * 60)
    print("🎉 Example completed!")
    print("For real usage, provide substantial training data and viral examples.")
    print("The analyzer will provide much more accurate predictions with proper data.")


if __name__ == "__main__":
    main()
