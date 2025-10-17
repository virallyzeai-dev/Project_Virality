"""Advanced example showing causal analysis and benchmarking."""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from virality_analyzer import (
    CausalAnalyzer,
    ContentData,
    ContentMetrics,
    ContentType,
    Platform,
    ViralityAnalyzer,
)


def generate_synthetic_dataset(n_samples=200):
    """Generate a synthetic dataset for demonstration."""
    np.random.seed(42)

    content_data_list = []
    platforms = [
        Platform.TWITTER,
        Platform.INSTAGRAM,
        Platform.YOUTUBE,
        Platform.TIKTOK,
    ]
    content_types = [ContentType.TEXT, ContentType.IMAGE, ContentType.VIDEO]

    for i in range(n_samples):
        # Generate synthetic metrics with some realistic patterns
        base_impressions = np.random.randint(1000, 50000)

        # Simulate that better content gets more engagement
        content_quality = np.random.random()
        engagement_multiplier = 0.5 + content_quality * 1.5

        likes = int(
            base_impressions * np.random.uniform(0.01, 0.1) * engagement_multiplier
        )
        shares = int(likes * np.random.uniform(0.1, 0.5))
        comments = int(likes * np.random.uniform(0.05, 0.3))

        # Posting time affects virality
        posting_hour = np.random.randint(0, 24)
        time_multiplier = 1.5 if posting_hour in [9, 12, 15, 18, 21] else 1.0

        # Platform affects performance
        platform = random.choice(platforms)
        platform_multiplier = {
            Platform.TWITTER: 1.0,
            Platform.INSTAGRAM: 1.2,
            Platform.YOUTUBE: 0.8,
            Platform.TIKTOK: 1.5,
        }[platform]

        # Apply multipliers
        final_multiplier = time_multiplier * platform_multiplier
        likes = int(likes * final_multiplier)
        shares = int(shares * final_multiplier)
        comments = int(comments * final_multiplier)

        # Generate text with some patterns
        text_templates = [
            "Check out this amazing {topic}! #trending #viral",
            "You won't believe what happened when {event}! Thread below 👇",
            "BREAKING: {news} - this changes everything! 🚨",
            "Simple {tip} that will change your life! #lifehack",
            "Behind the scenes of {process} ✨ #bts",
        ]

        topics = [
            "AI breakthrough",
            "life hack",
            "travel destination",
            "recipe",
            "workout",
        ]
        events = ["I tried this", "we launched", "the results came in", "I discovered"]
        news = ["New study reveals", "Scientists discover", "Company announces"]
        tips = ["trick", "method", "technique", "strategy"]
        processes = ["content creation", "product development", "team building"]

        template = random.choice(text_templates)
        if "{topic}" in template:
            text = template.format(topic=random.choice(topics))
        elif "{event}" in template:
            text = template.format(event=random.choice(events))
        elif "{news}" in template:
            text = template.format(news=random.choice(news))
        elif "{tip}" in template:
            text = template.format(tip=random.choice(tips))
        elif "{process}" in template:
            text = template.format(process=random.choice(processes))
        else:
            text = template

        # Create content data
        content_data = ContentData(
            content_id=f"synthetic_{i + 1:03d}",
            text=text,
            content_type=random.choice(content_types),
            platform=platform,
            metrics=ContentMetrics(
                likes=likes,
                shares=shares,
                comments=comments,
                impressions=base_impressions,
                saves=int(likes * np.random.uniform(0.02, 0.08))
                if np.random.random() > 0.5
                else None,
            ),
            posting_time=datetime.now() - timedelta(days=np.random.randint(1, 365)),
            hashtags=[f"#tag{j}" for j in range(np.random.randint(1, 6))],
            metadata={
                "follower_count": np.random.randint(100, 100000),
                "account_age_days": np.random.randint(30, 1800),
                "previous_viral_count": np.random.randint(0, 10),
            },
        )

        content_data_list.append(content_data)

    return content_data_list


def demonstrate_causal_analysis():
    """Demonstrate causal analysis capabilities."""
    print("🔬 Causal Analysis Demonstration\n")

    # Generate synthetic dataset
    print("📊 Generating synthetic dataset...")
    dataset = generate_synthetic_dataset(300)

    # Initialize causal analyzer
    causal_analyzer = CausalAnalyzer(virality_threshold=2.0)

    print("🧪 Performing causal analysis...")

    try:
        # Analyze causal relationships
        causal_results = causal_analyzer.analyze_feature_causality(
            dataset,
            key_features=[
                "sentiment_polarity",
                "emotional_intensity",
                "hashtag_count",
                "posting_hour",
                "follower_count_log",
            ],
        )

        print("✅ Causal analysis completed!\n")

        # Display results
        print("🎯 Key Causal Effects:")
        for effect in causal_results.causal_effects[:5]:
            significance = (
                "✅ Significant" if effect.is_significant else "❌ Not significant"
            )
            print(f"• {effect.factor}: {effect.effect_size:+.3f} ({significance})")

        print(
            f"\n🔍 Potential confounders identified: {len(causal_results.confounders)}"
        )

    except Exception as e:
        print(f"❌ Causal analysis failed: {e}")
        print(
            "This is expected with synthetic data. Use real data for meaningful results."
        )


def demonstrate_advanced_benchmarking():
    """Demonstrate advanced benchmarking features."""
    print("\n" + "=" * 60)
    print("🎯 Advanced Benchmarking Demonstration\n")

    # Generate datasets
    print("📊 Generating content datasets...")
    regular_content = generate_synthetic_dataset(100)

    # Create high-performing viral examples
    viral_examples = []
    for i in range(20):
        viral_content = ContentData(
            content_id=f"viral_{i + 1:03d}",
            text="VIRAL: This amazing discovery will blow your mind! 🤯 #viral #trending #mustread",
            content_type=ContentType.TEXT,
            platform=Platform.TWITTER,
            metrics=ContentMetrics(
                likes=np.random.randint(10000, 100000),
                shares=np.random.randint(5000, 50000),
                comments=np.random.randint(1000, 10000),
                impressions=np.random.randint(100000, 1000000),
            ),
            hashtags=["#viral", "#trending", "#mustread"],
            metadata={
                "follower_count": np.random.randint(50000, 1000000),
                "account_age_days": np.random.randint(365, 2000),
                "previous_viral_count": np.random.randint(5, 50),
            },
        )
        viral_examples.append(viral_content)

    # Initialize analyzer with benchmarking
    print("🎯 Setting up analyzer with viral benchmarks...")
    analyzer = ViralityAnalyzer()

    try:
        analyzer.train(regular_content + viral_examples[:10], viral_examples)
        print("✅ Analyzer trained with viral benchmarks!\n")

        # Analyze a piece of content against benchmarks
        test_content = regular_content[0]

        print(f"🔍 Analyzing content: '{test_content.text[:50]}...'\n")

        # Get comprehensive analysis
        report = analyzer.analyze(test_content)

        print("📊 Analysis Results:")
        print(f"Virality Score: {report.virality_prediction.virality_score:.2%}")
        print(
            f"Prediction: {'🚀 VIRAL' if report.virality_prediction.is_viral_prediction else '📉 Non-Viral'}"
        )

        if report.benchmark_comparison:
            print(
                f"Similarity to viral content: {report.benchmark_comparison.similarity_score:.2%}"
            )

            if report.benchmark_comparison.improvement_suggestions:
                print("\n💡 Improvement Suggestions:")
                for suggestion in report.benchmark_comparison.improvement_suggestions[
                    :3
                ]:
                    print(f"• {suggestion}")

        # Analyze viral patterns
        print("\n🌟 Viral Content Patterns Analysis:")
        patterns = analyzer.diagnostic_engine.benchmark.analyze_viral_patterns()

        if patterns:
            top_features = patterns.get("top_features", {})
            if top_features:
                print("Top features in viral content:")
                for feature, value in list(top_features.items())[:5]:
                    print(f"• {feature}: {value:.3f}")

    except Exception as e:
        print(f"❌ Advanced benchmarking failed: {e}")
        print(
            "This is expected with synthetic data. Use real viral examples for best results."
        )


def demonstrate_clustering():
    """Demonstrate content clustering capabilities."""
    print("\n" + "=" * 60)
    print("🎯 Content Clustering Demonstration\n")

    # Generate diverse dataset
    print("📊 Generating diverse content dataset...")
    dataset = generate_synthetic_dataset(150)

    # Initialize analyzer
    analyzer = ViralityAnalyzer()

    try:
        # Simple training (might fail with synthetic data, that's ok)
        print("🎯 Training analyzer...")
        analyzer.train(dataset[:100])

        print("🔄 Performing clustering analysis...")

        # Get batch analysis with clustering
        batch_results = analyzer.analyze_batch(dataset[100:])

        clustering_results = batch_results.get("clustering", {})

        if clustering_results:
            content_clusters = clustering_results.get("content_clusters", {})
            behavior_clusters = clustering_results.get("behavior_clusters", {})

            print("✅ Clustering analysis completed!\n")

            # Display content cluster insights
            if content_clusters:
                print(
                    f"📊 Identified {content_clusters.get('n_clusters', 0)} content clusters"
                )

                cluster_analysis = content_clusters.get("cluster_analysis", {})
                for cluster_name, info in cluster_analysis.items():
                    print(
                        f"• {cluster_name}: {info['size']} items, {info['viral_rate']:.1%} viral rate"
                    )

            # Display behavior patterns
            if behavior_clusters:
                print("\n👥 Behavior Patterns:")
                patterns = behavior_clusters.get("behavior_patterns", {})
                for pattern_name, pattern_info in patterns.items():
                    print(f"• {pattern_name}: {pattern_info['description']}")

            # Display recommendations
            insights = clustering_results.get("insights", {})
            if insights:
                print("\n💡 Cluster Insights:")
                for cluster, insight in insights.items():
                    print(f"• {insight}")

    except Exception as e:
        print(f"❌ Clustering analysis failed: {e}")
        print(
            "This is expected with synthetic data. Use real content data for meaningful clusters."
        )


def main():
    """Run all advanced demonstrations."""
    print("🚀 Advanced Content Virality Analysis Examples\n")
    print("This demonstration shows advanced features using synthetic data.")
    print("For production use, replace with real historical content data.\n")

    # Run demonstrations
    demonstrate_causal_analysis()
    demonstrate_advanced_benchmarking()
    demonstrate_clustering()

    print("\n" + "=" * 60)
    print("🎉 Advanced examples completed!")
    print("\n📚 Key Takeaways:")
    print("• Causal analysis helps identify what actually drives virality")
    print("• Benchmarking against viral content provides actionable insights")
    print("• Clustering reveals patterns in content and audience behavior")
    print("• Real data with sufficient volume is crucial for accurate results")


if __name__ == "__main__":
    main()
