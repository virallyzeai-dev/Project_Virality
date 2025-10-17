"""Streamlit web application for virality analysis."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
from typing import Dict, List, Optional

# Import our analyzer components
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from virality_analyzer.core.models import (
    ContentData,
    ContentMetrics,
    ContentType,
    Platform,
)
from virality_analyzer.analysis import DiagnosticEngine
from virality_analyzer.models import ViralityPredictor


class ViralityAnalyzerApp:
    """Streamlit web application for content virality analysis."""

    def __init__(self):
        """Initialize the application."""
        self.diagnostic_engine = None
        self.setup_page_config()

    def setup_page_config(self):
        """Configure Streamlit page."""
        st.set_page_config(
            page_title="Content Virality Analyzer",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def run(self):
        """Run the main application."""
        st.title("🚀 Content Virality Analyzer")
        st.markdown(
            """
        Analyze your content's viral potential using AI-powered predictions, 
        explainable insights, and benchmarking against successful viral content.
        """
        )

        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Choose Analysis Type",
            [
                "Single Content Analysis",
                "Batch Analysis",
                "Training & Setup",
                "Analytics Dashboard",
            ],
        )

        if page == "Single Content Analysis":
            self.single_content_page()
        elif page == "Batch Analysis":
            self.batch_analysis_page()
        elif page == "Training & Setup":
            self.training_setup_page()
        elif page == "Analytics Dashboard":
            self.analytics_dashboard_page()

    def single_content_page(self):
        """Page for analyzing single piece of content."""
        st.header("📊 Single Content Analysis")

        # Check if model is trained
        if not self._is_model_ready():
            st.warning(
                "⚠️ Please train the model first in the 'Training & Setup' page."
            )
            return

        # Content input form
        with st.form("content_form"):
            col1, col2 = st.columns(2)

            with col1:
                content_id = st.text_input(
                    "Content ID",
                    value=f"content_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                )
                platform = st.selectbox("Platform", [p.value for p in Platform])
                content_type = st.selectbox(
                    "Content Type", [c.value for c in ContentType]
                )

            with col2:
                posting_time = st.datetime_input("Posting Time", datetime.now())
                hashtags = st.text_input(
                    "Hashtags (comma-separated)", placeholder="#viral, #trending"
                )
                mentions = st.text_input(
                    "Mentions (comma-separated)", placeholder="@user1, @user2"
                )

            # Content input
            text_content = st.text_area(
                "Text Content",
                height=100,
                placeholder="Enter your content text here...",
            )

            col3, col4 = st.columns(2)
            with col3:
                image_url = st.text_input(
                    "Image URL (optional)", placeholder="https://example.com/image.jpg"
                )
            with col4:
                video_url = st.text_input(
                    "Video URL (optional)", placeholder="https://example.com/video.mp4"
                )

            # Metrics input
            st.subheader("📈 Current Metrics (optional)")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

            with metric_col1:
                likes = st.number_input("Likes", min_value=0, value=0)
                shares = st.number_input("Shares", min_value=0, value=0)

            with metric_col2:
                comments = st.number_input("Comments", min_value=0, value=0)
                impressions = st.number_input("Impressions", min_value=1, value=1000)

            with metric_col3:
                saves = st.number_input("Saves", min_value=0, value=0)
                clicks = st.number_input("Clicks", min_value=0, value=0)

            with metric_col4:
                total_views = st.number_input("Total Views", min_value=0, value=0)
                completed_views = st.number_input(
                    "Completed Views", min_value=0, value=0
                )

            submitted = st.form_submit_button("🔍 Analyze Content")

        if submitted:
            # Create content data object
            content_data = self._create_content_data(
                content_id=content_id,
                platform=platform,
                content_type=content_type,
                text=text_content,
                image_url=image_url or None,
                video_url=video_url or None,
                posting_time=posting_time,
                hashtags=hashtags,
                mentions=mentions,
                likes=likes,
                shares=shares,
                comments=comments,
                impressions=impressions,
                saves=saves,
                clicks=clicks,
                total_views=total_views,
                completed_views=completed_views,
            )

            # Perform analysis
            with st.spinner("🔄 Analyzing content..."):
                report = self.diagnostic_engine.diagnose_single_content(content_data)

            # Display results
            self._display_single_analysis_results(report)

    def batch_analysis_page(self):
        """Page for batch content analysis."""
        st.header("📊 Batch Content Analysis")

        if not self._is_model_ready():
            st.warning(
                "⚠️ Please train the model first in the 'Training & Setup' page."
            )
            return

        # File upload
        uploaded_file = st.file_uploader(
            "Upload Content Data (CSV)",
            type=["csv"],
            help="CSV should contain columns: content_id, text, platform, content_type, likes, shares, comments, impressions",
        )

        if uploaded_file is not None:
            # Load and preview data
            df = pd.read_csv(uploaded_file)
            st.subheader("📋 Data Preview")
            st.dataframe(df.head())

            # Validate required columns
            required_cols = [
                "content_id",
                "text",
                "platform",
                "likes",
                "shares",
                "comments",
                "impressions",
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                return

            # Analysis options
            col1, col2 = st.columns(2)
            with col1:
                include_clustering = st.checkbox(
                    "Include Clustering Analysis", value=True
                )
            with col2:
                include_causal = st.checkbox("Include Causal Analysis", value=True)

            if st.button("🚀 Start Batch Analysis"):
                # Convert DataFrame to ContentData objects
                content_data_list = self._dataframe_to_content_data(df)

                # Perform batch analysis
                with st.spinner("🔄 Performing batch analysis..."):
                    batch_results = self.diagnostic_engine.diagnose_batch_content(
                        content_data_list,
                        include_clustering=include_clustering,
                        include_causal=include_causal,
                    )

                # Display results
                self._display_batch_analysis_results(batch_results)

    def training_setup_page(self):
        """Page for training models and setup."""
        st.header("🛠️ Training & Setup")

        tab1, tab2, tab3 = st.tabs(
            ["Model Training", "Viral Content Database", "Settings"]
        )

        with tab1:
            st.subheader("🎯 Train Prediction Model")

            # Training data upload
            training_file = st.file_uploader(
                "Upload Training Data (CSV)",
                type=["csv"],
                key="training_upload",
                help="CSV with historical content and their performance metrics",
            )

            if training_file is not None:
                training_df = pd.read_csv(training_file)
                st.write("Training Data Preview:")
                st.dataframe(training_df.head())

                # Model configuration
                col1, col2 = st.columns(2)
                with col1:
                    model_type = st.selectbox(
                        "Model Type",
                        ["xgboost", "catboost", "random_forest", "neural_net"],
                    )
                    virality_threshold = st.number_input(
                        "Virality Threshold", min_value=0.1, max_value=10.0, value=2.0
                    )

                with col2:
                    validation_split = st.slider("Validation Split", 0.1, 0.5, 0.2)
                    perform_cv = st.checkbox("Perform Cross-Validation", value=True)

                if st.button("🎯 Train Model"):
                    try:
                        # Initialize diagnostic engine
                        self.diagnostic_engine = DiagnosticEngine(
                            virality_threshold=virality_threshold
                        )
                        self.diagnostic_engine.predictor.model_type = model_type
                        self.diagnostic_engine.predictor._initialize_model()

                        # Convert training data
                        training_content_data = self._dataframe_to_content_data(
                            training_df
                        )

                        with st.spinner("🔄 Training model..."):
                            # Train the model
                            self.diagnostic_engine.setup_predictor(
                                training_content_data
                            )

                        st.success("✅ Model trained successfully!")

                        # Store model in session state
                        st.session_state["diagnostic_engine"] = self.diagnostic_engine

                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")

        with tab2:
            st.subheader("🌟 Viral Content Database")

            viral_file = st.file_uploader(
                "Upload Viral Content Examples (CSV)",
                type=["csv"],
                key="viral_upload",
                help="CSV with known viral content for benchmarking",
            )

            if viral_file is not None:
                viral_df = pd.read_csv(viral_file)
                st.write("Viral Content Preview:")
                st.dataframe(viral_df.head())

                if st.button("📚 Setup Benchmark Database"):
                    if self._is_model_ready():
                        try:
                            viral_content_data = self._dataframe_to_content_data(
                                viral_df
                            )

                            with st.spinner("🔄 Setting up benchmark..."):
                                self.diagnostic_engine.setup_benchmark(
                                    viral_content_data
                                )

                            st.success("✅ Benchmark database setup successfully!")

                        except Exception as e:
                            st.error(f"❌ Setup failed: {str(e)}")
                    else:
                        st.warning("⚠️ Please train the model first.")

        with tab3:
            st.subheader("⚙️ Settings")

            # API Keys
            openai_key = st.text_input(
                "OpenAI API Key (optional)",
                type="password",
                help="For enhanced explanations",
            )

            # Display current model status
            if self._is_model_ready():
                st.success("✅ Model is trained and ready")

                # Model info
                model_info = st.expander("📊 Model Information")
                with model_info:
                    st.write(
                        f"Model Type: {self.diagnostic_engine.predictor.model_type}"
                    )
                    st.write(
                        f"Virality Threshold: {self.diagnostic_engine.virality_threshold}"
                    )
                    st.write(
                        f"Feature Count: {len(self.diagnostic_engine.predictor.feature_processor.feature_names)}"
                    )
            else:
                st.warning("⚠️ No trained model available")

    def analytics_dashboard_page(self):
        """Analytics dashboard page."""
        st.header("📈 Analytics Dashboard")

        if not self._is_model_ready():
            st.warning("⚠️ Please train the model and analyze some content first.")
            return

        # Get analysis insights
        insights = self.diagnostic_engine.get_analysis_insights()

        if not insights:
            st.info("📊 No analysis history available. Analyze some content first!")
            return

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Analyses", insights.get("total_analyses", 0))

        with col2:
            avg_score = insights.get("avg_virality_score", 0)
            st.metric("Avg Virality Score", f"{avg_score:.2%}")

        with col3:
            viral_rate = insights.get("viral_prediction_rate", 0)
            st.metric("Viral Prediction Rate", f"{viral_rate:.2%}")

        with col4:
            improvement = insights.get("improvement_trends", {}).get("improvement", 0)
            st.metric("Score Trend", f"{improvement:+.3f}")

        # Most common factors
        st.subheader("🔍 Most Important Factors")
        common_factors = insights.get("most_common_factors", {})

        if common_factors:
            factors_df = pd.DataFrame(
                [
                    {"Factor": k, "Average Importance": v}
                    for k, v in common_factors.items()
                ]
            )

            fig = px.bar(
                factors_df,
                x="Average Importance",
                y="Factor",
                orientation="h",
                title="Top Contributing Factors to Virality",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Trends analysis
        if len(self.diagnostic_engine.analysis_history) >= 5:
            st.subheader("📊 Virality Score Trends")

            scores = [
                r.virality_prediction.virality_score
                for r in self.diagnostic_engine.analysis_history
            ]
            dates = [r.generated_at for r in self.diagnostic_engine.analysis_history]

            trend_df = pd.DataFrame({"Date": dates, "Virality Score": scores})

            fig = px.line(
                trend_df,
                x="Date",
                y="Virality Score",
                title="Virality Scores Over Time",
            )
            st.plotly_chart(fig, use_container_width=True)

    def _is_model_ready(self) -> bool:
        """Check if model is trained and ready."""
        if "diagnostic_engine" in st.session_state:
            self.diagnostic_engine = st.session_state["diagnostic_engine"]
            return self.diagnostic_engine.predictor.is_trained
        return False

    def _create_content_data(self, **kwargs) -> ContentData:
        """Create ContentData object from form inputs."""
        # Parse hashtags and mentions
        hashtags = None
        if kwargs.get("hashtags"):
            hashtags = [tag.strip() for tag in kwargs["hashtags"].split(",")]

        mentions = None
        if kwargs.get("mentions"):
            mentions = [mention.strip() for mention in kwargs["mentions"].split(",")]

        # Create metrics object
        metrics = ContentMetrics(
            likes=kwargs.get("likes", 0),
            shares=kwargs.get("shares", 0),
            comments=kwargs.get("comments", 0),
            impressions=kwargs.get("impressions", 1),
            saves=kwargs.get("saves"),
            clicks=kwargs.get("clicks"),
            total_views=kwargs.get("total_views"),
            completed_views=kwargs.get("completed_views"),
        )

        # Create content data
        content_data = ContentData(
            content_id=kwargs["content_id"],
            text=kwargs.get("text"),
            image_url=kwargs.get("image_url"),
            video_url=kwargs.get("video_url"),
            content_type=ContentType(kwargs["content_type"]),
            platform=Platform(kwargs["platform"]),
            metrics=metrics,
            posting_time=kwargs.get("posting_time"),
            hashtags=hashtags,
            mentions=mentions,
        )

        return content_data

    def _dataframe_to_content_data(self, df: pd.DataFrame) -> List[ContentData]:
        """Convert pandas DataFrame to list of ContentData objects."""
        content_data_list = []

        for _, row in df.iterrows():
            # Handle optional columns
            metrics = ContentMetrics(
                likes=int(row.get("likes", 0)),
                shares=int(row.get("shares", 0)),
                comments=int(row.get("comments", 0)),
                impressions=int(row.get("impressions", 1)),
                saves=int(row.get("saves", 0)) if pd.notna(row.get("saves")) else None,
                clicks=int(row.get("clicks", 0))
                if pd.notna(row.get("clicks"))
                else None,
                total_views=int(row.get("total_views", 0))
                if pd.notna(row.get("total_views"))
                else None,
                completed_views=int(row.get("completed_views", 0))
                if pd.notna(row.get("completed_views"))
                else None,
            )

            content_data = ContentData(
                content_id=str(row["content_id"]),
                text=str(row["text"]) if pd.notna(row["text"]) else None,
                image_url=str(row.get("image_url"))
                if pd.notna(row.get("image_url"))
                else None,
                video_url=str(row.get("video_url"))
                if pd.notna(row.get("video_url"))
                else None,
                content_type=ContentType(row.get("content_type", "text")),
                platform=Platform(row.get("platform", "generic")),
                metrics=metrics,
            )

            content_data_list.append(content_data)

        return content_data_list

    def _display_single_analysis_results(self, report):
        """Display results from single content analysis."""
        # Main prediction result
        pred = report.virality_prediction

        col1, col2, col3 = st.columns(3)

        with col1:
            if pred.is_viral_prediction:
                st.success(f"🚀 VIRAL PREDICTION")
            else:
                st.error(f"📉 NON-VIRAL PREDICTION")

        with col2:
            st.metric("Virality Score", f"{pred.virality_score:.2%}")

        with col3:
            st.metric("Confidence", f"{pred.confidence:.2%}")

        # Explanation
        if report.explanation:
            st.subheader("💡 AI Explanation")
            st.write(report.explanation.explanation_text)

            # Key factors
            if report.explanation.key_factors:
                st.subheader("🔑 Key Factors")
                for i, factor in enumerate(report.explanation.key_factors[:5], 1):
                    st.write(f"{i}. {factor.replace('_', ' ').title()}")

            # Recommendations
            if report.explanation.recommendations:
                st.subheader("📋 Recommendations")
                for i, rec in enumerate(report.explanation.recommendations, 1):
                    st.write(f"{i}. {rec}")

        # Feature importance chart
        if pred.feature_importance:
            st.subheader("📊 Feature Importance")

            feature_df = pd.DataFrame(
                [
                    {"Feature": k.replace("_", " ").title(), "Importance": v}
                    for k, v in pred.feature_importance.items()
                ]
            )

            fig = px.bar(
                feature_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="RdYlGn",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Benchmark comparison
        if report.benchmark_comparison:
            st.subheader("🎯 Benchmark Comparison")
            benchmark = report.benchmark_comparison

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Similarity to Viral Content", f"{benchmark.similarity_score:.2%}"
                )

            with col2:
                if benchmark.top_similar_viral_content:
                    st.write("Similar Viral Content:")
                    for content_id in benchmark.top_similar_viral_content[:3]:
                        st.write(f"• {content_id}")

            if benchmark.improvement_suggestions:
                st.write("**Improvement Suggestions:**")
                for suggestion in benchmark.improvement_suggestions:
                    st.write(f"• {suggestion}")

    def _display_batch_analysis_results(self, batch_results):
        """Display results from batch analysis."""
        # Summary statistics
        summary = batch_results.get("summary_stats", {})

        st.subheader("📊 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Content", summary.get("total_content", 0))

        with col2:
            viral_rate = summary.get("predicted_viral_rate", 0)
            st.metric("Viral Rate", f"{viral_rate:.2%}")

        with col3:
            avg_score = summary.get("avg_virality_score", 0)
            st.metric("Avg Score", f"{avg_score:.2%}")

        with col4:
            high_potential = summary.get("high_potential_count", 0)
            st.metric("High Potential", high_potential)

        # Individual results table
        st.subheader("📋 Individual Results")
        individual_reports = batch_results.get("individual_reports", [])

        if individual_reports:
            results_data = []
            for report in individual_reports:
                pred = report.virality_prediction
                results_data.append(
                    {
                        "Content ID": report.content_id,
                        "Viral Prediction": "✅ Viral"
                        if pred.is_viral_prediction
                        else "❌ Non-Viral",
                        "Virality Score": f"{pred.virality_score:.2%}",
                        "Confidence": f"{pred.confidence:.2%}",
                        "Top Factor": list(pred.feature_importance.keys())[0]
                        if pred.feature_importance
                        else "N/A",
                    }
                )

            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)

        # Patterns analysis
        patterns = batch_results.get("patterns", {})
        if patterns:
            st.subheader("🔍 Pattern Analysis")

            # Platform performance
            platform_perf = patterns.get("platform_performance", {})
            if platform_perf:
                st.write("**Platform Performance:**")
                platform_df = pd.DataFrame(
                    [
                        {
                            "Platform": k,
                            "Viral Rate": f"{v['viral_rate']:.2%}",
                            "Avg Score": f"{v['avg_score']:.2%}",
                        }
                        for k, v in platform_perf.items()
                    ]
                )
                st.dataframe(platform_df)

        # Clustering results
        clustering = batch_results.get("clustering", {})
        if clustering:
            st.subheader("🎯 Clustering Analysis")

            content_clusters = clustering.get("content_clusters", {})
            if content_clusters:
                st.write(
                    f"**Identified {content_clusters.get('n_clusters', 0)} content clusters**"
                )

                cluster_analysis = content_clusters.get("cluster_analysis", {})
                for cluster_name, cluster_info in cluster_analysis.items():
                    with st.expander(
                        f"{cluster_name.replace('_', ' ').title()} ({cluster_info['size']} items)"
                    ):
                        st.write(f"Viral Rate: {cluster_info['viral_rate']:.2%}")
                        st.write(
                            f"Avg Virality: {cluster_info['avg_virality_coefficient']:.3f}"
                        )

                        if cluster_info.get("top_distinctive_features"):
                            st.write("Top Distinctive Features:")
                            for feature, data in cluster_info[
                                "top_distinctive_features"
                            ][:3]:
                                st.write(
                                    f"• {feature}: {data['relative_difference']:+.1%} vs average"
                                )

        # Recommendations
        recommendations = batch_results.get("recommendations", [])
        if recommendations:
            st.subheader("💡 Recommendations")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")


def main():
    """Main function to run the Streamlit app."""
    app = ViralityAnalyzerApp()
    app.run()


if __name__ == "__main__":
    main()
