# Content Virality Analyzer - User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [Feature Overview](#feature-overview)
6. [Usage Examples](#usage-examples)
7. [Web Interface](#web-interface)
8. [API Usage](#api-usage)
9. [Advanced Features](#advanced-features)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)

## Introduction

The Content Virality Analyzer is an AI-powered system that predicts and explains why content goes viral. It combines multiple machine learning approaches, explainable AI, causal inference, and benchmarking to provide comprehensive insights into content performance.

### Key Features

- **Multi-modal Analysis**: Analyze text, images, and videos
- **Predictive Modeling**: XGBoost, CatBoost, Random Forest, Neural Networks
- **Explainable AI**: SHAP, LIME, and LLM-powered explanations
- **Causal Analysis**: Identify what actually drives virality
- **Benchmarking**: Compare against successful viral content
- **Clustering**: Discover content and audience patterns
- **Web Interface**: Easy-to-use Streamlit application
- **REST API**: Integrate with existing systems

## Installation

### Prerequisites

- Python 3.8 or higher
- Poetry (recommended) or pip
- Optional: OpenAI API key for enhanced explanations

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd Project_Virality

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Using pip

```bash
# Clone the repository
git clone <repository-url>
cd Project_Virality

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

### Optional Dependencies

For full functionality, install optional dependencies:

```bash
# For enhanced visual analysis
pip install opencv-python pillow

# For causal inference
pip install dowhy causalml

# For advanced similarity search
pip install faiss-cpu

# For OpenAI explanations
pip install openai
```

## Quick Start

### 1. Basic Python Usage

```python
from virality_analyzer import ViralityAnalyzer, ContentData, ContentMetrics, ContentType, Platform
from datetime import datetime

# Initialize analyzer
analyzer = ViralityAnalyzer(model_type="xgboost", virality_threshold=2.0)

# Create sample content
content = ContentData(
    content_id="example_001",
    text="Amazing breakthrough in AI technology! 🚀 #AI #Innovation",
    content_type=ContentType.TEXT,
    platform=Platform.TWITTER,
    metrics=ContentMetrics(
        likes=1500,
        shares=300,
        comments=89,
        impressions=25000
    ),
    posting_time=datetime.now(),
    hashtags=["#AI", "#Innovation"]
)

# Train with your data (you need historical data)
# analyzer.train(training_data, viral_examples)

# Analyze content
# report = analyzer.analyze(content)
# print(f"Virality Score: {report.virality_prediction.virality_score:.2%}")
```

### 2. Web Interface

```bash
# Start the Streamlit web app
streamlit run src/virality_analyzer/api/app.py
```

### 3. REST API

```bash
# Start the FastAPI server
uvicorn src.virality_analyzer.api.endpoints:app --reload
```

## Core Concepts

### Virality Metrics

The system uses several key metrics to determine virality:

- **Virality Coefficient**: `shares / unique_users_reached`
- **Engagement Rate**: `(likes + shares + comments) / impressions`
- **Share Rate**: `shares / impressions`
- **Completion Rate**: `completed_views / total_views` (for videos)

### Content Types

Supported content types:
- **Text**: Social media posts, articles
- **Image**: Photos, graphics, memes
- **Video**: Short-form and long-form videos

### Platforms

Supported platforms:
- Twitter
- Instagram
- YouTube
- TikTok
- LinkedIn
- Generic (for other platforms)

### Feature Categories

1. **Text Features**: Sentiment, readability, social elements
2. **Visual Features**: Composition, colors, faces, aesthetics
3. **Engagement Features**: Rates, temporal patterns
4. **Platform Features**: Platform-specific optimization

## Feature Overview

### Multi-Modal Feature Extraction

#### Text Analysis
- Sentiment analysis (polarity, subjectivity)
- Emotional intensity detection
- Readability scoring (Flesch-Kincaid, etc.)
- Social media elements (hashtags, mentions, emojis)
- Linguistic patterns (caps usage, punctuation)

#### Visual Analysis
- Face detection and counting
- Color analysis (brightness, contrast, dominance)
- Composition analysis (rule of thirds, symmetry)
- Aesthetic scoring
- Object detection (when available)

#### Engagement Analysis
- Engagement rate calculations
- Temporal pattern analysis
- Virality coefficient computation
- Platform-specific metrics

### Predictive Models

#### Available Algorithms
1. **XGBoost**: Gradient boosting, excellent for tabular data
2. **CatBoost**: Categorical boosting, handles categorical features well
3. **Random Forest**: Ensemble method, good baseline
4. **Neural Networks**: Deep learning for complex patterns

#### Model Selection
Choose based on your data characteristics:
- **XGBoost**: General-purpose, high performance
- **CatBoost**: Many categorical features
- **Random Forest**: Interpretable baseline
- **Neural Networks**: Large datasets, complex patterns

### Explainable AI

#### SHAP (SHapley Additive exPlanations)
- Global feature importance
- Local explanations for individual predictions
- Interaction effects

#### LIME (Local Interpretable Model-agnostic Explanations)
- Local explanations around predictions
- Model-agnostic approach

#### LLM-Powered Explanations
- Natural language explanations
- Actionable recommendations
- Context-aware insights

### Causal Analysis

#### DoWhy Integration
- Causal graph construction
- Treatment effect estimation
- Refutation testing

#### Features
- Confounder identification
- Treatment effect quantification
- Mediation analysis
- Heterogeneous effects

## Usage Examples

### Example 1: Single Content Analysis

```python
from virality_analyzer import ViralityAnalyzer, ContentData, ContentMetrics, ContentType, Platform
from datetime import datetime

# Initialize
analyzer = ViralityAnalyzer()

# Create content
content = ContentData(
    content_id="post_001",
    text="Breaking: Revolutionary AI breakthrough changes everything! 🚀 #AI #Tech",
    content_type=ContentType.TEXT,
    platform=Platform.TWITTER,
    metrics=ContentMetrics(
        likes=2500,
        shares=450,
        comments=125,
        impressions=35000
    ),
    posting_time=datetime(2024, 1, 15, 14, 30),  # Peak engagement time
    hashtags=["#AI", "#Tech"]
)

# Analyze (after training)
report = analyzer.analyze(content)

print(f"Virality Score: {report.virality_prediction.virality_score:.2%}")
print(f"Prediction: {'VIRAL' if report.virality_prediction.is_viral_prediction else 'Non-Viral'}")
print(f"Explanation: {report.explanation.explanation_text}")

# Get recommendations
for i, rec in enumerate(report.explanation.recommendations, 1):
    print(f"{i}. {rec}")
```

### Example 2: Batch Analysis

```python
import pandas as pd

# Load your data
df = pd.read_csv("content_data.csv")

# Convert to ContentData objects
content_list = []
for _, row in df.iterrows():
    content = ContentData(
        content_id=row['content_id'],
        text=row['text'],
        content_type=ContentType(row['content_type']),
        platform=Platform(row['platform']),
        metrics=ContentMetrics(
            likes=row['likes'],
            shares=row['shares'],
            comments=row['comments'],
            impressions=row['impressions']
        )
    )
    content_list.append(content)

# Perform batch analysis
batch_results = analyzer.analyze_batch(content_list)

# View summary
summary = batch_results['summary_stats']
print(f"Analyzed {summary['total_content']} pieces of content")
print(f"Viral prediction rate: {summary['predicted_viral_rate']:.2%}")
print(f"Average virality score: {summary['avg_virality_score']:.2%}")

# View clustering results
if 'clustering' in batch_results:
    clusters = batch_results['clustering']['content_clusters']
    print(f"Identified {clusters['n_clusters']} distinct content clusters")
```

### Example 3: Causal Analysis

```python
from virality_analyzer.models import CausalAnalyzer

# Initialize causal analyzer
causal_analyzer = CausalAnalyzer(virality_threshold=2.0)

# Analyze causal relationships
causal_results = causal_analyzer.analyze_feature_causality(
    content_list,
    key_features=['sentiment_polarity', 'hashtag_count', 'posting_hour']
)

# View causal effects
print("Causal Effects on Virality:")
for effect in causal_results.causal_effects:
    significance = "✓" if effect.is_significant else "✗"
    print(f"{significance} {effect.factor}: {effect.effect_size:+.3f}")
```

### Example 4: Benchmarking

```python
from virality_analyzer.analysis import ViralContentBenchmark

# Initialize benchmark
benchmark = ViralContentBenchmark()

# Add viral examples
viral_examples = load_viral_content()  # Your viral content
benchmark.add_viral_content(viral_examples)

# Benchmark your content
benchmark_result = benchmark.benchmark_content(content)

print(f"Similarity to viral content: {benchmark_result.similarity_score:.2%}")

# View improvement suggestions
for suggestion in benchmark_result.improvement_suggestions:
    print(f"• {suggestion}")
```

## Web Interface

### Starting the Web App

```bash
streamlit run src/virality_analyzer/api/app.py
```

### Features

1. **Single Content Analysis**
   - Input content details manually
   - Upload images/videos
   - Get instant predictions and explanations

2. **Batch Analysis**
   - Upload CSV files with multiple content pieces
   - Comprehensive clustering and pattern analysis
   - Export results

3. **Training & Setup**
   - Train models with your historical data
   - Add viral content for benchmarking
   - Configure settings

4. **Analytics Dashboard**
   - View analysis history
   - Track performance trends
   - Identify important factors

### Navigation

- **Single Content Analysis**: Analyze individual posts
- **Batch Analysis**: Upload and analyze multiple items
- **Training & Setup**: Configure and train models
- **Analytics Dashboard**: View insights and trends

## API Usage

### Starting the API Server

```bash
uvicorn src.virality_analyzer.api.endpoints:app --reload --host 0.0.0.0 --port 8000
```

### Basic API Usage

```python
import requests

# API base URL
BASE_URL = "http://localhost:8000"

# Analyze content
content_data = {
    "content_id": "api_test",
    "text": "Testing API functionality! #test",
    "content_type": "text",
    "platform": "twitter",
    "metrics": {
        "likes": 100,
        "shares": 20,
        "comments": 10,
        "impressions": 1000
    }
}

response = requests.post(f"{BASE_URL}/analyze", json=content_data)
result = response.json()

print(f"Virality Score: {result['prediction']['virality_score']:.2%}")
```

For detailed API documentation, see [API Documentation](api.md).

## Advanced Features

### Custom Feature Engineering

```python
from virality_analyzer.features import TextFeatureExtractor

# Extend text feature extractor
class CustomTextExtractor(TextFeatureExtractor):
    def extract_custom_features(self, text):
        features = {}
        
        # Add your custom features
        features['custom_metric'] = self.calculate_custom_metric(text)
        
        return features
    
    def calculate_custom_metric(self, text):
        # Your custom logic
        return len(text.split())
```

### Custom Models

```python
from virality_analyzer.models import ViralityPredictor
from sklearn.ensemble import GradientBoostingClassifier

# Use custom model
predictor = ViralityPredictor()
predictor.model = GradientBoostingClassifier()
```

### Hyperparameter Tuning

```python
# Define parameter grid
param_grid = {
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200]
}

# Tune hyperparameters
tuning_results = predictor.hyperparameter_tuning(
    training_data, 
    param_grid=param_grid
)

print("Best parameters:", tuning_results['best_params'])
print("Best score:", tuning_results['best_score'])
```

## Best Practices

### Data Preparation

1. **Quality over Quantity**: Ensure clean, representative data
2. **Balanced Dataset**: Include both viral and non-viral examples
3. **Temporal Relevance**: Use recent data for current trends
4. **Platform Diversity**: Include data from multiple platforms

### Feature Engineering

1. **Domain Knowledge**: Leverage social media expertise
2. **Feature Selection**: Remove irrelevant or redundant features
3. **Scaling**: Ensure features are properly scaled
4. **Validation**: Test features on held-out data

### Model Training

1. **Cross-Validation**: Use proper validation techniques
2. **Regularization**: Prevent overfitting
3. **Ensemble Methods**: Combine multiple models
4. **Regular Updates**: Retrain with fresh data

### Interpretation

1. **Multiple Perspectives**: Use both global and local explanations
2. **Causal Thinking**: Distinguish correlation from causation
3. **Context Matters**: Consider platform and audience differences
4. **Actionable Insights**: Focus on implementable recommendations

## Troubleshooting

### Common Issues

#### Installation Problems

**Issue**: Poetry installation fails
```bash
# Solution: Update Poetry
curl -sSL https://install.python-poetry.org | python3 -
```

**Issue**: Missing dependencies
```bash
# Solution: Install all optional dependencies
poetry install --all-extras
```

#### Training Issues

**Issue**: "Not enough data" error
- **Solution**: Provide at least 50-100 examples
- **Alternative**: Use pre-trained models if available

**Issue**: Poor model performance
- **Solutions**:
  - Check data quality and labels
  - Try different model types
  - Adjust virality threshold
  - Add more diverse training data

#### API Issues

**Issue**: API server won't start
```bash
# Check if port is in use
lsof -i :8000

# Use different port
uvicorn src.virality_analyzer.api.endpoints:app --port 8001
```

**Issue**: CORS errors in browser
- **Solution**: The API includes CORS middleware, but ensure your frontend properly handles requests

#### Web Interface Issues

**Issue**: Streamlit app crashes
```bash
# Clear Streamlit cache
streamlit cache clear

# Restart with verbose logging
streamlit run src/virality_analyzer/api/app.py --logger.level=debug
```

### Performance Optimization

#### Large Datasets

1. **Batch Processing**: Process data in chunks
2. **Feature Caching**: Cache extracted features
3. **Model Persistence**: Save and load trained models

```python
# Save trained model
analyzer.diagnostic_engine.predictor.save_model("model.joblib")

# Load trained model
analyzer.diagnostic_engine.predictor.load_model("model.joblib")
```

#### Memory Management

```python
# Process large datasets in batches
batch_size = 100
for i in range(0, len(large_dataset), batch_size):
    batch = large_dataset[i:i + batch_size]
    results = analyzer.analyze_batch(batch)
    # Process results
```

### Getting Help

1. **Documentation**: Check this guide and API docs
2. **Examples**: Review example scripts
3. **Issues**: Submit GitHub issues for bugs
4. **Community**: Join discussions in project forums

### Performance Monitoring

```python
# Monitor analysis performance
import time

start_time = time.time()
report = analyzer.analyze(content)
analysis_time = time.time() - start_time

print(f"Analysis completed in {analysis_time:.2f} seconds")
```

## Conclusion

The Content Virality Analyzer provides powerful tools for understanding and predicting content performance. By combining multiple AI approaches with explainable insights, it helps content creators optimize their strategies for maximum viral potential.

For the best results:
- Use high-quality, diverse training data
- Regularly update models with fresh data
- Combine multiple analysis approaches
- Focus on actionable insights
- Consider platform-specific factors

Happy analyzing! 🚀
