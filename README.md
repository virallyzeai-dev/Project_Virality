# Content Virality Analyzer

An AI-powered system to analyze and predict content virality, providing insights into why content goes viral or fails to achieve viral status.

## Features

- **Predictive Modeling**: XGBoost, CatBoost, and neural networks for virality prediction
- **Causal Analysis**: Identify causal relationships between content features and virality
- **Explainable AI**: SHAP and LIME for understanding model decisions
- **Multi-modal Analysis**: Text, image, and video content analysis
- **Automated Diagnostics**: Generate actionable insights and recommendations
- **Benchmarking**: Compare against viral content in your domain
- **Real-time Analysis**: Streamlit web interface for interactive analysis

## Project Structure

```
src/
├── virality_analyzer/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── metrics.py
│   │   └── models.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collectors.py
│   │   ├── preprocessors.py
│   │   └── validators.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── text_features.py
│   │   ├── visual_features.py
│   │   ├── engagement_features.py
│   │   └── platform_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── predictive.py
│   │   ├── causal.py
│   │   └── explainer.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── benchmarking.py
│   │   ├── clustering.py
│   │   └── diagnostics.py
│   └── api/
│       ├── __init__.py
│       ├── app.py
│       └── endpoints.py
tests/
examples/
docs/
```

## Installation

1. Make sure you have Python 3.9+ and Poetry installed
2. Clone this repository
3. Install dependencies:

```bash
poetry install
```

4. Activate the virtual environment:

```bash
poetry shell
```

5. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Quick Start

```python
from virality_analyzer import ViralityAnalyzer

# Initialize the analyzer
analyzer = ViralityAnalyzer()

# Analyze content
content = {
    "text": "Your content text here",
    "image_url": "path/to/image.jpg",
    "metrics": {
        "likes": 100,
        "shares": 20,
        "comments": 15,
        "impressions": 5000
    }
}

# Get virality prediction and explanation
result = analyzer.analyze(content)
print(f"Virality Score: {result.virality_score}")
print(f"Explanation: {result.explanation}")
```

## Usage Examples

### 1. Content Analysis
```python
# Analyze why content didn't go viral
analysis = analyzer.diagnose_non_viral(content)
print(analysis.recommendations)
```

### 2. Benchmarking
```python
# Compare against viral content
benchmark = analyzer.benchmark_against_viral(content, category="tech")
print(benchmark.similarity_score)
```

### 3. Causal Analysis
```python
# Identify causal factors
causal_results = analyzer.causal_analysis(dataset)
print(causal_results.causal_effects)
```

## Web Interface

Launch the Streamlit web interface:

```bash
streamlit run src/virality_analyzer/api/app.py
```

## Configuration

Edit the `.env` file to configure:
- OpenAI API key for LLM-powered explanations
- Social media API credentials
- Model hyperparameters
- Feature extraction settings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `poetry run pytest`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
