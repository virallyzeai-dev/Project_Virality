# API Documentation

## Content Virality Analyzer REST API

The Virality Analyzer provides both a web interface and REST API for analyzing content virality. This document covers the REST API endpoints.

### Base URL

```
http://localhost:8000
```

### Authentication

Currently, no authentication is required for the API endpoints.

## Endpoints

### Health Check

#### GET `/health`

Check the health status of the API and model.

**Response:**
```json
{
  "status": "healthy",
  "model_trained": true,
  "timestamp": "2024-01-15T10:30:00"
}
```

### Model Training

#### POST `/train`

Train the virality prediction model with historical data.

**Request Body:**
```json
{
  "content_list": [
    {
      "content_id": "train_001",
      "text": "Sample content text",
      "content_type": "text",
      "platform": "twitter",
      "metrics": {
        "likes": 100,
        "shares": 50,
        "comments": 25,
        "impressions": 1000
      },
      "posting_time": "2024-01-15T14:30:00",
      "hashtags": ["#example"],
      "mentions": ["@user"]
    }
  ],
  "model_type": "xgboost",
  "virality_threshold": 2.0,
  "validation_split": 0.2
}
```

**Response:**
```json
{
  "success": true,
  "message": "Model trained successfully",
  "performance_metrics": {
    "f1_score": 0.85,
    "accuracy": 0.82,
    "precision": 0.88,
    "recall": 0.83
  }
}
```

### Prediction

#### POST `/predict`

Get virality prediction for a single piece of content.

**Request Body:**
```json
{
  "content_id": "test_001",
  "text": "Content to analyze",
  "content_type": "text",
  "platform": "twitter",
  "metrics": {
    "likes": 10,
    "shares": 5,
    "comments": 2,
    "impressions": 100
  }
}
```

**Response:**
```json
{
  "content_id": "test_001",
  "virality_score": 0.75,
  "is_viral_prediction": true,
  "confidence": 0.85,
  "feature_importance": {
    "sentiment_polarity": 0.2,
    "hashtag_count": 0.15,
    "engagement_rate": 0.18
  }
}
```

### Comprehensive Analysis

#### POST `/analyze`

Perform comprehensive analysis including prediction and explanation.

**Request Body:** Same as `/predict`

**Response:**
```json
{
  "content_id": "test_001",
  "prediction": {
    "content_id": "test_001",
    "virality_score": 0.75,
    "is_viral_prediction": true,
    "confidence": 0.85,
    "feature_importance": {...}
  },
  "explanation": {
    "content_id": "test_001",
    "explanation_text": "This content is predicted to be viral due to high emotional intensity and optimal posting time.",
    "key_factors": ["emotional_intensity", "posting_hour", "hashtag_count"],
    "recommendations": [
      "Consider adding more visual elements",
      "Optimize hashtag usage for better reach"
    ]
  },
  "benchmark_similarity": 0.82,
  "generated_at": "2024-01-15T10:30:00"
}
```

### Batch Analysis

#### POST `/analyze/batch`

Start batch analysis of multiple content pieces.

**Request Body:**
```json
{
  "content_list": [...], // Array of content objects
  "include_clustering": true,
  "include_causal": true
}
```

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "started"
}
```

#### GET `/analyze/batch/{job_id}`

Get status and results of batch analysis job.

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "results": {
    "individual_reports": [...],
    "summary_stats": {...},
    "clustering": {...},
    "causal_analysis": {...},
    "recommendations": [...]
  }
}
```

### Benchmark Management

#### POST `/benchmark/add`

Add viral content examples to the benchmark database.

**Request Body:**
```json
{
  "content_list": [...] // Array of viral content objects
}
```

**Response:**
```json
{
  "message": "Added 10 viral content examples to benchmark",
  "total_examples": 150
}
```

### Analytics

#### GET `/analytics/insights`

Get analytics insights from analysis history.

**Response:**
```json
{
  "total_analyses": 250,
  "avg_virality_score": 0.45,
  "viral_prediction_rate": 0.32,
  "most_common_factors": {
    "sentiment_polarity": 0.25,
    "hashtag_count": 0.22,
    "emotional_intensity": 0.20
  },
  "improvement_trends": {
    "trend_direction": "improving",
    "improvement": 0.05
  }
}
```

### Model Information

#### GET `/model/info`

Get information about the current trained model.

**Response:**
```json
{
  "trained": true,
  "model_type": "xgboost",
  "virality_threshold": 2.0,
  "feature_count": 45,
  "benchmark_examples": 150,
  "analysis_count": 250
}
```

## Data Models

### ContentData

```json
{
  "content_id": "string",
  "text": "string (optional)",
  "image_url": "string (optional)",
  "video_url": "string (optional)",
  "content_type": "text|image|video",
  "platform": "twitter|instagram|youtube|tiktok|linkedin|generic",
  "metrics": {
    "likes": "integer",
    "shares": "integer", 
    "comments": "integer",
    "impressions": "integer",
    "saves": "integer (optional)",
    "clicks": "integer (optional)",
    "total_views": "integer (optional)",
    "completed_views": "integer (optional)"
  },
  "posting_time": "ISO datetime (optional)",
  "hashtags": ["string"] (optional),
  "mentions": ["string"] (optional),
  "metadata": {} (optional)
}
```

### Error Responses

All endpoints may return error responses in the following format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

Common HTTP status codes:
- `400`: Bad Request (invalid input, model not trained)
- `404`: Not Found (job not found)
- `500`: Internal Server Error (processing failed)

## Usage Examples

### Python Client Example

```python
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

# Train model
training_data = {
    "content_list": [...],  # Your training data
    "model_type": "xgboost"
}

response = requests.post(f"{BASE_URL}/train", json=training_data)
print(response.json())

# Analyze content
content_data = {
    "content_id": "test_001",
    "text": "Amazing new product launch! #innovation",
    "content_type": "text",
    "platform": "twitter",
    "metrics": {
        "likes": 150,
        "shares": 30,
        "comments": 20,
        "impressions": 2000
    }
}

response = requests.post(f"{BASE_URL}/analyze", json=content_data)
result = response.json()

print(f"Virality Score: {result['prediction']['virality_score']:.2%}")
print(f"Prediction: {result['prediction']['is_viral_prediction']}")
print(f"Explanation: {result['explanation']['explanation_text']}")
```

### JavaScript Client Example

```javascript
const BASE_URL = 'http://localhost:8000';

// Analyze content
const contentData = {
  content_id: 'test_001',
  text: 'Amazing new product launch! #innovation',
  content_type: 'text',
  platform: 'twitter',
  metrics: {
    likes: 150,
    shares: 30,
    comments: 20,
    impressions: 2000
  }
};

fetch(`${BASE_URL}/analyze`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(contentData)
})
.then(response => response.json())
.then(result => {
  console.log('Virality Score:', result.prediction.virality_score);
  console.log('Prediction:', result.prediction.is_viral_prediction ? 'Viral' : 'Non-Viral');
  console.log('Explanation:', result.explanation.explanation_text);
});
```

## Rate Limiting

Currently, no rate limiting is implemented. For production use, consider implementing appropriate rate limiting based on your usage patterns.

## Support

For issues or questions about the API, please refer to the project documentation or submit an issue to the project repository.
