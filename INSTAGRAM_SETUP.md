# Instagram Virality Analysis Setup Guide

## Overview

This guide shows you how to set up and use the Content Virality Analyzer with Instagram posts using your Instagram access token.

## Prerequisites

### 1. Instagram Access Token

You'll need an Instagram access token to fetch post data. You can get this through:

**Option A: Instagram Basic Display API (Recommended for Personal Use)**
1. Go to [Facebook Developers](https://developers.facebook.com/)
2. Create a new app
3. Add "Instagram Basic Display" product
4. Configure OAuth redirect URIs
5. Get a User Access Token

**Option B: Instagram Graph API (For Business Accounts)**
1. Create a Facebook App
2. Add Instagram Graph API product
3. Connect your Instagram Business Account
4. Generate access token with required permissions

### Required Permissions:
- `instagram_graph_user_profile`
- `instagram_graph_user_media`
- `pages_show_list` (for Instagram Graph API)

## Installation

### 1. Clone and Setup the Project

```powershell
# Clone the repository
git clone <your-repository-url>
cd Project_Virality

# Install dependencies using Poetry (recommended)
poetry install

# Or using pip
pip install -e .

# Install optional dependencies for enhanced features
poetry install --extras "all"
```

### 2. Environment Configuration

Copy the example environment file and configure your tokens:

```powershell
copy .env.example .env
```

Edit `.env` file with your credentials:

```bash
# Instagram API Configuration
INSTAGRAM_ACCESS_TOKEN=your_actual_instagram_access_token_here

# Optional: OpenAI for enhanced explanations
OPENAI_API_KEY=your_openai_api_key

# Model Configuration
VIRALITY_THRESHOLD=2.0
MODEL_TYPE=xgboost

# Other settings
LOG_LEVEL=INFO
USE_GPU=false
```

### 3. Verify Setup

Run the verification script:

```powershell
poetry run python verify_setup.py
```

## Usage

### Basic Analysis

Run the Instagram analysis script:

```powershell
poetry run python examples/instagram_virality_analysis.py
```

This will:
1. Connect to Instagram API using your token
2. Fetch your recent posts (default: 10 posts)
3. Analyze each post for virality potential
4. Provide recommendations for improvement

### Custom Analysis

You can also use the analyzer programmatically:

```python
from examples.instagram_virality_analysis import InstagramDataCollector, analyze_instagram_virality
import os

# Load your access token
access_token = os.getenv("INSTAGRAM_ACCESS_TOKEN")

# Analyze specific number of posts
analyze_instagram_virality(access_token, num_posts=20)
```

## Understanding the Results

### Virality Score
- **0.0 - 1.0**: Low virality potential
- **1.0 - 2.0**: Moderate potential 
- **2.0 - 5.0**: High potential (considered "viral")
- **5.0+**: Exceptional viral potential

### Metrics Analyzed
- **Engagement Rate**: (Likes + Comments) / Impressions
- **Save Rate**: Saves / Impressions (if available)
- **Comment Ratio**: Comments / Likes
- **Completion Rate**: For videos, views that watched to completion

### Common Recommendations
- Use 5-11 relevant hashtags
- Post during peak hours (10 AM - 9 PM)
- Include questions in captions to encourage comments
- Hook viewers in first 3 seconds for videos
- Write descriptive captions (50+ characters)

## Troubleshooting

### Common Issues

**1. "Instagram access token not found"**
- Ensure `INSTAGRAM_ACCESS_TOKEN` is set in your `.env` file
- Check that there are no extra spaces or quotes around the token

**2. "Error fetching Instagram data"**
- Verify your access token is valid and not expired
- Check that you have the required permissions
- Ensure your Instagram account has posts to analyze

**3. "No posts found"**
- Your Instagram account might be private
- Check if you have recent posts
- Verify the token has correct permissions

**4. Module import errors**
- Run `poetry install` to ensure all dependencies are installed
- Activate the virtual environment: `poetry shell`

### Getting Help

If you encounter issues:

1. Check the [Instagram Basic Display API documentation](https://developers.facebook.com/docs/instagram-basic-display-api)
2. Verify your app is in "Live" mode (not Development)
3. Test your access token using Instagram's Graph API Explorer
4. Check the application logs for detailed error messages

## Advanced Features

### Batch Analysis
Analyze multiple Instagram accounts or historical data:

```python
# Collect data from multiple timeframes
collector = InstagramDataCollector(access_token)
posts = collector.get_user_media(limit=50)  # Get more posts

# Analyze in batches
for batch in chunks(posts, 10):
    # Process each batch
    analyze_batch(batch)
```

### Custom Virality Thresholds
Adjust what you consider "viral" based on your account size:

```python
# For smaller accounts (< 10K followers)
analyzer = ViralityAnalyzer(virality_threshold=1.0)

# For larger accounts (> 100K followers)  
analyzer = ViralityAnalyzer(virality_threshold=3.0)
```

### Integration with Other Platforms
The analyzer supports multiple platforms. You can compare your Instagram performance against other social media:

```python
# Mix Instagram with other platform data
all_content = [
    instagram_posts,
    twitter_posts, 
    youtube_videos
]

# Comparative analysis
results = analyzer.analyze_batch(all_content)
```

## Next Steps

1. **Collect Historical Data**: Gather more posts for better ML model training
2. **Track Performance**: Monitor how implemented recommendations affect your metrics
3. **Custom Models**: Train models specific to your niche or industry
4. **Automated Monitoring**: Set up scheduled analysis of new posts
5. **A/B Testing**: Test different content strategies based on recommendations

## API Limits and Best Practices

### Instagram API Limits
- **Basic Display API**: 200 requests per hour per user
- **Graph API**: Varies by app usage and verification status

### Best Practices
- Cache analysis results to avoid redundant API calls
- Implement rate limiting in your scripts
- Store historical data for trend analysis
- Regular token refresh for long-term usage

---

For more detailed documentation, see:
- [User Guide](../docs/user_guide.md)  
- [API Documentation](../docs/api.md)
- [Instagram API Documentation](https://developers.facebook.com/docs/instagram-api/)