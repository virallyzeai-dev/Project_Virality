"""Core configuration and settings for the Virality Analyzer."""

import os
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings."""
    
    # API Keys
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    twitter_api_key: Optional[str] = Field(None, env="TWITTER_API_KEY")
    twitter_api_secret: Optional[str] = Field(None, env="TWITTER_API_SECRET")
    instagram_access_token: Optional[str] = Field(None, env="INSTAGRAM_ACCESS_TOKEN")
    youtube_api_key: Optional[str] = Field(None, env="YOUTUBE_API_KEY")
    
    # Model Configuration
    virality_threshold: float = Field(2.0, env="VIRALITY_THRESHOLD")
    model_type: str = Field("xgboost", env="MODEL_TYPE")
    feature_extraction_batch_size: int = Field(32, env="FEATURE_EXTRACTION_BATCH_SIZE")
    
    # Database
    database_url: str = Field("sqlite:///virality_data.db", env="DATABASE_URL")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # Feature Extraction
    use_gpu: bool = Field(False, env="USE_GPU")
    max_text_length: int = Field(512, env="MAX_TEXT_LENGTH")
    image_size: int = Field(224, env="IMAGE_SIZE")
    
    # Benchmarking
    viral_content_db_path: str = Field("data/viral_benchmark.faiss", env="VIRAL_CONTENT_DB_PATH")
    similarity_threshold: float = Field(0.8, env="SIMILARITY_THRESHOLD")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


class ModelConfig:
    """Configuration for different model types."""
    
    XGBOOST_PARAMS = {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }
    
    CATBOOST_PARAMS = {
        "iterations": 100,
        "learning_rate": 0.1,
        "depth": 6,
        "l2_leaf_reg": 3,
        "random_seed": 42,
        "verbose": False
    }
    
    NEURAL_NET_PARAMS = {
        "hidden_layers": [256, 128, 64],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "early_stopping_patience": 10
    }


class FeatureConfig:
    """Configuration for feature extraction."""
    
    TEXT_FEATURES = [
        "sentiment_polarity",
        "sentiment_subjectivity", 
        "emotional_intensity",
        "readability_score",
        "text_length",
        "word_count",
        "sentence_count",
        "hashtag_count",
        "mention_count",
        "url_count",
        "emoji_count"
    ]
    
    VISUAL_FEATURES = [
        "brightness",
        "contrast",
        "saturation",
        "color_dominance",
        "face_count",
        "object_count",
        "visual_complexity",
        "aesthetic_score"
    ]
    
    ENGAGEMENT_FEATURES = [
        "likes_per_impression",
        "shares_per_impression", 
        "comments_per_impression",
        "click_through_rate",
        "watch_time_avg",
        "completion_rate",
        "save_rate",
        "virality_coefficient"
    ]
    
    PLATFORM_FEATURES = [
        "posting_hour",
        "posting_day_of_week",
        "hashtag_popularity",
        "trending_topic_alignment",
        "follower_count",
        "account_age",
        "previous_viral_count"
    ]
