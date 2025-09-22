"""Core configuration and settings for the Virality Analyzer."""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # API Keys
    openai_api_key: Optional[str] = Field(default=None)
    twitter_api_key: Optional[str] = Field(default=None)
    twitter_api_secret: Optional[str] = Field(default=None)
    instagram_access_token: Optional[str] = Field(default=None)
    youtube_api_key: Optional[str] = Field(default=None)

    # Model Configuration
    virality_threshold: float = Field(default=2.0)
    model_type: str = Field(default="xgboost")
    feature_extraction_batch_size: int = Field(default=32)

    # Database
    database_url: str = Field(default="sqlite:///virality_data.db")

    # Logging
    log_level: str = Field(default="INFO")

    # Feature Extraction
    use_gpu: bool = Field(default=False)
    max_text_length: int = Field(default=512)
    image_size: int = Field(default=224)

    # Benchmarking
    viral_content_db_path: str = Field(default="data/viral_benchmark.faiss")
    similarity_threshold: float = Field(default=0.8)


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
        "random_state": 42,
    }

    CATBOOST_PARAMS = {
        "iterations": 100,
        "learning_rate": 0.1,
        "depth": 6,
        "l2_leaf_reg": 3,
        "random_seed": 42,
        "verbose": False,
    }

    NEURAL_NET_PARAMS = {
        "hidden_layers": [256, 128, 64],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "early_stopping_patience": 10,
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
        "emoji_count",
    ]

    VISUAL_FEATURES = [
        "brightness",
        "contrast",
        "saturation",
        "color_dominance",
        "face_count",
        "object_count",
        "visual_complexity",
        "aesthetic_score",
    ]

    ENGAGEMENT_FEATURES = [
        "likes_per_impression",
        "shares_per_impression",
        "comments_per_impression",
        "click_through_rate",
        "watch_time_avg",
        "completion_rate",
        "save_rate",
        "virality_coefficient",
    ]

    PLATFORM_FEATURES = [
        "posting_hour",
        "posting_day_of_week",
        "hashtag_popularity",
        "trending_topic_alignment",
        "follower_count",
        "account_age",
        "previous_viral_count",
    ]
