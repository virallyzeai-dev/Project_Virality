"""Core data models for the virality analyzer."""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ContentType(Enum):
    """Types of content that can be analyzed."""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    MIXED = "mixed"


class Platform(Enum):
    """Social media platforms."""

    TWITTER = "twitter"
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    FACEBOOK = "facebook"
    LINKEDIN = "linkedin"
    GENERIC = "generic"


@dataclass
class ContentMetrics:
    """Raw metrics for a piece of content."""

    likes: int = 0
    shares: int = 0
    comments: int = 0
    impressions: int = 0
    saves: Optional[int] = None
    clicks: Optional[int] = None
    total_views: Optional[int] = None
    completed_views: Optional[int] = None
    unique_users_reached: Optional[int] = None
    impressions_timeline: Optional[List[int]] = None
    time_intervals: Optional[List[float]] = None


@dataclass
class ContentData:
    """Complete content data for analysis."""

    content_id: str
    text: Optional[str] = None
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    content_type: ContentType = ContentType.TEXT
    platform: Platform = Platform.GENERIC
    metrics: ContentMetrics = field(default_factory=ContentMetrics)
    posting_time: Optional[datetime] = None
    hashtags: Optional[List[str]] = None
    mentions: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureVector:
    """Extracted features for ML models."""

    text_features: Dict[str, float] = field(default_factory=dict)
    visual_features: Dict[str, float] = field(default_factory=dict)
    engagement_features: Dict[str, float] = field(default_factory=dict)
    platform_features: Dict[str, float] = field(default_factory=dict)
    embeddings: Optional[List[float]] = None


@dataclass
class PredictionResult:
    """Result of virality prediction."""

    content_id: str
    virality_score: float
    is_viral_prediction: bool
    confidence: float
    feature_importance: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExplanationResult:
    """Explanation of why content is/isn't viral."""

    content_id: str
    explanation_text: str
    key_factors: List[str]
    recommendations: List[str]
    benchmark_comparison: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class CausalEffect:
    """Result of causal analysis."""

    factor: str
    effect_size: float
    p_value: float
    confidence_interval: tuple
    is_significant: bool


@dataclass
class CausalAnalysisResult:
    """Complete causal analysis result."""

    content_id: str
    causal_effects: List[CausalEffect]
    treatment_effects: Dict[str, float] = field(default_factory=dict)
    confounders: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Result of benchmarking against viral content."""

    content_id: str
    similarity_score: float
    top_similar_viral_content: List[str]
    feature_gaps: Dict[str, float] = field(default_factory=dict)
    improvement_suggestions: List[str] = field(default_factory=list)


@dataclass
class DiagnosticReport:
    """Complete diagnostic report for content."""

    content_id: str
    virality_prediction: PredictionResult
    explanation: ExplanationResult
    causal_analysis: Optional[CausalAnalysisResult] = None
    benchmark_comparison: Optional[BenchmarkResult] = None
    generated_at: datetime = field(default_factory=datetime.now)


class AnalysisStatus(Enum):
    """Status of content analysis."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AnalysisJob:
    """Represents an analysis job."""

    job_id: str
    content_data: ContentData
    status: AnalysisStatus = AnalysisStatus.PENDING
    result: Optional[DiagnosticReport] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class ModelPerformance:
    """Model performance metrics."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confusion_matrix: List[List[int]]
    feature_importance: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Result of model training."""

    model_type: str
    performance: ModelPerformance
    model_path: str
    training_data_size: int
    validation_data_size: int
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    trained_at: datetime = field(default_factory=datetime.now)


# Type aliases for convenience
Features = Dict[str, Union[float, int, str]]
Predictions = Dict[str, float]
Embeddings = List[float]
