"""Core package initialization."""

from .config import settings, ModelConfig, FeatureConfig
from .metrics import MetricsCalculator, ViralityMetrics, BenchmarkMetrics
from .models import (
    ContentType, Platform, ContentData, ContentMetrics, 
    FeatureVector, PredictionResult, ExplanationResult,
    DiagnosticReport, AnalysisJob, AnalysisStatus
)

__version__ = "0.1.0"

__all__ = [
    "settings",
    "ModelConfig", 
    "FeatureConfig",
    "MetricsCalculator",
    "ViralityMetrics",
    "BenchmarkMetrics", 
    "ContentType",
    "Platform",
    "ContentData",
    "ContentMetrics",
    "FeatureVector",
    "PredictionResult", 
    "ExplanationResult",
    "DiagnosticReport",
    "AnalysisJob",
    "AnalysisStatus"
]
