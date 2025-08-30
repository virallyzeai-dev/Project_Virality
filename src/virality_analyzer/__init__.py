"""Content Virality Analyzer package initialization."""

from .core import (
    settings, ContentData, ContentMetrics, ContentType, Platform,
    ViralityMetrics, DiagnosticReport
)
from .features import (
    TextFeatureExtractor, VisualFeatureExtractor,
    EngagementFeatureExtractor, PlatformFeatureExtractor
)
from .models import (
    ViralityPredictor, FeatureProcessor, CausalAnalyzer, ModelExplainer
)
from .analysis import (
    ViralContentBenchmark, ContentClusterAnalyzer, DiagnosticEngine
)

__version__ = "0.1.0"

# Main analyzer class for easy usage
class ViralityAnalyzer:
    """Main analyzer class that provides a simple interface to all functionality."""
    
    def __init__(self, virality_threshold: float = 2.0, model_type: str = "xgboost"):
        """Initialize the virality analyzer."""
        self.diagnostic_engine = DiagnosticEngine(virality_threshold=virality_threshold)
        self.diagnostic_engine.predictor.model_type = model_type
        self.diagnostic_engine.predictor._initialize_model()
        
    def train(self, training_data, viral_examples=None):
        """Train the analyzer with historical data."""
        # Setup predictor
        self.diagnostic_engine.setup_predictor(training_data)
        
        # Setup benchmark if viral examples provided
        if viral_examples:
            self.diagnostic_engine.setup_benchmark(viral_examples)
        
        return self
    
    def analyze(self, content_data):
        """Analyze a single piece of content."""
        return self.diagnostic_engine.diagnose_single_content(content_data)
    
    def analyze_batch(self, content_data_list):
        """Analyze multiple pieces of content."""
        return self.diagnostic_engine.diagnose_batch_content(content_data_list)
    
    def predict(self, content_data):
        """Get virality prediction for content."""
        return self.diagnostic_engine.predictor.predict(content_data)
    
    def explain(self, content_data):
        """Get explanation for content's virality potential."""
        if self.diagnostic_engine.explainer:
            return self.diagnostic_engine.explainer.explain_prediction(content_data)
        return None
    
    def benchmark(self, content_data, category=None):
        """Benchmark content against viral examples."""
        return self.diagnostic_engine.benchmark.benchmark_content(content_data, category)
    
    def get_insights(self):
        """Get analytics insights from analysis history."""
        return self.diagnostic_engine.get_analysis_insights()


__all__ = [
    # Main class
    "ViralityAnalyzer",
    
    # Core components
    "settings",
    "ContentData", 
    "ContentMetrics",
    "ContentType",
    "Platform",
    "ViralityMetrics",
    "DiagnosticReport",
    
    # Feature extractors
    "TextFeatureExtractor",
    "VisualFeatureExtractor", 
    "EngagementFeatureExtractor",
    "PlatformFeatureExtractor",
    
    # Models
    "ViralityPredictor",
    "FeatureProcessor",
    "CausalAnalyzer",
    "ModelExplainer",
    
    # Analysis
    "ViralContentBenchmark",
    "ContentClusterAnalyzer", 
    "DiagnosticEngine"
]
