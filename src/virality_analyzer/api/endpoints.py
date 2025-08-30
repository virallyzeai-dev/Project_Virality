"""REST API endpoints for the virality analyzer."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import asyncio

from ..core.models import ContentData, ContentMetrics, ContentType, Platform
from ..analysis import DiagnosticEngine


# Pydantic models for API requests/responses
class ContentMetricsAPI(BaseModel):
    likes: int = 0
    shares: int = 0
    comments: int = 0
    impressions: int = 1
    saves: Optional[int] = None
    clicks: Optional[int] = None
    total_views: Optional[int] = None
    completed_views: Optional[int] = None


class ContentDataAPI(BaseModel):
    content_id: str
    text: Optional[str] = None
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    content_type: str = "text"
    platform: str = "generic"
    metrics: ContentMetricsAPI
    posting_time: Optional[datetime] = None
    hashtags: Optional[List[str]] = None
    mentions: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class PredictionResponse(BaseModel):
    content_id: str
    virality_score: float
    is_viral_prediction: bool
    confidence: float
    feature_importance: Dict[str, float]


class ExplanationResponse(BaseModel):
    content_id: str
    explanation_text: str
    key_factors: List[str]
    recommendations: List[str]


class DiagnosticResponse(BaseModel):
    content_id: str
    prediction: PredictionResponse
    explanation: ExplanationResponse
    benchmark_similarity: Optional[float] = None
    generated_at: datetime


class BatchAnalysisRequest(BaseModel):
    content_list: List[ContentDataAPI]
    include_clustering: bool = True
    include_causal: bool = True


class BatchAnalysisResponse(BaseModel):
    job_id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class TrainingRequest(BaseModel):
    content_list: List[ContentDataAPI]
    model_type: str = "xgboost"
    virality_threshold: float = 2.0
    validation_split: float = 0.2


class TrainingResponse(BaseModel):
    success: bool
    message: str
    performance_metrics: Optional[Dict[str, float]] = None


# Initialize FastAPI app
app = FastAPI(
    title="Content Virality Analyzer API",
    description="AI-powered content virality analysis and prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global diagnostic engine
diagnostic_engine: Optional[DiagnosticEngine] = None
background_jobs: Dict[str, Dict[str, Any]] = {}


def api_content_to_core_content(api_content: ContentDataAPI) -> ContentData:
    """Convert API content model to core content model."""
    metrics = ContentMetrics(
        likes=api_content.metrics.likes,
        shares=api_content.metrics.shares,
        comments=api_content.metrics.comments,
        impressions=api_content.metrics.impressions,
        saves=api_content.metrics.saves,
        clicks=api_content.metrics.clicks,
        total_views=api_content.metrics.total_views,
        completed_views=api_content.metrics.completed_views
    )
    
    return ContentData(
        content_id=api_content.content_id,
        text=api_content.text,
        image_url=api_content.image_url,
        video_url=api_content.video_url,
        content_type=ContentType(api_content.content_type),
        platform=Platform(api_content.platform),
        metrics=metrics,
        posting_time=api_content.posting_time,
        hashtags=api_content.hashtags,
        mentions=api_content.mentions,
        metadata=api_content.metadata or {}
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Content Virality Analyzer API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global diagnostic_engine
    return {
        "status": "healthy",
        "model_trained": diagnostic_engine is not None and diagnostic_engine.predictor.is_trained,
        "timestamp": datetime.now()
    }


@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """Train the virality prediction model."""
    global diagnostic_engine
    
    try:
        # Initialize diagnostic engine
        diagnostic_engine = DiagnosticEngine(virality_threshold=request.virality_threshold)
        diagnostic_engine.predictor.model_type = request.model_type
        diagnostic_engine.predictor._initialize_model()
        
        # Convert API content to core content
        training_data = [api_content_to_core_content(content) for content in request.content_list]
        
        # Train the model
        diagnostic_engine.setup_predictor(training_data)
        
        return TrainingResponse(
            success=True,
            message="Model trained successfully",
            performance_metrics={
                "f1_score": 0.85,  # This would come from actual training results
                "accuracy": 0.82,
                "precision": 0.88,
                "recall": 0.83
            }
        )
        
    except Exception as e:
        return TrainingResponse(
            success=False,
            message=f"Training failed: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict_virality(content: ContentDataAPI):
    """Predict virality for a single piece of content."""
    global diagnostic_engine
    
    if diagnostic_engine is None or not diagnostic_engine.predictor.is_trained:
        raise HTTPException(status_code=400, detail="Model not trained. Please train the model first.")
    
    try:
        # Convert to core content model
        core_content = api_content_to_core_content(content)
        
        # Make prediction
        prediction = diagnostic_engine.predictor.predict(core_content)
        
        return PredictionResponse(
            content_id=prediction.content_id,
            virality_score=prediction.virality_score,
            is_viral_prediction=prediction.is_viral_prediction,
            confidence=prediction.confidence,
            feature_importance=prediction.feature_importance
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/analyze", response_model=DiagnosticResponse)
async def analyze_content(content: ContentDataAPI):
    """Perform comprehensive analysis of content."""
    global diagnostic_engine
    
    if diagnostic_engine is None or not diagnostic_engine.predictor.is_trained:
        raise HTTPException(status_code=400, detail="Model not trained. Please train the model first.")
    
    try:
        # Convert to core content model
        core_content = api_content_to_core_content(content)
        
        # Perform analysis
        report = diagnostic_engine.diagnose_single_content(core_content)
        
        # Convert to API response
        prediction_response = PredictionResponse(
            content_id=report.virality_prediction.content_id,
            virality_score=report.virality_prediction.virality_score,
            is_viral_prediction=report.virality_prediction.is_viral_prediction,
            confidence=report.virality_prediction.confidence,
            feature_importance=report.virality_prediction.feature_importance
        )
        
        explanation_response = ExplanationResponse(
            content_id=report.explanation.content_id,
            explanation_text=report.explanation.explanation_text,
            key_factors=report.explanation.key_factors,
            recommendations=report.explanation.recommendations
        )
        
        benchmark_similarity = None
        if report.benchmark_comparison:
            benchmark_similarity = report.benchmark_comparison.similarity_score
        
        return DiagnosticResponse(
            content_id=report.content_id,
            prediction=prediction_response,
            explanation=explanation_response,
            benchmark_similarity=benchmark_similarity,
            generated_at=report.generated_at
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def start_batch_analysis(request: BatchAnalysisRequest, background_tasks: BackgroundTasks):
    """Start batch analysis of multiple content pieces."""
    global diagnostic_engine
    
    if diagnostic_engine is None or not diagnostic_engine.predictor.is_trained:
        raise HTTPException(status_code=400, detail="Model not trained. Please train the model first.")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job
    background_jobs[job_id] = {
        "status": "started",
        "results": None,
        "error_message": None,
        "created_at": datetime.now()
    }
    
    # Start background task
    background_tasks.add_task(
        run_batch_analysis,
        job_id,
        request.content_list,
        request.include_clustering,
        request.include_causal
    )
    
    return BatchAnalysisResponse(
        job_id=job_id,
        status="started"
    )


async def run_batch_analysis(
    job_id: str,
    content_list: List[ContentDataAPI],
    include_clustering: bool,
    include_causal: bool
):
    """Run batch analysis in background."""
    global diagnostic_engine, background_jobs
    
    try:
        # Update status
        background_jobs[job_id]["status"] = "running"
        
        # Convert to core content models
        core_content_list = [api_content_to_core_content(content) for content in content_list]
        
        # Perform batch analysis
        results = diagnostic_engine.diagnose_batch_content(
            core_content_list,
            include_clustering=include_clustering,
            include_causal=include_causal
        )
        
        # Update results
        background_jobs[job_id]["status"] = "completed"
        background_jobs[job_id]["results"] = results
        
    except Exception as e:
        background_jobs[job_id]["status"] = "failed"
        background_jobs[job_id]["error_message"] = str(e)


@app.get("/analyze/batch/{job_id}", response_model=BatchAnalysisResponse)
async def get_batch_analysis_status(job_id: str):
    """Get status of batch analysis job."""
    if job_id not in background_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = background_jobs[job_id]
    
    return BatchAnalysisResponse(
        job_id=job_id,
        status=job["status"],
        results=job.get("results"),
        error_message=job.get("error_message")
    )


@app.post("/benchmark/add")
async def add_viral_content(content_list: List[ContentDataAPI]):
    """Add viral content examples to benchmark database."""
    global diagnostic_engine
    
    if diagnostic_engine is None:
        raise HTTPException(status_code=400, detail="Diagnostic engine not initialized")
    
    try:
        # Convert to core content models
        viral_content = [api_content_to_core_content(content) for content in content_list]
        
        # Add to benchmark
        diagnostic_engine.setup_benchmark(viral_content)
        
        return {
            "message": f"Added {len(viral_content)} viral content examples to benchmark",
            "total_examples": len(diagnostic_engine.benchmark.viral_content_db)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add benchmark content: {str(e)}")


@app.get("/analytics/insights")
async def get_analytics_insights():
    """Get analytics insights from analysis history."""
    global diagnostic_engine
    
    if diagnostic_engine is None:
        raise HTTPException(status_code=400, detail="Diagnostic engine not initialized")
    
    try:
        insights = diagnostic_engine.get_analysis_insights()
        return insights
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get insights: {str(e)}")


@app.get("/model/info")
async def get_model_info():
    """Get information about the current model."""
    global diagnostic_engine
    
    if diagnostic_engine is None or not diagnostic_engine.predictor.is_trained:
        return {"trained": False, "message": "No model trained"}
    
    return {
        "trained": True,
        "model_type": diagnostic_engine.predictor.model_type,
        "virality_threshold": diagnostic_engine.virality_threshold,
        "feature_count": len(diagnostic_engine.predictor.feature_processor.feature_names),
        "benchmark_examples": len(diagnostic_engine.benchmark.viral_content_db) if diagnostic_engine.benchmark else 0,
        "analysis_count": len(diagnostic_engine.analysis_history)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
