"""Automated diagnostics and report generation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from ..core.models import (
    ContentData, DiagnosticReport, PredictionResult, 
    ExplanationResult, BenchmarkResult, CausalAnalysisResult
)
from ..models import ViralityPredictor, ModelExplainer, CausalAnalyzer
from .benchmarking import ViralContentBenchmark
from .clustering import ContentClusterAnalyzer


class DiagnosticEngine:
    """Main engine for automated content virality diagnostics."""
    
    def __init__(self, virality_threshold: float = 2.0):
        """Initialize the diagnostic engine."""
        self.virality_threshold = virality_threshold
        
        # Initialize components
        self.predictor = ViralityPredictor(virality_threshold=virality_threshold)
        self.explainer = None  # Will be initialized after predictor is trained
        self.causal_analyzer = CausalAnalyzer(virality_threshold=virality_threshold)
        self.benchmark = ViralContentBenchmark()
        self.cluster_analyzer = ContentClusterAnalyzer()
        
        # Track analysis history
        self.analysis_history = []
        
    def setup_predictor(self, training_data: List[ContentData]) -> None:
        """Train the predictor and setup explainer."""
        print("Training virality predictor...")
        training_result = self.predictor.train(training_data)
        print(f"Training completed with F1 score: {training_result.performance.f1_score:.3f}")
        
        # Setup explainer
        self.explainer = ModelExplainer(self.predictor)
        
        # Prepare background data for SHAP
        feature_vectors = []
        for content_data in training_data[:100]:  # Use subset for efficiency
            fv = self.predictor.feature_processor.extract_features(content_data)
            feature_vectors.append(fv)
        
        background_data = self.predictor.feature_processor.transform_features(feature_vectors)
        self.explainer.setup_shap_explainer(background_data)
        self.explainer.setup_lime_explainer(background_data)
    
    def setup_benchmark(self, viral_content_data: List[ContentData]) -> None:
        """Setup the benchmark database with viral content examples."""
        print("Setting up viral content benchmark...")
        self.benchmark.add_viral_content(viral_content_data)
        print(f"Added {len(self.benchmark.viral_content_db)} viral examples to benchmark")
    
    def diagnose_single_content(
        self, 
        content_data: ContentData,
        include_causal: bool = False,
        include_benchmark: bool = True
    ) -> DiagnosticReport:
        """Generate comprehensive diagnostic report for a single piece of content."""
        
        if not self.predictor.is_trained:
            raise ValueError("Predictor must be trained before running diagnostics")
        
        # 1. Prediction
        prediction_result = self.predictor.predict(content_data)
        
        # 2. Explanation
        explanation_result = None
        if self.explainer:
            explanation_result = self.explainer.explain_prediction(content_data)
        else:
            # Fallback explanation
            explanation_result = ExplanationResult(
                content_id=content_data.content_id,
                explanation_text=f"Content predicted as {'VIRAL' if prediction_result.is_viral_prediction else 'NON-VIRAL'} with {prediction_result.virality_score:.1%} confidence.",
                key_factors=list(prediction_result.feature_importance.keys())[:5],
                recommendations=["Train explainer for detailed recommendations"]
            )
        
        # 3. Benchmark comparison
        benchmark_result = None
        if include_benchmark and self.benchmark.viral_content_db:
            benchmark_result = self.benchmark.benchmark_content(content_data)
        
        # 4. Causal analysis (optional)
        causal_result = None
        if include_causal:
            # Note: Single content causal analysis is limited
            # In practice, you'd need a dataset for meaningful causal inference
            causal_result = CausalAnalysisResult(
                content_id=content_data.content_id,
                causal_effects=[],
                treatment_effects={},
                confounders=[]
            )
        
        # Create diagnostic report
        report = DiagnosticReport(
            content_id=content_data.content_id,
            virality_prediction=prediction_result,
            explanation=explanation_result,
            causal_analysis=causal_result,
            benchmark_comparison=benchmark_result
        )
        
        # Store in history
        self.analysis_history.append(report)
        
        return report
    
    def diagnose_batch_content(
        self, 
        content_data_list: List[ContentData],
        include_clustering: bool = True,
        include_causal: bool = True
    ) -> Dict[str, Any]:
        """Diagnose multiple pieces of content and find patterns."""
        
        print(f"Analyzing {len(content_data_list)} pieces of content...")
        
        # Individual diagnostics
        individual_reports = []
        for content_data in content_data_list:
            report = self.diagnose_single_content(
                content_data, 
                include_causal=False,  # Will do batch causal analysis
                include_benchmark=True
            )
            individual_reports.append(report)
        
        # Batch analyses
        batch_results = {
            'individual_reports': individual_reports,
            'summary_stats': self._calculate_summary_stats(individual_reports)
        }
        
        # Clustering analysis
        if include_clustering:
            print("Performing clustering analysis...")
            cluster_results = self.cluster_analyzer.cluster_content(content_data_list)
            behavior_results = self.cluster_analyzer.cluster_audience_behavior(content_data_list)
            
            batch_results['clustering'] = {
                'content_clusters': cluster_results,
                'behavior_clusters': behavior_results,
                'insights': self.cluster_analyzer.get_cluster_insights()
            }
        
        # Causal analysis
        if include_causal:
            print("Performing causal analysis...")
            causal_results = self.causal_analyzer.analyze_feature_causality(content_data_list)
            batch_results['causal_analysis'] = causal_results
        
        # Pattern analysis
        batch_results['patterns'] = self._analyze_content_patterns(content_data_list, individual_reports)
        
        # Overall recommendations
        batch_results['recommendations'] = self._generate_batch_recommendations(batch_results)
        
        return batch_results
    
    def _calculate_summary_stats(self, reports: List[DiagnosticReport]) -> Dict[str, Any]:
        """Calculate summary statistics from diagnostic reports."""
        if not reports:
            return {}
        
        virality_scores = [r.virality_prediction.virality_score for r in reports]
        viral_predictions = [r.virality_prediction.is_viral_prediction for r in reports]
        
        # Benchmark similarities
        benchmark_scores = []
        for report in reports:
            if report.benchmark_comparison:
                benchmark_scores.append(report.benchmark_comparison.similarity_score)
        
        summary = {
            'total_content': len(reports),
            'predicted_viral_count': sum(viral_predictions),
            'predicted_viral_rate': sum(viral_predictions) / len(viral_predictions),
            'avg_virality_score': np.mean(virality_scores),
            'median_virality_score': np.median(virality_scores),
            'virality_score_std': np.std(virality_scores),
            'high_potential_count': sum(1 for score in virality_scores if score > 0.7),
            'low_potential_count': sum(1 for score in virality_scores if score < 0.3)
        }
        
        if benchmark_scores:
            summary['avg_benchmark_similarity'] = np.mean(benchmark_scores)
            summary['high_similarity_count'] = sum(1 for score in benchmark_scores if score > 0.8)
        
        return summary
    
    def _analyze_content_patterns(
        self, 
        content_data_list: List[ContentData], 
        reports: List[DiagnosticReport]
    ) -> Dict[str, Any]:
        """Analyze patterns across content."""
        patterns = {
            'platform_performance': {},
            'content_type_performance': {},
            'common_success_factors': {},
            'common_failure_factors': {}
        }
        
        # Platform performance
        platform_stats = {}
        for content_data, report in zip(content_data_list, reports):
            platform = content_data.platform.value
            if platform not in platform_stats:
                platform_stats[platform] = {'count': 0, 'viral_count': 0, 'scores': []}
            
            platform_stats[platform]['count'] += 1
            platform_stats[platform]['scores'].append(report.virality_prediction.virality_score)
            if report.virality_prediction.is_viral_prediction:
                platform_stats[platform]['viral_count'] += 1
        
        for platform, stats in platform_stats.items():
            patterns['platform_performance'][platform] = {
                'viral_rate': stats['viral_count'] / stats['count'],
                'avg_score': np.mean(stats['scores']),
                'count': stats['count']
            }
        
        # Content type performance
        type_stats = {}
        for content_data, report in zip(content_data_list, reports):
            content_type = content_data.content_type.value
            if content_type not in type_stats:
                type_stats[content_type] = {'count': 0, 'viral_count': 0, 'scores': []}
            
            type_stats[content_type]['count'] += 1
            type_stats[content_type]['scores'].append(report.virality_prediction.virality_score)
            if report.virality_prediction.is_viral_prediction:
                type_stats[content_type]['viral_count'] += 1
        
        for content_type, stats in type_stats.items():
            patterns['content_type_performance'][content_type] = {
                'viral_rate': stats['viral_count'] / stats['count'],
                'avg_score': np.mean(stats['scores']),
                'count': stats['count']
            }
        
        # Common factors
        all_factors = {}
        viral_factors = {}
        non_viral_factors = {}
        
        for report in reports:
            is_viral = report.virality_prediction.is_viral_prediction
            feature_importance = report.virality_prediction.feature_importance
            
            for factor, importance in feature_importance.items():
                if factor not in all_factors:
                    all_factors[factor] = []
                all_factors[factor].append(importance)
                
                if is_viral:
                    if factor not in viral_factors:
                        viral_factors[factor] = []
                    viral_factors[factor].append(importance)
                else:
                    if factor not in non_viral_factors:
                        non_viral_factors[factor] = []
                    non_viral_factors[factor].append(importance)
        
        # Top factors for viral content
        if viral_factors:
            viral_factor_means = {k: np.mean(v) for k, v in viral_factors.items()}
            patterns['common_success_factors'] = dict(
                sorted(viral_factor_means.items(), key=lambda x: x[1], reverse=True)[:5]
            )
        
        # Top factors for non-viral content
        if non_viral_factors:
            non_viral_factor_means = {k: np.mean(v) for k, v in non_viral_factors.items()}
            patterns['common_failure_factors'] = dict(
                sorted(non_viral_factor_means.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            )
        
        return patterns
    
    def _generate_batch_recommendations(self, batch_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on batch analysis."""
        recommendations = []
        
        summary = batch_results.get('summary_stats', {})
        patterns = batch_results.get('patterns', {})
        
        # Overall performance recommendations
        viral_rate = summary.get('predicted_viral_rate', 0)
        if viral_rate < 0.1:
            recommendations.append("Overall viral potential is low. Focus on improving content quality and engagement factors.")
        elif viral_rate > 0.5:
            recommendations.append("Strong viral potential detected. Maintain current content strategy while optimizing timing and distribution.")
        
        # Platform recommendations
        platform_perf = patterns.get('platform_performance', {})
        if platform_perf:
            best_platform = max(platform_perf.items(), key=lambda x: x[1]['viral_rate'])
            worst_platform = min(platform_perf.items(), key=lambda x: x[1]['viral_rate'])
            
            recommendations.append(f"Best performing platform: {best_platform[0]} ({best_platform[1]['viral_rate']:.1%} viral rate)")
            if worst_platform[1]['viral_rate'] < 0.1:
                recommendations.append(f"Consider strategy adjustment for {worst_platform[0]} (only {worst_platform[1]['viral_rate']:.1%} viral rate)")
        
        # Factor-based recommendations
        success_factors = patterns.get('common_success_factors', {})
        if success_factors:
            top_factor = list(success_factors.keys())[0]
            recommendations.append(f"Key success factor identified: {top_factor}. Focus on optimizing this across all content.")
        
        # Clustering recommendations
        clustering = batch_results.get('clustering', {})
        if clustering and 'behavior_clusters' in clustering:
            behavior_recs = clustering['behavior_clusters'].get('recommendations', [])
            recommendations.extend(behavior_recs[:2])  # Add top 2 behavior recommendations
        
        return recommendations[:8]  # Limit to top 8 recommendations
    
    def generate_report_summary(self, report: DiagnosticReport) -> str:
        """Generate a human-readable summary of a diagnostic report."""
        summary_parts = []
        
        # Main prediction
        pred = report.virality_prediction
        summary_parts.append(
            f"Content {report.content_id} is predicted to be "
            f"{'VIRAL' if pred.is_viral_prediction else 'NON-VIRAL'} "
            f"with {pred.virality_score:.1%} probability."
        )
        
        # Key explanation
        if report.explanation:
            summary_parts.append(f"Key insight: {report.explanation.explanation_text}")
            
            if report.explanation.recommendations:
                top_rec = report.explanation.recommendations[0]
                summary_parts.append(f"Top recommendation: {top_rec}")
        
        # Benchmark comparison
        if report.benchmark_comparison:
            similarity = report.benchmark_comparison.similarity_score
            summary_parts.append(
                f"Similarity to viral content: {similarity:.1%}"
            )
            
            if similarity < 0.5:
                summary_parts.append("Content differs significantly from successful viral examples.")
        
        return " ".join(summary_parts)
    
    def save_report(self, report: DiagnosticReport, filepath: str) -> None:
        """Save diagnostic report to file."""
        # Convert to JSON-serializable format
        report_dict = {
            'content_id': report.content_id,
            'generated_at': report.generated_at.isoformat(),
            'virality_prediction': {
                'virality_score': report.virality_prediction.virality_score,
                'is_viral_prediction': report.virality_prediction.is_viral_prediction,
                'confidence': report.virality_prediction.confidence,
                'feature_importance': report.virality_prediction.feature_importance
            },
            'explanation': {
                'explanation_text': report.explanation.explanation_text,
                'key_factors': report.explanation.key_factors,
                'recommendations': report.explanation.recommendations
            } if report.explanation else None,
            'benchmark_comparison': {
                'similarity_score': report.benchmark_comparison.similarity_score,
                'top_similar_content': report.benchmark_comparison.top_similar_viral_content,
                'improvement_suggestions': report.benchmark_comparison.improvement_suggestions
            } if report.benchmark_comparison else None,
            'summary': self.generate_report_summary(report)
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)
    
    def load_report(self, filepath: str) -> Dict[str, Any]:
        """Load diagnostic report from file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def get_analysis_insights(self) -> Dict[str, Any]:
        """Get insights from analysis history."""
        if not self.analysis_history:
            return {}
        
        insights = {
            'total_analyses': len(self.analysis_history),
            'avg_virality_score': np.mean([r.virality_prediction.virality_score for r in self.analysis_history]),
            'viral_prediction_rate': np.mean([r.virality_prediction.is_viral_prediction for r in self.analysis_history]),
            'most_common_factors': self._get_most_common_factors(),
            'improvement_trends': self._analyze_improvement_trends()
        }
        
        return insights
    
    def _get_most_common_factors(self) -> Dict[str, float]:
        """Get most commonly important factors across all analyses."""
        all_factors = {}
        
        for report in self.analysis_history:
            for factor, importance in report.virality_prediction.feature_importance.items():
                if factor not in all_factors:
                    all_factors[factor] = []
                all_factors[factor].append(abs(importance))
        
        # Calculate average importance
        factor_averages = {k: np.mean(v) for k, v in all_factors.items()}
        
        # Return top 10 factors
        return dict(sorted(factor_averages.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def _analyze_improvement_trends(self) -> Dict[str, Any]:
        """Analyze trends in virality scores over time."""
        if len(self.analysis_history) < 5:
            return {}
        
        scores = [r.virality_prediction.virality_score for r in self.analysis_history]
        
        # Simple trend analysis
        first_half = scores[:len(scores)//2]
        second_half = scores[len(scores)//2:]
        
        trend = {
            'first_half_avg': np.mean(first_half),
            'second_half_avg': np.mean(second_half),
            'improvement': np.mean(second_half) - np.mean(first_half),
            'trend_direction': 'improving' if np.mean(second_half) > np.mean(first_half) else 'declining'
        }
        
        return trend
