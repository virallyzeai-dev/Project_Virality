"""Clustering analysis for audience behavior and content patterns."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

from ..core.models import ContentData
from ..features import FeatureProcessor


class ContentClusterAnalyzer:
    """Analyze content and audience patterns using clustering."""
    
    def __init__(self):
        """Initialize the cluster analyzer."""
        self.feature_processor = FeatureProcessor()
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.cluster_models = {}
        self.cluster_labels = {}
        self.feature_data = None
        
    def prepare_clustering_data(self, content_data_list: List[ContentData]) -> pd.DataFrame:
        """Prepare data for clustering analysis."""
        feature_rows = []
        
        for content_data in content_data_list:
            # Extract features
            feature_vector = self.feature_processor.extract_features(content_data)
            
            # Combine all features
            row = {'content_id': content_data.content_id}
            row.update(feature_vector.text_features)
            row.update(feature_vector.visual_features)
            row.update(feature_vector.engagement_features)
            row.update(feature_vector.platform_features)
            
            # Add metadata
            row['platform'] = content_data.platform.value
            row['content_type'] = content_data.content_type.value
            
            # Add virality metrics
            virality_coeff = self.feature_processor.engagement_extractor.metrics_calculator.calculate_virality_coefficient(
                content_data.metrics.shares,
                content_data.metrics.unique_users_reached or content_data.metrics.impressions
            )
            row['virality_coefficient'] = virality_coeff
            row['is_viral'] = 1 if virality_coeff >= 2.0 else 0
            
            feature_rows.append(row)
        
        return pd.DataFrame(feature_rows)
    
    def find_optimal_clusters(
        self, 
        data: pd.DataFrame,
        max_clusters: int = 10,
        method: str = "kmeans"
    ) -> int:
        """Find optimal number of clusters using elbow method and silhouette score."""
        # Prepare numerical data
        numerical_data = data.select_dtypes(include=[np.number])
        feature_cols = [col for col in numerical_data.columns 
                       if col not in ['content_id', 'is_viral', 'virality_coefficient']]
        
        X = numerical_data[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Reduce dimensions if needed
        if X_scaled.shape[1] > 50:
            X_scaled = self.pca.fit_transform(X_scaled)
        
        # Calculate metrics for different cluster numbers
        inertias = []
        silhouette_scores = []
        
        k_range = range(2, min(max_clusters + 1, len(data) // 2))
        
        for k in k_range:
            if method == "kmeans":
                clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
            else:
                clusterer = AgglomerativeClustering(n_clusters=k)
            
            cluster_labels = clusterer.fit_predict(X_scaled)
            
            if method == "kmeans":
                inertias.append(clusterer.inertia_)
            
            # Calculate silhouette score
            sil_score = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(sil_score)
        
        # Find optimal number using silhouette score
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        return optimal_k
    
    def cluster_content(
        self, 
        content_data_list: List[ContentData],
        n_clusters: Optional[int] = None,
        method: str = "kmeans"
    ) -> Dict[str, Any]:
        """Cluster content based on features."""
        # Prepare data
        data = self.prepare_clustering_data(content_data_list)
        self.feature_data = data
        
        # Prepare numerical features for clustering
        numerical_data = data.select_dtypes(include=[np.number])
        feature_cols = [col for col in numerical_data.columns 
                       if col not in ['content_id', 'is_viral', 'virality_coefficient']]
        
        X = numerical_data[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Reduce dimensions if needed
        if X_scaled.shape[1] > 50:
            X_scaled = self.pca.fit_transform(X_scaled)
        
        # Find optimal clusters if not specified
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(data, method=method)
        
        # Perform clustering
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == "dbscan":
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        else:
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        
        cluster_labels = clusterer.fit_predict(X_scaled)
        
        # Store results
        self.cluster_models[method] = clusterer
        self.cluster_labels[method] = cluster_labels
        
        # Add cluster labels to data
        data[f'cluster_{method}'] = cluster_labels
        
        # Analyze clusters
        cluster_analysis = self._analyze_clusters(data, cluster_labels, method)
        
        return {
            'n_clusters': len(np.unique(cluster_labels[cluster_labels >= 0])),  # Exclude noise for DBSCAN
            'cluster_labels': cluster_labels.tolist(),
            'cluster_analysis': cluster_analysis,
            'silhouette_score': silhouette_score(X_scaled, cluster_labels) if len(np.unique(cluster_labels)) > 1 else 0
        }
    
    def _analyze_clusters(
        self, 
        data: pd.DataFrame, 
        cluster_labels: np.ndarray,
        method: str
    ) -> Dict[str, Any]:
        """Analyze characteristics of each cluster."""
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = cluster_labels
        
        cluster_analysis = {}
        unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])
        
        for cluster_id in unique_clusters:
            cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
            
            # Basic statistics
            cluster_info = {
                'size': len(cluster_data),
                'viral_rate': cluster_data['is_viral'].mean(),
                'avg_virality_coefficient': cluster_data['virality_coefficient'].mean(),
                'platform_distribution': cluster_data['platform'].value_counts().to_dict(),
                'content_type_distribution': cluster_data['content_type'].value_counts().to_dict()
            }
            
            # Feature characteristics
            numerical_cols = cluster_data.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numerical_cols 
                           if col not in ['content_id', 'is_viral', 'virality_coefficient', 'cluster']]
            
            # Compare cluster features to overall average
            overall_means = data[feature_cols].mean()
            cluster_means = cluster_data[feature_cols].mean()
            
            # Find distinctive features (features where cluster differs significantly from overall)
            distinctive_features = {}
            for feature in feature_cols:
                cluster_val = cluster_means[feature]
                overall_val = overall_means[feature]
                
                if overall_val != 0:
                    relative_diff = (cluster_val - overall_val) / abs(overall_val)
                    if abs(relative_diff) > 0.2:  # 20% difference threshold
                        distinctive_features[feature] = {
                            'cluster_value': cluster_val,
                            'overall_value': overall_val,
                            'relative_difference': relative_diff
                        }
            
            cluster_info['distinctive_features'] = distinctive_features
            cluster_info['top_distinctive_features'] = sorted(
                distinctive_features.items(), 
                key=lambda x: abs(x[1]['relative_difference']), 
                reverse=True
            )[:5]
            
            cluster_analysis[f'cluster_{cluster_id}'] = cluster_info
        
        return cluster_analysis
    
    def cluster_audience_behavior(
        self, 
        content_data_list: List[ContentData]
    ) -> Dict[str, Any]:
        """Cluster content based on audience behavior patterns."""
        engagement_data = []
        
        for content_data in content_data_list:
            metrics = content_data.metrics
            
            if metrics.impressions > 0:
                row = {
                    'content_id': content_data.content_id,
                    'like_rate': metrics.likes / metrics.impressions,
                    'share_rate': metrics.shares / metrics.impressions,
                    'comment_rate': metrics.comments / metrics.impressions,
                    'engagement_rate': (metrics.likes + metrics.shares + metrics.comments) / metrics.impressions,
                    'virality_coefficient': self.feature_processor.engagement_extractor.metrics_calculator.calculate_virality_coefficient(
                        metrics.shares, metrics.unique_users_reached or metrics.impressions
                    ),
                    'platform': content_data.platform.value,
                    'content_type': content_data.content_type.value
                }
                
                # Add completion rate for videos
                if metrics.total_views and metrics.completed_views:
                    row['completion_rate'] = metrics.completed_views / metrics.total_views
                else:
                    row['completion_rate'] = 0.0
                
                engagement_data.append(row)
        
        if not engagement_data:
            return {}
        
        engagement_df = pd.DataFrame(engagement_data)
        
        # Cluster based on engagement patterns
        feature_cols = ['like_rate', 'share_rate', 'comment_rate', 'completion_rate']
        X = engagement_df[feature_cols].fillna(0)
        X_scaled = StandardScaler().fit_transform(X)
        
        # Find optimal clusters
        optimal_k = min(5, len(engagement_data) // 3)  # Max 5 clusters
        if optimal_k < 2:
            optimal_k = 2
        
        # Perform clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        engagement_df['behavior_cluster'] = cluster_labels
        
        # Analyze behavior patterns
        behavior_patterns = {}
        for cluster_id in range(optimal_k):
            cluster_data = engagement_df[engagement_df['behavior_cluster'] == cluster_id]
            
            behavior_patterns[f'behavior_cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'avg_like_rate': cluster_data['like_rate'].mean(),
                'avg_share_rate': cluster_data['share_rate'].mean(),
                'avg_comment_rate': cluster_data['comment_rate'].mean(),
                'avg_completion_rate': cluster_data['completion_rate'].mean(),
                'avg_virality': cluster_data['virality_coefficient'].mean(),
                'description': self._describe_behavior_cluster(cluster_data)
            }
        
        return {
            'behavior_patterns': behavior_patterns,
            'cluster_labels': cluster_labels.tolist(),
            'recommendations': self._generate_behavior_recommendations(behavior_patterns)
        }
    
    def _describe_behavior_cluster(self, cluster_data: pd.DataFrame) -> str:
        """Generate description for behavior cluster."""
        like_rate = cluster_data['like_rate'].mean()
        share_rate = cluster_data['share_rate'].mean()
        comment_rate = cluster_data['comment_rate'].mean()
        virality = cluster_data['virality_coefficient'].mean()
        
        if virality >= 2.0:
            return "High viral potential - strong sharing behavior"
        elif share_rate > like_rate and share_rate > comment_rate:
            return "Share-focused audience - content gets amplified"
        elif comment_rate > like_rate and comment_rate > share_rate:
            return "Discussion-focused audience - generates conversations"
        elif like_rate > share_rate and like_rate > comment_rate:
            return "Appreciation-focused audience - shows support but limited sharing"
        else:
            return "Low engagement - passive audience"
    
    def _generate_behavior_recommendations(
        self, 
        behavior_patterns: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on behavior clusters."""
        recommendations = []
        
        for cluster_name, pattern in behavior_patterns.items():
            if "High viral potential" in pattern['description']:
                recommendations.append(f"Target {cluster_name}: Focus on shareable content with emotional appeal")
            elif "Share-focused" in pattern['description']:
                recommendations.append(f"Target {cluster_name}: Create content that encourages sharing")
            elif "Discussion-focused" in pattern['description']:
                recommendations.append(f"Target {cluster_name}: Ask questions and encourage comments")
            elif "Appreciation-focused" in pattern['description']:
                recommendations.append(f"Target {cluster_name}: Focus on quality over viral potential")
        
        return recommendations
    
    def visualize_clusters(
        self, 
        method: str = "kmeans",
        save_path: Optional[str] = None
    ) -> None:
        """Visualize clusters using t-SNE or PCA."""
        if method not in self.cluster_labels or self.feature_data is None:
            print("No clustering results available. Run cluster_content first.")
            return
        
        # Prepare data
        numerical_data = self.feature_data.select_dtypes(include=[np.number])
        feature_cols = [col for col in numerical_data.columns 
                       if col not in ['content_id', 'is_viral', 'virality_coefficient']]
        
        X = numerical_data[feature_cols].fillna(0)
        X_scaled = StandardScaler().fit_transform(X)
        
        # Reduce to 2D for visualization
        if X_scaled.shape[1] > 2:
            if X_scaled.shape[0] > 1000:
                # Use PCA for large datasets
                reducer = PCA(n_components=2)
            else:
                # Use t-SNE for smaller datasets
                reducer = TSNE(n_components=2, random_state=42)
            
            X_2d = reducer.fit_transform(X_scaled)
        else:
            X_2d = X_scaled
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        cluster_labels = self.cluster_labels[method]
        unique_labels = np.unique(cluster_labels)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            if label == -1:  # Noise points in DBSCAN
                color = 'black'
                marker = 'x'
                label_name = 'Noise'
            else:
                color = colors[i]
                marker = 'o'
                label_name = f'Cluster {label}'
            
            mask = cluster_labels == label
            plt.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=[color], marker=marker, 
                label=label_name, alpha=0.7, s=50
            )
        
        plt.title(f'Content Clusters ({method.upper()})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_cluster_insights(self, method: str = "kmeans") -> Dict[str, str]:
        """Get human-readable insights about clusters."""
        if method not in self.cluster_labels:
            return {}
        
        cluster_labels = self.cluster_labels[method]
        unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])
        
        insights = {}
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_data = self.feature_data[cluster_mask]
            
            # Analyze cluster characteristics
            viral_rate = cluster_data['is_viral'].mean()
            avg_virality = cluster_data['virality_coefficient'].mean()
            size = len(cluster_data)
            
            # Generate insight
            if viral_rate > 0.5:
                insight = f"High-performing cluster ({size} contents) with {viral_rate:.1%} viral rate"
            elif viral_rate > 0.2:
                insight = f"Moderate-performing cluster ({size} contents) with {viral_rate:.1%} viral rate"
            else:
                insight = f"Low-performing cluster ({size} contents) with {viral_rate:.1%} viral rate"
            
            # Add platform/type info
            top_platform = cluster_data['platform'].mode().iloc[0] if not cluster_data['platform'].mode().empty else "Mixed"
            top_type = cluster_data['content_type'].mode().iloc[0] if not cluster_data['content_type'].mode().empty else "Mixed"
            
            insight += f". Primarily {top_platform} {top_type} content."
            
            insights[f'cluster_{cluster_id}'] = insight
        
        return insights
