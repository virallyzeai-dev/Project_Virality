"""Benchmarking against viral content using similarity search."""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# FAISS for similarity search
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print(
        "Warning: FAISS not available. Similarity search will use basic implementation."
    )

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from ..core.models import BenchmarkResult, ContentData
from ..models.predictive import FeatureProcessor


class ViralContentBenchmark:
    """Benchmark content against known viral examples."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the benchmark system."""
        self.embedding_model_name = embedding_model
        self.sentence_transformer = SentenceTransformer(embedding_model)
        self.feature_processor = FeatureProcessor()

        # Storage for viral content data
        self.viral_content_db = []
        self.viral_embeddings = None
        self.viral_features = None
        self.feature_scaler = StandardScaler()

        # FAISS index for fast similarity search
        self.faiss_index = None
        self.content_id_mapping = {}

    def add_viral_content(self, content_data_list: List[ContentData]) -> None:
        """Add viral content examples to the benchmark database."""
        for content_data in content_data_list:
            # Only add if it's actually viral
            virality_coeff = self.feature_processor.engagement_extractor.metrics_calculator.calculate_virality_coefficient(
                content_data.metrics.shares,
                content_data.metrics.unique_users_reached
                or content_data.metrics.impressions,
            )

            if virality_coeff >= 2.0:  # Viral threshold
                self.viral_content_db.append(content_data)

        # Rebuild embeddings and features
        self._build_embeddings_and_features()

    def _build_embeddings_and_features(self) -> None:
        """Build embeddings and feature vectors for viral content."""
        if not self.viral_content_db:
            return

        # Extract text embeddings
        embeddings = []
        feature_vectors = []

        for i, content_data in enumerate(self.viral_content_db):
            # Text embeddings
            text = content_data.text or ""
            if len(text.strip()) == 0:
                text = f"Content from {content_data.platform.value}"  # Fallback

            embedding = self.sentence_transformer.encode(text)
            embeddings.append(embedding)

            # Feature vectors
            feature_vector = self.feature_processor.extract_features(content_data)
            combined_features = {}
            combined_features.update(feature_vector.text_features)
            combined_features.update(feature_vector.visual_features)
            combined_features.update(feature_vector.engagement_features)
            combined_features.update(feature_vector.platform_features)

            feature_vectors.append(combined_features)
            self.content_id_mapping[i] = content_data.content_id

        # Convert to numpy arrays
        self.viral_embeddings = np.array(embeddings)

        # Convert features to array format
        if feature_vectors:
            # Get all feature names
            all_features = set()
            for fv in feature_vectors:
                all_features.update(fv.keys())

            feature_names = sorted(all_features)
            feature_matrix = []

            for fv in feature_vectors:
                feature_row = [fv.get(name, 0.0) for name in feature_names]
                feature_matrix.append(feature_row)

            self.viral_features = np.array(feature_matrix)
            self.feature_names = feature_names

            # Fit scaler
            self.feature_scaler.fit(self.viral_features)

        # Build FAISS index if available
        self._build_faiss_index()

    def _build_faiss_index(self) -> None:
        """Build FAISS index for fast similarity search."""
        if not FAISS_AVAILABLE or self.viral_embeddings is None:
            return

        dimension = self.viral_embeddings.shape[1]

        # Create FAISS index (L2 distance)
        self.faiss_index = faiss.IndexFlatL2(dimension)

        # Normalize embeddings for cosine similarity
        normalized_embeddings = self.viral_embeddings / np.linalg.norm(
            self.viral_embeddings, axis=1, keepdims=True
        )

        # Add to index
        self.faiss_index.add(normalized_embeddings.astype("float32"))

    def find_similar_viral_content(
        self,
        content_data: ContentData,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ) -> List[Tuple[str, float]]:
        """Find similar viral content using embedding similarity."""
        if not self.viral_content_db:
            return []

        # Get embedding for input content
        text = content_data.text or f"Content from {content_data.platform.value}"
        query_embedding = self.sentence_transformer.encode([text])

        if FAISS_AVAILABLE and self.faiss_index is not None:
            # Use FAISS for fast search
            normalized_query = query_embedding / np.linalg.norm(query_embedding)

            # Search
            distances, indices = self.faiss_index.search(
                normalized_query.astype("float32"),
                min(top_k, len(self.viral_content_db)),
            )

            # Convert distances to similarities (cosine similarity)
            similarities = (
                1 - distances[0]
            )  # Since we're using L2 on normalized vectors

            # Filter by threshold and return results
            results = []
            for idx, similarity in zip(indices[0], similarities):
                if similarity >= similarity_threshold:
                    content_id = self.content_id_mapping[idx]
                    results.append((content_id, float(similarity)))

            return results
        else:
            # Fallback to sklearn cosine similarity
            similarities = cosine_similarity(query_embedding, self.viral_embeddings)[0]

            # Get top similar content
            similar_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in similar_indices:
                similarity = similarities[idx]
                if similarity >= similarity_threshold:
                    content_id = self.viral_content_db[idx].content_id
                    results.append((content_id, float(similarity)))

            return results

    def compare_features(
        self, content_data: ContentData, viral_content_id: str
    ) -> Dict[str, float]:
        """Compare features between content and a specific viral example."""
        # Find the viral content
        viral_content = None
        for vc in self.viral_content_db:
            if vc.content_id == viral_content_id:
                viral_content = vc
                break

        if viral_content is None:
            return {}

        # Extract features for both contents
        query_features = self.feature_processor.extract_features(content_data)
        viral_features = self.feature_processor.extract_features(viral_content)

        # Combine features
        query_combined = {}
        query_combined.update(query_features.text_features)
        query_combined.update(query_features.visual_features)
        query_combined.update(query_features.engagement_features)
        query_combined.update(query_features.platform_features)

        viral_combined = {}
        viral_combined.update(viral_features.text_features)
        viral_combined.update(viral_features.visual_features)
        viral_combined.update(viral_features.engagement_features)
        viral_combined.update(viral_features.platform_features)

        # Calculate feature gaps
        feature_gaps = {}
        all_features = set(query_combined.keys()) | set(viral_combined.keys())

        for feature in all_features:
            query_val = query_combined.get(feature, 0.0)
            viral_val = viral_combined.get(feature, 0.0)

            # Calculate relative gap (negative means query is below viral)
            if viral_val != 0:
                gap = (query_val - viral_val) / abs(viral_val)
            else:
                gap = query_val

            feature_gaps[feature] = gap

        return feature_gaps

    def analyze_viral_patterns(self, category: Optional[str] = None) -> Dict[str, Any]:
        """Analyze patterns in viral content."""
        if not self.viral_content_db:
            return {}

        # Filter by category if specified
        content_to_analyze = self.viral_content_db
        if category:
            content_to_analyze = [
                vc
                for vc in self.viral_content_db
                if vc.platform.value == category or vc.content_type.value == category
            ]

        if not content_to_analyze:
            return {}

        # Extract features for all viral content
        all_features = []
        for content in content_to_analyze:
            feature_vector = self.feature_processor.extract_features(content)
            combined_features = {}
            combined_features.update(feature_vector.text_features)
            combined_features.update(feature_vector.visual_features)
            combined_features.update(feature_vector.engagement_features)
            combined_features.update(feature_vector.platform_features)
            all_features.append(combined_features)

        # Convert to DataFrame for analysis
        df = pd.DataFrame(all_features)

        # Calculate statistics
        patterns = {
            "mean_values": df.mean().to_dict(),
            "median_values": df.median().to_dict(),
            "std_values": df.std().to_dict(),
            "top_features": df.mean().nlargest(10).to_dict(),
            "most_variable_features": df.std().nlargest(10).to_dict(),
            "sample_size": len(content_to_analyze),
        }

        # Platform distribution
        platform_dist = {}
        for content in content_to_analyze:
            platform = content.platform.value
            platform_dist[platform] = platform_dist.get(platform, 0) + 1

        patterns["platform_distribution"] = platform_dist

        # Content type distribution
        type_dist = {}
        for content in content_to_analyze:
            content_type = content.content_type.value
            type_dist[content_type] = type_dist.get(content_type, 0) + 1

        patterns["content_type_distribution"] = type_dist

        return patterns

    def benchmark_content(
        self, content_data: ContentData, category: Optional[str] = None
    ) -> BenchmarkResult:
        """Comprehensive benchmarking of content against viral examples."""

        # Find similar viral content
        similar_content = self.find_similar_viral_content(content_data, top_k=5)

        if not similar_content:
            return BenchmarkResult(
                content_id=content_data.content_id,
                similarity_score=0.0,
                top_similar_viral_content=[],
                feature_gaps={},
                improvement_suggestions=[],
            )

        # Get the most similar content for detailed comparison
        most_similar_id, max_similarity = similar_content[0]
        feature_gaps = self.compare_features(content_data, most_similar_id)

        # Analyze patterns to get benchmarks
        viral_patterns = self.analyze_viral_patterns(category)

        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(
            feature_gaps, viral_patterns
        )

        return BenchmarkResult(
            content_id=content_data.content_id,
            similarity_score=max_similarity,
            top_similar_viral_content=[content_id for content_id, _ in similar_content],
            feature_gaps=feature_gaps,
            improvement_suggestions=improvement_suggestions,
        )

    def _generate_improvement_suggestions(
        self, feature_gaps: Dict[str, float], viral_patterns: Dict[str, Any]
    ) -> List[str]:
        """Generate improvement suggestions based on feature gaps."""
        suggestions = []

        # Identify features where content is significantly below viral benchmarks
        negative_gaps = {k: v for k, v in feature_gaps.items() if v < -0.2}  # 20% below

        # Sort by largest gaps
        sorted_gaps = sorted(negative_gaps.items(), key=lambda x: x[1])

        for feature, gap in sorted_gaps[:5]:  # Top 5 improvement areas
            suggestion = self._get_feature_suggestion(feature, gap)
            if suggestion:
                suggestions.append(suggestion)

        return suggestions

    def _get_feature_suggestion(self, feature: str, gap: float) -> Optional[str]:
        """Get specific suggestion for a feature gap."""
        suggestions_map = {
            "sentiment_polarity": f"Increase positive sentiment (currently {gap:.1%} below viral content)",
            "emotional_intensity": f"Add more emotional intensity (currently {gap:.1%} below viral content)",
            "readability_score": f"Improve readability (currently {gap:.1%} below viral content)",
            "hashtag_count": f"Optimize hashtag usage (currently {gap:.1%} different from viral content)",
            "face_count": f"Consider adding faces to visuals (currently {gap:.1%} below viral content)",
            "brightness": f"Improve image brightness (currently {gap:.1%} below viral content)",
            "aesthetic_score": f"Enhance visual appeal (currently {gap:.1%} below viral content)",
            "engagement_rate": f"Focus on driving engagement (currently {gap:.1%} below viral content)",
        }

        return suggestions_map.get(feature)

    def save_benchmark_db(self, filepath: str) -> None:
        """Save the viral content database."""
        data = {
            "viral_content_db": self.viral_content_db,
            "viral_embeddings": self.viral_embeddings,
            "viral_features": self.viral_features,
            "feature_names": getattr(self, "feature_names", []),
            "content_id_mapping": self.content_id_mapping,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load_benchmark_db(self, filepath: str) -> None:
        """Load the viral content database."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.viral_content_db = data["viral_content_db"]
        self.viral_embeddings = data["viral_embeddings"]
        self.viral_features = data["viral_features"]
        self.feature_names = data.get("feature_names", [])
        self.content_id_mapping = data["content_id_mapping"]

        # Rebuild FAISS index
        self._build_faiss_index()
