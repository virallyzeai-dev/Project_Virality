"""Text feature extraction for content analysis."""

import re
import numpy as np
from typing import Dict, List, Optional
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderAnalyzer
import textstat
from transformers import pipeline
from sentence_transformers import SentenceTransformer


class TextFeatureExtractor:
    """Extract features from text content."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the text feature extractor."""
        self.vader_analyzer = VaderAnalyzer()
        self.emotion_classifier = None
        self.sentence_transformer = None
        self.model_name = model_name
        
        # Download required NLTK data
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        # Initialize models lazily
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize heavy models lazily."""
        try:
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=-1  # Use CPU
            )
            self.sentence_transformer = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"Warning: Could not initialize some models: {e}")
    
    def extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """Extract sentiment-related features."""
        if not text:
            return {
                "sentiment_polarity": 0.0,
                "sentiment_subjectivity": 0.0,
                "sentiment_compound": 0.0,
                "sentiment_positive": 0.0,
                "sentiment_negative": 0.0,
                "sentiment_neutral": 0.0
            }
        
        # VADER sentiment analysis
        vader_scores = self.vader_analyzer.polarity_scores(text)
        
        return {
            "sentiment_polarity": vader_scores['compound'],
            "sentiment_subjectivity": abs(vader_scores['compound']),  # Approximation
            "sentiment_compound": vader_scores['compound'],
            "sentiment_positive": vader_scores['pos'],
            "sentiment_negative": vader_scores['neg'],
            "sentiment_neutral": vader_scores['neu']
        }
    
    def extract_emotional_intensity(self, text: str) -> Dict[str, float]:
        """Extract emotional intensity features."""
        if not text or not self.emotion_classifier:
            return {"emotional_intensity": 0.0}
        
        try:
            emotions = self.emotion_classifier(text)
            # Get the confidence score of the predicted emotion
            max_intensity = max([emotion['score'] for emotion in emotions])
            return {"emotional_intensity": max_intensity}
        except Exception:
            return {"emotional_intensity": 0.0}
    
    def extract_readability_features(self, text: str) -> Dict[str, float]:
        """Extract readability and complexity features."""
        if not text:
            return {
                "flesch_reading_ease": 0.0,
                "flesch_kincaid_grade": 0.0,
                "readability_score": 0.0
            }
        
        try:
            flesch_ease = textstat.flesch_reading_ease(text)
            flesch_grade = textstat.flesch_kincaid_grade(text)
            
            # Normalize readability score (higher = more readable)
            readability_score = max(0, min(100, flesch_ease)) / 100
            
            return {
                "flesch_reading_ease": flesch_ease,
                "flesch_kincaid_grade": flesch_grade,
                "readability_score": readability_score
            }
        except Exception:
            return {
                "flesch_reading_ease": 0.0,
                "flesch_kincaid_grade": 0.0,
                "readability_score": 0.0
            }
    
    def extract_structural_features(self, text: str) -> Dict[str, float]:
        """Extract structural features from text."""
        if not text:
            return {
                "text_length": 0.0,
                "word_count": 0.0,
                "sentence_count": 0.0,
                "avg_word_length": 0.0,
                "avg_sentence_length": 0.0
            }
        
        # Basic counts
        text_length = len(text)
        words = text.split()
        word_count = len(words)
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Averages
        avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        return {
            "text_length": float(text_length),
            "word_count": float(word_count),
            "sentence_count": float(sentence_count),
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length
        }
    
    def extract_social_features(self, text: str) -> Dict[str, float]:
        """Extract social media specific features."""
        if not text:
            return {
                "hashtag_count": 0.0,
                "mention_count": 0.0,
                "url_count": 0.0,
                "emoji_count": 0.0,
                "exclamation_count": 0.0,
                "question_count": 0.0,
                "caps_ratio": 0.0
            }
        
        # Count various social media elements
        hashtag_count = len(re.findall(r'#\w+', text))
        mention_count = len(re.findall(r'@\w+', text))
        url_count = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
        
        # Emoji count (simple approximation)
        emoji_pattern = re.compile("["
                                 u"\U0001F600-\U0001F64F"  # emoticons
                                 u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                 u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                 u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                 "]+", flags=re.UNICODE)
        emoji_count = len(emoji_pattern.findall(text))
        
        # Punctuation counts
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        # Caps ratio
        caps_chars = sum(1 for c in text if c.isupper())
        total_chars = sum(1 for c in text if c.isalpha())
        caps_ratio = caps_chars / max(total_chars, 1)
        
        return {
            "hashtag_count": float(hashtag_count),
            "mention_count": float(mention_count),
            "url_count": float(url_count),
            "emoji_count": float(emoji_count),
            "exclamation_count": float(exclamation_count),
            "question_count": float(question_count),
            "caps_ratio": caps_ratio
        }
    
    def get_text_embeddings(self, text: str) -> Optional[List[float]]:
        """Get sentence embeddings for the text."""
        if not text or not self.sentence_transformer:
            return None
        
        try:
            embeddings = self.sentence_transformer.encode(text)
            return embeddings.tolist()
        except Exception:
            return None
    
    def extract_all_features(self, text: str) -> Dict[str, float]:
        """Extract all text features."""
        features = {}
        
        # Combine all feature types
        features.update(self.extract_sentiment_features(text))
        features.update(self.extract_emotional_intensity(text))
        features.update(self.extract_readability_features(text))
        features.update(self.extract_structural_features(text))
        features.update(self.extract_social_features(text))
        
        return features
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract keywords from text (simple frequency-based)."""
        if not text:
            return []
        
        # Simple keyword extraction based on word frequency
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top k
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k]]
