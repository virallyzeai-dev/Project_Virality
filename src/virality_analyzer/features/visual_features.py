"""Visual feature extraction for images and videos."""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageStat
import requests
from io import BytesIO


class VisualFeatureExtractor:
    """Extract features from visual content (images and videos)."""
    
    def __init__(self):
        """Initialize the visual feature extractor."""
        self.face_cascade = None
        self._initialize_opencv_models()
    
    def _initialize_opencv_models(self) -> None:
        """Initialize OpenCV models for face detection."""
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except Exception as e:
            print(f"Warning: Could not initialize face detection: {e}")
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from file path or URL."""
        try:
            if image_path.startswith(('http://', 'https://')):
                # Load from URL
                response = requests.get(image_path, timeout=10)
                image = Image.open(BytesIO(response.content))
                return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                # Load from file
                return cv2.imread(image_path)
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def extract_color_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract color-related features from image."""
        if image is None:
            return {
                "brightness": 0.0,
                "contrast": 0.0,
                "saturation": 0.0,
                "dominant_color_r": 0.0,
                "dominant_color_g": 0.0,
                "dominant_color_b": 0.0,
                "color_diversity": 0.0
            }
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Brightness (average of V channel in HSV)
        brightness = np.mean(hsv[:, :, 2]) / 255.0
        
        # Contrast (standard deviation of lightness)
        contrast = np.std(lab[:, :, 0]) / 100.0
        
        # Saturation (average of S channel in HSV)
        saturation = np.mean(hsv[:, :, 1]) / 255.0
        
        # Dominant color (mean RGB values)
        dominant_color = np.mean(image, axis=(0, 1))
        dominant_color_r = dominant_color[2] / 255.0  # BGR to RGB
        dominant_color_g = dominant_color[1] / 255.0
        dominant_color_b = dominant_color[0] / 255.0
        
        # Color diversity (number of unique colors normalized)
        resized = cv2.resize(image, (50, 50))  # Reduce for performance
        unique_colors = len(np.unique(resized.reshape(-1, 3), axis=0))
        color_diversity = unique_colors / (50 * 50)
        
        return {
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "dominant_color_r": dominant_color_r,
            "dominant_color_g": dominant_color_g,
            "dominant_color_b": dominant_color_b,
            "color_diversity": color_diversity
        }
    
    def extract_composition_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract composition and structural features."""
        if image is None:
            return {
                "aspect_ratio": 0.0,
                "edge_density": 0.0,
                "visual_complexity": 0.0,
                "symmetry_score": 0.0
            }
        
        height, width = image.shape[:2]
        aspect_ratio = width / height
        
        # Edge detection for complexity
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Visual complexity (based on edge density and color diversity)
        visual_complexity = edge_density
        
        # Simple symmetry score (vertical symmetry)
        left_half = gray[:, :width//2]
        right_half = cv2.flip(gray[:, width//2:], 1)
        
        # Resize to match if needed
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        symmetry_score = 1.0 - (np.mean(np.abs(left_half - right_half)) / 255.0)
        
        return {
            "aspect_ratio": aspect_ratio,
            "edge_density": edge_density,
            "visual_complexity": visual_complexity,
            "symmetry_score": symmetry_score
        }
    
    def extract_object_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract object and face detection features."""
        if image is None or self.face_cascade is None:
            return {
                "face_count": 0.0,
                "face_area_ratio": 0.0,
                "largest_face_ratio": 0.0
            }
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Face detection
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        face_count = len(faces)
        total_area = image.shape[0] * image.shape[1]
        
        if face_count > 0:
            # Calculate total face area
            face_areas = [w * h for (x, y, w, h) in faces]
            total_face_area = sum(face_areas)
            face_area_ratio = total_face_area / total_area
            
            # Largest face ratio
            largest_face_area = max(face_areas)
            largest_face_ratio = largest_face_area / total_area
        else:
            face_area_ratio = 0.0
            largest_face_ratio = 0.0
        
        return {
            "face_count": float(face_count),
            "face_area_ratio": face_area_ratio,
            "largest_face_ratio": largest_face_ratio
        }
    
    def extract_aesthetic_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract aesthetic quality features."""
        if image is None:
            return {
                "sharpness": 0.0,
                "exposure_quality": 0.0,
                "color_harmony": 0.0,
                "aesthetic_score": 0.0
            }
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(laplacian_var / 1000.0, 1.0)  # Normalize
        
        # Exposure quality (avoid over/under exposure)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        # Check for over-exposure (too many white pixels)
        overexposed = np.sum(hist[240:]) / np.sum(hist)
        # Check for under-exposure (too many dark pixels)
        underexposed = np.sum(hist[:20]) / np.sum(hist)
        exposure_quality = 1.0 - max(overexposed, underexposed)
        
        # Color harmony (simple metric based on color distribution)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hue_peaks = len([i for i in range(1, 179) if hue_hist[i] > hue_hist[i-1] and hue_hist[i] > hue_hist[i+1]])
        color_harmony = 1.0 / (1.0 + hue_peaks * 0.1)  # Fewer peaks = more harmony
        
        # Overall aesthetic score (weighted combination)
        aesthetic_score = (sharpness * 0.3 + exposure_quality * 0.4 + color_harmony * 0.3)
        
        return {
            "sharpness": sharpness,
            "exposure_quality": exposure_quality,
            "color_harmony": color_harmony,
            "aesthetic_score": aesthetic_score
        }
    
    def extract_all_features(self, image_path: str) -> Dict[str, float]:
        """Extract all visual features from an image."""
        image = self.load_image(image_path)
        
        features = {}
        features.update(self.extract_color_features(image))
        features.update(self.extract_composition_features(image))
        features.update(self.extract_object_features(image))
        features.update(self.extract_aesthetic_features(image))
        
        return features
    
    def extract_video_features(self, video_path: str, sample_frames: int = 10) -> Dict[str, float]:
        """Extract features from video by sampling frames."""
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            
            if frame_count == 0:
                cap.release()
                return {"video_duration": 0.0, "video_fps": 0.0}
            
            # Sample frames evenly throughout the video
            frame_indices = np.linspace(0, frame_count - 1, min(sample_frames, frame_count), dtype=int)
            
            all_features = []
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    frame_features = {}
                    frame_features.update(self.extract_color_features(frame))
                    frame_features.update(self.extract_composition_features(frame))
                    frame_features.update(self.extract_object_features(frame))
                    frame_features.update(self.extract_aesthetic_features(frame))
                    all_features.append(frame_features)
            
            cap.release()
            
            if not all_features:
                return {"video_duration": duration, "video_fps": fps}
            
            # Aggregate features across frames
            aggregated_features = {}
            feature_keys = all_features[0].keys()
            
            for key in feature_keys:
                values = [f[key] for f in all_features]
                aggregated_features[f"avg_{key}"] = np.mean(values)
                aggregated_features[f"std_{key}"] = np.std(values)
            
            # Add video-specific features
            aggregated_features["video_duration"] = duration
            aggregated_features["video_fps"] = fps
            
            return aggregated_features
            
        except Exception as e:
            print(f"Error processing video: {e}")
            return {"video_duration": 0.0, "video_fps": 0.0}
