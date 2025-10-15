"""
Crosshair Position Prediction using DINOv3 Feature Matching

Production implementation for crosshair position prediction from minimap crops.
Uses DINOv3 visual features and similarity matching for accurate position detection.
"""

from .feature_extractor import FeatureExtractor
from .feature_database import FeatureDatabase
from .matcher import CrosshairMatcher
from .inference import CrosshairPredictor

__all__ = [
    'FeatureExtractor',
    'FeatureDatabase',
    'CrosshairMatcher',
    'CrosshairPredictor',
]

