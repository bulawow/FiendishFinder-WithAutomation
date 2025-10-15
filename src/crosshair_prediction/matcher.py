"""
Crosshair Position Matcher

Main matching logic that combines feature extraction and database querying
to predict crosshair positions from minimap crops.
"""

import numpy as np
import logging
from typing import Tuple, Optional, List
from PIL import Image

from .feature_extractor import FeatureExtractor
from .feature_database import FeatureDatabase

logger = logging.getLogger(__name__)


class CrosshairMatcher:
    """Match minimap crops to positions using feature similarity."""
    
    def __init__(
        self,
        database_path: str,
        model_name: str = 'dinov3_vitb16',
        device: Optional[str] = None,
        use_interpolation: bool = True,
        top_k_matches: int = 10,
        temperature: float = 0.01
    ):
        """
        Initialize the crosshair matcher.

        Args:
            database_path: Path to the feature database
            model_name: DINOv3 model variant to use (default: dinov3_vitb16)
            device: Device to run on ('cuda', 'cpu', or None for auto)
            use_interpolation: Whether to interpolate position from top-k matches
            top_k_matches: Number of top matches to use for interpolation
            temperature: Temperature for softmax interpolation (lower = more weight on best match)
        """
        self.database_path = database_path
        self.use_interpolation = use_interpolation
        self.top_k_matches = top_k_matches
        self.temperature = temperature
        
        logger.info("Initializing CrosshairMatcher...")
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            model_name=model_name,
            device=device
        )
        
        # Load feature database
        self.database = FeatureDatabase(database_path=database_path)
        self.database.load()
        
        logger.info("CrosshairMatcher initialized successfully")
    
    def predict_position(
        self,
        minimap_crop: Image.Image,
        floor: int
    ) -> Tuple[float, float, float]:
        """
        Predict the crosshair position from a minimap crop.
        
        Args:
            minimap_crop: PIL Image of the minimap crop
            floor: Floor number
        
        Returns:
            Tuple of (x, y, confidence) where confidence is the similarity score
        """
        # Extract features from the query crop
        query_features = self.feature_extractor.extract_features(
            minimap_crop,
            normalize=True
        )
        
        # Query the database
        matches = self.database.query(
            query_features,
            floor=floor,
            top_k=self.top_k_matches
        )
        
        if not matches:
            raise ValueError(f"No matches found for floor {floor}")
        
        # Get the best match
        best_x, best_y, best_sim = matches[0]
        
        # If not using interpolation, return the best match
        if not self.use_interpolation or len(matches) == 1:
            return best_x, best_y, best_sim
        
        # Interpolate position from top-k matches using weighted average
        x_pred, y_pred, confidence = self._interpolate_position(matches)
        
        return x_pred, y_pred, confidence
    
    def _interpolate_position(
        self,
        matches: List[Tuple[float, float, float]]
    ) -> Tuple[float, float, float]:
        """
        Interpolate position from multiple matches using weighted average.
        
        Args:
            matches: List of (x, y, similarity) tuples
        
        Returns:
            Tuple of (x, y, confidence)
        """
        # Extract positions and similarities
        positions = np.array([(x, y) for x, y, _ in matches])
        similarities = np.array([sim for _, _, sim in matches])

        # Use softmax to convert similarities to weights
        # Temperature parameter controls how much to focus on top matches
        exp_sims = np.exp(similarities / self.temperature)
        weights = exp_sims / np.sum(exp_sims)
        
        # Weighted average of positions
        x_pred = np.sum(positions[:, 0] * weights)
        y_pred = np.sum(positions[:, 1] * weights)
        
        # Confidence is the weighted average of similarities
        confidence = np.sum(similarities * weights)
        
        return float(x_pred), float(y_pred), float(confidence)
    
    def predict_with_floor_detection(
        self,
        minimap_crop: Image.Image,
        candidate_floors: Optional[List[int]] = None
    ) -> Tuple[float, float, int, float]:
        """
        Predict position and floor by finding the best match across floors.
        
        Args:
            minimap_crop: PIL Image of the minimap crop
            candidate_floors: List of floors to search (None = all floors)
        
        Returns:
            Tuple of (x, y, floor, confidence)
        """
        # Extract features once
        query_features = self.feature_extractor.extract_features(
            minimap_crop,
            normalize=True
        )
        
        # Determine which floors to search
        if candidate_floors is None:
            candidate_floors = self.database.metadata.get('floors', list(range(16)))
        
        # Find best match across all candidate floors
        best_floor = None
        best_x = None
        best_y = None
        best_confidence = -1.0
        
        for floor in candidate_floors:
            if floor not in self.database.features:
                continue
            
            # Query this floor
            matches = self.database.query(
                query_features,
                floor=floor,
                top_k=self.top_k_matches
            )
            
            if not matches:
                continue
            
            # Get position for this floor
            if self.use_interpolation and len(matches) > 1:
                x, y, confidence = self._interpolate_position(matches)
            else:
                x, y, confidence = matches[0]
            
            # Update best if this is better
            if confidence > best_confidence:
                best_confidence = confidence
                best_x = x
                best_y = y
                best_floor = floor
        
        if best_floor is None:
            raise ValueError("No matches found on any floor")
        
        return best_x, best_y, best_floor, best_confidence

