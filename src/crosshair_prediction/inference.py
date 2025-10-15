"""
Crosshair Position Predictor - Production Interface

Provides a simple interface for crosshair position prediction from screenshots
or minimap crops using DINOv3 feature matching.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image

from .matcher import CrosshairMatcher

logger = logging.getLogger(__name__)


class CrosshairPredictor:
    """
    Crosshair position predictor using DINOv3 feature matching.
    
    Production interface for automatic monster tracking and position detection.
    """
    
    def __init__(
        self,
        database_path: Optional[str] = None,
        model_name: str = 'dinov3_vitb16',
        device: Optional[str] = None
    ):
        """
        Initialize the crosshair predictor.

        Args:
            database_path: Path to the feature database (default: auto-detect best available)
            model_name: DINOv3 model variant (default: dinov3_vitb16 - best accuracy)
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        logger.info("Initializing CrosshairPredictor...")

        # Auto-detect database if not provided
        if database_path is None:
            database_path = self._auto_detect_database()
        
        # Initialize the matcher with optimal settings
        self.matcher = CrosshairMatcher(
            database_path=database_path,
            model_name=model_name,
            device=device,
            use_interpolation=True,
            top_k_matches=2,
            temperature=0.01  # Optimal temperature for best accuracy
        )
        
        logger.info("CrosshairPredictor ready")
    
    def _auto_detect_database(self) -> str:
        """Auto-detect the best available database."""
        # Get the path to the crosshair_prediction module
        module_dir = Path(__file__).parent

        # Priority order: vitb16 > vits16plus > vits14
        candidates = [
            module_dir / "database" / "feature_db_dinov3_vitb16.pkl",
            module_dir / "database" / "feature_db_dinov3.pkl",
            module_dir / "database" / "feature_db.pkl"
        ]

        for db_path in candidates:
            if db_path.exists():
                logger.info(f"Auto-detected database: {db_path}")
                return str(db_path)

        raise FileNotFoundError(
            "No feature database found. Expected location:\n"
            f"{module_dir / 'database' / 'feature_db_dinov3_vitb16.pkl'}\n\n"
            "Please ensure the database file exists in the crosshair_prediction/database/ folder."
        )
    
    def predict_from_screenshot(
        self,
        screenshot_path,
        minimap_region: Tuple[int, int, int, int],
        floor: Optional[int] = None
    ) -> Tuple[float, float, int]:
        """
        Predict crosshair position from a screenshot.

        Args:
            screenshot_path: Path to the screenshot image (str or Path)
            minimap_region: Tuple of (x, y, width, height) for minimap region
            floor: Optional floor number (if None, will detect floor)

        Returns:
            Tuple of (x, y, floor) coordinates
        """
        try:
            # Convert to Path if string
            if isinstance(screenshot_path, str):
                screenshot_path = Path(screenshot_path)

            # Load screenshot
            screenshot = Image.open(screenshot_path)

            # Extract minimap region
            x, y, width, height = minimap_region
            minimap_crop = screenshot.crop((x, y, x + width, y + height))

            logger.info(f"Cropped minimap region: {minimap_region} from {screenshot_path.name}")
            
            # Predict position
            if floor is not None:
                # Floor is known, predict position only
                pred_x, pred_y, confidence = self.matcher.predict_position(
                    minimap_crop,
                    floor=floor
                )
                pred_floor = floor
            else:
                # Detect floor as well
                pred_x, pred_y, pred_floor, confidence = self.matcher.predict_with_floor_detection(
                    minimap_crop
                )
            
            logger.info(
                f"Predicted position: ({pred_x:.1f}, {pred_y:.1f}) on floor {pred_floor} "
                f"(confidence: {confidence:.3f})"
            )
            
            return pred_x, pred_y, pred_floor
            
        except Exception as e:
            logger.error(f"Error predicting from screenshot: {e}")
            raise
    
    def predict_from_crop(
        self,
        minimap_crop: Image.Image,
        floor: Optional[int] = None
    ) -> Tuple[float, float, int, float]:
        """
        Predict crosshair position from a minimap crop.
        
        Args:
            minimap_crop: PIL Image of the minimap crop
            floor: Optional floor number (if None, will detect floor)
        
        Returns:
            Tuple of (x, y, floor, confidence)
        """
        if floor is not None:
            # Floor is known, predict position only
            pred_x, pred_y, confidence = self.matcher.predict_position(
                minimap_crop,
                floor=floor
            )
            pred_floor = floor
        else:
            # Detect floor as well
            pred_x, pred_y, pred_floor, confidence = self.matcher.predict_with_floor_detection(
                minimap_crop
            )
        
        return pred_x, pred_y, pred_floor, confidence

