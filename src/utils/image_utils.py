#!/usr/bin/env python3
"""
Image Utilities for FiendishFinder

Common image loading, conversion, and processing functions used across
the application for minimap images, screenshots, and floor maps.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict
from PIL import Image
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

logger = logging.getLogger(__name__)


# ============================================================================
# FLOOR IMAGE LOADING
# ============================================================================

def load_floor_image(floor: int, 
                    minimap_dir: Path = Path("processed_minimap"),
                    cache: Optional[Dict[int, Image.Image]] = None,
                    target_mode: str = 'RGBA') -> Optional[Image.Image]:
    """
    Load a floor image from the processed minimap directory.
    
    Args:
        floor: Floor number (0-15)
        minimap_dir: Directory containing floor images
        cache: Optional cache dictionary to store loaded images
        target_mode: Target PIL image mode ('RGB' or 'RGBA')
    
    Returns:
        PIL Image or None if loading failed
    """
    # Check cache first
    if cache is not None and floor in cache:
        return cache[floor]
    
    # Construct floor file path
    floor_file = minimap_dir / f"floor_{floor:02d}.png"
    
    if not floor_file.exists():
        logger.error(f"Floor image not found: {floor_file}")
        return None
    
    try:
        image = Image.open(floor_file)
        
        # Convert to target mode if needed
        image = ensure_image_mode(image, target_mode)
        
        # Cache if cache provided
        if cache is not None:
            cache[floor] = image
        
        return image
        
    except Exception as e:
        logger.error(f"Error loading floor {floor} from {floor_file}: {e}")
        return None


# ============================================================================
# PIXMAP LOADING (PyQt6)
# ============================================================================

def load_pixmap(file_path: Path,
               scale_to: Optional[Tuple[int, int]] = None,
               keep_aspect_ratio: bool = True) -> Optional[QPixmap]:
    """
    Load a QPixmap from a file with optional scaling.
    
    Args:
        file_path: Path to image file
        scale_to: Optional (width, height) to scale to
        keep_aspect_ratio: Whether to maintain aspect ratio when scaling
    
    Returns:
        QPixmap or None if loading failed
    """
    try:
        pixmap = QPixmap(str(file_path))
        
        if pixmap.isNull():
            logger.error(f"Failed to load pixmap from {file_path}")
            return None
        
        # Scale if requested
        if scale_to is not None:
            width, height = scale_to
            aspect_mode = (Qt.AspectRatioMode.KeepAspectRatio if keep_aspect_ratio 
                          else Qt.AspectRatioMode.IgnoreAspectRatio)
            
            pixmap = pixmap.scaled(
                width, height,
                aspect_mode,
                Qt.TransformationMode.SmoothTransformation
            )
        
        return pixmap
        
    except Exception as e:
        logger.error(f"Error loading pixmap from {file_path}: {e}")
        return None


# ============================================================================
# IMAGE MODE CONVERSION
# ============================================================================

def ensure_image_mode(image: Image.Image, target_mode: str = 'RGBA') -> Image.Image:
    """
    Ensure a PIL Image is in the specified mode.
    
    Args:
        image: PIL Image to convert
        target_mode: Target mode ('RGB', 'RGBA', 'L', etc.)
    
    Returns:
        Image in the target mode (may be the same object if already correct)
    """
    if image.mode == target_mode:
        return image
    
    return image.convert(target_mode)


# ============================================================================
# PIXEL COLOR EXTRACTION
# ============================================================================

def get_pixel_rgb(image: Image.Image, 
                 x: int, y: int,
                 check_alpha: bool = True,
                 alpha_threshold: int = 128) -> Optional[Tuple[int, int, int]]:
    """
    Get the RGB color of a pixel, handling different image modes.
    
    Args:
        image: PIL Image
        x: X coordinate
        y: Y coordinate
        check_alpha: If True, return None for transparent pixels
        alpha_threshold: Alpha value below which pixel is considered transparent
    
    Returns:
        (R, G, B) tuple or None if pixel is transparent/invalid
    """
    try:
        pixel = image.getpixel((x, y))
    except IndexError:
        logger.warning(f"Pixel coordinates ({x}, {y}) out of bounds for image size {image.size}")
        return None
    
    # Handle different pixel formats
    if isinstance(pixel, int):
        # Grayscale
        return (pixel, pixel, pixel)
    
    elif len(pixel) == 3:
        # RGB
        return pixel
    
    elif len(pixel) == 4:
        # RGBA
        r, g, b, a = pixel
        
        if check_alpha and a < alpha_threshold:
            return None  # Transparent pixel
        
        return (r, g, b)
    
    else:
        logger.warning(f"Unexpected pixel format: {pixel}")
        return None

