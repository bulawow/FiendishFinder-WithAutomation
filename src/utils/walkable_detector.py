#!/usr/bin/env python3
"""
Walkable Area Detection for FiendishFinder

Detects walkable areas on Tibia minimap by filtering out non-walkable colors.
Used for generating valid crosshair dataset positions.
"""

import json
from pathlib import Path
from typing import Set, Tuple, List
from PIL import Image
import numpy as np


class WalkableDetector:
    """Detects walkable areas on Tibia minimap images."""

    # Default non-walkable colors in Tibia minimap (RGB values)
    # These colors represent areas where the player cannot stand
    DEFAULT_NON_WALKABLE_COLORS: Set[Tuple[int, int, int]] = {
        # Pure black - missing/unexplored areas
        (0, 0, 0),

        # Water colors - various shades of blue/cyan used for water
        (0, 0, 255),      # Pure blue
        (0, 128, 255),    # Light blue
        (0, 64, 255),     # Medium blue
        (0, 191, 255),    # Deep sky blue
        (64, 164, 223),   # Cornflower blue
        (72, 160, 216),   # Water blue
        (80, 156, 208),   # Water blue variant
        (88, 152, 200),   # Water blue variant

        # Deep water / ocean
        (0, 0, 128),      # Navy blue
        (0, 0, 139),      # Dark blue
        (0, 0, 205),      # Medium blue

        # Lava colors - red/orange tones
        (255, 0, 0),      # Pure red
        (255, 69, 0),     # Red-orange
        (255, 140, 0),    # Dark orange
        (220, 20, 60),    # Crimson
        (178, 34, 34),    # Firebrick

        # Mountain/cliff colors - typically gray/brown
        (128, 128, 128),  # Gray
        (105, 105, 105),  # Dim gray
        (169, 169, 169),  # Dark gray
        (192, 192, 192),  # Silver

        # Add more non-walkable colors as needed
    }

    def __init__(self, tolerance: int = 5, settings_dir: str = "settings"):
        """Initialize the walkable detector.

        Args:
            tolerance: Color matching tolerance (0-255). Higher values allow
                      more variation in color matching.
            settings_dir: Directory to store color settings
        """
        self.tolerance = tolerance
        self.settings_dir = Path(settings_dir)
        self.settings_dir.mkdir(exist_ok=True)
        self.colors_file = self.settings_dir / "non_walkable_colors.json"

        # Instance-specific color set (loaded from file or defaults)
        self.NON_WALKABLE_COLORS: Set[Tuple[int, int, int]] = set()

        # Floor-specific walkable exceptions
        # Format: {(r, g, b): [list of floors where this color IS walkable]}
        # RGB(51, 102, 153) is walkable on floors 8-15 but not on floors 0-7
        self.FLOOR_WALKABLE_EXCEPTIONS = {
            (51, 102, 153): list(range(8, 16))  # Floors 8, 9, 10, 11, 12, 13, 14, 15
        }

        self.load_colors()
    
    def is_color_walkable(self, rgb: Tuple[int, int, int], floor: int = None) -> bool:
        """Check if a specific RGB color is walkable.

        Args:
            rgb: RGB color tuple (r, g, b)
            floor: Optional floor number for floor-specific rules

        Returns:
            True if the color is walkable, False otherwise
        """
        r, g, b = rgb

        # Check against all non-walkable colors with tolerance
        for non_walkable in self.NON_WALKABLE_COLORS:
            nr, ng, nb = non_walkable

            # Calculate color distance
            distance = abs(r - nr) + abs(g - ng) + abs(b - nb)

            if distance <= self.tolerance * 3:  # tolerance per channel
                # Check if this color has a floor-specific exception
                if floor is not None and non_walkable in self.FLOOR_WALKABLE_EXCEPTIONS:
                    walkable_floors = self.FLOOR_WALKABLE_EXCEPTIONS[non_walkable]
                    if floor in walkable_floors:
                        # This color is walkable on this specific floor
                        continue
                return False

        return True
    
    def is_position_walkable(self, image: Image.Image, x: int, y: int, floor: int = None) -> bool:
        """Check if a specific position on the map is walkable.

        Args:
            image: PIL Image of the floor map
            x: X coordinate
            y: Y coordinate
            floor: Optional floor number for floor-specific rules

        Returns:
            True if the position is walkable, False otherwise
        """
        # Check bounds
        if x < 0 or y < 0 or x >= image.width or y >= image.height:
            return False

        # Get pixel color
        pixel = image.getpixel((x, y))

        # Handle different image modes
        if isinstance(pixel, int):
            # Grayscale
            rgb = (pixel, pixel, pixel)
        elif len(pixel) == 3:
            # RGB
            rgb = pixel
        elif len(pixel) == 4:
            # RGBA - check alpha channel
            r, g, b, a = pixel
            if a < 128:  # Transparent areas are not walkable
                return False
            rgb = (r, g, b)
        else:
            return False

        return self.is_color_walkable(rgb, floor)
    
    def get_walkable_positions(self, image: Image.Image,
                               sample_rate: int = 10, floor: int = None) -> List[Tuple[int, int]]:
        """Get all walkable positions on a floor map.

        Args:
            image: PIL Image of the floor map
            sample_rate: Check every Nth pixel (higher = faster but less accurate)
            floor: Optional floor number for floor-specific rules

        Returns:
            List of (x, y) coordinates that are walkable
        """
        walkable_positions = []

        # Convert to RGB if needed
        if image.mode != 'RGB' and image.mode != 'RGBA':
            image = image.convert('RGB')

        # Sample the image
        for y in range(0, image.height, sample_rate):
            for x in range(0, image.width, sample_rate):
                if self.is_position_walkable(image, x, y, floor):
                    walkable_positions.append((x, y))

        return walkable_positions
    
    def get_walkable_area_mask(self, image: Image.Image, floor: int = None) -> np.ndarray:
        """Generate a boolean mask of walkable areas.

        Args:
            image: PIL Image of the floor map
            floor: Optional floor number for floor-specific rules

        Returns:
            Boolean numpy array where True = walkable, False = non-walkable
        """
        # Convert to RGB if needed
        if image.mode != 'RGB' and image.mode != 'RGBA':
            image = image.convert('RGB')

        # Convert to numpy array
        img_array = np.array(image)

        # Create mask
        mask = np.ones((image.height, image.width), dtype=bool)

        # Check each pixel
        for y in range(image.height):
            for x in range(image.width):
                if not self.is_position_walkable(image, x, y, floor):
                    mask[y, x] = False

        return mask
    
    def load_colors(self) -> None:
        """Load non-walkable colors from JSON file."""
        if not self.colors_file.exists():
            # Use defaults if no file exists
            self.NON_WALKABLE_COLORS = self.DEFAULT_NON_WALKABLE_COLORS.copy()
            self.save_colors()
            return

        try:
            with open(self.colors_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert list of lists back to set of tuples
            colors_list = data.get('non_walkable_colors', [])
            self.NON_WALKABLE_COLORS = {tuple(color) for color in colors_list}

            # If empty, use defaults
            if not self.NON_WALKABLE_COLORS:
                self.NON_WALKABLE_COLORS = self.DEFAULT_NON_WALKABLE_COLORS.copy()

        except Exception as e:
            print(f"Error loading non-walkable colors: {e}")
            self.NON_WALKABLE_COLORS = self.DEFAULT_NON_WALKABLE_COLORS.copy()

    def save_colors(self) -> bool:
        """Save non-walkable colors to JSON file.

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Convert set of tuples to list of lists for JSON serialization
            colors_list = [list(color) for color in sorted(self.NON_WALKABLE_COLORS)]

            data = {
                'non_walkable_colors': colors_list
            }

            with open(self.colors_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            return True
        except Exception as e:
            print(f"Error saving non-walkable colors: {e}")
            return False

    def add_non_walkable_color(self, rgb: Tuple[int, int, int]):
        """Add a new non-walkable color to the detector.

        Args:
            rgb: RGB color tuple (r, g, b)
        """
        self.NON_WALKABLE_COLORS.add(rgb)
        self.save_colors()

    def remove_non_walkable_color(self, rgb: Tuple[int, int, int]):
        """Remove a color from the non-walkable list.

        Args:
            rgb: RGB color tuple (r, g, b)
        """
        self.NON_WALKABLE_COLORS.discard(rgb)
        self.save_colors()

    def get_non_walkable_colors(self) -> Set[Tuple[int, int, int]]:
        """Get the current set of non-walkable colors.

        Returns:
            Set of RGB color tuples
        """
        return self.NON_WALKABLE_COLORS.copy()

