#!/usr/bin/env python3
"""
Minimap Viewer Component for FiendishFinder

A PyQt6-based minimap viewer with zoom functionality, floor navigation,
and camera position preservation for Tibia minimap exploration.
"""

import sys
import math
import json
import uuid
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QPushButton,
    QComboBox, QLabel, QSlider, QFrame, QSizePolicy, QMessageBox,
    QGraphicsLineItem, QGraphicsRectItem, QGraphicsPolygonItem,
    QGraphicsItem, QCheckBox, QLineEdit, QFormLayout,
    QScrollArea, QGroupBox, QStyleOptionGraphicsItem, QMenuBar, QMenu,
    QTabWidget, QTextEdit, QSpinBox, QListWidget, QListWidgetItem,
    QDialog, QFileDialog, QGraphicsTextItem, QSplitter, QStackedWidget
)
from PyQt6.QtCore import Qt, QRectF, pyqtSignal, QPointF, QTimer
from PyQt6.QtGui import QPixmap, QWheelEvent, QMouseEvent, QPainter, QKeyEvent, QPen, QColor, QBrush, QPolygon, QPolygonF, QPainterPath, QTransform

# Import canonical enums from dataset_models
from src.models.dataset_models import MonsterDifficulty, ExivaDirection, ExivaRange

# Import Exiva mechanics utilities
from src.utils.exiva_mechanics import (
    get_direction_ranges,
    get_boundary_angles,
    calculate_angle_between_points,
    EXIVA_DISTANCE_RANGES
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AreaData:
    """Data structure for storing area information."""
    area_id: str
    floor: int
    name: str
    coordinates: List[Tuple[float, float]]  # List of (x, y) points for polygon
    difficulty_levels: List[str]  # List of MonsterDifficulty values
    color: str  # Hex color string
    transparency: float  # 0.0 to 1.0
    metadata: Dict[str, Any]  # Additional custom fields
    route: str  # Description of how to get to this area
    created_timestamp: float
    modified_timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AreaData':
        """Create AreaData from dictionary."""
        # Remove area_type field for backward compatibility
        data = data.copy()
        data.pop('area_type', None)

        # Migrate from old 'monsters' field to 'route' field for backward compatibility
        if 'monsters' in data and 'route' not in data:
            # Convert old monsters list to a simple string (or empty if no monsters)
            data['route'] = ''
            data.pop('monsters')

        # Add route field for backward compatibility if not present
        if 'route' not in data:
            data['route'] = ''

        return cls(**data)


@dataclass
class ExivaReading:
    """Data structure for storing a single Exiva spell reading."""
    direction: ExivaDirection
    distance: ExivaRange  # Changed from ExivaDistance to ExivaRange
    crosshair_position: Tuple[float, float]  # (x, y) position where reading was taken
    timestamp: float
    monster_difficulty: MonsterDifficulty = MonsterDifficulty.UNKNOWN  # Monster difficulty type

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'direction': self.direction.value,
            'distance': self.distance.value,
            'crosshair_position': self.crosshair_position,
            'timestamp': self.timestamp,
            'monster_difficulty': self.monster_difficulty.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExivaReading':
        """Create ExivaReading from dictionary."""
        return cls(
            direction=ExivaDirection(data['direction']),
            distance=ExivaRange(data['distance']),  # Changed from ExivaDistance to ExivaRange
            crosshair_position=tuple(data['crosshair_position']),
            timestamp=data['timestamp'],
            monster_difficulty=MonsterDifficulty(data.get('monster_difficulty', 'unknown'))  # Default to unknown for backward compatibility
        )


@dataclass
class ExivaSession:
    """Data structure for managing an active monster hunt session."""
    session_id: str
    floor: int
    readings: List[ExivaReading]
    created_timestamp: float
    is_active: bool = True
    last_difficulty: MonsterDifficulty = MonsterDifficulty.UNKNOWN

    def add_reading(self, reading: ExivaReading):
        """Add a new Exiva reading to the session."""
        self.readings.append(reading)
        # Update last known difficulty
        if reading.monster_difficulty != MonsterDifficulty.UNKNOWN:
            self.last_difficulty = reading.monster_difficulty

    def clear_readings(self):
        """Clear all readings from the session."""
        self.readings.clear()
        self.last_difficulty = MonsterDifficulty.UNKNOWN

    def get_last_difficulty(self) -> MonsterDifficulty:
        """Get the last known monster difficulty from readings."""
        return self.last_difficulty

    def should_restart_on_difficulty_change(self, new_difficulty: MonsterDifficulty) -> bool:
        """Check if session should restart due to difficulty change.

        Returns True if difficulty changed in ANY way:
        - UNKNOWN → Specific (just identified the monster)
        - Specific → UNKNOWN (new unknown monster spawned)
        - Specific → Different specific (different monster spawned)

        Does NOT restart if:
        - UNKNOWN → UNKNOWN (both unknown)
        - Same → Same (same difficulty)

        Args:
            new_difficulty: The new difficulty being added

        Returns:
            True if session should restart, False otherwise
        """
        # Don't restart if both are UNKNOWN
        if new_difficulty == MonsterDifficulty.UNKNOWN and self.last_difficulty == MonsterDifficulty.UNKNOWN:
            return False

        # Restart if we had UNKNOWN and now have a specific difficulty
        # This means we just identified the monster - clear uncertain readings
        if self.last_difficulty == MonsterDifficulty.UNKNOWN and new_difficulty != MonsterDifficulty.UNKNOWN:
            logger.info(f"Monster identified: unknown → {new_difficulty.value} - restarting to clear uncertain readings")
            return True

        # Restart if we had a specific difficulty and now it's UNKNOWN
        # This means a different monster spawned and we don't know what it is yet
        if self.last_difficulty != MonsterDifficulty.UNKNOWN and new_difficulty == MonsterDifficulty.UNKNOWN:
            logger.info(f"Monster changed: {self.last_difficulty.value} → unknown - different monster spawned")
            return True

        # Restart if difficulty changed from one specific value to another
        if new_difficulty != self.last_difficulty:
            logger.info(f"Difficulty changed: {self.last_difficulty.value} → {new_difficulty.value} - monster changed")
            return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'session_id': self.session_id,
            'floor': self.floor,
            'readings': [reading.to_dict() for reading in self.readings],
            'created_timestamp': self.created_timestamp,
            'is_active': self.is_active,
            'last_difficulty': self.last_difficulty.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExivaSession':
        """Create ExivaSession from dictionary."""
        return cls(
            session_id=data['session_id'],
            floor=data['floor'],
            readings=[ExivaReading.from_dict(r) for r in data['readings']],
            created_timestamp=data['created_timestamp'],
            is_active=data.get('is_active', True),
            last_difficulty=MonsterDifficulty(data.get('last_difficulty', 'unknown'))
        )


class AreaDataManager:
    """Manages area data persistence and storage."""

    def __init__(self, data_dir: str = "area_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.areas_file = self.data_dir / "areas.json"
        self.settings_file = self.data_dir / "settings.json"
        self.areas: Dict[int, List[AreaData]] = {}  # floor -> list of areas
        self.global_transparency: float = 0.5  # Default global transparency
        self.load_settings()
        self.load_areas()

    def load_settings(self) -> None:
        """Load global settings from JSON file."""
        if not self.settings_file.exists():
            logger.info("No existing settings found, using defaults")
            return

        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.global_transparency = data.get('global_transparency', 0.5)
            logger.info(f"Loaded global transparency: {self.global_transparency}")

        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            self.global_transparency = 0.5

    def save_settings(self) -> None:
        """Save global settings to JSON file."""
        try:
            data = {
                'global_transparency': self.global_transparency
            }

            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved global transparency: {self.global_transparency}")

        except Exception as e:
            logger.error(f"Failed to save settings: {e}")

    def load_areas(self) -> None:
        """Load areas from JSON file."""
        if not self.areas_file.exists():
            logger.info("No existing area data found, starting with empty areas")
            return

        try:
            with open(self.areas_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.areas = {}
            for floor_str, areas_list in data.items():
                floor = int(floor_str)
                self.areas[floor] = [AreaData.from_dict(area_dict) for area_dict in areas_list]

            total_areas = sum(len(areas) for areas in self.areas.values())
            logger.info(f"Loaded {total_areas} areas across {len(self.areas)} floors")

        except Exception as e:
            logger.error(f"Failed to load area data: {e}")
            self.areas = {}

    def save_areas(self) -> None:
        """Save areas to JSON file."""
        try:
            data = {}
            for floor, areas_list in self.areas.items():
                data[str(floor)] = [area.to_dict() for area in areas_list]

            with open(self.areas_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            total_areas = sum(len(areas) for areas in self.areas.values())
            logger.info(f"Saved {total_areas} areas to {self.areas_file}")

        except Exception as e:
            logger.error(f"Failed to save area data: {e}")

    def add_area(self, area: AreaData) -> None:
        """Add a new area."""
        if area.floor not in self.areas:
            self.areas[area.floor] = []
        self.areas[area.floor].append(area)
        self.save_areas()

    def remove_area(self, floor: int, area_id: str) -> bool:
        """Remove an area by ID."""
        if floor not in self.areas:
            return False

        original_count = len(self.areas[floor])
        self.areas[floor] = [area for area in self.areas[floor] if area.area_id != area_id]

        if len(self.areas[floor]) < original_count:
            self.save_areas()
            return True
        return False

    def update_area(self, area: AreaData) -> bool:
        """Update an existing area."""
        if area.floor not in self.areas:
            return False

        for i, existing_area in enumerate(self.areas[area.floor]):
            if existing_area.area_id == area.area_id:
                self.areas[area.floor][i] = area
                self.save_areas()
                return True
        return False

    def get_areas_for_floor(self, floor: int) -> List[AreaData]:
        """Get all areas for a specific floor."""
        return self.areas.get(floor, [])

    def get_area_by_id(self, floor: int, area_id: str) -> Optional[AreaData]:
        """Get a specific area by ID."""
        for area in self.get_areas_for_floor(floor):
            if area.area_id == area_id:
                return area
        return None

    def get_default_colors(self) -> Dict[str, str]:
        """Get default colors for each difficulty level."""
        return {
            MonsterDifficulty.NONE.value: "#808080",        # Gray
            MonsterDifficulty.HARMLESS.value: "#00FF00",    # Green
            MonsterDifficulty.TRIVIAL.value: "#7FFF00",     # Chartreuse
            MonsterDifficulty.EASY.value: "#FFFF00",        # Yellow
            MonsterDifficulty.MEDIUM.value: "#FFA500",      # Orange
            MonsterDifficulty.HARD.value: "#FF4500",        # Red-Orange
            MonsterDifficulty.CHALLENGING.value: "#FF0000", # Red
        }


class ExivaOverlayItem(QGraphicsRectItem):
    """Graphics item for rendering Exiva overlay that dims areas not matching readings."""

    def __init__(self, scene_rect: QRectF, readings: List[ExivaReading], parent=None):
        super().__init__(scene_rect, parent)
        self.readings = readings
        self.scene_rect = scene_rect

        # Performance optimization: Cache the overlay pattern
        self.cached_overlay_pixmap: Optional[QPixmap] = None
        self.cache_valid = False
        self.base_grid_size = 20.0  # Increased base grid size for better performance
        self.current_zoom_level = 1.0  # Current zoom level for LOD
        self.cached_zoom_level = 1.0  # Zoom level when cache was generated

        # Zoom debouncing for performance
        self.zoom_update_pending = False

        # Asynchronous cache generation
        self.cache_generation_timer = QTimer()
        self.cache_generation_timer.setSingleShot(True)
        self.cache_generation_timer.timeout.connect(self._generate_cache_async)

        # Performance optimization: Spatial indexing for readings
        self.reading_spatial_index: Dict[Tuple[int, int], List[int]] = {}
        self.reading_bounds: List[Tuple[float, float, float, float]] = []  # (min_x, min_y, max_x, max_y) for each reading

        # Use pre-computed lookup tables from exiva_mechanics module
        self.distance_ranges = EXIVA_DISTANCE_RANGES
        self.direction_angles = get_direction_ranges()

        # Dimming percentage tracking
        self.last_intersection_path: Optional[QPainterPath] = None
        self.dimmed_percentage: float = 0.0

        # Set Z-value to render above map but below crosshairs and area polygons
        self.setZValue(500)

        # Set up the overlay appearance
        self.update_overlay()

        # Generate initial cache immediately to prevent floor change disappearing bug
        self.generate_cached_overlay()

    def point_matches_exiva_reading(self, point: QPointF, reading: ExivaReading) -> bool:
        """Check if a point matches the given Exiva reading."""
        crosshair_pos = QPointF(reading.crosshair_position[0], reading.crosshair_position[1])

        # Calculate distance and angle from crosshair to point
        distance = self.calculate_distance_to_point(crosshair_pos, point)
        angle = self.calculate_angle_to_point(crosshair_pos, point)

        # Check if both distance and direction match
        distance_matches = self.is_distance_in_range(distance, reading.distance)
        direction_matches = self.is_angle_in_direction_range(angle, reading.direction)

        return distance_matches and direction_matches

    def calculate_distance_to_point(self, from_pos: QPointF, to_pos: QPointF) -> float:
        """Calculate the square-based distance between two points (Tibia mechanics)."""
        dx = abs(to_pos.x() - from_pos.x())
        dy = abs(to_pos.y() - from_pos.y())
        return max(dx, dy)  # Square-based distance (Chebyshev distance)

    def calculate_angle_to_point(self, from_pos: QPointF, to_pos: QPointF) -> float:
        """Calculate the angle from one point to another in degrees (0-360)."""
        return calculate_angle_between_points(from_pos.x(), from_pos.y(), to_pos.x(), to_pos.y())

    def is_distance_in_range(self, distance: float, exiva_range: ExivaRange) -> bool:
        """Check if a distance falls within the range for a given Exiva range."""
        min_dist, max_dist = self.distance_ranges[exiva_range]
        return min_dist <= distance <= max_dist

    def is_angle_in_direction_range(self, angle: float, direction: ExivaDirection) -> bool:
        """Check if an angle falls within the range for a given direction."""
        min_angle, max_angle = self.get_direction_angle_range(direction)

        # Handle the special case of NORTH direction which wraps around 0 degrees
        if direction == ExivaDirection.NORTH:
            return angle >= min_angle or angle <= max_angle
        else:
            return min_angle <= angle <= max_angle

    def get_direction_angle_range(self, direction: ExivaDirection) -> Tuple[float, float]:
        """Get the angle range (in degrees) for a given Exiva direction."""
        return self.direction_angles[direction]

    def update_zoom_level(self, zoom_factor: float, force_update: bool = False):
        """Update zoom level for level-of-detail optimization with smart caching."""
        self.current_zoom_level = zoom_factor

        # Only invalidate cache if zoom level crosses significant thresholds
        # This prevents cache invalidation during smooth zoom operations
        zoom_thresholds = [0.25, 0.5, 1.0, 2.0, 4.0]

        current_tier = self._get_zoom_tier(self.current_zoom_level)
        cached_tier = self._get_zoom_tier(self.cached_zoom_level)

        if force_update or current_tier != cached_tier:
            self.cache_valid = False
            self.cached_zoom_level = zoom_factor

    def _get_zoom_tier(self, zoom_level: float) -> int:
        """Get the zoom tier for a given zoom level to determine cache invalidation."""
        if zoom_level < 0.25:
            return 0  # Very low zoom
        elif zoom_level < 0.5:
            return 1  # Low zoom
        elif zoom_level < 1.0:
            return 2  # Medium zoom
        elif zoom_level < 2.0:
            return 3  # High zoom
        elif zoom_level < 4.0:
            return 4  # Very high zoom
        else:
            return 5  # Ultra high zoom

    def get_adaptive_grid_size(self) -> float:
        """Calculate adaptive grid size based on zoom level with stable precision."""
        # Stabilized grid sizing to prevent glitching at high zoom levels
        # Use larger multipliers for better performance while maintaining visual quality
        if self.current_zoom_level < 0.25:
            return self.base_grid_size * 6.0  # Very coarse at very low zoom
        elif self.current_zoom_level < 0.5:
            return self.base_grid_size * 3.0  # Coarse at low zoom
        elif self.current_zoom_level < 1.0:
            return self.base_grid_size * 2.0  # Medium detail at medium zoom
        elif self.current_zoom_level < 2.0:
            return self.base_grid_size * 1.5  # Standard detail at high zoom
        elif self.current_zoom_level < 8.0:  # Extended range for better stability
            return self.base_grid_size * 1.2  # Fine detail at very high zoom
        else:
            # At ultra-high zoom levels, maintain reasonable detail
            return self.base_grid_size * 1.0  # Standard detail for extreme zoom

    def update_overlay(self):
        """Update the overlay based on current readings."""
        # Create a semi-transparent black overlay
        overlay_color = QColor(0, 0, 0, 180)  # Black with 70% opacity
        brush = QBrush(overlay_color)
        self.setBrush(brush)

        # No outline
        pen = QPen(Qt.PenStyle.NoPen)
        self.setPen(pen)

        # Invalidate cache when overlay properties change
        self.cache_valid = False

    def build_spatial_index(self, scale_factor: float, grid_size: float):
        """Build spatial index for readings to optimize grid processing."""
        self.reading_spatial_index.clear()
        self.reading_bounds.clear()

        for reading_idx, reading in enumerate(self.readings):
            # Convert to pixmap coordinates
            scene_x = (reading.crosshair_position[0] - self.scene_rect.left()) * scale_factor
            scene_y = (reading.crosshair_position[1] - self.scene_rect.top()) * scale_factor

            # Calculate bounding box for this reading
            min_dist, max_dist = self.distance_ranges[reading.distance]
            min_dist_scaled = min_dist * scale_factor
            max_dist_scaled = max_dist * scale_factor if max_dist != float('inf') else max(self.scene_rect.width(), self.scene_rect.height()) * scale_factor

            # Bounding box for the reading's influence area
            min_x = scene_x - max_dist_scaled
            max_x = scene_x + max_dist_scaled
            min_y = scene_y - max_dist_scaled
            max_y = scene_y + max_dist_scaled

            self.reading_bounds.append((min_x, min_y, max_x, max_y))

            # Map reading to grid cells it could affect
            grid_min_col = max(0, int(min_x // grid_size))
            grid_max_col = int(max_x // grid_size) + 1
            grid_min_row = max(0, int(min_y // grid_size))
            grid_max_row = int(max_y // grid_size) + 1

            for col in range(grid_min_col, grid_max_col):
                for row in range(grid_min_row, grid_max_row):
                    grid_key = (col, row)
                    if grid_key not in self.reading_spatial_index:
                        self.reading_spatial_index[grid_key] = []
                    self.reading_spatial_index[grid_key].append(reading_idx)

    def generate_cached_overlay(self):
        """Generate and cache the overlay pattern with optimized polygon-based rendering."""
        if not self.readings or self.scene_rect.isEmpty():
            self.cached_overlay_pixmap = None
            self.cache_valid = True
            return

        # Stabilized resolution scaling to prevent zoom glitching
        original_width = self.scene_rect.width()
        original_height = self.scene_rect.height()

        # Use consistent rounding to avoid precision issues
        pixmap_width = max(1, round(original_width))
        pixmap_height = max(1, round(original_height))

        # Simplified max dimension logic to reduce zoom-dependent scaling changes
        # Higher resolution at high zoom levels for small areas
        if self.current_zoom_level >= 4.0:
            max_dimension = 2048  # Increased for better small area detail
        elif self.current_zoom_level >= 2.0:
            max_dimension = 2048  # Consistent high resolution
        elif self.current_zoom_level >= 1.0:
            max_dimension = 2048  # Consistent medium resolution
        else:
            max_dimension = 3072  # Low zoom - higher resolution needed

        # More stable scale factor calculation
        scale_factor = 1.0
        if pixmap_width > max_dimension or pixmap_height > max_dimension:
            scale_factor = min(max_dimension / pixmap_width, max_dimension / pixmap_height)
            # Use precise rounding to maintain grid alignment
            pixmap_width = max(1, round(pixmap_width * scale_factor))
            pixmap_height = max(1, round(pixmap_height * scale_factor))

        # OPTIMIZED APPROACH: Use polygon-based rendering instead of grid-based
        # This provides much better performance than grid processing

        self.cached_overlay_pixmap = QPixmap(pixmap_width, pixmap_height)
        self.cached_overlay_pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(self.cached_overlay_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)  # Disable for performance

        # Create a single large polygon covering the entire scene
        full_scene_polygon = QPolygonF([
            QPointF(0, 0),
            QPointF(pixmap_width, 0),
            QPointF(pixmap_width, pixmap_height),
            QPointF(0, pixmap_height)
        ])

        # Fill the entire scene with dimming
        overlay_color = QColor(0, 0, 0, 180)
        painter.setBrush(QBrush(overlay_color))
        painter.setPen(QPen(Qt.PenStyle.NoPen))
        painter.drawPolygon(full_scene_polygon)

        # Create intersection of all reading sectors (areas that match ALL readings)
        # This ensures only areas that satisfy all readings are undimmed
        intersection_path = None

        # OPTIMIZED APPROACH: Smart geometric regions with accurate boundaries
        # Create accurate regions but with much better performance

        intersection_path = None

        for reading_idx, reading in enumerate(self.readings):
            # Create an optimized region for this reading
            region_path = self._create_optimized_region_path(reading, scale_factor, pixmap_width, pixmap_height)

            if region_path and not region_path.isEmpty():
                if intersection_path is None:
                    # First reading - use its region as the starting intersection
                    intersection_path = region_path
                else:
                    # Subsequent readings - intersect with existing intersection
                    intersection_path = intersection_path.intersected(region_path)

        # Use the intersection path to "cut out" areas from the dimming overlay
        # Only areas that match ALL readings will be undimmed
        if intersection_path is not None and not intersection_path.isEmpty():
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
            painter.setBrush(QBrush(QColor(255, 255, 255, 255)))  # Full opacity for cutting
            painter.drawPath(intersection_path)

        painter.end()
        self.cache_valid = True
        # Update cached zoom level to current zoom level for accurate tracking
        self.cached_zoom_level = self.current_zoom_level

        # Store the intersection path in SCENE coordinates (not pixmap coordinates)
        # The intersection_path is currently in pixmap coordinates, so we need to scale it back
        if intersection_path is not None and not intersection_path.isEmpty():
            # Create a transform to convert from pixmap coordinates to scene coordinates
            transform = QTransform()
            transform.scale(1.0 / scale_factor, 1.0 / scale_factor)
            self.last_intersection_path = transform.map(intersection_path)
        else:
            self.last_intersection_path = None

        self._calculate_dimmed_percentage()

    def create_region_polygon_for_reading(self, reading: ExivaReading, scene_rect: QRectF, grid_size: float = 2.0) -> QPolygonF:
        """Create a polygon representing the region that matches an Exiva reading.

        This creates a pixel-perfect region by densely sampling the scene and creating
        a polygon that accurately represents the square-based Exiva regions.
        """
        # Use very fine grid sampling to capture exact square boundaries
        # This is critical for matching the square-based distance calculation

        # Create a bitmap-like representation of the matching region
        matching_rects = []

        # Sample with a fine grid to capture square boundaries accurately
        x = scene_rect.left()
        while x < scene_rect.right():
            y = scene_rect.top()
            while y < scene_rect.bottom():
                test_point = QPointF(x + grid_size/2, y + grid_size/2)  # Sample center of grid cell
                if self.point_matches_exiva_reading(test_point, reading):
                    # Add a small rectangle representing this grid cell
                    rect_points = [
                        QPointF(x, y),
                        QPointF(x + grid_size, y),
                        QPointF(x + grid_size, y + grid_size),
                        QPointF(x, y + grid_size)
                    ]
                    matching_rects.append(rect_points)
                y += grid_size
            x += grid_size

        if not matching_rects:
            return QPolygonF()  # No matching region found

        # Create a polygon that encompasses all matching rectangles
        # Find the overall bounding area
        all_points = []
        for rect in matching_rects:
            all_points.extend(rect)

        if not all_points:
            return QPolygonF()

        # Find bounds
        min_x = min(p.x() for p in all_points)
        max_x = max(p.x() for p in all_points)
        min_y = min(p.y() for p in all_points)
        max_y = max(p.y() for p in all_points)

        # Create a bounding polygon - this will be refined by the intersection logic
        boundary_points = [
            QPointF(min_x, min_y),
            QPointF(max_x, min_y),
            QPointF(max_x, max_y),
            QPointF(min_x, max_y)
        ]

        return QPolygonF(boundary_points)

    def _create_exiva_region_boundary(self, center: QPointF, direction: ExivaDirection,
                                    min_dist: float, max_dist: float, scene_rect: QRectF) -> List[QPointF]:
        """Create boundary points for an Exiva region using square-based areas bounded by diagonal lines."""
        import math

        # Exiva creates SQUARE-BASED regions, not wedge-shaped regions
        # The region is the intersection of:
        # 1. A square annulus (square ring) for the distance range
        # 2. A directional sector bounded by two diagonal lines

        # For very far distances, extend to scene boundaries
        if max_dist == float('inf'):
            max_dist = max(scene_rect.width(), scene_rect.height()) * 2

        # Create a large square region that covers the entire distance range and direction
        # We'll sample this region using the point matching logic

        # Get the angle range for this direction
        min_angle, max_angle = self.get_direction_angle_range(direction)
        min_angle_rad = math.radians(min_angle)
        max_angle_rad = math.radians(max_angle)

        # Calculate the maximum extent we need to cover
        # For square-based distance, we need to extend far enough in both X and Y directions
        extent = max_dist * 1.5  # Extra margin to ensure we cover the full square

        # Create a bounding rectangle that encompasses the entire possible region
        left = center.x() - extent
        right = center.x() + extent
        top = center.y() - extent
        bottom = center.y() + extent

        # Constrain to scene boundaries
        left = max(left, scene_rect.left())
        right = min(right, scene_rect.right())
        top = max(top, scene_rect.top())
        bottom = min(bottom, scene_rect.bottom())

        # For now, return a large rectangular region
        # The actual filtering will be done by the point matching logic
        boundary_points = [
            QPointF(left, top),
            QPointF(right, top),
            QPointF(right, bottom),
            QPointF(left, bottom)
        ]

        return boundary_points

    def _group_points_into_square_regions(self, points: List[QPointF], grid_size: float) -> List[List[QPointF]]:
        """Group matching points into square regions for better representation of Exiva areas."""
        if not points:
            return []

        # For now, create a simple bounding rectangle that encompasses all points
        # This ensures we cover the full square-based region
        min_x = min(p.x() for p in points)
        max_x = max(p.x() for p in points)
        min_y = min(p.y() for p in points)
        max_y = max(p.y() for p in points)

        # Expand the bounds slightly to ensure we cover edge cases
        margin = grid_size * 2
        min_x -= margin
        max_x += margin
        min_y -= margin
        max_y += margin

        # Create a single rectangular region
        region_points = [
            QPointF(min_x, min_y),
            QPointF(max_x, min_y),
            QPointF(max_x, max_y),
            QPointF(min_x, max_y)
        ]

        return [region_points]

    def _create_boundary_polygon_from_points(self, points: List[QPointF], grid_size: float) -> QPolygonF:
        """Create a boundary polygon from a set of points representing square regions."""
        if not points:
            return QPolygonF()

        # Find the overall bounding rectangle
        min_x = min(p.x() for p in points)
        max_x = max(p.x() for p in points)
        min_y = min(p.y() for p in points)
        max_y = max(p.y() for p in points)

        # Create a rectangular polygon that covers the entire region
        boundary_points = [
            QPointF(min_x, min_y),
            QPointF(max_x, min_y),
            QPointF(max_x, max_y),
            QPointF(min_x, max_y)
        ]

        return QPolygonF(boundary_points)

    def _create_optimized_region_path(self, reading: ExivaReading, scale_factor: float,
                                    pixmap_width: int, pixmap_height: int) -> QPainterPath:
        """Create an accurate region path using pure geometry for best performance."""
        import math

        # Get reading parameters
        crosshair_pos = QPointF(reading.crosshair_position[0], reading.crosshair_position[1])
        min_dist, max_dist = self.distance_ranges[reading.distance]

        # Convert to pixmap coordinates
        center_x = (crosshair_pos.x() - self.scene_rect.left()) * scale_factor
        center_y = (crosshair_pos.y() - self.scene_rect.top()) * scale_factor

        # Handle infinite distance
        if max_dist == float('inf'):
            max_dist_scaled = max(pixmap_width, pixmap_height) * 2
        else:
            max_dist_scaled = max_dist * scale_factor

        min_dist_scaled = min_dist * scale_factor

        # Get direction boundaries using the correct 2.42:1 ratio
        min_angle, max_angle = self.get_direction_angle_range(reading.direction)

        # Create the region using accurate geometric approach
        path = QPainterPath()

        # STEP 1: Create the Chebyshev distance region (square annulus)
        # For Chebyshev distance, the region is a square centered at the crosshair

        # Outer square boundary
        outer_left = center_x - max_dist_scaled
        outer_right = center_x + max_dist_scaled
        outer_top = center_y - max_dist_scaled
        outer_bottom = center_y + max_dist_scaled

        # Inner square boundary (if min_dist > 0)
        inner_left = center_x - min_dist_scaled
        inner_right = center_x + min_dist_scaled
        inner_top = center_y - min_dist_scaled
        inner_bottom = center_y + min_dist_scaled

        # STEP 2: Create directional sector boundaries
        # Convert angles to radians
        min_angle_rad = math.radians(min_angle)
        max_angle_rad = math.radians(max_angle)

        # Create a large polygon that represents the directional sector
        # Extend far enough to cover the entire distance range
        extend_distance = max_dist_scaled * 2

        # Calculate direction vectors
        min_dx = math.sin(min_angle_rad)
        min_dy = -math.cos(min_angle_rad)
        max_dx = math.sin(max_angle_rad)
        max_dy = -math.cos(max_angle_rad)

        # STEP 3: Create the intersection of square annulus and directional sector
        # This is the key to accuracy - we need to properly intersect these shapes

        # Create the square annulus first
        outer_rect = QRectF(outer_left, outer_top,
                           outer_right - outer_left, outer_bottom - outer_top)
        path.addRect(outer_rect)

        # Subtract inner square if needed
        if min_dist > 0:
            inner_rect = QRectF(inner_left, inner_top,
                               inner_right - inner_left, inner_bottom - inner_top)
            inner_path = QPainterPath()
            inner_path.addRect(inner_rect)
            path = path.subtracted(inner_path)

        # Create directional sector
        sector_path = self._create_accurate_directional_sector(
            center_x, center_y, extend_distance, min_angle, max_angle,
            reading.direction == ExivaDirection.NORTH
        )

        # Intersect the square annulus with the directional sector
        if not sector_path.isEmpty():
            path = path.intersected(sector_path)

        return path

    def _create_accurate_directional_sector(self, center_x: float, center_y: float, radius: float,
                                           min_angle: float, max_angle: float, is_north: bool) -> QPainterPath:
        """Create a more accurate directional sector that properly handles the 2.42:1 ratio boundaries."""
        import math

        path = QPainterPath()

        # Convert angles to radians
        min_angle_rad = math.radians(min_angle)
        max_angle_rad = math.radians(max_angle)

        # Create a polygon that represents the directional sector
        # Start from center and create boundary lines

        if is_north:
            # North direction wraps around 0 degrees
            # Create a polygon that covers the north sector

            # Start from center
            points = [QPointF(center_x, center_y)]

            # Add points along the min_angle boundary line
            for distance in [radius * 0.1, radius]:
                x = center_x + distance * math.sin(min_angle_rad)
                y = center_y - distance * math.cos(min_angle_rad)
                points.append(QPointF(x, y))

            # Add points along the arc from min_angle to 360, then 0 to max_angle
            # Use more points for smoother boundary
            for angle_deg in range(int(min_angle), 360, 2):
                angle_rad = math.radians(angle_deg)
                x = center_x + radius * math.sin(angle_rad)
                y = center_y - radius * math.cos(angle_rad)
                points.append(QPointF(x, y))

            for angle_deg in range(0, int(max_angle) + 1, 2):
                angle_rad = math.radians(angle_deg)
                x = center_x + radius * math.sin(angle_rad)
                y = center_y - radius * math.cos(angle_rad)
                points.append(QPointF(x, y))

            # Add points along the max_angle boundary line (back to center)
            for distance in [radius, radius * 0.1]:
                x = center_x + distance * math.sin(max_angle_rad)
                y = center_y - distance * math.cos(max_angle_rad)
                points.append(QPointF(x, y))

        else:
            # Normal direction (not wrapping around 0)
            points = [QPointF(center_x, center_y)]

            # Add points along the min_angle boundary line
            for distance in [radius * 0.1, radius]:
                x = center_x + distance * math.sin(min_angle_rad)
                y = center_y - distance * math.cos(min_angle_rad)
                points.append(QPointF(x, y))

            # Add points along the arc from min_angle to max_angle
            for angle_deg in range(int(min_angle), int(max_angle) + 1, 2):
                angle_rad = math.radians(angle_deg)
                x = center_x + radius * math.sin(angle_rad)
                y = center_y - radius * math.cos(angle_rad)
                points.append(QPointF(x, y))

            # Add points along the max_angle boundary line (back to center)
            for distance in [radius, radius * 0.1]:
                x = center_x + distance * math.sin(max_angle_rad)
                y = center_y - distance * math.cos(max_angle_rad)
                points.append(QPointF(x, y))

        # Create polygon from points
        if len(points) > 2:
            polygon = QPolygonF(points)
            path.addPolygon(polygon)

        return path

    def _create_directional_sector_path(self, center_x: float, center_y: float, radius: float,
                                      min_angle: float, max_angle: float, is_north: bool) -> QPainterPath:
        """Create a directional sector path for the given angle range."""
        import math

        path = QPainterPath()

        # Convert angles to radians
        min_angle_rad = math.radians(min_angle)
        max_angle_rad = math.radians(max_angle)

        # Create a large sector that extends well beyond the needed area
        # Start from center
        path.moveTo(center_x, center_y)

        # Handle north direction wrapping
        if is_north:
            # North wraps around 0 degrees
            # Create two lines extending from center

            # Line 1: min_angle direction
            end_x1 = center_x + radius * math.sin(min_angle_rad)
            end_y1 = center_y - radius * math.cos(min_angle_rad)
            path.lineTo(end_x1, end_y1)

            # Arc from min_angle to 360, then from 0 to max_angle
            # For simplicity, create a large polygon that covers the north sector
            points = []

            # Add points along the north sector boundary
            for angle_deg in range(int(min_angle), 360, 5):
                angle_rad = math.radians(angle_deg)
                x = center_x + radius * math.sin(angle_rad)
                y = center_y - radius * math.cos(angle_rad)
                points.append(QPointF(x, y))

            for angle_deg in range(0, int(max_angle) + 1, 5):
                angle_rad = math.radians(angle_deg)
                x = center_x + radius * math.sin(angle_rad)
                y = center_y - radius * math.cos(angle_rad)
                points.append(QPointF(x, y))

            # Add all points to path
            for point in points:
                path.lineTo(point)

        else:
            # Normal direction (not wrapping)
            # Line 1: min_angle direction
            end_x1 = center_x + radius * math.sin(min_angle_rad)
            end_y1 = center_y - radius * math.cos(min_angle_rad)
            path.lineTo(end_x1, end_y1)

            # Arc from min_angle to max_angle
            points = []
            for angle_deg in range(int(min_angle), int(max_angle) + 1, 5):
                angle_rad = math.radians(angle_deg)
                x = center_x + radius * math.sin(angle_rad)
                y = center_y - radius * math.cos(angle_rad)
                points.append(QPointF(x, y))

            # Add all points to path
            for point in points:
                path.lineTo(point)

        # Close the path back to center
        path.closeSubpath()

        return path

    def paint(self, painter: QPainter, option, widget=None):
        """Optimized paint method using cached overlay with viewport culling."""
        if not self.readings:
            return

        # CRITICAL FIX: Generate cache immediately if needed to prevent floor change disappearing bug
        if not self.cache_valid or self.cached_overlay_pixmap is None:
            # For critical operations (like floor changes), generate cache immediately
            # For zoom operations, the cache should already be valid due to tier-based invalidation
            self.generate_cached_overlay()

            # If cache generation failed, don't render anything
            if self.cached_overlay_pixmap is None:
                return

        # Draw cached overlay if available
        if self.cached_overlay_pixmap and not self.cached_overlay_pixmap.isNull():
            painter.save()

            # Viewport culling: only draw the visible portion
            exposed_rect = option.exposedRect if option else self.scene_rect

            # Calculate intersection of exposed area with our scene rect
            visible_rect = exposed_rect.intersected(self.scene_rect)

            if not visible_rect.isEmpty():
                # Calculate source rectangle in pixmap coordinates
                pixmap_rect = QRectF(self.cached_overlay_pixmap.rect())

                # Map visible scene rect to pixmap coordinates
                scale_x = pixmap_rect.width() / self.scene_rect.width()
                scale_y = pixmap_rect.height() / self.scene_rect.height()

                source_x = (visible_rect.left() - self.scene_rect.left()) * scale_x
                source_y = (visible_rect.top() - self.scene_rect.top()) * scale_y
                source_width = visible_rect.width() * scale_x
                source_height = visible_rect.height() * scale_y

                source_rect = QRectF(source_x, source_y, source_width, source_height)

                # Only draw the visible portion
                painter.drawPixmap(visible_rect, self.cached_overlay_pixmap, source_rect)

            painter.restore()

    def _schedule_cache_generation(self):
        """Schedule cache generation - now immediate for reliability."""
        # CRITICAL FIX: Generate cache immediately to prevent disappearing overlay bugs
        # The tier-based caching system already prevents excessive regeneration during zoom
        self.generate_cached_overlay()

    def _generate_cache_async(self):
        """Generate cache asynchronously - now immediate for reliability."""
        self.generate_cached_overlay()

    def update_readings(self, readings: List[ExivaReading]):
        """Update the readings and refresh the overlay."""
        self.readings = readings
        self.cache_valid = False  # Invalidate cache
        self.update()  # Trigger a repaint

    def _calculate_dimmed_percentage(self):
        """Calculate the percentage of the map that is dimmed (not in intersection)."""
        if not self.scene_rect or self.scene_rect.isEmpty():
            self.dimmed_percentage = 0.0
            return

        # Calculate total scene area
        total_area = self.scene_rect.width() * self.scene_rect.height()

        if total_area <= 0:
            self.dimmed_percentage = 0.0
            return

        # Calculate undimmed area (intersection path area)
        undimmed_area = 0.0
        if self.last_intersection_path and not self.last_intersection_path.isEmpty():
            # Use bounding rect as approximation for performance
            # For more accuracy, we could use polygon area calculation
            bounding_rect = self.last_intersection_path.boundingRect()

            # Intersect with scene rect to ensure we don't count area outside the scene
            scene_qrect = QRectF(self.scene_rect)
            intersection_rect = bounding_rect.intersected(scene_qrect)

            if not intersection_rect.isEmpty():
                undimmed_area = intersection_rect.width() * intersection_rect.height()

        # Calculate dimmed percentage
        dimmed_area = total_area - undimmed_area
        self.dimmed_percentage = (dimmed_area / total_area) * 100.0

        # Clamp to 0-100 range
        self.dimmed_percentage = max(0.0, min(100.0, self.dimmed_percentage))

        logger.debug(f"Dimmed percentage: {self.dimmed_percentage:.2f}% (undimmed area: {undimmed_area:.0f}, total: {total_area:.0f})")

    def get_dimmed_percentage(self) -> float:
        """Get the current dimmed percentage of the map.

        Returns:
            Percentage of map that is dimmed (0-100)
        """
        return self.dimmed_percentage


class AreaGraphicsItem(QGraphicsPolygonItem):
    """Graphics item for rendering areas on the minimap."""

    def __init__(self, area_data: AreaData, global_transparency: float = 0.5, parent=None):
        super().__init__(parent)
        self.area_data = area_data
        self.global_transparency = global_transparency  # Use global transparency instead of per-area
        self.is_selected_area = False
        self.hover_effect = False

        # Set up the polygon from coordinates
        polygon = QPolygonF()
        for x, y in area_data.coordinates:
            polygon.append(QPointF(x, y))
        self.setPolygon(polygon)

        # Set z-value to render above map and crosshair ranges but below crosshair lines
        self.setZValue(998.5)

        # Enable hover and selection
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)

        # Default colors for difficulty levels
        self.default_colors = {
            MonsterDifficulty.HARMLESS.value: "#00FF00",    # Green
            MonsterDifficulty.TRIVIAL.value: "#7FFF00",     # Chartreuse
            MonsterDifficulty.EASY.value: "#FFFF00",        # Yellow
            MonsterDifficulty.MEDIUM.value: "#FFA500",      # Orange
            MonsterDifficulty.HARD.value: "#FF4500",        # Red-Orange
            MonsterDifficulty.CHALLENGING.value: "#FF0000", # Red
        }

        self.update_appearance()

    def set_global_transparency(self, transparency: float):
        """Update the global transparency value and refresh appearance."""
        self.global_transparency = transparency
        self.update_appearance()

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None):
        """Custom paint method to handle multi-monster stripe patterns."""
        if len(self.area_data.difficulty_levels) > 1:
            # Multi-monster area - use custom stripe pattern
            self.paint_striped_pattern(painter, option, widget)
        else:
            # Single monster area - use default rendering
            super().paint(painter, option, widget)

    def paint_striped_pattern(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None):
        """Paint diagonal stripes for multi-monster areas."""
        painter.save()

        # Get the polygon bounds
        polygon = self.polygon()
        bounding_rect = polygon.boundingRect()

        # Set up clipping to the polygon shape
        painter.setClipPath(self.shape())

        # Calculate stripe parameters
        stripe_width = 8.0  # Width of each stripe
        stripe_spacing = stripe_width * 2  # Total spacing between same-color stripes

        # Get colors for all difficulty levels
        colors = [QColor(self.default_colors[difficulty])
                 for difficulty in self.area_data.difficulty_levels
                 if difficulty in self.default_colors]

        if not colors:
            # Fallback to default rendering if no valid colors
            super().paint(painter, option, widget)
            painter.restore()
            return

        # Set transparency
        alpha = int(self.global_transparency * 255)
        for color in colors:
            color.setAlpha(alpha)

        # Draw diagonal stripes
        angle = 45  # 45-degree diagonal stripes

        # Calculate the diagonal extent needed to cover the entire bounding rect
        diagonal_length = max(bounding_rect.width(), bounding_rect.height()) * 1.5

        # Starting position for stripes
        start_x = bounding_rect.left() - diagonal_length
        start_y = bounding_rect.top()

        color_index = 0
        stripe_position = 0

        while stripe_position < diagonal_length * 2:
            # Calculate stripe endpoints
            x1 = start_x + stripe_position
            y1 = start_y
            x2 = x1 + diagonal_length
            y2 = y1 + diagonal_length

            # Set up pen for this stripe
            pen = QPen(colors[color_index % len(colors)])
            pen.setWidthF(stripe_width)
            pen.setCosmetic(True)  # Keep consistent width regardless of zoom
            painter.setPen(pen)

            # Draw the stripe line
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

            # Move to next stripe and color
            stripe_position += stripe_width
            color_index += 1

        # Draw the outline
        self.paint_outline(painter)

        painter.restore()

    def paint_outline(self, painter: QPainter):
        """Paint the outline for the area."""
        # Determine outline color (use first difficulty color)
        if self.area_data.difficulty_levels and self.area_data.difficulty_levels[0] in self.default_colors:
            outline_color = QColor(self.default_colors[self.area_data.difficulty_levels[0]])
        else:
            outline_color = QColor("#808080")

        alpha = int(self.global_transparency * 255)
        outline_color.setAlpha(min(255, alpha + 100))  # Make outline more opaque

        pen_width = 2.0 if self.is_selected_area else 1.0
        if self.hover_effect:
            pen_width += 1.0

        pen = QPen(outline_color)
        pen.setWidthF(pen_width)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))

        painter.drawPolygon(self.polygon())

    def update_appearance(self):
        """Update the visual appearance based on area data and state."""
        # For single monster areas, set up brush and pen for default rendering
        # Multi-monster areas will be handled by the custom paint method
        if len(self.area_data.difficulty_levels) == 1:
            # Single monster type - use solid color
            alpha = int(self.global_transparency * 255)
            color = QColor(self.default_colors[self.area_data.difficulty_levels[0]])
            color.setAlpha(alpha)
            brush = QBrush(color)
            self.setBrush(brush)

            # Create pen for outline
            outline_color = QColor(self.default_colors[self.area_data.difficulty_levels[0]])
            outline_color.setAlpha(min(255, alpha + 100))  # Make outline more opaque

            pen_width = 2.0 if self.is_selected_area else 1.0
            if self.hover_effect:
                pen_width += 1.0

            pen = QPen(outline_color)
            pen.setWidthF(pen_width)
            pen.setCosmetic(True)  # Keep consistent width regardless of zoom
            self.setPen(pen)
        elif len(self.area_data.difficulty_levels) == 0:
            # No difficulty levels - use default gray
            alpha = int(self.global_transparency * 255)
            color = QColor("#808080")
            color.setAlpha(alpha)
            brush = QBrush(color)
            self.setBrush(brush)
            outline_color = QColor("#808080")
            outline_color.setAlpha(min(255, alpha + 100))

            pen_width = 2.0 if self.is_selected_area else 1.0
            if self.hover_effect:
                pen_width += 1.0

            pen = QPen(outline_color)
            pen.setWidthF(pen_width)
            pen.setCosmetic(True)
            self.setPen(pen)
        else:
            # Multiple monster types - will be handled by custom paint method
            # Set transparent brush so default rendering doesn't interfere
            self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            self.setPen(QPen(Qt.PenStyle.NoPen))

        # Trigger a repaint to apply changes
        self.update()

    def set_selected_area(self, selected: bool):
        """Set the selection state of this area."""
        self.is_selected_area = selected
        self.update_appearance()

    def hoverEnterEvent(self, event):
        """Handle mouse hover enter."""
        self.hover_effect = True
        self.update_appearance()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Handle mouse hover leave."""
        self.hover_effect = False
        self.update_appearance()
        super().hoverLeaveEvent(event)

    def update_from_area_data(self, area_data: AreaData):
        """Update the graphics item from new area data."""
        self.area_data = area_data

        # Update polygon
        polygon = QPolygonF()
        for x, y in area_data.coordinates:
            polygon.append(QPointF(x, y))
        self.setPolygon(polygon)

        self.update_appearance()


class AreaEditingMode(Enum):
    """Area editing modes."""
    DISABLED = "disabled"
    POLYGON = "polygon"


class MinimapGraphicsView(QGraphicsView):
    """Custom QGraphicsView for minimap display with zoom, pan, crosshair, and area editing capabilities."""

    viewTransformed = pyqtSignal(float, QPointF)
    areaCreated = pyqtSignal(AreaData)
    areaSelected = pyqtSignal(str)  # area_id
    crosshairPlaced = pyqtSignal(QPointF)  # Emitted when crosshair is placed
    crosshairCleared = pyqtSignal()  # Emitted when crosshair is cleared
    difficultyChangeRestart = pyqtSignal(str, str)  # Emitted when session restarts due to difficulty change (old_difficulty, new_difficulty)
    fullDimRestart = pyqtSignal(float)  # Emitted when session restarts due to 100% dimming (dimmed_percentage)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 20.0
        self.zoom_step = 1.15

        self.pan_enabled = True
        self.last_pan_point = QPointF()
        self.is_panning = False

        self.floor_camera_positions = {}
        self.current_floor_id = None

        self.crosshair_diagonals: List[Optional[QGraphicsLineItem]] = [None] * 8
        self.crosshair_center_square: Optional[QGraphicsRectItem] = None
        self.crosshair_inner_range: Optional[QGraphicsRectItem] = None
        self.crosshair_outer_range: Optional[QGraphicsRectItem] = None
        self.global_crosshair_position: Optional[QPointF] = None

        # Area editing functionality
        self.area_editing_mode = AreaEditingMode.DISABLED
        self.area_graphics_items: Dict[str, AreaGraphicsItem] = {}
        self.selected_area_id: Optional[str] = None

        # Crosshair visibility state for area editing mode
        self.crosshairs_hidden_for_area_editing = False
        self.saved_crosshair_position_for_area_editing: Optional[QPointF] = None

        # Area creation state
        self.creating_area = False
        self.area_creation_points: List[QPointF] = []
        self.temp_area_item: Optional[QGraphicsPolygonItem] = None
        self.area_data_manager = AreaDataManager()

        # Exiva session management
        self.current_exiva_session: Optional[ExivaSession] = None
        self.exiva_overlay_item: Optional[ExivaOverlayItem] = None

        # Zoom debouncing for performance optimization
        self.zoom_debounce_timer = QTimer()
        self.zoom_debounce_timer.setSingleShot(True)
        self.zoom_debounce_timer.timeout.connect(self._apply_debounced_zoom_update)
        self.pending_zoom_factor = None
        
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel events for zooming with performance optimization."""
        # Reduce sensitivity for smoother zooming
        delta = event.angleDelta().y()

        # Only process significant wheel movements to reduce update frequency
        if abs(delta) < 60:  # Ignore very small wheel movements
            return

        zoom_in = delta > 0
        self.zoom(zoom_in, event.position())

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events for panning, crosshair placement, and area editing."""
        mouse_pos = event.position()
        mouse_point = QPointF(mouse_pos.x(), mouse_pos.y())
        scene_pos = self.mapToScene(mouse_point.toPoint())

        # Handle middle mouse button panning (works regardless of area editing mode)
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = True
            self.last_pan_point = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        # Handle area editing mode
        if self.area_editing_mode != AreaEditingMode.DISABLED:
            if event.button() == Qt.MouseButton.LeftButton:
                # Check if clicking on an existing area first
                item = self.itemAt(mouse_point.toPoint())
                if isinstance(item, AreaGraphicsItem):
                    # Clicking on an existing area - select it
                    self.select_area(item.area_data.area_id)
                    return
                elif self.selected_area_id is not None:
                    # Clicking on empty space while an area is selected - deselect it
                    # Only deselect if not currently creating an area
                    if not self.creating_area:
                        self.deselect_area()
                        return

                # Handle area creation (only if no area was selected and deselected)
                self.handle_area_creation_click(scene_pos)
                return
            elif event.button() == Qt.MouseButton.RightButton:
                self.finish_area_creation()
                return

        # Handle area selection (when not in area editing mode)
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(mouse_point.toPoint())
            if isinstance(item, AreaGraphicsItem):
                self.select_area(item.area_data.area_id)
                return
            elif self.selected_area_id is not None:
                # Clicking on empty space while an area is selected - deselect it
                self.deselect_area()
                return

        # Default behavior for panning and crosshairs
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = True
            self.last_pan_point = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.RightButton:
            snapped_pos = QPointF(
                math.floor(scene_pos.x()) + 0.5,
                math.floor(scene_pos.y()) + 0.5
            )
            self.place_crosshairs(snapped_pos)
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events for panning."""
        if self.is_panning:
            delta = event.position() - self.last_pan_point
            self.last_pan_point = event.position()

            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events."""
        if event.button() == Qt.MouseButton.LeftButton or event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.emit_view_transformed()
        else:
            super().mouseReleaseEvent(event)

    def zoom(self, zoom_in: bool, center_point: Optional[QPointF] = None):
        """Zoom the view in or out."""
        factor = self.zoom_step if zoom_in else 1.0 / self.zoom_step
        new_zoom = self.zoom_factor * factor

        if new_zoom < self.min_zoom or new_zoom > self.max_zoom:
            return

        self.zoom_factor = new_zoom

        if center_point:
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
            center_pointf = QPointF(center_point.x(), center_point.y())
            old_pos = self.mapToScene(center_pointf.toPoint())
            self.scale(factor, factor)
            new_pos = self.mapToScene(center_pointf.toPoint())
            delta = new_pos - old_pos
            self.translate(delta.x(), delta.y())
        else:
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
            self.scale(factor, factor)

        self.emit_view_transformed()
        self.update_crosshair_appearance()

    def zoom_to_factor(self, target_zoom: float):
        """Set zoom to a specific factor using absolute transformation."""
        if target_zoom < self.min_zoom or target_zoom > self.max_zoom:
            return

        self.resetTransform()
        self.scale(target_zoom, target_zoom)
        self.zoom_factor = target_zoom
        self.emit_view_transformed()
        self.update_crosshair_appearance()

    def fit_in_view_with_margin(self, margin_percent: float = 0.1):
        """Fit the scene in view with a margin."""
        if self.scene():
            scene_rect = self.scene().itemsBoundingRect()
            if not scene_rect.isEmpty():
                margin_x = scene_rect.width() * margin_percent
                margin_y = scene_rect.height() * margin_percent
                scene_rect.adjust(-margin_x, -margin_y, margin_x, margin_y)

                self.fitInView(scene_rect, Qt.AspectRatioMode.KeepAspectRatio)

                transform = self.transform()
                self.zoom_factor = transform.m11()
                self.emit_view_transformed()
                self.update_crosshair_appearance()

    def save_camera_position(self, floor_id=None):
        """Save current camera position (zoom is global)."""
        if floor_id is None:
            floor_id = self.current_floor_id

        if floor_id is None:
            logger.warning("Cannot save camera position - no floor ID specified")
            return

        viewport_center = self.viewport().rect().center()
        scene_center = self.mapToScene(viewport_center)

        if not scene_center.isNull() and self.scene():
            self.floor_camera_positions[floor_id] = {
                'center': QPointF(scene_center)
            }
        else:
            logger.warning("Could not save camera position - invalid scene coordinates")
    
    def restore_camera_position(self, floor_id=None):
        """Restore previously saved camera position (keep current global zoom)."""
        if floor_id is None:
            floor_id = self.current_floor_id

        if floor_id is None or floor_id not in self.floor_camera_positions:
            logger.info(f"No saved camera position for floor {floor_id}")
            return

        saved_data = self.floor_camera_positions[floor_id]
        saved_center = QPointF(saved_data['center'])

        scene_rect_before = self.scene().sceneRect() if self.scene() else QRectF()
        center_before = self.mapToScene(self.viewport().rect().center())

        viewport_rect = self.viewport().rect()
        target_viewport_center = self.mapFromScene(saved_center)

        current_viewport_center = viewport_rect.center()
        offset_x = target_viewport_center.x() - current_viewport_center.x()
        offset_y = target_viewport_center.y() - current_viewport_center.y()

        h_scroll = self.horizontalScrollBar()
        v_scroll = self.verticalScrollBar()
        h_scroll.setValue(h_scroll.value() + int(offset_x))
        v_scroll.setValue(v_scroll.value() + int(offset_y))

        center_after = self.mapToScene(self.viewport().rect().center())
        logger.info(f"RESTORE Floor {floor_id} - Target: ({saved_center.x():.2f}, {saved_center.y():.2f}), "
                   f"Before: ({center_before.x():.2f}, {center_before.y():.2f}), "
                   f"After: ({center_after.x():.2f}, {center_after.y():.2f}), "
                   f"Scene: {scene_rect_before.width():.0f}x{scene_rect_before.height():.0f}, "
                   f"Zoom: {self.zoom_factor:.4f} (GLOBAL - unchanged)")

        self.emit_view_transformed()
    
    def emit_view_transformed(self):
        """Emit signal when view is transformed."""
        center = self.mapToScene(self.viewport().rect().center())
        self.viewTransformed.emit(self.zoom_factor, center)

        # Debounce Exiva overlay zoom updates for better performance
        self._debounce_zoom_update(self.zoom_factor)

    def _debounce_zoom_update(self, zoom_factor: float):
        """Debounce zoom updates to prevent excessive overlay regeneration."""
        self.pending_zoom_factor = zoom_factor

        # Stop any existing timer and start a new one
        self.zoom_debounce_timer.stop()
        self.zoom_debounce_timer.start(150)  # 150ms debounce delay

    def _apply_debounced_zoom_update(self):
        """Apply the debounced zoom update to the overlay."""
        if self.exiva_overlay_item and self.pending_zoom_factor is not None:
            # Always update zoom level - the tier system will handle cache invalidation intelligently
            self.exiva_overlay_item.update_zoom_level(self.pending_zoom_factor)
            self.pending_zoom_factor = None

    def calculate_crosshair_pen_width(self, base_width: float) -> float:
        """Calculate appropriate pen width based on current zoom level."""
        min_visible_width = 1.0

        if self.zoom_factor >= 1.0:
            adjusted_base = max(base_width * 2.0, min_visible_width)
            return adjusted_base
        else:
            scale_factor = max(1.0 / self.zoom_factor, 1.0)
            scale_factor = min(scale_factor, 8.0)
            scaled_width = base_width * scale_factor
            return max(scaled_width, min_visible_width)

    def get_exiva_diagonal_angles(self) -> List[float]:
        """Get the 8 diagonal boundary angles used by Tibia's Exiva spell."""
        return get_boundary_angles()

    def get_crosshair_color(self) -> QColor:
        """Get the appropriate crosshair color based on current floor."""
        if self.current_floor_id == 7:
            return QColor(0, 0, 0, 255)  # Black for floor 07
        else:
            return QColor(255, 255, 255, 255)  # White for all other floors

    def calculate_boundary_line_endpoints(self, center_pos: QPointF, angle_deg: float) -> tuple[QPointF, QPointF]:
        """Calculate the endpoints of a diagonal line that extends to the scene boundaries."""
        if not self.scene():
            angle_rad = math.radians(angle_deg)
            half_length = 100.0
            start_x = center_pos.x() - half_length * math.sin(angle_rad)
            start_y = center_pos.y() + half_length * math.cos(angle_rad)
            end_x = center_pos.x() + half_length * math.sin(angle_rad)
            end_y = center_pos.y() - half_length * math.cos(angle_rad)
            return QPointF(start_x, start_y), QPointF(end_x, end_y)

        scene_rect = self.scene().sceneRect()
        angle_rad = math.radians(angle_deg)
        dx = math.sin(angle_rad)
        dy = -math.cos(angle_rad)

        intersections = []

        if dx != 0:
            t = (scene_rect.left() - center_pos.x()) / dx
            y = center_pos.y() + t * dy
            if scene_rect.top() <= y <= scene_rect.bottom():
                intersections.append(QPointF(scene_rect.left(), y))

        if dx != 0:
            t = (scene_rect.right() - center_pos.x()) / dx
            y = center_pos.y() + t * dy
            if scene_rect.top() <= y <= scene_rect.bottom():
                intersections.append(QPointF(scene_rect.right(), y))

        if dy != 0:
            t = (scene_rect.top() - center_pos.y()) / dy
            x = center_pos.x() + t * dx
            if scene_rect.left() <= x <= scene_rect.right():
                intersections.append(QPointF(x, scene_rect.top()))

        if dy != 0:
            t = (scene_rect.bottom() - center_pos.y()) / dy
            x = center_pos.x() + t * dx
            if scene_rect.left() <= x <= scene_rect.right():
                intersections.append(QPointF(x, scene_rect.bottom()))

        unique_intersections = []
        for point in intersections:
            is_duplicate = False
            for existing in unique_intersections:
                if abs(point.x() - existing.x()) < 0.1 and abs(point.y() - existing.y()) < 0.1:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_intersections.append(point)

        if len(unique_intersections) >= 2:
            def get_t_value(point):
                if abs(dx) > abs(dy):
                    return (point.x() - center_pos.x()) / dx if dx != 0 else 0
                else:
                    return (point.y() - center_pos.y()) / dy if dy != 0 else 0

            unique_intersections.sort(key=get_t_value)
            return unique_intersections[0], unique_intersections[-1]

        max_dimension = max(scene_rect.width(), scene_rect.height())
        half_length = max_dimension
        start_x = center_pos.x() - half_length * dx
        start_y = center_pos.y() - half_length * dy
        end_x = center_pos.x() + half_length * dx
        end_y = center_pos.y() + half_length * dy

        return QPointF(start_x, start_y), QPointF(end_x, end_y)

    def calculate_diagonal_line_length(self) -> float:
        """Calculate the length of diagonal lines based on current zoom level."""
        base_length = 100.0

        if self.zoom_factor >= 1.0:
            return base_length * min(2.0, self.zoom_factor)
        else:
            scale_factor = max(1.0, 1.0 / self.zoom_factor)
            scale_factor = min(scale_factor, 10.0)
            return base_length * scale_factor

    def calculate_exiva_range_size(self, range_squares: int) -> float:
        """Calculate the size of an Exiva range rectangle in scene coordinates."""
        return float((range_squares * 2) + 1)

    def get_exiva_distance_ranges(self) -> Dict[ExivaRange, Tuple[float, float]]:
        """Get distance ranges for Exiva spell readings in scene coordinates (accurate Tibia mechanics)."""
        return EXIVA_DISTANCE_RANGES

    def get_direction_angle_range(self, direction: ExivaDirection) -> Tuple[float, float]:
        """Get the angle range (in degrees) for a given Exiva direction."""
        return get_direction_ranges()[direction]

    def calculate_distance_to_point(self, from_pos: QPointF, to_pos: QPointF) -> float:
        """Calculate the distance between two points using Tibia's square-based distance system."""
        dx = abs(to_pos.x() - from_pos.x())
        dy = abs(to_pos.y() - from_pos.y())
        # Tibia uses square-based distance (Chebyshev distance), not Euclidean
        return max(dx, dy)

    def calculate_angle_to_point(self, from_pos: QPointF, to_pos: QPointF) -> float:
        """Calculate the angle from one point to another in degrees (0-360)."""
        return calculate_angle_between_points(from_pos.x(), from_pos.y(), to_pos.x(), to_pos.y())

    def is_angle_in_direction_range(self, angle: float, direction: ExivaDirection) -> bool:
        """Check if an angle falls within the range for a given direction."""
        min_angle, max_angle = self.get_direction_angle_range(direction)

        # Handle the special case of NORTH direction which wraps around 0 degrees
        if direction == ExivaDirection.NORTH:
            return angle >= min_angle or angle <= max_angle
        else:
            return min_angle <= angle <= max_angle

    def is_distance_in_range(self, distance: float, exiva_range: ExivaRange) -> bool:
        """Check if a distance falls within the range for a given Exiva range."""
        min_dist, max_dist = self.get_exiva_distance_ranges()[exiva_range]
        return min_dist <= distance <= max_dist

    def point_matches_exiva_reading(self, point: QPointF, reading: ExivaReading) -> bool:
        """Check if a point matches the given Exiva reading."""
        crosshair_pos = QPointF(reading.crosshair_position[0], reading.crosshair_position[1])

        # Calculate distance and angle from crosshair to point
        distance = self.calculate_distance_to_point(crosshair_pos, point)
        angle = self.calculate_angle_to_point(crosshair_pos, point)

        # Check if both distance and direction match
        distance_matches = self.is_distance_in_range(distance, reading.distance)
        direction_matches = self.is_angle_in_direction_range(angle, reading.direction)

        return distance_matches and direction_matches

    def area_matches_exiva_readings(self, area_polygon: QPolygonF, readings: List[ExivaReading]) -> bool:
        """Check if any part of an area polygon matches ALL given Exiva readings."""
        if not readings:
            return True  # No readings means no restrictions

        # Sample points within the polygon to test
        bounding_rect = area_polygon.boundingRect()
        sample_points = []

        # Create a grid of sample points within the bounding rectangle
        step_size = 5.0  # Sample every 5 pixels
        x = bounding_rect.left()
        while x <= bounding_rect.right():
            y = bounding_rect.top()
            while y <= bounding_rect.bottom():
                point = QPointF(x, y)
                if area_polygon.containsPoint(point, Qt.FillRule.OddEvenFill):
                    sample_points.append(point)
                y += step_size
            x += step_size

        # If no sample points found, test the polygon vertices
        if not sample_points:
            sample_points = [area_polygon.at(i) for i in range(area_polygon.size())]

        # Check if any sample point matches ALL readings
        for point in sample_points:
            matches_all = True
            for reading in readings:
                if not self.point_matches_exiva_reading(point, reading):
                    matches_all = False
                    break

            if matches_all:
                return True  # Found at least one point that matches all readings

        return False  # No point in the area matches all readings

    def get_scene_areas_matching_readings(self, readings: List[ExivaReading]) -> List[QRectF]:
        """Get rectangular areas of the scene that match all given Exiva readings."""
        if not readings or not self.scene():
            return []

        scene_rect = self.scene().sceneRect()
        matching_areas = []

        # Create a grid of test rectangles across the scene
        grid_size = 20.0  # Size of each test rectangle
        x = scene_rect.left()
        while x < scene_rect.right():
            y = scene_rect.top()
            while y < scene_rect.bottom():
                test_rect = QRectF(x, y, grid_size, grid_size)
                test_polygon = QPolygonF(test_rect)

                if self.area_matches_exiva_readings(test_polygon, readings):
                    matching_areas.append(test_rect)

                y += grid_size
            x += grid_size

        return matching_areas

    def place_crosshairs(self, scene_pos: QPointF):
        """Place Exiva-style crosshairs with 8-directional lines and distance ranges."""
        if not self.scene():
            return

        self.remove_crosshairs()
        self.global_crosshair_position = QPointF(scene_pos)

        # Emit signal that crosshair was placed
        self.crosshairPlaced.emit(QPointF(scene_pos))

        # If in area editing mode, save position but don't display crosshairs
        if self.area_editing_mode != AreaEditingMode.DISABLED:
            self.saved_crosshair_position_for_area_editing = QPointF(scene_pos)
            self.crosshairs_hidden_for_area_editing = True
            logger.info(f"Crosshair position saved at ({scene_pos.x():.2f}, {scene_pos.y():.2f}) but hidden due to area editing mode")
            return

        crosshair_width = self.calculate_crosshair_pen_width(0.3)
        angles = self.get_exiva_diagonal_angles()

        crosshair_color = self.get_crosshair_color()
        crosshair_pen = QPen(crosshair_color)
        crosshair_pen.setWidthF(crosshair_width)
        crosshair_pen.setStyle(Qt.PenStyle.SolidLine)
        crosshair_pen.setCosmetic(True)

        for i, angle_deg in enumerate(angles):
            start_point, end_point = self.calculate_boundary_line_endpoints(scene_pos, angle_deg)

            diagonal_line = QGraphicsLineItem(start_point.x(), start_point.y(), end_point.x(), end_point.y())
            diagonal_line.setPen(crosshair_pen)
            diagonal_line.setZValue(999)
            self.scene().addItem(diagonal_line)
            self.crosshair_diagonals[i] = diagonal_line

        self.create_exiva_ranges(scene_pos, crosshair_width)

        base_square_size = 1.0
        square_size = base_square_size * max(1.0, min(2.0, 1.0 / self.zoom_factor)) if self.zoom_factor < 1.0 else base_square_size

        self.crosshair_center_square = QGraphicsRectItem(
            scene_pos.x() - square_size/2, scene_pos.y() - square_size/2,
            square_size, square_size
        )
        self.crosshair_center_square.setPen(crosshair_pen)
        self.crosshair_center_square.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.crosshair_center_square.setZValue(1001)
        self.scene().addItem(self.crosshair_center_square)

        logger.info(f"Exiva crosshairs placed at ({scene_pos.x():.2f}, {scene_pos.y():.2f}) on floor {self.current_floor_id} with 8 diagonal boundary lines and distance ranges")

    def create_exiva_ranges(self, center_pos: QPointF, range_width: float):
        """Create Exiva spell distance range overlays."""
        crosshair_color = self.get_crosshair_color()
        range_pen = QPen(crosshair_color)
        range_pen.setWidthF(range_width)
        range_pen.setStyle(Qt.PenStyle.SolidLine)
        range_pen.setCosmetic(True)

        inner_size = self.calculate_exiva_range_size(100)

        self.crosshair_inner_range = QGraphicsRectItem(
            center_pos.x() - inner_size/2, center_pos.y() - inner_size/2,
            inner_size, inner_size
        )
        self.crosshair_inner_range.setPen(range_pen)
        self.crosshair_inner_range.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.crosshair_inner_range.setZValue(998)
        self.scene().addItem(self.crosshair_inner_range)

        outer_size = self.calculate_exiva_range_size(250)

        self.crosshair_outer_range = QGraphicsRectItem(
            center_pos.x() - outer_size/2, center_pos.y() - outer_size/2,
            outer_size, outer_size
        )
        self.crosshair_outer_range.setPen(range_pen)
        self.crosshair_outer_range.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.crosshair_outer_range.setZValue(997)
        self.scene().addItem(self.crosshair_outer_range)

    def update_crosshair_appearance(self):
        """Update the appearance of existing Exiva crosshairs based on current zoom level."""
        if not self.scene() or self.global_crosshair_position is None:
            return

        # Don't update crosshairs if they are hidden for area editing
        if self.crosshairs_hidden_for_area_editing:
            return

        has_diagonals = any(line is not None for line in self.crosshair_diagonals)
        if not (has_diagonals or self.crosshair_center_square):
            return

        crosshair_width = self.calculate_crosshair_pen_width(0.3)

        crosshair_color = self.get_crosshair_color()
        crosshair_pen = QPen(crosshair_color)
        crosshair_pen.setWidthF(crosshair_width)
        crosshair_pen.setStyle(Qt.PenStyle.SolidLine)
        crosshair_pen.setCosmetic(True)

        angles = self.get_exiva_diagonal_angles()
        scene_pos = self.global_crosshair_position

        for diagonal_line, angle_deg in zip(self.crosshair_diagonals, angles):
            if diagonal_line is not None:
                diagonal_line.setPen(crosshair_pen)
                start_point, end_point = self.calculate_boundary_line_endpoints(scene_pos, angle_deg)
                diagonal_line.setLine(start_point.x(), start_point.y(), end_point.x(), end_point.y())

        if self.crosshair_inner_range:
            self.crosshair_inner_range.setPen(crosshair_pen)

        if self.crosshair_outer_range:
            self.crosshair_outer_range.setPen(crosshair_pen)

        if self.crosshair_center_square:
            self.crosshair_center_square.setPen(crosshair_pen)

            base_square_size = 1.0
            square_size = base_square_size * max(1.0, min(2.0, 1.0 / self.zoom_factor)) if self.zoom_factor < 1.0 else base_square_size

            self.crosshair_center_square.setRect(
                scene_pos.x() - square_size/2, scene_pos.y() - square_size/2,
                square_size, square_size
            )

        logger.debug(f"Updated Exiva crosshair appearance: crosshair width {crosshair_width:.3f}, lines extend to scene boundaries")

    def remove_crosshairs(self):
        """Remove existing Exiva crosshairs from the scene."""
        for i in range(8):
            if self.crosshair_diagonals[i] and self.scene():
                try:
                    self.scene().removeItem(self.crosshair_diagonals[i])
                except RuntimeError:
                    pass
                self.crosshair_diagonals[i] = None

        if self.crosshair_center_square and self.scene():
            try:
                self.scene().removeItem(self.crosshair_center_square)
            except RuntimeError:
                pass
            self.crosshair_center_square = None

        if self.crosshair_inner_range and self.scene():
            try:
                self.scene().removeItem(self.crosshair_inner_range)
            except RuntimeError:
                pass
            self.crosshair_inner_range = None

        if self.crosshair_outer_range and self.scene():
            try:
                self.scene().removeItem(self.crosshair_outer_range)
            except RuntimeError:
                pass
            self.crosshair_outer_range = None

        self.global_crosshair_position = None

        # Emit signal that crosshair was cleared
        self.crosshairCleared.emit()

        # Reset area editing crosshair state
        self.crosshairs_hidden_for_area_editing = False
        self.saved_crosshair_position_for_area_editing = None

    def restore_crosshairs(self, _floor_id: int = None):
        """Restore global crosshairs if they exist."""
        if not self.scene():
            logger.warning("Cannot restore crosshairs - no scene available")
            return

        if self.global_crosshair_position is not None:
            # If in area editing mode, don't display crosshairs but ensure state is correct
            if self.area_editing_mode != AreaEditingMode.DISABLED:
                self.saved_crosshair_position_for_area_editing = QPointF(self.global_crosshair_position)
                self.crosshairs_hidden_for_area_editing = True
                logger.info(f"Crosshairs position preserved but hidden due to area editing mode on floor {_floor_id}")
            else:
                self.place_crosshairs(self.global_crosshair_position)

    def save_crosshair_position(self, _floor_id: int = None):
        """Save current global crosshair position (no-op since crosshairs are already global)."""
        pass

    def clear_all_crosshairs(self):
        """Clear global crosshairs."""
        self.remove_crosshairs()

    def hide_crosshairs_for_area_editing(self):
        """Hide crosshairs when entering area editing mode."""
        if self.global_crosshair_position is not None and not self.crosshairs_hidden_for_area_editing:
            # Save the current crosshair position
            self.saved_crosshair_position_for_area_editing = QPointF(self.global_crosshair_position)
            # Remove crosshairs from scene but don't clear the global position
            self._remove_crosshair_graphics_items()
            self.crosshairs_hidden_for_area_editing = True
            logger.info("Crosshairs hidden for area editing mode")

    def show_crosshairs_for_area_editing(self):
        """Show crosshairs when exiting area editing mode."""
        if self.crosshairs_hidden_for_area_editing and self.saved_crosshair_position_for_area_editing is not None:
            # Restore crosshairs at the saved position
            self.place_crosshairs(self.saved_crosshair_position_for_area_editing)
            self.crosshairs_hidden_for_area_editing = False
            self.saved_crosshair_position_for_area_editing = None
            logger.info("Crosshairs restored after area editing mode")

    def _remove_crosshair_graphics_items(self):
        """Remove crosshair graphics items from scene without clearing position state."""
        for i in range(8):
            if self.crosshair_diagonals[i] and self.scene():
                try:
                    self.scene().removeItem(self.crosshair_diagonals[i])
                except RuntimeError:
                    pass
                self.crosshair_diagonals[i] = None

        if self.crosshair_center_square and self.scene():
            try:
                self.scene().removeItem(self.crosshair_center_square)
            except RuntimeError:
                pass
            self.crosshair_center_square = None

        if self.crosshair_inner_range and self.scene():
            try:
                self.scene().removeItem(self.crosshair_inner_range)
            except RuntimeError:
                pass
            self.crosshair_inner_range = None

        if self.crosshair_outer_range and self.scene():
            try:
                self.scene().removeItem(self.crosshair_outer_range)
            except RuntimeError:
                pass
            self.crosshair_outer_range = None

    # Area editing methods
    def set_area_editing_mode(self, mode: AreaEditingMode):
        """Set the area editing mode."""
        previous_mode = self.area_editing_mode
        self.area_editing_mode = mode

        # Handle crosshair visibility based on mode change
        if previous_mode == AreaEditingMode.DISABLED and mode != AreaEditingMode.DISABLED:
            # Entering area editing mode - hide crosshairs
            self.hide_crosshairs_for_area_editing()
        elif previous_mode != AreaEditingMode.DISABLED and mode == AreaEditingMode.DISABLED:
            # Exiting area editing mode - show crosshairs
            self.show_crosshairs_for_area_editing()

        if mode == AreaEditingMode.DISABLED:
            self.finish_area_creation()
        logger.info(f"Area editing mode set to: {mode.value}")

    def handle_area_creation_click(self, scene_pos: QPointF):
        """Handle mouse clicks during area creation."""
        if self.area_editing_mode == AreaEditingMode.POLYGON:
            if not self.creating_area:
                # Start polygon creation
                self.creating_area = True
                self.area_creation_points = [scene_pos]
                self.create_temp_area_preview()
            else:
                # Add point to polygon
                self.area_creation_points.append(scene_pos)
                self.update_temp_area_preview()

    def create_temp_area_preview(self):
        """Create a temporary preview of the area being created."""
        if self.temp_area_item:
            self.scene().removeItem(self.temp_area_item)

        polygon = QPolygonF()
        for point in self.area_creation_points:
            polygon.append(point)

        self.temp_area_item = QGraphicsPolygonItem(polygon)
        pen = QPen(QColor("#FF0000"))
        pen.setWidthF(2.0)
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self.temp_area_item.setPen(pen)
        self.temp_area_item.setBrush(QBrush(QColor(255, 0, 0, 50)))
        self.temp_area_item.setZValue(600)
        self.scene().addItem(self.temp_area_item)

    def update_temp_area_preview(self):
        """Update the temporary area preview."""
        if self.temp_area_item and self.area_creation_points:
            polygon = QPolygonF()
            for point in self.area_creation_points:
                polygon.append(point)
            self.temp_area_item.setPolygon(polygon)

    def finish_area_creation(self):
        """Finish creating an area and add it to the data."""
        if not self.creating_area or len(self.area_creation_points) < 2:
            self.cancel_area_creation()
            return

        # Create area data
        area_id = str(uuid.uuid4())
        coordinates = [(point.x(), point.y()) for point in self.area_creation_points]



        import time
        default_colors = self.area_data_manager.get_default_colors()
        default_difficulty = MonsterDifficulty.MEDIUM.value
        default_color = default_colors[default_difficulty]

        area_data = AreaData(
            area_id=area_id,
            floor=self.current_floor_id,
            name=f"Area {len(self.area_graphics_items) + 1}",
            coordinates=coordinates,
            difficulty_levels=[default_difficulty],
            color=default_color,
            transparency=0.5,
            metadata={},
            route='',
            created_timestamp=time.time(),
            modified_timestamp=time.time()
        )

        # Add to data manager
        self.area_data_manager.add_area(area_data)

        # Create graphics item
        self.add_area_graphics_item(area_data)

        # Emit signal
        self.areaCreated.emit(area_data)

        # Clean up
        self.cancel_area_creation()
        logger.info(f"Created new area: {area_data.name} on floor {area_data.floor}")

    def undo_last_area_point(self):
        """Remove the last point from the current area being created."""
        if not self.creating_area or len(self.area_creation_points) <= 1:
            # Can't undo if not creating an area or if only one point remains
            return False

        # Remove the last point
        self.area_creation_points.pop()

        # Update the preview
        if len(self.area_creation_points) == 0:
            # No points left, cancel area creation
            self.cancel_area_creation()
        else:
            # Update the preview with remaining points
            self.update_temp_area_preview()

        logger.info(f"Undid last area point. Remaining points: {len(self.area_creation_points)}")
        return True

    def cancel_area_creation(self):
        """Cancel area creation and clean up."""
        self.creating_area = False
        self.area_creation_points = []

        if self.temp_area_item:
            self.scene().removeItem(self.temp_area_item)
            self.temp_area_item = None

    def add_area_graphics_item(self, area_data: AreaData):
        """Add a graphics item for an area."""
        graphics_item = AreaGraphicsItem(area_data, self.area_data_manager.global_transparency)
        self.area_graphics_items[area_data.area_id] = graphics_item
        self.scene().addItem(graphics_item)

    def update_global_transparency(self, transparency: float):
        """Update the global transparency for all area graphics items."""
        self.area_data_manager.global_transparency = transparency

        # Update all existing area graphics items
        for graphics_item in self.area_graphics_items.values():
            graphics_item.set_global_transparency(transparency)

    def remove_area_graphics_item(self, area_id: str):
        """Remove a graphics item for an area."""
        if area_id in self.area_graphics_items:
            graphics_item = self.area_graphics_items[area_id]
            self.scene().removeItem(graphics_item)
            del self.area_graphics_items[area_id]

    def select_area(self, area_id: str):
        """Select an area."""
        # Deselect previous area
        if self.selected_area_id and self.selected_area_id in self.area_graphics_items:
            self.area_graphics_items[self.selected_area_id].set_selected_area(False)

        # Select new area
        self.selected_area_id = area_id
        if area_id in self.area_graphics_items:
            self.area_graphics_items[area_id].set_selected_area(True)
            self.areaSelected.emit(area_id)
            logger.info(f"Selected area: {area_id}")

    def deselect_area(self):
        """Deselect the currently selected area."""
        if self.selected_area_id and self.selected_area_id in self.area_graphics_items:
            self.area_graphics_items[self.selected_area_id].set_selected_area(False)
            logger.info(f"Deselected area: {self.selected_area_id}")

        self.selected_area_id = None
        # Emit signal with empty area_id to indicate deselection
        self.areaSelected.emit("")

    def load_areas_for_floor(self, floor: int):
        """Load and display areas for a specific floor."""
        # Clear existing area graphics (only if they haven't been cleared already)
        # This handles both normal area loading and floor switches where scene.clear() was called
        for graphics_item in list(self.area_graphics_items.values()):
            try:
                if self.scene() and graphics_item.scene() == self.scene():
                    self.scene().removeItem(graphics_item)
            except RuntimeError:
                # Graphics item was already deleted (e.g., by scene.clear())
                pass
        self.area_graphics_items.clear()

        # Load areas for this floor
        areas = self.area_data_manager.get_areas_for_floor(floor)
        for area_data in areas:
            self.add_area_graphics_item(area_data)

    # Exiva session management methods
    def start_exiva_session(self) -> str:
        """Start a new Exiva monster hunt session."""
        import time
        import uuid

        # End any existing session
        self.end_exiva_session()

        # Create new session
        session_id = str(uuid.uuid4())
        self.current_exiva_session = ExivaSession(
            session_id=session_id,
            floor=self.current_floor_id,
            readings=[],
            created_timestamp=time.time(),
            is_active=True
        )

        logger.info(f"Started new Exiva session: {session_id} on floor {self.current_floor_id}")
        return session_id

    def end_exiva_session(self):
        """End the current Exiva session and clear overlays."""
        if self.current_exiva_session:
            self.current_exiva_session.is_active = False
            logger.info(f"Ended Exiva session: {self.current_exiva_session.session_id}")
            self.current_exiva_session = None

        # Remove overlay
        self.clear_exiva_overlay()

    def add_exiva_reading(self, direction: ExivaDirection, distance: ExivaRange,
                         monster_difficulty: MonsterDifficulty = MonsterDifficulty.UNKNOWN,
                         auto_restart_on_difficulty_change: bool = True) -> bool:
        """Add a new Exiva reading to the current session.

        Args:
            direction: Exiva direction reading
            distance: Exiva distance range reading
            monster_difficulty: Monster difficulty level
            auto_restart_on_difficulty_change: If True, automatically restart session when difficulty changes

        Returns:
            True if reading was added successfully, False otherwise
        """
        if not self.current_exiva_session:
            logger.warning("No active Exiva session to add reading to")
            return False

        if not self.global_crosshair_position:
            logger.warning("No crosshair position set for Exiva reading")
            return False

        # Check if we should restart due to difficulty change
        if auto_restart_on_difficulty_change and self.current_exiva_session.should_restart_on_difficulty_change(monster_difficulty):
            old_difficulty = self.current_exiva_session.last_difficulty
            logger.info(f"Monster difficulty changed from {old_difficulty.value} to {monster_difficulty.value} - restarting hunt session")

            # Store the current floor before restarting
            current_floor = self.current_exiva_session.floor

            # Restart the session
            self.start_exiva_session()

            # Emit signal to notify UI about the restart
            self.difficultyChangeRestart.emit(old_difficulty.value, monster_difficulty.value)

        import time
        reading = ExivaReading(
            direction=direction,
            distance=distance,
            crosshair_position=(self.global_crosshair_position.x(), self.global_crosshair_position.y()),
            timestamp=time.time(),
            monster_difficulty=monster_difficulty
        )

        self.current_exiva_session.add_reading(reading)
        self.update_exiva_overlay()

        logger.info(f"Added Exiva reading: {direction.value} {distance.value} ({monster_difficulty.value}) from ({reading.crosshair_position[0]:.1f}, {reading.crosshair_position[1]:.1f})")

        # Check if area is 100% dimmed and restart if needed
        if self.exiva_overlay_item:
            dimmed_pct = self.exiva_overlay_item.get_dimmed_percentage()
            # Restart ONLY if exactly 100% dimmed
            if dimmed_pct == 100.0:
                logger.info(f"Area 100% dimmed - restarting hunt session")

                # Save the last reading and crosshair position before restarting
                last_reading = reading
                last_crosshair_pos = QPointF(self.global_crosshair_position) if self.global_crosshair_position else None

                # Restart the session
                self.start_exiva_session()

                # Re-add the last reading to the new session to keep it visible
                if last_reading and last_crosshair_pos:
                    self.current_exiva_session.add_reading(last_reading)
                    self.update_exiva_overlay()
                    # Restore the crosshair position
                    self.place_crosshairs(last_crosshair_pos)
                    logger.info(f"Preserved last reading and crosshair after 100% dim restart")

                # Emit signal to notify UI about the restart
                self.fullDimRestart.emit(dimmed_pct)

        return True

    def clear_exiva_readings(self):
        """Clear all readings from the current session."""
        if self.current_exiva_session:
            self.current_exiva_session.clear_readings()
            self.update_exiva_overlay()
            logger.info("Cleared all Exiva readings from current session")

    def update_exiva_overlay(self):
        """Update the Exiva overlay based on current readings."""
        if not self.scene():
            return

        # Remove existing overlay
        self.clear_exiva_overlay()

        # Create new overlay if we have readings
        if self.current_exiva_session and self.current_exiva_session.readings:
            scene_rect = self.scene().sceneRect()
            self.exiva_overlay_item = ExivaOverlayItem(scene_rect, self.current_exiva_session.readings)
            # Update zoom level for LOD optimization
            self.exiva_overlay_item.update_zoom_level(self.zoom_factor)
            self.scene().addItem(self.exiva_overlay_item)
            logger.debug(f"Updated Exiva overlay with {len(self.current_exiva_session.readings)} readings")

    def clear_exiva_overlay(self):
        """Remove the Exiva overlay from the scene."""
        if self.exiva_overlay_item:
            if self.scene():
                try:
                    self.scene().removeItem(self.exiva_overlay_item)
                except RuntimeError:
                    # Item was already removed (e.g., by scene.clear())
                    pass
            self.exiva_overlay_item = None

    def get_current_exiva_session(self) -> Optional[ExivaSession]:
        """Get the current active Exiva session."""
        return self.current_exiva_session

    def has_active_exiva_session(self) -> bool:
        """Check if there is an active Exiva session."""
        return self.current_exiva_session is not None and self.current_exiva_session.is_active


class EntryDetailsWindow(QMainWindow):
    """Window to display detailed information about a dataset entry."""

    def __init__(self, entry, dataset_type: str, dataset_manager, parent=None):
        super().__init__(parent)
        self.entry = entry
        self.dataset_type = dataset_type
        self.dataset_manager = dataset_manager
        self.setWindowTitle(f"Entry Details - {dataset_type.capitalize()}")
        self.setMinimumSize(500, 400)

        # Set window flags to make it a separate window
        self.setWindowFlags(Qt.WindowType.Window)

        self.setup_ui()

    def setup_ui(self):
        """Set up the details window UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Title
        title_label = QLabel(f"{self.dataset_type.capitalize()} Dataset Entry")
        title_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(title_label)

        # Create horizontal splitter for screenshot and details
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Screenshot preview (left side)
        screenshot_widget = QWidget()
        screenshot_layout = QVBoxLayout(screenshot_widget)
        screenshot_layout.setContentsMargins(0, 0, 0, 0)

        screenshot_label = QLabel("Screenshot Preview")
        screenshot_label.setStyleSheet("font-weight: bold;")
        screenshot_layout.addWidget(screenshot_label)

        self.screenshot_view = QGraphicsView()
        self.screenshot_scene = QGraphicsScene()
        self.screenshot_view.setScene(self.screenshot_scene)
        self.screenshot_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.screenshot_view.setMinimumWidth(400)
        screenshot_layout.addWidget(self.screenshot_view)

        splitter.addWidget(screenshot_widget)

        # Details text (right side)
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        details_layout.setContentsMargins(0, 0, 0, 0)

        details_label = QLabel("Entry Details")
        details_label.setStyleSheet("font-weight: bold;")
        details_layout.addWidget(details_label)

        details_text = QTextEdit()
        details_text.setReadOnly(True)
        details_text.setMinimumWidth(300)

        # Build details string
        if self.dataset_type == "minimap":
            details = f"Entry ID: {self.entry.entry_id}\n\n"
            details += f"Screenshot: {self.entry.screenshot_path}\n\n"
            details += f"Crosshair Position:\n"
            details += f"  X: {self.entry.crosshair_x:.1f}\n"
            details += f"  Y: {self.entry.crosshair_y:.1f}\n\n"
            details += f"Floor: {self.entry.floor}\n\n"
            details += f"Image Size: {self.entry.image_width} x {self.entry.image_height}\n\n"
            details += f"Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.entry.created_timestamp))}\n"
            details += f"Modified: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.entry.modified_timestamp))}\n"
            if self.entry.notes:
                details += f"\nNotes:\n{self.entry.notes}"
        else:  # exiva
            details = f"Entry ID: {self.entry.entry_id}\n\n"
            details += f"Screenshot: {self.entry.screenshot_path}\n\n"
            details += f"Range: {self.entry.range}\n"
            details += f"Direction: {self.entry.direction}\n"
            details += f"Difficulty: {self.entry.difficulty}\n"
            details += f"Floor Indication: {self.entry.floor_indication}\n"
            details += f"Standing Next To: {self.entry.standing_next_to}\n\n"
            details += f"Image Size: {self.entry.image_width} x {self.entry.image_height}\n\n"
            details += f"Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.entry.created_timestamp))}\n"
            details += f"Modified: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.entry.modified_timestamp))}\n"
            if self.entry.raw_text:
                details += f"\nRaw Text:\n{self.entry.raw_text}\n"
            if self.entry.notes:
                details += f"\nNotes:\n{self.entry.notes}"

        details_text.setText(details)
        details_layout.addWidget(details_text)

        splitter.addWidget(details_widget)

        # Set initial splitter sizes (60% screenshot, 40% details)
        splitter.setSizes([600, 400])

        layout.addWidget(splitter)

        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

        # Load the screenshot
        self.load_screenshot()

    def load_screenshot(self):
        """Load and display the screenshot."""
        screenshot_path = self.dataset_manager.get_screenshot_full_path(self.entry.screenshot_path)

        if not screenshot_path.exists():
            # Show error message in the scene
            text_item = self.screenshot_scene.addText(f"Screenshot not found:\n{self.entry.screenshot_path}")
            text_item.setDefaultTextColor(Qt.GlobalColor.red)
            return

        # Load and display screenshot
        pixmap = QPixmap(str(screenshot_path))
        if pixmap.isNull():
            text_item = self.screenshot_scene.addText("Failed to load screenshot")
            text_item.setDefaultTextColor(Qt.GlobalColor.red)
            return

        self.screenshot_scene.clear()
        pixmap_item = QGraphicsPixmapItem(pixmap)
        self.screenshot_scene.addItem(pixmap_item)
        self.screenshot_scene.setSceneRect(pixmap_item.boundingRect())
        self.screenshot_view.fitInView(pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)


class ZoomableGraphicsView(QGraphicsView):
    """Custom QGraphicsView with smooth zoom functionality."""

    def __init__(self, scene, parent_dialog):
        super().__init__(scene)
        self.parent_dialog = parent_dialog
        self._zoom = 1.0
        self._zoom_step = 1.15
        self._min_zoom = 0.1
        self._max_zoom = 10.0

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        # Get the position of the mouse in scene coordinates
        old_pos = self.mapToScene(event.position().toPoint())

        # Calculate zoom factor
        if event.angleDelta().y() > 0:
            zoom_factor = self._zoom_step
        else:
            zoom_factor = 1 / self._zoom_step

        # Apply zoom with limits
        new_zoom = self._zoom * zoom_factor
        if new_zoom < self._min_zoom or new_zoom > self._max_zoom:
            return

        self._zoom = new_zoom

        # Apply the scale
        self.setTransform(self.transform().scale(zoom_factor, zoom_factor))

        # Get the new position after zoom
        new_pos = self.mapToScene(event.position().toPoint())

        # Calculate the difference and adjust
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

        # Update parent dialog's zoom factor and label
        self.parent_dialog.zoom_factor = self._zoom
        self.parent_dialog.update_zoom_label()

        event.accept()

    def reset_zoom(self):
        """Reset zoom to 1.0."""
        self._zoom = 1.0


class RegionSelectionDialog(QDialog):
    """Dialog for selecting minimap and exiva regions from a screenshot."""

    def __init__(self, dataset_manager, parent=None):
        super().__init__(parent)
        self.dataset_manager = dataset_manager
        self.screenshot_path = None
        self.pixmap = None

        # Region selection state
        self.current_region_type = "minimap"  # "minimap" or "exiva"
        self.selecting = False
        self.start_point = None
        self.current_rect = None

        # Stored regions (x, y, width, height)
        self.minimap_region = None
        self.exiva_region = None

        # Dragging state for moving regions
        self.dragging_region = None  # "minimap" or "exiva" or None
        self.drag_start_pos = None
        self.drag_start_region = None

        # Graphics items for regions
        self.minimap_rect_item = None
        self.minimap_text_item = None
        self.exiva_rect_item = None
        self.exiva_text_item = None

        # Zoom state
        self.zoom_factor = 1.0

        self.setWindowTitle("Select Global Regions")
        self.resize(1000, 700)
        self.setModal(False)
        self.setup_ui()
        self.load_existing_regions()

    def setup_ui(self):
        """Set up the UI."""
        layout = QVBoxLayout(self)

        # Instructions
        instructions = QLabel(
            "Load a screenshot and select the minimap and exiva regions.\n"
            "These regions will be used globally for all dataset entries."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Load screenshot button
        load_btn = QPushButton("Load Screenshot")
        load_btn.clicked.connect(self.load_screenshot)
        layout.addWidget(load_btn)

        # Region selection buttons
        button_layout = QHBoxLayout()

        self.minimap_btn = QPushButton("Draw Minimap Region")
        self.minimap_btn.setCheckable(True)
        self.minimap_btn.clicked.connect(lambda: self.set_region_type("minimap"))
        button_layout.addWidget(self.minimap_btn)

        self.exiva_btn = QPushButton("Draw Exiva Region")
        self.exiva_btn.setCheckable(True)
        self.exiva_btn.clicked.connect(lambda: self.set_region_type("exiva"))
        button_layout.addWidget(self.exiva_btn)

        layout.addLayout(button_layout)

        # Manual region input controls
        controls_group = QGroupBox("Manual Region Controls")
        controls_layout = QVBoxLayout()

        # Minimap region controls
        minimap_controls = QHBoxLayout()
        minimap_controls.addWidget(QLabel("Minimap:"))

        minimap_controls.addWidget(QLabel("X:"))
        self.minimap_x_spin = QSpinBox()
        self.minimap_x_spin.setRange(0, 10000)
        self.minimap_x_spin.setFixedWidth(80)
        self.minimap_x_spin.valueChanged.connect(self.on_minimap_manual_change)
        minimap_controls.addWidget(self.minimap_x_spin)

        minimap_controls.addWidget(QLabel("Y:"))
        self.minimap_y_spin = QSpinBox()
        self.minimap_y_spin.setRange(0, 10000)
        self.minimap_y_spin.setFixedWidth(80)
        self.minimap_y_spin.valueChanged.connect(self.on_minimap_manual_change)
        minimap_controls.addWidget(self.minimap_y_spin)

        minimap_controls.addWidget(QLabel("Width:"))
        self.minimap_w_spin = QSpinBox()
        self.minimap_w_spin.setRange(1, 10000)
        self.minimap_w_spin.setFixedWidth(80)
        self.minimap_w_spin.valueChanged.connect(self.on_minimap_manual_change)
        minimap_controls.addWidget(self.minimap_w_spin)

        minimap_controls.addWidget(QLabel("Height:"))
        self.minimap_h_spin = QSpinBox()
        self.minimap_h_spin.setRange(1, 10000)
        self.minimap_h_spin.setFixedWidth(80)
        self.minimap_h_spin.valueChanged.connect(self.on_minimap_manual_change)
        minimap_controls.addWidget(self.minimap_h_spin)

        minimap_controls.addStretch()
        controls_layout.addLayout(minimap_controls)

        # Exiva region controls
        exiva_controls = QHBoxLayout()
        exiva_controls.addWidget(QLabel("Exiva:"))

        exiva_controls.addWidget(QLabel("X:"))
        self.exiva_x_spin = QSpinBox()
        self.exiva_x_spin.setRange(0, 10000)
        self.exiva_x_spin.setFixedWidth(80)
        self.exiva_x_spin.valueChanged.connect(self.on_exiva_manual_change)
        exiva_controls.addWidget(self.exiva_x_spin)

        exiva_controls.addWidget(QLabel("Y:"))
        self.exiva_y_spin = QSpinBox()
        self.exiva_y_spin.setRange(0, 10000)
        self.exiva_y_spin.setFixedWidth(80)
        self.exiva_y_spin.valueChanged.connect(self.on_exiva_manual_change)
        exiva_controls.addWidget(self.exiva_y_spin)

        exiva_controls.addWidget(QLabel("Width:"))
        self.exiva_w_spin = QSpinBox()
        self.exiva_w_spin.setRange(1, 10000)
        self.exiva_w_spin.setFixedWidth(80)
        self.exiva_w_spin.valueChanged.connect(self.on_exiva_manual_change)
        exiva_controls.addWidget(self.exiva_w_spin)

        exiva_controls.addWidget(QLabel("Height:"))
        self.exiva_h_spin = QSpinBox()
        self.exiva_h_spin.setRange(1, 10000)
        self.exiva_h_spin.setFixedWidth(80)
        self.exiva_h_spin.valueChanged.connect(self.on_exiva_manual_change)
        exiva_controls.addWidget(self.exiva_h_spin)

        exiva_controls.addStretch()
        controls_layout.addLayout(exiva_controls)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Graphics view for screenshot with custom zoom handling
        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView(self.scene, self)
        self.view.setMouseTracking(True)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.viewport().installEventFilter(self)
        layout.addWidget(self.view)

        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom:"))

        zoom_out_btn = QPushButton("-")
        zoom_out_btn.setFixedSize(30, 30)
        zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(zoom_out_btn)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(60)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        zoom_layout.addWidget(self.zoom_label)

        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedSize(30, 30)
        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(zoom_in_btn)

        reset_zoom_btn = QPushButton("Reset")
        reset_zoom_btn.clicked.connect(self.reset_zoom)
        zoom_layout.addWidget(reset_zoom_btn)

        zoom_layout.addStretch()
        layout.addLayout(zoom_layout)

        # Status labels
        status_layout = QFormLayout()
        self.minimap_status = QLabel("Not set")
        self.exiva_status = QLabel("Not set")
        status_layout.addRow("Minimap Region:", self.minimap_status)
        status_layout.addRow("Exiva Region:", self.exiva_status)
        layout.addLayout(status_layout)

        # Save and close buttons
        bottom_layout = QHBoxLayout()
        save_btn = QPushButton("Save Regions")
        save_btn.clicked.connect(self.save_regions)
        bottom_layout.addWidget(save_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        bottom_layout.addWidget(close_btn)

        layout.addLayout(bottom_layout)

    def load_existing_regions(self):
        """Load existing global regions if they exist."""
        regions = self.dataset_manager.load_global_regions()
        self.minimap_region = regions['minimap_region']
        self.exiva_region = regions['exiva_region']
        self.update_status_labels()
        self.update_spinboxes()

    def load_screenshot(self):
        """Load a screenshot for region selection."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Screenshot", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            self.screenshot_path = file_path
            self.pixmap = QPixmap(file_path)
            self.scene.clear()
            self.scene.addPixmap(self.pixmap)
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.draw_regions()

    def set_region_type(self, region_type):
        """Set the current region type for selection."""
        self.current_region_type = region_type
        self.minimap_btn.setChecked(region_type == "minimap")
        self.exiva_btn.setChecked(region_type == "exiva")

        # Update cursor to indicate selection mode
        if self.minimap_btn.isChecked() or self.exiva_btn.isChecked():
            self.view.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.view.setCursor(Qt.CursorShape.ArrowCursor)

    def on_minimap_manual_change(self):
        """Handle manual changes to minimap region spinboxes."""
        x = self.minimap_x_spin.value()
        y = self.minimap_y_spin.value()
        w = self.minimap_w_spin.value()
        h = self.minimap_h_spin.value()
        self.minimap_region = (x, y, w, h)
        self.update_status_labels()
        self.draw_regions()

    def on_exiva_manual_change(self):
        """Handle manual changes to exiva region spinboxes."""
        x = self.exiva_x_spin.value()
        y = self.exiva_y_spin.value()
        w = self.exiva_w_spin.value()
        h = self.exiva_h_spin.value()
        self.exiva_region = (x, y, w, h)
        self.update_status_labels()
        self.draw_regions()

    def update_spinboxes(self):
        """Update spinbox values from current regions."""
        # Block signals to prevent triggering updates
        self.minimap_x_spin.blockSignals(True)
        self.minimap_y_spin.blockSignals(True)
        self.minimap_w_spin.blockSignals(True)
        self.minimap_h_spin.blockSignals(True)
        self.exiva_x_spin.blockSignals(True)
        self.exiva_y_spin.blockSignals(True)
        self.exiva_w_spin.blockSignals(True)
        self.exiva_h_spin.blockSignals(True)

        if self.minimap_region:
            x, y, w, h = self.minimap_region
            self.minimap_x_spin.setValue(x)
            self.minimap_y_spin.setValue(y)
            self.minimap_w_spin.setValue(w)
            self.minimap_h_spin.setValue(h)
        else:
            self.minimap_x_spin.setValue(0)
            self.minimap_y_spin.setValue(0)
            self.minimap_w_spin.setValue(100)
            self.minimap_h_spin.setValue(100)

        if self.exiva_region:
            x, y, w, h = self.exiva_region
            self.exiva_x_spin.setValue(x)
            self.exiva_y_spin.setValue(y)
            self.exiva_w_spin.setValue(w)
            self.exiva_h_spin.setValue(h)
        else:
            self.exiva_x_spin.setValue(0)
            self.exiva_y_spin.setValue(0)
            self.exiva_w_spin.setValue(100)
            self.exiva_h_spin.setValue(100)

        # Unblock signals
        self.minimap_x_spin.blockSignals(False)
        self.minimap_y_spin.blockSignals(False)
        self.minimap_w_spin.blockSignals(False)
        self.minimap_h_spin.blockSignals(False)
        self.exiva_x_spin.blockSignals(False)
        self.exiva_y_spin.blockSignals(False)
        self.exiva_w_spin.blockSignals(False)
        self.exiva_h_spin.blockSignals(False)

    def eventFilter(self, obj, event):
        """Handle mouse events for region selection and dragging."""
        if obj == self.view.viewport() and self.pixmap:
            # Check if we're in selection mode (one of the buttons is checked)
            in_selection_mode = self.minimap_btn.isChecked() or self.exiva_btn.isChecked()

            if event.type() == event.Type.MouseButtonPress:
                scene_pos = self.view.mapToScene(event.pos())
                if self.scene.sceneRect().contains(scene_pos):
                    if in_selection_mode:
                        # Drawing mode - create new region
                        self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
                        self.selecting = True
                        self.start_point = scene_pos
                        return True
                    else:
                        # Check if clicking on an existing region to drag it
                        clicked_region = self.get_region_at_pos(scene_pos)
                        if clicked_region:
                            self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
                            self.dragging_region = clicked_region
                            self.drag_start_pos = scene_pos
                            if clicked_region == "minimap":
                                self.drag_start_region = self.minimap_region
                            else:
                                self.drag_start_region = self.exiva_region
                            self.view.setCursor(Qt.CursorShape.ClosedHandCursor)
                            return True

            elif event.type() == event.Type.MouseMove:
                scene_pos = self.view.mapToScene(event.pos())
                if self.selecting:
                    self.draw_selection_rect(scene_pos)
                    return True
                elif self.dragging_region:
                    self.drag_region(scene_pos)
                    return True
                elif not in_selection_mode:
                    # Update cursor when hovering over regions
                    if self.get_region_at_pos(scene_pos):
                        self.view.setCursor(Qt.CursorShape.OpenHandCursor)
                    else:
                        self.view.setCursor(Qt.CursorShape.ArrowCursor)

            elif event.type() == event.Type.MouseButtonRelease:
                if self.selecting:
                    scene_pos = self.view.mapToScene(event.pos())
                    self.finish_selection(scene_pos)
                    self.selecting = False
                    self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                    return True
                elif self.dragging_region:
                    self.dragging_region = None
                    self.drag_start_pos = None
                    self.drag_start_region = None
                    self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                    self.view.setCursor(Qt.CursorShape.ArrowCursor)
                    return True

        return super().eventFilter(obj, event)

    def get_region_at_pos(self, pos):
        """Check if position is inside a region and return which one."""
        x, y = pos.x(), pos.y()

        # Check minimap region
        if self.minimap_region:
            rx, ry, rw, rh = self.minimap_region
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return "minimap"

        # Check exiva region
        if self.exiva_region:
            rx, ry, rw, rh = self.exiva_region
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return "exiva"

        return None

    def drag_region(self, current_pos):
        """Drag the selected region to a new position."""
        if not self.drag_start_pos or not self.drag_start_region:
            return

        # Calculate offset
        dx = current_pos.x() - self.drag_start_pos.x()
        dy = current_pos.y() - self.drag_start_pos.y()

        # Update region position
        old_x, old_y, w, h = self.drag_start_region
        new_x = int(old_x + dx)
        new_y = int(old_y + dy)

        # Clamp to scene bounds
        new_x = max(0, min(new_x, int(self.scene.width()) - w))
        new_y = max(0, min(new_y, int(self.scene.height()) - h))

        if self.dragging_region == "minimap":
            self.minimap_region = (new_x, new_y, w, h)
        else:
            self.exiva_region = (new_x, new_y, w, h)

        self.update_status_labels()
        self.update_spinboxes()
        self.draw_regions()

    def draw_selection_rect(self, end_point):
        """Draw the current selection rectangle."""
        if not self.start_point:
            return

        # Remove previous selection rect
        if self.current_rect:
            self.scene.removeItem(self.current_rect)

        # Draw new selection rect
        from PyQt6.QtGui import QPen, QColor
        rect = QRectF(self.start_point, end_point).normalized()
        pen = QPen(QColor(255, 255, 0), 2)
        self.current_rect = self.scene.addRect(rect, pen)

    def finish_selection(self, end_point):
        """Finish the region selection."""
        if not self.start_point:
            return

        rect = QRectF(self.start_point, end_point).normalized()
        x, y, w, h = int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())

        if self.current_region_type == "minimap":
            self.minimap_region = (x, y, w, h)
        else:
            self.exiva_region = (x, y, w, h)

        self.update_status_labels()
        self.update_spinboxes()
        self.draw_regions()

        # Uncheck buttons and restore cursor
        self.minimap_btn.setChecked(False)
        self.exiva_btn.setChecked(False)
        self.view.setCursor(Qt.CursorShape.ArrowCursor)

    def draw_regions(self):
        """Draw the saved regions on the screenshot."""
        # Clear previous region drawings
        if self.minimap_rect_item:
            self.scene.removeItem(self.minimap_rect_item)
            self.minimap_rect_item = None
        if self.minimap_text_item:
            self.scene.removeItem(self.minimap_text_item)
            self.minimap_text_item = None
        if self.exiva_rect_item:
            self.scene.removeItem(self.exiva_rect_item)
            self.exiva_rect_item = None
        if self.exiva_text_item:
            self.scene.removeItem(self.exiva_text_item)
            self.exiva_text_item = None

        # Draw minimap region
        if self.minimap_region:
            x, y, w, h = self.minimap_region
            pen = QPen(QColor(0, 255, 0), 3)
            brush = QBrush(QColor(0, 255, 0, 30))
            self.minimap_rect_item = self.scene.addRect(x, y, w, h, pen, brush)

            # Add label
            self.minimap_text_item = QGraphicsTextItem("Minimap (drag to move)")
            self.minimap_text_item.setPos(x, y - 20)
            self.minimap_text_item.setDefaultTextColor(QColor(0, 255, 0))
            self.scene.addItem(self.minimap_text_item)

        # Draw exiva region
        if self.exiva_region:
            x, y, w, h = self.exiva_region
            pen = QPen(QColor(255, 0, 0), 3)
            brush = QBrush(QColor(255, 0, 0, 30))
            self.exiva_rect_item = self.scene.addRect(x, y, w, h, pen, brush)

            # Add label
            self.exiva_text_item = QGraphicsTextItem("Exiva (drag to move)")
            self.exiva_text_item.setPos(x, y - 20)
            self.exiva_text_item.setDefaultTextColor(QColor(255, 0, 0))
            self.scene.addItem(self.exiva_text_item)

    def update_status_labels(self):
        """Update the status labels."""
        if self.minimap_region:
            x, y, w, h = self.minimap_region
            self.minimap_status.setText(f"({x}, {y}, {w}, {h})")
        else:
            self.minimap_status.setText("Not set")

        if self.exiva_region:
            x, y, w, h = self.exiva_region
            self.exiva_status.setText(f"({x}, {y}, {w}, {h})")
        else:
            self.exiva_status.setText("Not set")

    def save_regions(self):
        """Save the regions to the dataset manager."""
        if self.dataset_manager.save_global_regions(self.minimap_region, self.exiva_region):
            QMessageBox.information(self, "Success", "Global regions saved successfully!")
        else:
            QMessageBox.critical(self, "Error", "Failed to save global regions.")

    def zoom_in(self):
        """Zoom in the view."""
        # Simulate wheel event for zoom in
        from PyQt6.QtCore import QPointF
        from PyQt6.QtGui import QWheelEvent

        center = self.view.viewport().rect().center()
        wheel_event = QWheelEvent(
            QPointF(center),
            self.view.mapToGlobal(center),
            QPointF(0, 0),
            QPointF(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False
        )
        self.view.wheelEvent(wheel_event)

    def zoom_out(self):
        """Zoom out the view."""
        # Simulate wheel event for zoom out
        from PyQt6.QtCore import QPointF
        from PyQt6.QtGui import QWheelEvent

        center = self.view.viewport().rect().center()
        wheel_event = QWheelEvent(
            QPointF(center),
            self.view.mapToGlobal(center),
            QPointF(0, 0),
            QPointF(0, -120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False
        )
        self.view.wheelEvent(wheel_event)

    def reset_zoom(self):
        """Reset zoom to 100%."""
        self.view.resetTransform()
        self.view.reset_zoom()
        self.zoom_factor = 1.0
        self.update_zoom_label()

    def update_zoom_label(self):
        """Update the zoom percentage label."""
        zoom_percent = int(self.zoom_factor * 100)
        self.zoom_label.setText(f"{zoom_percent}%")


class OptionsPanel(QWidget):
    """Panel for application options and settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings_manager = None
        self.dataset_manager = None
        self.setup_ui()
        self.load_settings_manager()
        self.load_dataset_manager()

    def load_settings_manager(self):
        """Load the settings manager."""
        from src.models.settings_manager import SettingsManager
        self.settings_manager = SettingsManager()
        self.load_current_settings()

    def load_dataset_manager(self):
        """Load the dataset manager."""
        from src.models.dataset_models import DatasetManager
        self.dataset_manager = DatasetManager()
        self.load_global_regions()

    def setup_ui(self):
        """Set up the options panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title
        title_label = QLabel("Application Options")
        title_label.setStyleSheet("font-weight: bold; font-size: 18px;")
        layout.addWidget(title_label)

        # Settings group
        settings_group = QGroupBox("Hotkey Settings")
        settings_layout = QFormLayout(settings_group)
        settings_layout.setSpacing(10)
        settings_layout.setContentsMargins(15, 15, 15, 15)

        # Screenshot hotkey setting
        self.hotkey_input = QLineEdit()
        self.hotkey_input.setMaxLength(1)
        self.hotkey_input.setPlaceholderText(".")
        self.hotkey_input.setToolTip("Single key to use as screenshot hotkey")
        self.hotkey_input.setFixedWidth(100)
        self.hotkey_input.textChanged.connect(self.on_hotkey_changed)

        settings_layout.addRow("Screenshot Button:", self.hotkey_input)

        layout.addWidget(settings_group)

        # Paths group
        paths_group = QGroupBox("Folder Paths")
        paths_layout = QVBoxLayout(paths_group)
        paths_layout.setSpacing(10)
        paths_layout.setContentsMargins(15, 15, 15, 15)

        # Tibia screenshot folder
        tibia_folder_label = QLabel("Tibia Screenshot Folder:")
        tibia_folder_label.setStyleSheet("font-weight: bold;")
        paths_layout.addWidget(tibia_folder_label)

        tibia_folder_layout = QHBoxLayout()
        self.tibia_folder_input = QLineEdit()
        self.tibia_folder_input.setPlaceholderText(r"C:\Users\...\Tibia\screenshots")
        self.tibia_folder_input.setToolTip("Path to Tibia screenshots folder")
        self.tibia_folder_input.textChanged.connect(self.on_tibia_folder_changed)
        tibia_folder_layout.addWidget(self.tibia_folder_input)

        browse_btn = QPushButton("Browse...")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self.browse_tibia_folder)
        tibia_folder_layout.addWidget(browse_btn)

        paths_layout.addLayout(tibia_folder_layout)

        layout.addWidget(paths_group)

        # Global Regions group
        regions_group = QGroupBox("Global Regions")
        regions_layout = QVBoxLayout(regions_group)
        regions_layout.setSpacing(10)
        regions_layout.setContentsMargins(15, 15, 15, 15)

        # Description
        desc_label = QLabel("Configure global minimap and exiva regions that apply to all dataset entries.")
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666; font-size: 11px;")
        regions_layout.addWidget(desc_label)

        # Minimap region display
        minimap_label = QLabel("Minimap Region:")
        minimap_label.setStyleSheet("font-weight: bold;")
        regions_layout.addWidget(minimap_label)

        self.minimap_region_value = QLabel("Not set")
        self.minimap_region_value.setStyleSheet("margin-left: 10px; color: #444;")
        regions_layout.addWidget(self.minimap_region_value)

        # Exiva region display
        exiva_label = QLabel("Exiva Region:")
        exiva_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        regions_layout.addWidget(exiva_label)

        self.exiva_region_value = QLabel("Not set")
        self.exiva_region_value.setStyleSheet("margin-left: 10px; color: #444;")
        regions_layout.addWidget(self.exiva_region_value)

        # Configure button
        configure_btn = QPushButton("Configure Global Regions")
        configure_btn.setToolTip("Open dialog to select minimap and exiva regions from a screenshot")
        configure_btn.clicked.connect(self.open_region_settings)
        regions_layout.addWidget(configure_btn)

        layout.addWidget(regions_group)
        layout.addStretch()

    def load_current_settings(self):
        """Load current settings into UI."""
        if not self.settings_manager:
            return

        hotkey = self.settings_manager.get_screenshot_hotkey()
        self.hotkey_input.setText(hotkey)

        tibia_folder = self.settings_manager.get_tibia_screenshot_folder()
        self.tibia_folder_input.setText(tibia_folder)

    def load_global_regions(self):
        """Load and display global regions."""
        if not self.dataset_manager:
            return

        regions = self.dataset_manager.load_global_regions()
        minimap_region = regions.get('minimap_region')
        exiva_region = regions.get('exiva_region')

        # Update minimap region display
        if minimap_region:
            x, y, w, h = minimap_region
            self.minimap_region_value.setText(f"x={x}, y={y}, width={w}, height={h}")
        else:
            self.minimap_region_value.setText("Not set")

        # Update exiva region display
        if exiva_region:
            x, y, w, h = exiva_region
            self.exiva_region_value.setText(f"x={x}, y={y}, width={w}, height={h}")
        else:
            self.exiva_region_value.setText("Not set")

    def open_region_settings(self):
        """Open the region settings dialog."""
        if not self.dataset_manager:
            QMessageBox.warning(self, "Error", "Dataset manager not initialized.")
            return

        dialog = RegionSelectionDialog(self.dataset_manager, self)
        dialog.finished.connect(self.on_region_dialog_closed)
        dialog.show()

    def on_region_dialog_closed(self):
        """Refresh region display when dialog is closed."""
        self.load_global_regions()

    def on_hotkey_changed(self, text: str):
        """Handle hotkey input change."""
        if not self.settings_manager:
            return

        # Save the hotkey (only save if not empty)
        if text:
            self.settings_manager.set_screenshot_hotkey(text)

    def on_tibia_folder_changed(self, text: str):
        """Handle Tibia folder input change."""
        if not self.settings_manager:
            return

        # Save the folder path (only save if not empty)
        if text:
            self.settings_manager.set_tibia_screenshot_folder(text)

    def browse_tibia_folder(self):
        """Browse for Tibia screenshot folder."""
        from PyQt6.QtWidgets import QFileDialog

        current_folder = self.tibia_folder_input.text()
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Tibia Screenshot Folder",
            current_folder if current_folder else ""
        )

        if folder:
            self.tibia_folder_input.setText(folder)


class MinimapComparisonWindow(QDialog):
    """Standalone window for side-by-side comparison of game and synthetic minimap images."""

    def __init__(self, game_image, synthetic_image, parent=None):
        """Initialize the comparison window.

        Args:
            game_image: PIL Image from game screenshot
            synthetic_image: PIL Image from dataset
            parent: Parent widget
        """
        super().__init__(parent)
        self.game_image = game_image
        self.synthetic_image = synthetic_image

        self.setWindowTitle("Minimap Comparison - Game vs Synthetic")
        self.setMinimumSize(900, 700)
        self.resize(1100, 800)

        # Make it a proper window, not modal
        self.setWindowFlags(Qt.WindowType.Window)

        self.setup_ui()

    def setup_ui(self):
        """Set up the comparison window UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Title
        title_label = QLabel("Visual Comparison: Game Screenshot vs Synthetic Dataset")
        title_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #333;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Instructions
        instructions = QLabel(
            "Compare the two minimap images to identify any differences. "
            "Use the zoom slider to inspect details."
        )
        instructions.setStyleSheet("color: #666; font-size: 11px; padding: 5px;")
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Zoom controls at top for easy access
        zoom_layout = QHBoxLayout()
        zoom_layout.setSpacing(10)

        zoom_label = QLabel("Zoom:")
        zoom_label.setStyleSheet("font-weight: bold;")
        zoom_layout.addWidget(zoom_label)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setMinimum(100)
        self.zoom_slider.setMaximum(800)
        self.zoom_slider.setValue(300)
        self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.zoom_slider.setTickInterval(100)
        self.zoom_slider.valueChanged.connect(self.update_zoom)
        zoom_layout.addWidget(self.zoom_slider)

        self.zoom_value_label = QLabel("300%")
        self.zoom_value_label.setStyleSheet("font-weight: bold; min-width: 50px;")
        zoom_layout.addWidget(self.zoom_value_label)

        layout.addLayout(zoom_layout)

        # Image comparison area with scroll areas
        comparison_layout = QHBoxLayout()
        comparison_layout.setSpacing(15)

        # Game minimap section
        game_section = QVBoxLayout()
        game_section.setSpacing(5)

        game_label = QLabel("Game Screenshot Minimap")
        game_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #2196F3; padding: 5px;")
        game_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        game_section.addWidget(game_label)

        # Scroll area for game image
        self.game_scroll = QScrollArea()
        self.game_scroll.setWidgetResizable(False)
        self.game_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.game_scroll.setStyleSheet("QScrollArea { border: 3px solid #2196F3; background-color: #f5f5f5; }")
        self.game_scroll.setMinimumSize(400, 400)
        self.game_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.game_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.game_image_label = QLabel()
        self.game_image_label.setScaledContents(False)
        self.game_scroll.setWidget(self.game_image_label)

        game_section.addWidget(self.game_scroll)

        # Game image info
        game_info = QLabel(f"Original Size: {self.game_image.width} x {self.game_image.height} pixels")
        game_info.setStyleSheet("color: #666; font-size: 10px; padding: 3px;")
        game_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        game_section.addWidget(game_info)

        comparison_layout.addLayout(game_section, 1)

        # Synthetic minimap section
        synthetic_section = QVBoxLayout()
        synthetic_section.setSpacing(5)

        synthetic_label = QLabel("Synthetic Dataset Minimap")
        synthetic_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #4CAF50; padding: 5px;")
        synthetic_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        synthetic_section.addWidget(synthetic_label)

        # Scroll area for synthetic image
        self.synthetic_scroll = QScrollArea()
        self.synthetic_scroll.setWidgetResizable(False)
        self.synthetic_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.synthetic_scroll.setStyleSheet("QScrollArea { border: 3px solid #4CAF50; background-color: #f5f5f5; }")
        self.synthetic_scroll.setMinimumSize(400, 400)
        self.synthetic_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.synthetic_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.synthetic_image_label = QLabel()
        self.synthetic_image_label.setScaledContents(False)
        self.synthetic_scroll.setWidget(self.synthetic_image_label)

        synthetic_section.addWidget(self.synthetic_scroll)

        # Synthetic image info
        synthetic_info = QLabel(f"Original Size: {self.synthetic_image.width} x {self.synthetic_image.height} pixels")
        synthetic_info.setStyleSheet("color: #666; font-size: 10px; padding: 3px;")
        synthetic_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        synthetic_section.addWidget(synthetic_info)

        comparison_layout.addLayout(synthetic_section, 1)

        layout.addLayout(comparison_layout, 1)

        # Analysis section
        analysis_group = QGroupBox("Analysis Results")
        analysis_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        analysis_layout = QVBoxLayout(analysis_group)
        analysis_layout.setContentsMargins(10, 10, 10, 10)

        self.analysis_label = QLabel()
        self.analysis_label.setStyleSheet("color: #333; font-size: 11px; line-height: 1.4;")
        self.analysis_label.setWordWrap(True)
        analysis_layout.addWidget(self.analysis_label)

        layout.addWidget(analysis_group)

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        # Reset zoom button
        reset_zoom_btn = QPushButton("Reset Zoom (100%)")
        reset_zoom_btn.clicked.connect(lambda: self.zoom_slider.setValue(100))
        reset_zoom_btn.setMaximumWidth(150)
        button_layout.addWidget(reset_zoom_btn)

        button_layout.addStretch()

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        close_btn.setMaximumWidth(100)
        close_btn.setStyleSheet("QPushButton { padding: 8px 20px; font-weight: bold; }")
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

        # Display images
        self.update_zoom(self.zoom_slider.value())
        self.perform_analysis()

    def update_zoom(self, value):
        """Update the zoom level of displayed images."""
        from PyQt6.QtGui import QPixmap
        from PIL import Image
        from PIL.ImageQt import ImageQt

        zoom_percent = value
        self.zoom_value_label.setText(f"{zoom_percent}%")

        # Calculate new size
        scale_factor = zoom_percent / 100.0

        # Game image
        game_width = int(self.game_image.width * scale_factor)
        game_height = int(self.game_image.height * scale_factor)
        game_resized = self.game_image.resize((game_width, game_height), Image.Resampling.NEAREST)
        game_qimage = ImageQt(game_resized)
        game_pixmap = QPixmap.fromImage(game_qimage)
        self.game_image_label.setPixmap(game_pixmap)
        self.game_image_label.setFixedSize(game_pixmap.size())

        # Synthetic image
        synthetic_width = int(self.synthetic_image.width * scale_factor)
        synthetic_height = int(self.synthetic_image.height * scale_factor)
        synthetic_resized = self.synthetic_image.resize((synthetic_width, synthetic_height), Image.Resampling.NEAREST)
        synthetic_qimage = ImageQt(synthetic_resized)
        synthetic_pixmap = QPixmap.fromImage(synthetic_qimage)
        self.synthetic_image_label.setPixmap(synthetic_pixmap)
        self.synthetic_image_label.setFixedSize(synthetic_pixmap.size())

    def perform_analysis(self):
        """Perform quick analysis comparing the two images."""
        import numpy as np

        # Convert to numpy arrays
        game_array = np.array(self.game_image)
        synthetic_array = np.array(self.synthetic_image)

        analysis_parts = []

        # Size comparison
        analysis_parts.append("<b>Dimension Check:</b>")
        if game_array.shape == synthetic_array.shape:
            analysis_parts.append("  ✓ Image dimensions match perfectly")
        else:
            analysis_parts.append(f"  ⚠ <span style='color: #f44336;'>Dimensions differ:</span> Game {game_array.shape[:2]} vs Synthetic {synthetic_array.shape[:2]}")

        # Pixel-perfect comparison (if same size)
        if game_array.shape == synthetic_array.shape:
            identical_pixels = np.all(game_array == synthetic_array, axis=-1).sum()
            total_pixels = game_array.shape[0] * game_array.shape[1]
            match_percent = (identical_pixels / total_pixels) * 100

            analysis_parts.append("")
            analysis_parts.append("<b>Pixel Match Analysis:</b>")

            if match_percent == 100:
                analysis_parts.append(f"  ✓ <span style='color: #4CAF50; font-weight: bold;'>{match_percent:.2f}% match</span> - Images are pixel-perfect identical!")
            elif match_percent >= 95:
                analysis_parts.append(f"  ✓ <span style='color: #8BC34A; font-weight: bold;'>{match_percent:.2f}% match</span> - Very similar (minor differences)")
            elif match_percent >= 90:
                analysis_parts.append(f"  ⚠ <span style='color: #FF9800; font-weight: bold;'>{match_percent:.2f}% match</span> - Some differences detected")
            else:
                analysis_parts.append(f"  ⚠ <span style='color: #f44336; font-weight: bold;'>{match_percent:.2f}% match</span> - Significant differences!")

            analysis_parts.append(f"  Identical pixels: {identical_pixels:,} / {total_pixels:,}")
            different_pixels = total_pixels - identical_pixels
            if different_pixels > 0:
                analysis_parts.append(f"  Different pixels: {different_pixels:,}")

        # Color statistics
        game_mean = game_array.mean(axis=(0, 1))
        synthetic_mean = synthetic_array.mean(axis=(0, 1))

        analysis_parts.append("")
        analysis_parts.append("<b>Color Statistics (RGB Mean):</b>")
        analysis_parts.append(f"  Game:      <span style='color: #2196F3;'>R={game_mean[0]:.1f}, G={game_mean[1]:.1f}, B={game_mean[2]:.1f}</span>")
        analysis_parts.append(f"  Synthetic: <span style='color: #4CAF50;'>R={synthetic_mean[0]:.1f}, G={synthetic_mean[1]:.1f}, B={synthetic_mean[2]:.1f}</span>")

        # Color difference
        color_diff = np.abs(game_mean - synthetic_mean)
        max_diff = color_diff.max()
        if max_diff < 1.0:
            analysis_parts.append(f"  ✓ Color difference: <span style='color: #4CAF50;'>Negligible (max {max_diff:.2f})</span>")
        elif max_diff < 5.0:
            analysis_parts.append(f"  ✓ Color difference: <span style='color: #8BC34A;'>Minor (max {max_diff:.2f})</span>")
        else:
            analysis_parts.append(f"  ⚠ Color difference: <span style='color: #FF9800;'>Noticeable (max {max_diff:.2f})</span>")

        self.analysis_label.setText("<br>".join(analysis_parts))


class DatasetTestsPanel(QWidget):
    """Panel for testing datasets by navigating through entries."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent  # Store reference to MinimapViewerWidget
        self.dataset_manager = None
        self.settings_manager = None
        self.current_dataset_type = "minimap"  # "minimap" or "exiva"
        self.current_entries = []
        self.current_index = 0

        # Visual comparison state
        self.game_minimap_image = None  # PIL Image from game screenshot
        self.synthetic_minimap_image = None  # PIL Image from dataset

        self.setup_ui()
        self.load_dataset_manager()
        self.load_settings_manager()

    def load_dataset_manager(self):
        """Load the dataset manager."""
        from src.models.dataset_models import DatasetManager
        self.dataset_manager = DatasetManager()
        self.load_current_dataset()

    def load_settings_manager(self):
        """Load the settings manager."""
        from src.models.settings_manager import SettingsManager
        self.settings_manager = SettingsManager()

    def setup_ui(self):
        """Set up the dataset tests panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Title
        title_label = QLabel("Dataset Tests")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)

        # Dataset type selection
        dataset_group = QGroupBox("Dataset Selection")
        dataset_layout = QVBoxLayout(dataset_group)
        dataset_layout.setContentsMargins(8, 8, 8, 8)
        dataset_layout.setSpacing(5)

        # Dataset type combo box
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Dataset Type:"))
        self.dataset_type_combo = QComboBox()
        self.dataset_type_combo.addItems(["Minimap", "Exiva"])
        self.dataset_type_combo.currentTextChanged.connect(self.on_dataset_type_changed)
        type_layout.addWidget(self.dataset_type_combo)
        dataset_layout.addLayout(type_layout)

        # Dataset stats
        self.stats_label = QLabel("Entries: 0")
        self.stats_label.setStyleSheet("font-style: italic; color: #666;")
        dataset_layout.addWidget(self.stats_label)

        layout.addWidget(dataset_group)

        # Entry list
        list_group = QGroupBox("Entries")
        list_layout = QVBoxLayout(list_group)
        list_layout.setContentsMargins(8, 8, 8, 8)
        list_layout.setSpacing(5)

        # Pagination controls
        pagination_layout = QHBoxLayout()

        self.page_size_label = QLabel("Page size:")
        pagination_layout.addWidget(self.page_size_label)

        self.page_size_spin = QSpinBox()
        self.page_size_spin.setMinimum(10)
        self.page_size_spin.setMaximum(10000)
        self.page_size_spin.setValue(100)
        self.page_size_spin.setSingleStep(50)
        self.page_size_spin.setToolTip("Number of entries to load per page (lower = faster)")
        self.page_size_spin.valueChanged.connect(self.on_page_size_changed)
        pagination_layout.addWidget(self.page_size_spin)

        pagination_layout.addStretch()

        self.prev_page_btn = QPushButton("◀ Previous")
        self.prev_page_btn.clicked.connect(self.load_previous_page)
        self.prev_page_btn.setEnabled(False)
        pagination_layout.addWidget(self.prev_page_btn)

        self.page_info_label = QLabel("Page 1 of 1")
        self.page_info_label.setStyleSheet("font-weight: bold;")
        pagination_layout.addWidget(self.page_info_label)

        self.next_page_btn = QPushButton("Next ▶")
        self.next_page_btn.clicked.connect(self.load_next_page)
        self.next_page_btn.setEnabled(False)
        pagination_layout.addWidget(self.next_page_btn)

        list_layout.addLayout(pagination_layout)

        self.entry_list = QListWidget()
        self.entry_list.currentRowChanged.connect(self.on_entry_selected)
        self.entry_list.itemDoubleClicked.connect(self.on_entry_double_clicked)
        list_layout.addWidget(self.entry_list)

        # Instruction label
        instruction_label = QLabel("Double-click an entry to view screenshot and details")
        instruction_label.setStyleSheet("font-style: italic; color: #888; font-size: 11px;")
        instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        list_layout.addWidget(instruction_label)

        layout.addWidget(list_group)

        # Initialize pagination state
        self.current_page = 0
        self.total_entries = 0

        # Tests section
        tests_group = QGroupBox("Tests")
        tests_layout = QVBoxLayout(tests_group)
        tests_layout.setContentsMargins(8, 8, 8, 8)
        tests_layout.setSpacing(5)

        # Test selected entry button
        self.test_selected_btn = QPushButton("Test Selected Entry")
        self.test_selected_btn.setToolTip("Run test on the currently selected entry")
        self.test_selected_btn.clicked.connect(self.test_selected_entry)
        tests_layout.addWidget(self.test_selected_btn)

        # Test all entries button
        self.test_all_btn = QPushButton("Test All Entries")
        self.test_all_btn.setToolTip("Run tests on all entries in the dataset")
        self.test_all_btn.clicked.connect(self.test_all_entries)
        tests_layout.addWidget(self.test_all_btn)

        layout.addWidget(tests_group)

        # Dataset Coverage Visualization
        coverage_group = QGroupBox("Dataset Coverage Visualization")
        coverage_layout = QVBoxLayout(coverage_group)
        coverage_layout.setContentsMargins(8, 8, 8, 8)
        coverage_layout.setSpacing(5)

        # Instructions
        coverage_instructions = QLabel(
            "Show pink pixels on the map where dataset samples exist.\n"
            "This helps identify coverage gaps and plan data generation."
        )
        coverage_instructions.setStyleSheet("color: gray; font-size: 10px;")
        coverage_instructions.setWordWrap(True)
        coverage_layout.addWidget(coverage_instructions)

        # Toggle button
        self.coverage_toggle_btn = QPushButton("Show Coverage Overlay")
        self.coverage_toggle_btn.setCheckable(True)
        self.coverage_toggle_btn.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold; padding: 10px;")
        self.coverage_toggle_btn.clicked.connect(self.toggle_coverage_overlay)
        coverage_layout.addWidget(self.coverage_toggle_btn)

        # Status label
        self.coverage_status_label = QLabel("")
        self.coverage_status_label.setStyleSheet("color: gray; font-size: 10px;")
        self.coverage_status_label.setWordWrap(True)
        coverage_layout.addWidget(self.coverage_status_label)

        layout.addWidget(coverage_group)

        # Coverage overlay state
        self.coverage_overlay_items = []  # Store coverage pixel items
        self.coverage_overlay_visible = False

        # Visual Comparison Tool
        comparison_group = QGroupBox("Visual Comparison Tool")
        comparison_layout = QVBoxLayout(comparison_group)
        comparison_layout.setContentsMargins(8, 8, 8, 8)
        comparison_layout.setSpacing(5)

        # Instructions
        comparison_instructions = QLabel(
            "Compare game-captured minimap screenshots with synthetic dataset images.\n"
            "This helps identify differences between real and generated data."
        )
        comparison_instructions.setStyleSheet("color: gray; font-size: 10px;")
        comparison_instructions.setWordWrap(True)
        comparison_layout.addWidget(comparison_instructions)

        # Capture game minimap button
        self.capture_game_minimap_btn = QPushButton("Capture Game Minimap from Screenshot")
        self.capture_game_minimap_btn.setToolTip("Load latest Tibia screenshot and crop minimap region")
        self.capture_game_minimap_btn.clicked.connect(self.capture_game_minimap)
        comparison_layout.addWidget(self.capture_game_minimap_btn)

        # Load synthetic minimap button
        self.load_synthetic_minimap_btn = QPushButton("Load Synthetic Minimap from Dataset")
        self.load_synthetic_minimap_btn.setToolTip("Select a synthetic minimap image from the dataset")
        self.load_synthetic_minimap_btn.clicked.connect(self.load_synthetic_minimap)
        comparison_layout.addWidget(self.load_synthetic_minimap_btn)

        # Show comparison button
        self.show_comparison_btn = QPushButton("Show Side-by-Side Comparison")
        self.show_comparison_btn.setToolTip("Display both images side-by-side for visual comparison")
        self.show_comparison_btn.clicked.connect(self.show_comparison_window)
        self.show_comparison_btn.setEnabled(False)
        comparison_layout.addWidget(self.show_comparison_btn)

        # Status label
        self.comparison_status_label = QLabel("")
        self.comparison_status_label.setStyleSheet("color: gray; font-size: 10px;")
        self.comparison_status_label.setWordWrap(True)
        comparison_layout.addWidget(self.comparison_status_label)

        layout.addWidget(comparison_group)

        # Add stretch to push everything to the top
        layout.addStretch()

    def on_dataset_type_changed(self, text: str):
        """Handle dataset type selection change."""
        self.current_dataset_type = text.lower()
        self.current_page = 0  # Reset to first page
        self.load_current_dataset()

    def load_current_dataset(self):
        """Load the currently selected dataset with pagination."""
        if not self.dataset_manager:
            return

        # Get total count first
        if self.current_dataset_type == "minimap":
            self.total_entries = self.dataset_manager.get_minimap_dataset_count()
        else:
            # For exiva, we still need to load all (usually smaller dataset)
            all_entries = self.dataset_manager.load_exiva_dataset()
            self.total_entries = len(all_entries)
            self.current_entries = all_entries
            self.update_stats()
            self.populate_entry_list()
            self.update_pagination_controls()
            if self.current_entries:
                self.entry_list.setCurrentRow(0)
            return

        # Load paginated data for minimap dataset
        page_size = self.page_size_spin.value()
        offset = self.current_page * page_size

        self.current_entries = self.dataset_manager.load_minimap_dataset_paginated(offset, page_size)

        # Update UI
        self.update_stats()
        self.populate_entry_list()
        self.update_pagination_controls()

        # Select first entry if available
        if self.current_entries:
            self.entry_list.setCurrentRow(0)

    def update_stats(self):
        """Update the statistics label."""
        if self.current_dataset_type == "minimap":
            # Show total count and current page range
            page_size = self.page_size_spin.value()
            start = self.current_page * page_size + 1
            end = min(start + len(self.current_entries) - 1, self.total_entries)
            self.stats_label.setText(f"Total Entries: {self.total_entries:,} (Showing {start:,}-{end:,})")
        else:
            count = len(self.current_entries)
            self.stats_label.setText(f"Entries: {count}")

    def update_pagination_controls(self):
        """Update pagination button states and page info."""
        if self.current_dataset_type != "minimap":
            # Hide pagination for non-minimap datasets
            self.prev_page_btn.setEnabled(False)
            self.next_page_btn.setEnabled(False)
            self.page_info_label.setText("")
            return

        page_size = self.page_size_spin.value()
        total_pages = max(1, (self.total_entries + page_size - 1) // page_size)
        current_page_display = self.current_page + 1

        # Update page info
        self.page_info_label.setText(f"Page {current_page_display} of {total_pages}")

        # Update button states
        self.prev_page_btn.setEnabled(self.current_page > 0)
        self.next_page_btn.setEnabled(self.current_page < total_pages - 1)

    def load_previous_page(self):
        """Load the previous page of entries."""
        if self.current_page > 0:
            self.current_page -= 1
            self.load_current_dataset()

    def load_next_page(self):
        """Load the next page of entries."""
        page_size = self.page_size_spin.value()
        total_pages = (self.total_entries + page_size - 1) // page_size
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self.load_current_dataset()

    def on_page_size_changed(self, value: int):
        """Handle page size change."""
        # Reset to first page when page size changes
        self.current_page = 0
        self.load_current_dataset()

    def populate_entry_list(self):
        """Populate the entry list widget with current page entries."""
        self.entry_list.clear()

        page_size = self.page_size_spin.value()
        offset = self.current_page * page_size

        for i, entry in enumerate(self.current_entries):
            global_index = offset + i
            if self.current_dataset_type == "minimap":
                # Show entry number, crosshair position, and floor
                list_text = f"Entry {global_index + 1} - ({entry.crosshair_x:.0f}, {entry.crosshair_y:.0f}, {entry.floor})"
            else:  # exiva
                # Show entry number, range, and direction
                list_text = f"Entry {global_index + 1} - {entry.range} {entry.direction}"

            item = QListWidgetItem(list_text)
            self.entry_list.addItem(item)

    def on_entry_selected(self, row: int):
        """Handle entry selection from the list."""
        if row >= 0 and row < len(self.current_entries):
            self.current_index = row

    def on_entry_double_clicked(self, item: QListWidgetItem):
        """Handle double-click on an entry to show details window."""
        if self.current_index >= 0 and self.current_index < len(self.current_entries):
            entry = self.current_entries[self.current_index]
            details_window = EntryDetailsWindow(entry, self.current_dataset_type, self.dataset_manager, self)
            details_window.show()

    def test_selected_entry(self):
        """Test the currently selected entry."""
        # Placeholder for testing selected entry
        pass

    def test_all_entries(self):
        """Test all entries in the current dataset."""
        # Placeholder for testing all entries
        pass

    def toggle_coverage_overlay(self):
        """Toggle the dataset coverage overlay on/off."""
        if self.coverage_toggle_btn.isChecked():
            self.show_coverage_overlay()
        else:
            self.hide_coverage_overlay()

    def show_coverage_overlay(self):
        """Show pink pixels on the map where dataset samples exist."""
        # Get reference to the main viewer
        main_viewer = self.parent_widget
        if not main_viewer or not hasattr(main_viewer, 'graphics_view'):
            QMessageBox.warning(
                self,
                "No Map Loaded",
                "Please load a floor map first."
            )
            self.coverage_toggle_btn.setChecked(False)
            return

        graphics_view = main_viewer.graphics_view
        if not graphics_view or not graphics_view.scene():
            QMessageBox.warning(
                self,
                "No Map Loaded",
                "Please load a floor map first."
            )
            self.coverage_toggle_btn.setChecked(False)
            return

        # Get current floor
        current_floor = main_viewer.current_floor if hasattr(main_viewer, 'current_floor') else None

        if current_floor is None:
            QMessageBox.warning(
                self,
                "No Floor Selected",
                "Please select a floor first."
            )
            self.coverage_toggle_btn.setChecked(False)
            return

        self.coverage_status_label.setText("Loading dataset positions...")
        QApplication.processEvents()

        try:
            # Load only positions for current floor (optimized - doesn't load full entries)
            positions = self.dataset_manager.get_minimap_positions_for_floor(current_floor)

            if not positions:
                self.coverage_status_label.setText(f"No dataset entries found for floor {current_floor}")
                self.coverage_toggle_btn.setChecked(False)
                return

            # Clear any existing coverage overlay
            self.hide_coverage_overlay()

            # Create pink pixels at each crosshair position
            from PyQt6.QtWidgets import QGraphicsRectItem
            from PyQt6.QtGui import QColor, QPen, QBrush
            from PyQt6.QtCore import Qt

            pink_color = QColor(255, 20, 147)  # Deep pink
            pen = QPen(Qt.PenStyle.NoPen)  # No border for cleaner look
            brush = QBrush(pink_color)

            for crosshair_x, crosshair_y in positions:
                # Convert float coordinates to integer pixel positions
                pixel_x = int(crosshair_x)
                pixel_y = int(crosshair_y)

                # Create a small rectangle (1x1 pixel) at this position
                rect_item = QGraphicsRectItem(pixel_x, pixel_y, 1, 1)
                rect_item.setPen(pen)
                rect_item.setBrush(brush)
                rect_item.setZValue(1000)  # Draw on top of everything

                graphics_view.scene().addItem(rect_item)
                self.coverage_overlay_items.append(rect_item)

            self.coverage_overlay_visible = True
            self.coverage_status_label.setText(
                f"Showing {len(positions):,} dataset positions on floor {current_floor}"
            )
            self.coverage_toggle_btn.setText("Hide Coverage Overlay")
            self.coverage_toggle_btn.setStyleSheet("background-color: #E91E63; color: white; font-weight: bold; padding: 10px;")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load coverage overlay: {str(e)}"
            )
            self.coverage_toggle_btn.setChecked(False)
            self.coverage_status_label.setText("")

    def hide_coverage_overlay(self):
        """Hide the coverage overlay."""
        # Get reference to the main viewer
        main_viewer = self.parent_widget
        if not main_viewer or not hasattr(main_viewer, 'graphics_view'):
            return

        graphics_view = main_viewer.graphics_view
        if not graphics_view or not graphics_view.scene():
            return

        # Remove all coverage overlay items
        for item in self.coverage_overlay_items:
            try:
                graphics_view.scene().removeItem(item)
            except RuntimeError:
                # Item was already removed
                pass

        self.coverage_overlay_items.clear()
        self.coverage_overlay_visible = False
        self.coverage_status_label.setText("")
        self.coverage_toggle_btn.setText("Show Coverage Overlay")
        self.coverage_toggle_btn.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold; padding: 10px;")

    def on_floor_changed(self):
        """Handle floor change event - refresh coverage overlay if visible."""
        if self.coverage_overlay_visible and self.coverage_toggle_btn.isChecked():
            # Refresh the overlay for the new floor
            self.show_coverage_overlay()

    def capture_game_minimap(self):
        """Capture and crop minimap from the latest Tibia screenshot."""
        try:
            from pathlib import Path
            from PIL import Image

            # Get screenshot directory from settings
            if not self.settings_manager:
                QMessageBox.warning(
                    self,
                    "Settings Not Loaded",
                    "Settings manager not initialized."
                )
                return

            screenshot_dir = Path(self.settings_manager.get_tibia_screenshot_folder())

            if not screenshot_dir.exists():
                QMessageBox.warning(
                    self,
                    "Screenshot Directory Not Found",
                    f"Screenshot directory does not exist:\n{screenshot_dir}\n\n"
                    "Please configure the Tibia screenshot folder in Options/Settings."
                )
                return

            # Get minimap region from global settings
            regions = self.dataset_manager.load_global_regions()
            minimap_region = regions.get('minimap_region')

            if not minimap_region:
                QMessageBox.warning(
                    self,
                    "Minimap Region Not Set",
                    "Please configure the minimap region in Options/Settings before capturing."
                )
                return

            # Find the latest screenshot
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
            screenshot_files = []

            for ext in image_extensions:
                screenshot_files.extend(screenshot_dir.glob(f'*{ext}'))

            if not screenshot_files:
                QMessageBox.warning(
                    self,
                    "No Screenshots Found",
                    f"No screenshot files found in:\n{screenshot_dir}"
                )
                return

            # Sort by modification time (most recent first)
            screenshot_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            latest_screenshot = screenshot_files[0]

            # Load and crop the screenshot
            screenshot = Image.open(latest_screenshot)
            x, y, width, height = minimap_region

            # Crop the minimap region (PIL crop uses left, top, right, bottom)
            self.game_minimap_image = screenshot.crop((x, y, x + width, y + height))

            # Update status
            self.comparison_status_label.setText(
                f"✓ Game minimap captured from: {latest_screenshot.name}"
            )
            self.comparison_status_label.setStyleSheet("color: green; font-size: 10px;")

            # Enable comparison button if both images are loaded
            self.update_comparison_button_state()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to capture game minimap:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()

    def load_synthetic_minimap(self):
        """Load a synthetic minimap image from the dataset."""
        try:
            from pathlib import Path
            from PIL import Image

            # Get the dataset directory
            dataset_dir = Path("datasets/minimap/screenshots")

            if not dataset_dir.exists():
                QMessageBox.warning(
                    self,
                    "Dataset Not Found",
                    f"Dataset directory does not exist:\n{dataset_dir}"
                )
                return

            # Get list of all synthetic minimap images
            image_files = list(dataset_dir.glob("*.png"))

            if not image_files:
                QMessageBox.warning(
                    self,
                    "No Dataset Images",
                    "No synthetic minimap images found in the dataset."
                )
                return

            # Create a dialog to select an image
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, QLabel

            dialog = QDialog(self)
            dialog.setWindowTitle("Select Synthetic Minimap")
            dialog.setMinimumSize(500, 400)

            layout = QVBoxLayout(dialog)

            # Instructions
            label = QLabel(f"Select a synthetic minimap image ({len(image_files)} available):")
            layout.addWidget(label)

            # List widget
            list_widget = QListWidget()

            # Load dataset entries to show more info
            entries = self.dataset_manager.load_minimap_dataset()
            entry_dict = {entry.screenshot_path.split('/')[-1]: entry for entry in entries}

            for img_file in sorted(image_files):
                filename = img_file.name
                entry = entry_dict.get(filename)

                if entry:
                    item_text = f"{filename} - Floor {entry.floor} - ({entry.crosshair_x:.0f}, {entry.crosshair_y:.0f})"
                else:
                    item_text = filename

                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, img_file)
                list_widget.addItem(item)

            layout.addWidget(list_widget)

            # Buttons
            button_layout = QHBoxLayout()
            select_btn = QPushButton("Select")
            cancel_btn = QPushButton("Cancel")

            button_layout.addStretch()
            button_layout.addWidget(select_btn)
            button_layout.addWidget(cancel_btn)

            layout.addLayout(button_layout)

            # Connect buttons
            selected_file = None

            def on_select():
                nonlocal selected_file
                current_item = list_widget.currentItem()
                if current_item:
                    selected_file = current_item.data(Qt.ItemDataRole.UserRole)
                    dialog.accept()

            def on_cancel():
                dialog.reject()

            select_btn.clicked.connect(on_select)
            cancel_btn.clicked.connect(on_cancel)
            list_widget.itemDoubleClicked.connect(on_select)

            # Show dialog
            if dialog.exec() == QDialog.DialogCode.Accepted and selected_file:
                # Load the selected image
                self.synthetic_minimap_image = Image.open(selected_file)

                # Update status
                self.comparison_status_label.setText(
                    f"✓ Synthetic minimap loaded: {selected_file.name}"
                )
                self.comparison_status_label.setStyleSheet("color: green; font-size: 10px;")

                # Enable comparison button if both images are loaded
                self.update_comparison_button_state()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load synthetic minimap:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()

    def update_comparison_button_state(self):
        """Enable/disable the comparison button based on loaded images."""
        if self.game_minimap_image and self.synthetic_minimap_image:
            self.show_comparison_btn.setEnabled(True)
            self.show_comparison_btn.setStyleSheet(
                "background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;"
            )
        else:
            self.show_comparison_btn.setEnabled(False)
            self.show_comparison_btn.setStyleSheet("")

    def show_comparison_window(self):
        """Show a window with side-by-side comparison of the two images."""
        if not self.game_minimap_image or not self.synthetic_minimap_image:
            QMessageBox.warning(
                self,
                "Images Not Loaded",
                "Please load both game and synthetic minimap images first."
            )
            return

        # Create and show the comparison window
        comparison_window = MinimapComparisonWindow(
            self.game_minimap_image,
            self.synthetic_minimap_image,
            self
        )
        comparison_window.show()


class MonsterTrackingPanel(QWidget):
    """Panel for monster tracking using Exiva spell."""

    def __init__(self, parent_widget, parent=None):
        super().__init__(parent)
        self.parent_widget = parent_widget  # Reference to MinimapViewerWidget
        self.setup_ui()

    def setup_ui(self):
        """Set up the monster tracking panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Title
        title_label = QLabel("Monster Tracking")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)

        # Automated tracking group
        auto_tracking_group = QGroupBox("Automated Crosshair Detection")
        auto_tracking_layout = QVBoxLayout(auto_tracking_group)
        auto_tracking_layout.setContentsMargins(8, 8, 8, 8)
        auto_tracking_layout.setSpacing(5)

        # Start/Stop button
        self.start_tracking_button = QPushButton("Start Monster Tracking")
        self.start_tracking_button.clicked.connect(self.parent_widget.start_automated_tracking)
        auto_tracking_layout.addWidget(self.start_tracking_button)

        self.stop_tracking_button = QPushButton("Stop Monster Tracking")
        self.stop_tracking_button.clicked.connect(self.parent_widget.stop_automated_tracking)
        self.stop_tracking_button.setEnabled(False)
        auto_tracking_layout.addWidget(self.stop_tracking_button)

        # Status display
        self.tracking_status_label = QLabel("Not tracking")
        self.tracking_status_label.setStyleSheet(
            "font-style: italic; color: #666; padding: 5px; "
            "background-color: #f0f0f0; border-radius: 3px;"
        )
        self.tracking_status_label.setWordWrap(True)
        auto_tracking_layout.addWidget(self.tracking_status_label)

        # Crosshair position display
        position_label = QLabel("Detected Crosshair Position:")
        position_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
        auto_tracking_layout.addWidget(position_label)

        self.crosshair_position_label = QLabel("Not detected")
        self.crosshair_position_label.setStyleSheet("color: #666; padding: 3px;")
        auto_tracking_layout.addWidget(self.crosshair_position_label)

        # Exiva extraction display
        exiva_extract_label = QLabel("Extracted Exiva Reading:")
        exiva_extract_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
        auto_tracking_layout.addWidget(exiva_extract_label)

        self.exiva_extraction_label = QLabel("Not detected")
        self.exiva_extraction_label.setStyleSheet("color: #666; padding: 3px;")
        self.exiva_extraction_label.setWordWrap(True)
        auto_tracking_layout.addWidget(self.exiva_extraction_label)

        layout.addWidget(auto_tracking_group)

        # Exiva tracking group
        exiva_group = QGroupBox("Exiva Spell Tracking")
        exiva_group_layout = QVBoxLayout(exiva_group)
        exiva_group_layout.setContentsMargins(8, 8, 8, 8)
        exiva_group_layout.setSpacing(5)

        # Session control buttons
        session_buttons_layout = QHBoxLayout()
        self.start_hunt_button = QPushButton("Start Hunt")
        self.start_hunt_button.clicked.connect(self.parent_widget.start_monster_hunt)
        session_buttons_layout.addWidget(self.start_hunt_button)

        self.reset_hunt_button = QPushButton("Reset Hunt")
        self.reset_hunt_button.clicked.connect(self.parent_widget.reset_monster_hunt)
        self.reset_hunt_button.setEnabled(False)
        session_buttons_layout.addWidget(self.reset_hunt_button)

        exiva_group_layout.addLayout(session_buttons_layout)

        # Direction selection
        direction_label = QLabel("Direction:")
        direction_label.setStyleSheet("font-weight: bold;")
        exiva_group_layout.addWidget(direction_label)

        self.direction_combo = QComboBox()
        self.direction_combo.addItem("North", ExivaDirection.NORTH)
        self.direction_combo.addItem("Northeast", ExivaDirection.NORTHEAST)
        self.direction_combo.addItem("East", ExivaDirection.EAST)
        self.direction_combo.addItem("Southeast", ExivaDirection.SOUTHEAST)
        self.direction_combo.addItem("South", ExivaDirection.SOUTH)
        self.direction_combo.addItem("Southwest", ExivaDirection.SOUTHWEST)
        self.direction_combo.addItem("West", ExivaDirection.WEST)
        self.direction_combo.addItem("Northwest", ExivaDirection.NORTHWEST)
        self.direction_combo.setEnabled(False)
        exiva_group_layout.addWidget(self.direction_combo)

        # Distance selection
        distance_label = QLabel("Distance:")
        distance_label.setStyleSheet("font-weight: bold;")
        exiva_group_layout.addWidget(distance_label)

        self.distance_combo = QComboBox()
        self.distance_combo.addItem("Close (5-100)", ExivaRange.TO_THE)
        self.distance_combo.addItem("Far (101-250)", ExivaRange.FAR)
        self.distance_combo.addItem("Very Far (251+)", ExivaRange.VERY_FAR)
        self.distance_combo.setEnabled(False)
        exiva_group_layout.addWidget(self.distance_combo)

        # Monster difficulty selection
        difficulty_label = QLabel("Monster Difficulty:")
        difficulty_label.setStyleSheet("font-weight: bold;")
        exiva_group_layout.addWidget(difficulty_label)

        self.difficulty_combo = QComboBox()
        self.difficulty_combo.addItem("Unknown (?)", MonsterDifficulty.UNKNOWN)
        self.difficulty_combo.addItem("Harmless (1)", MonsterDifficulty.HARMLESS)
        self.difficulty_combo.addItem("Trivial (5)", MonsterDifficulty.TRIVIAL)
        self.difficulty_combo.addItem("Easy (15)", MonsterDifficulty.EASY)
        self.difficulty_combo.addItem("Medium (25)", MonsterDifficulty.MEDIUM)
        self.difficulty_combo.addItem("Hard (50)", MonsterDifficulty.HARD)
        self.difficulty_combo.addItem("Challenging (100)", MonsterDifficulty.CHALLENGING)
        self.difficulty_combo.setEnabled(False)
        exiva_group_layout.addWidget(self.difficulty_combo)

        # Add reading button
        self.add_reading_button = QPushButton("Add Exiva Reading")
        self.add_reading_button.clicked.connect(self.parent_widget.add_exiva_reading)
        self.add_reading_button.setEnabled(False)
        exiva_group_layout.addWidget(self.add_reading_button)

        # Auto-restart on difficulty change checkbox
        self.auto_restart_checkbox = QCheckBox("Auto-restart on difficulty change")
        self.auto_restart_checkbox.setChecked(True)  # Enabled by default
        self.auto_restart_checkbox.setToolTip(
            "Automatically restart the hunt session when monster difficulty changes.\n"
            "Smoothly clears old readings and overlay when the monster changes.\n"
            "Useful when someone else kills the monster you're tracking."
        )
        exiva_group_layout.addWidget(self.auto_restart_checkbox)

        # Session status
        self.session_status_label = QLabel("No active hunt session")
        self.session_status_label.setStyleSheet("font-style: italic; color: #666;")
        exiva_group_layout.addWidget(self.session_status_label)

        # Dimmed area percentage
        self.dimmed_percentage_label = QLabel("Dimmed area: 0.0%")
        self.dimmed_percentage_label.setStyleSheet("font-style: italic; color: #666; font-size: 11px;")
        self.dimmed_percentage_label.setToolTip("Percentage of the map that has been dimmed by Exiva readings")
        exiva_group_layout.addWidget(self.dimmed_percentage_label)

        layout.addWidget(exiva_group)

        # Add stretch to push content to top
        layout.addStretch()


class AreaPropertiesPanel(QWidget):
    """Properties panel for editing area attributes."""

    areaUpdated = pyqtSignal(AreaData)
    areaDeleted = pyqtSignal(str)  # area_id

    def __init__(self, area_data_manager: AreaDataManager, parent=None):
        super().__init__(parent)
        self.area_data_manager = area_data_manager
        self.current_area: Optional[AreaData] = None
        self.updating_ui = False

        self.setup_ui()
        self.setEnabled(False)  # Disabled until an area is selected

    def setup_ui(self):
        """Set up the properties panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Title
        title_label = QLabel("Area Properties")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)

        # Scroll area for properties
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        properties_widget = QWidget()
        properties_layout = QVBoxLayout(properties_widget)
        properties_layout.setContentsMargins(5, 5, 5, 5)
        properties_layout.setSpacing(15)

        # Basic Information Group
        basic_group = QGroupBox("Basic Information")
        basic_layout = QFormLayout(basic_group)

        self.name_edit = QLineEdit()
        self.name_edit.textChanged.connect(self.on_name_changed)
        basic_layout.addRow("Name:", self.name_edit)

        properties_layout.addWidget(basic_group)

        # Monster Difficulty Group
        difficulty_group = QGroupBox("Monster Difficulty")
        difficulty_layout = QVBoxLayout(difficulty_group)

        # Charm numbers for each difficulty level
        charm_numbers = {
            MonsterDifficulty.NONE: "-",
            MonsterDifficulty.UNKNOWN: "?",
            MonsterDifficulty.HARMLESS: "1",
            MonsterDifficulty.TRIVIAL: "5",
            MonsterDifficulty.EASY: "15",
            MonsterDifficulty.MEDIUM: "25",
            MonsterDifficulty.HARD: "50",
            MonsterDifficulty.CHALLENGING: "100",
        }

        self.difficulty_checkboxes = {}
        for difficulty in MonsterDifficulty:
            # Skip NONE difficulty - it's only for ML training data, not manual area editing
            if difficulty == MonsterDifficulty.NONE:
                continue

            charm_num = charm_numbers.get(difficulty, "?")
            label_text = f"{difficulty.value.title()} ({charm_num})"
            checkbox = QCheckBox(label_text)
            checkbox.stateChanged.connect(self.on_difficulty_changed)
            self.difficulty_checkboxes[difficulty.value] = checkbox
            difficulty_layout.addWidget(checkbox)

        properties_layout.addWidget(difficulty_group)

        # Route Group
        route_group = QGroupBox("Route")
        route_layout = QVBoxLayout(route_group)

        # Route description text edit
        self.route_text_edit = QTextEdit()
        self.route_text_edit.setPlaceholderText("Describe how to get to this area...")
        self.route_text_edit.setMaximumHeight(150)
        self.route_text_edit.textChanged.connect(self.on_route_changed)
        route_layout.addWidget(self.route_text_edit)

        properties_layout.addWidget(route_group)

        # Action buttons
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setContentsMargins(0, 0, 0, 0)

        self.delete_button = QPushButton("Delete Area")
        self.delete_button.setStyleSheet("background-color: #ff4444; color: white;")
        self.delete_button.clicked.connect(self.on_delete_clicked)
        button_layout.addWidget(self.delete_button)

        properties_layout.addWidget(button_widget)

        # Add stretch to push everything to the top
        properties_layout.addStretch()

        scroll_area.setWidget(properties_widget)
        layout.addWidget(scroll_area)

    def set_area(self, area_data: Optional[AreaData]):
        """Set the area to edit."""
        self.current_area = area_data
        self.setEnabled(area_data is not None)

        if area_data:
            self.updating_ui = True

            # Update basic information
            self.name_edit.setText(area_data.name)

            # Update difficulty checkboxes
            for difficulty_value, checkbox in self.difficulty_checkboxes.items():
                checkbox.setChecked(difficulty_value in area_data.difficulty_levels)

            # Update route text
            self.route_text_edit.setPlainText(area_data.route)

            self.updating_ui = False
        else:
            # Clear route text when no area is selected
            self.route_text_edit.clear()



    def on_name_changed(self):
        """Handle name change."""
        if self.updating_ui or not self.current_area:
            return

        self.current_area.name = self.name_edit.text()
        self.current_area.modified_timestamp = time.time()
        self.save_and_emit_update()

    def on_difficulty_changed(self):
        """Handle difficulty level change."""
        if self.updating_ui or not self.current_area:
            return

        selected_difficulties = []
        for difficulty_value, checkbox in self.difficulty_checkboxes.items():
            if checkbox.isChecked():
                selected_difficulties.append(difficulty_value)

        self.current_area.difficulty_levels = selected_difficulties

        # Auto-update color based on selected difficulties
        default_colors = self.area_data_manager.get_default_colors()
        if len(selected_difficulties) == 1:
            # Single monster type - use the specific color
            new_color = default_colors[selected_difficulties[0]]
            self.current_area.color = new_color
        elif len(selected_difficulties) > 1:
            # Multiple monster types - store all colors for multi-color rendering
            # Use the first difficulty's color as the primary color for backward compatibility
            # The actual multi-color rendering will be handled in AreaGraphicsItem
            self.current_area.color = default_colors[selected_difficulties[0]]

        self.current_area.modified_timestamp = time.time()
        self.save_and_emit_update()

    def on_route_changed(self):
        """Handle route text change."""
        if self.updating_ui or not self.current_area:
            return

        self.current_area.route = self.route_text_edit.toPlainText()
        self.current_area.modified_timestamp = time.time()
        self.save_and_emit_update()

    def on_delete_clicked(self):
        """Handle delete button click."""
        if not self.current_area:
            return

        reply = QMessageBox.question(
            self, "Delete Area",
            f"Are you sure you want to delete '{self.current_area.name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            area_id = self.current_area.area_id
            self.area_data_manager.remove_area(self.current_area.floor, area_id)
            self.areaDeleted.emit(area_id)
            self.set_area(None)

    def save_and_emit_update(self):
        """Save the area and emit update signal."""
        if self.current_area:
            self.area_data_manager.update_area(self.current_area)
            self.areaUpdated.emit(self.current_area)


class MinimapViewer(QWidget):
    """Main minimap viewer widget with floor navigation and zoom controls."""

    floorChanged = pyqtSignal(int)

    def __init__(self, minimap_dir: str = "processed_minimap", parent=None):
        super().__init__(parent)

        self.minimap_dir = Path(minimap_dir)
        self.current_floor = 7
        self.floor_images: Dict[int, QPixmap] = {}

        self.floor_order = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        ]

        # Timer for debouncing transparency changes
        self.transparency_save_timer = QTimer()
        self.transparency_save_timer.setSingleShot(True)
        self.transparency_save_timer.timeout.connect(self.save_transparency_setting)

        self.setup_ui()
        self.load_floor_images()
        self.set_floor(self.current_floor)
    
    def setup_ui(self):
        """Set up the user interface with vertical layout and sidebars."""
        # Main horizontal layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Left sidebar for controls
        left_sidebar = QFrame()
        left_sidebar.setFrameStyle(QFrame.Shape.StyledPanel)
        left_sidebar.setFixedWidth(200)
        left_sidebar.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        left_layout = QVBoxLayout(left_sidebar)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)

        # Floor controls section
        floor_group = QFrame()
        floor_group.setFrameStyle(QFrame.Shape.Box)
        floor_group_layout = QVBoxLayout(floor_group)
        floor_group_layout.setContentsMargins(8, 8, 8, 8)
        floor_group_layout.setSpacing(5)

        floor_title = QLabel("Floor Navigation")
        floor_title.setStyleSheet("font-weight: bold; font-size: 12px;")
        floor_group_layout.addWidget(floor_title)

        floor_label = QLabel("Floor:")
        floor_group_layout.addWidget(floor_label)

        # Floor navigation buttons and combo
        floor_nav_layout = QHBoxLayout()
        floor_nav_layout.setSpacing(8)  # Add proper spacing between elements

        self.floor_down_btn = QPushButton("-")
        self.floor_down_btn.setMaximumWidth(30)
        self.floor_down_btn.setToolTip("Go down one floor")
        self.floor_down_btn.clicked.connect(self.floor_down)

        self.floor_combo = QComboBox()
        self.floor_combo.setMinimumWidth(100)
        self.floor_combo.currentTextChanged.connect(self.on_floor_changed)

        self.floor_up_btn = QPushButton("+")
        self.floor_up_btn.setMaximumWidth(30)
        self.floor_up_btn.setToolTip("Go up one floor")
        self.floor_up_btn.clicked.connect(self.floor_up)

        floor_nav_layout.addWidget(self.floor_down_btn)
        floor_nav_layout.addWidget(self.floor_combo)
        floor_nav_layout.addWidget(self.floor_up_btn)
        floor_group_layout.addLayout(floor_nav_layout)

        # Area editing controls section
        area_group = QFrame()
        area_group.setFrameStyle(QFrame.Shape.Box)
        area_group_layout = QVBoxLayout(area_group)
        area_group_layout.setContentsMargins(8, 8, 8, 8)
        area_group_layout.setSpacing(5)

        area_title = QLabel("Area Editing")
        area_title.setStyleSheet("font-weight: bold; font-size: 12px;")
        area_group_layout.addWidget(area_title)

        # Area editing toggle button
        self.area_editing_toggle = QPushButton("Enable Area Editing")
        self.area_editing_toggle.setCheckable(True)
        self.area_editing_toggle.setChecked(False)
        self.area_editing_toggle.clicked.connect(self.on_area_editing_toggle)
        area_group_layout.addWidget(self.area_editing_toggle)

        # Possible Areas section
        possible_areas_group = QFrame()
        possible_areas_group.setFrameStyle(QFrame.Shape.Box)
        possible_areas_layout = QVBoxLayout(possible_areas_group)
        possible_areas_layout.setContentsMargins(8, 8, 8, 8)
        possible_areas_layout.setSpacing(5)

        possible_areas_title = QLabel("Possible Areas")
        possible_areas_title.setStyleSheet("font-weight: bold; font-size: 12px;")
        possible_areas_layout.addWidget(possible_areas_title)

        # List widget to display possible areas
        self.possible_areas_list = QListWidget()
        self.possible_areas_list.setMaximumHeight(200)
        self.possible_areas_list.setToolTip(
            "Shows areas matching current monster difficulty\n"
            "that are not inside dimmed regions.\n"
            "Click on an area to center the camera on it."
        )
        # Connect click handler
        self.possible_areas_list.itemClicked.connect(self.on_possible_area_clicked)
        possible_areas_layout.addWidget(self.possible_areas_list)

        # Always visible, but empty when no active hunt
        self.possible_areas_group = possible_areas_group

        # Initialize with placeholder text
        placeholder_item = QListWidgetItem("(No active hunt)")
        placeholder_item.setForeground(QColor("#999"))
        self.possible_areas_list.addItem(placeholder_item)

        # Add groups to left sidebar
        left_layout.addWidget(floor_group)
        left_layout.addWidget(area_group)
        left_layout.addWidget(possible_areas_group)
        left_layout.addStretch()  # Push content to top

        # Center area for main content
        center_frame = QFrame()
        center_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        center_layout = QVBoxLayout(center_frame)
        center_layout.setContentsMargins(0, 0, 0, 0)

        self.graphics_view = MinimapGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)

        # Now that graphics_view is created, add the global transparency slider
        transparency_label = QLabel("Area Transparency:")
        area_group_layout.addWidget(transparency_label)

        self.global_transparency_slider = QSlider(Qt.Orientation.Horizontal)
        self.global_transparency_slider.setRange(0, 100)
        self.global_transparency_slider.setValue(int(self.graphics_view.area_data_manager.global_transparency * 100))
        self.global_transparency_slider.valueChanged.connect(self.on_global_transparency_changed)

        self.global_transparency_label = QLabel(f"{int(self.graphics_view.area_data_manager.global_transparency * 100)}%")
        transparency_widget = QWidget()
        transparency_layout = QHBoxLayout(transparency_widget)
        transparency_layout.setContentsMargins(0, 0, 0, 0)
        transparency_layout.addWidget(self.global_transparency_slider)
        transparency_layout.addWidget(self.global_transparency_label)

        area_group_layout.addWidget(transparency_widget)

        # Connect area editing signals
        self.graphics_view.areaCreated.connect(self.on_area_created)
        self.graphics_view.areaSelected.connect(self.on_area_selected)

        # Connect difficulty change restart signal
        self.graphics_view.difficultyChangeRestart.connect(self.on_difficulty_change_restart)

        # Connect full dim restart signal
        self.graphics_view.fullDimRestart.connect(self.on_full_dim_restart)

        center_layout.addWidget(self.graphics_view)

        # Right sidebar for area properties and dataset tools
        right_sidebar = QFrame()
        right_sidebar.setFrameStyle(QFrame.Shape.StyledPanel)
        right_sidebar.setFixedWidth(300)
        right_sidebar.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        right_layout = QVBoxLayout(right_sidebar)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Tab widget for different panels
        self.right_tab_widget = QTabWidget()

        # Track previous tab for area selection/deselection
        self.previous_tab_index = 0

        # Monster tracking panel
        self.monster_tracking_panel = MonsterTrackingPanel(self)
        self.right_tab_widget.addTab(self.monster_tracking_panel, "Monster Tracking")

        # Area properties panel
        self.area_properties_panel = AreaPropertiesPanel(self.graphics_view.area_data_manager)
        self.area_properties_panel.areaUpdated.connect(self.on_area_updated)
        self.area_properties_panel.areaDeleted.connect(self.on_area_deleted)
        self.right_tab_widget.addTab(self.area_properties_panel, "Areas")

        # Dataset tests panel
        self.dataset_tests_panel = DatasetTestsPanel(self)
        self.right_tab_widget.addTab(self.dataset_tests_panel, "Dataset Tests")

        # Connect floor change signal to update coverage overlay
        self.floorChanged.connect(self.dataset_tests_panel.on_floor_changed)

        # Dataset creator panels (lazy loaded)
        self.minimap_dataset_panel = None
        self.exiva_dataset_panel = None

        right_layout.addWidget(self.right_tab_widget)

        # Add all sections to main layout
        main_layout.addWidget(left_sidebar)
        main_layout.addWidget(center_frame, 1)  # Center gets all extra space
        main_layout.addWidget(right_sidebar)

        self.current_minimap_item: Optional[QGraphicsPixmapItem] = None

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def load_floor_images(self):
        """Load all available floor images."""
        logger.info(f"Loading floor images from {self.minimap_dir}")

        if not self.minimap_dir.exists():
            logger.error(f"Minimap directory not found: {self.minimap_dir}")
            return

        loaded_floors = []
        for floor in self.floor_order:
            floor_file = self.minimap_dir / f"floor_{floor:02d}.png"
            if floor_file.exists():
                try:
                    pixmap = QPixmap(str(floor_file))
                    if not pixmap.isNull():
                        self.floor_images[floor] = pixmap
                        loaded_floors.append(floor)
                    else:
                        logger.warning(f"Failed to load floor image: {floor_file}")
                except Exception as e:
                    logger.error(f"Error loading floor {floor}: {e}")

        self.floor_combo.blockSignals(True)
        self.floor_combo.clear()
        for floor in self.floor_order:
            if floor in self.floor_images:
                if floor == 7:
                    self.floor_combo.addItem(f"Floor {floor:02d} (Main)", floor)
                else:
                    self.floor_combo.addItem(f"Floor {floor:02d}", floor)
        self.floor_combo.blockSignals(False)

        logger.info(f"Loaded {len(loaded_floors)} floor images: {loaded_floors}")

    def set_floor(self, floor: int):
        """Set the current floor and display its minimap."""
        if floor not in self.floor_images:
            logger.warning(f"Floor {floor} not available")
            return

        # Preserve area editing mode before floor switch
        current_area_editing_mode = self.graphics_view.area_editing_mode

        # Clean up any in-progress area creation before clearing the scene
        # This prevents crashes when temp_area_item gets removed from scene
        if self.graphics_view.creating_area:
            logger.info("Canceling in-progress area creation due to floor switch")
            self.graphics_view.cancel_area_creation()

        # Properly deselect any selected area before clearing graphics items
        # This ensures the visual selection state is cleared and deselection signal is emitted
        if self.graphics_view.selected_area_id is not None:
            self.graphics_view.deselect_area()

        # Clear area graphics items dictionary before clearing scene
        # This prevents crashes when scene.clear() deletes the graphics items
        self.graphics_view.area_graphics_items.clear()

        self.current_floor = floor
        self.graphics_view.current_floor_id = floor

        self.graphics_scene.clear()

        # Preserve Exiva session across floor changes - sessions should persist globally
        # Only the "Reset Hunt" button should clear Exiva sessions

        self.graphics_view.crosshair_diagonals = [None] * 8
        self.graphics_view.crosshair_center_square = None
        self.graphics_view.crosshair_inner_range = None
        self.graphics_view.crosshair_outer_range = None

        current_zoom = self.graphics_view.zoom_factor
        current_center = self.graphics_view.mapToScene(self.graphics_view.viewport().rect().center())

        pixmap = self.floor_images[floor]
        self.current_minimap_item = QGraphicsPixmapItem(pixmap)
        self.graphics_scene.addItem(self.current_minimap_item)

        new_scene_rect = self.current_minimap_item.boundingRect()
        self.graphics_scene.setSceneRect(new_scene_rect)

        logger.info(f"FLOOR_CHANGE - Floor {floor}: Scene rect {new_scene_rect.width():.0f}x{new_scene_rect.height():.0f}")

        if hasattr(self, '_first_floor_loaded'):
            self.graphics_view.zoom_to_factor(current_zoom)

            viewport_rect = self.graphics_view.viewport().rect()
            target_viewport_center = self.graphics_view.mapFromScene(current_center)
            current_viewport_center = viewport_rect.center()
            offset_x = target_viewport_center.x() - current_viewport_center.x()
            offset_y = target_viewport_center.y() - current_viewport_center.y()

            h_scroll = self.graphics_view.horizontalScrollBar()
            v_scroll = self.graphics_view.verticalScrollBar()
            h_scroll.setValue(h_scroll.value() + int(offset_x))
            v_scroll.setValue(v_scroll.value() + int(offset_y))

            logger.info(f"GLOBAL CAMERA - Floor {floor}: Zoom: {current_zoom:.4f}, Position: ({current_center.x():.2f}, {current_center.y():.2f}) - UNCHANGED")
        else:
            self.fit_view()
            self._first_floor_loaded = True

        self.graphics_view.restore_crosshairs(floor)

        # Load areas for this floor
        self.graphics_view.load_areas_for_floor(floor)

        # Recreate Exiva overlay if there's an active session
        # This ensures overlays appear on all floors when session is active
        if self.graphics_view.current_exiva_session is not None:
            self.graphics_view.update_exiva_overlay()

        # Restore area editing mode after floor switch
        # This preserves the user's editing mode selection across floor changes
        self.graphics_view.set_area_editing_mode(current_area_editing_mode)

        self.floor_combo.blockSignals(True)
        for i in range(self.floor_combo.count()):
            if self.floor_combo.itemData(i) == floor:
                self.floor_combo.setCurrentIndex(i)
                break
        self.floor_combo.blockSignals(False)

        logger.info(f"Switched to floor {floor}")
        self.floorChanged.emit(floor)

    def on_floor_changed(self, _floor_text: str):
        """Handle floor selection change from combo box."""
        current_data = self.floor_combo.currentData()
        if current_data is not None:
            self.set_floor(current_data)

    def floor_up(self):
        """Navigate to the next floor up (lower floor number)."""
        current_index = self.floor_order.index(self.current_floor)
        if current_index > 0:
            next_floor = self.floor_order[current_index - 1]
            if next_floor in self.floor_images:
                self.set_floor(next_floor)

    def floor_down(self):
        """Navigate to the next floor down (higher floor number)."""
        current_index = self.floor_order.index(self.current_floor)
        if current_index < len(self.floor_order) - 1:
            next_floor = self.floor_order[current_index + 1]
            if next_floor in self.floor_images:
                self.set_floor(next_floor)

    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key.Key_Plus or event.key() == Qt.Key.Key_Equal:
            self.floor_up()
        elif event.key() == Qt.Key.Key_Minus:
            self.floor_down()
        elif event.key() == Qt.Key.Key_C and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.clear_crosshairs()
        elif event.key() == Qt.Key.Key_Z and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.undo_area_point()
        else:
            super().keyPressEvent(event)

    def clear_crosshairs(self):
        """Clear global crosshairs."""
        self.graphics_view.clear_all_crosshairs()

    def undo_area_point(self):
        """Undo the last point placement during area creation."""
        self.graphics_view.undo_last_area_point()
        logger.info("Cleared global crosshairs")

    def fit_view(self):
        """Fit the minimap in the view."""
        self.graphics_view.fit_in_view_with_margin(0.05)

    def get_current_floor(self) -> int:
        """Get the currently displayed floor."""
        return self.current_floor

    def get_available_floors(self) -> list[int]:
        """Get list of available floors."""
        return list(self.floor_images.keys())

    def get_camera_info(self) -> dict:
        """Get current camera information."""
        center = self.graphics_view.mapToScene(self.graphics_view.viewport().rect().center())
        return {
            'floor': self.current_floor,
            'zoom_factor': self.graphics_view.zoom_factor,
            'center_x': center.x(),
            'center_y': center.y()
        }

    # Area editing methods
    def on_global_transparency_changed(self, value: int):
        """Handle global transparency slider change."""
        transparency = value / 100.0
        self.global_transparency_label.setText(f"{value}%")

        # Update the transparency immediately for visual feedback
        self.graphics_view.area_data_manager.global_transparency = transparency
        for graphics_item in self.graphics_view.area_graphics_items.values():
            graphics_item.set_global_transparency(transparency)

        # Debounce the save operation - only save after 500ms of no changes
        self.transparency_save_timer.stop()
        self.transparency_save_timer.start(500)

    def save_transparency_setting(self):
        """Save the transparency setting to disk (called by debounce timer)."""
        self.graphics_view.area_data_manager.save_settings()

    def on_area_editing_toggle(self):
        """Handle area editing toggle button."""
        if self.area_editing_toggle.isChecked():
            self.set_area_editing_mode(AreaEditingMode.POLYGON)
            self.area_editing_toggle.setText("Disable Area Editing")
        else:
            self.set_area_editing_mode(AreaEditingMode.DISABLED)
            self.area_editing_toggle.setText("Enable Area Editing")

    def set_area_editing_mode(self, mode: AreaEditingMode):
        """Set the area editing mode and update UI."""
        self.graphics_view.set_area_editing_mode(mode)

        # Update toggle button state
        if mode == AreaEditingMode.DISABLED:
            self.area_editing_toggle.setChecked(False)
            self.area_editing_toggle.setText("Enable Area Editing")
        elif mode == AreaEditingMode.POLYGON:
            self.area_editing_toggle.setChecked(True)
            self.area_editing_toggle.setText("Disable Area Editing")

    def on_area_created(self, area_data: AreaData):
        """Handle area creation."""
        logger.info(f"Area created: {area_data.name} on floor {area_data.floor}")

    def get_areas_tab_index(self) -> Optional[int]:
        """Get the index of the Areas tab in the right tab widget."""
        for i in range(self.right_tab_widget.count()):
            if self.right_tab_widget.tabText(i) == "Areas":
                return i
        return None

    def on_area_selected(self, area_id: str):
        """Handle area selection and deselection."""
        if area_id:
            # Area selected
            area_data = self.graphics_view.area_data_manager.get_area_by_id(self.current_floor, area_id)
            self.area_properties_panel.set_area(area_data)

            # Save current tab index if not already on Areas tab
            areas_tab_index = self.get_areas_tab_index()
            if areas_tab_index is not None and self.right_tab_widget.currentIndex() != areas_tab_index:
                self.previous_tab_index = self.right_tab_widget.currentIndex()

            # Switch to Areas tab
            if areas_tab_index is not None:
                self.right_tab_widget.setCurrentIndex(areas_tab_index)
        else:
            # Area deselected (empty area_id)
            self.area_properties_panel.set_area(None)

            # Restore previous tab
            areas_tab_index = self.get_areas_tab_index()
            if areas_tab_index is not None and self.right_tab_widget.currentIndex() == areas_tab_index:
                # Only restore if we're currently on the Areas tab
                self.right_tab_widget.setCurrentIndex(self.previous_tab_index)

    def on_area_updated(self, area_data: AreaData):
        """Handle area property updates."""
        if area_data.area_id in self.graphics_view.area_graphics_items:
            graphics_item = self.graphics_view.area_graphics_items[area_data.area_id]
            graphics_item.update_from_area_data(area_data)
        logger.info(f"Area updated: {area_data.name}")

    def on_area_deleted(self, area_id: str):
        """Handle area deletion."""
        # Properly deselect the area if it's currently selected
        if self.graphics_view.selected_area_id == area_id:
            self.graphics_view.deselect_area()

        self.graphics_view.remove_area_graphics_item(area_id)
        logger.info(f"Area deleted: {area_id}")

    # Exiva tracking methods
    def start_monster_hunt(self):
        """Start a new monster hunt session."""
        session_id = self.graphics_view.start_exiva_session()

        # Update UI state in monster tracking panel
        panel = self.monster_tracking_panel
        panel.start_hunt_button.setEnabled(False)
        panel.reset_hunt_button.setEnabled(True)
        panel.direction_combo.setEnabled(True)
        panel.distance_combo.setEnabled(True)
        panel.difficulty_combo.setEnabled(True)
        panel.add_reading_button.setEnabled(True)

        panel.session_status_label.setText(f"Active hunt session: {session_id[:8]}...")
        panel.session_status_label.setStyleSheet("font-style: italic; color: #008000;")

        # Reset dimmed percentage display
        self.update_dimmed_percentage_display()

        # Update possible areas list (will hide it since no readings yet)
        self.update_possible_areas_list()

        logger.info("Started new monster hunt session from UI")

    def reset_monster_hunt(self):
        """Reset the current monster hunt session."""
        self.graphics_view.end_exiva_session()

        # Update UI state in monster tracking panel
        panel = self.monster_tracking_panel
        panel.start_hunt_button.setEnabled(True)
        panel.reset_hunt_button.setEnabled(False)
        panel.direction_combo.setEnabled(False)
        panel.distance_combo.setEnabled(False)
        panel.difficulty_combo.setEnabled(False)
        panel.add_reading_button.setEnabled(False)

        panel.session_status_label.setText("No active hunt session")
        panel.session_status_label.setStyleSheet("font-style: italic; color: #666;")

        # Reset dimmed percentage display
        self.update_dimmed_percentage_display()

        # Update possible areas list (will hide it since no active session)
        self.update_possible_areas_list()

        logger.info("Reset monster hunt session from UI")

    def add_exiva_reading(self):
        """Add an Exiva reading to the current session."""
        if not self.graphics_view.has_active_exiva_session():
            logger.warning("No active hunt session to add reading to")
            return

        if not self.graphics_view.global_crosshair_position:
            # Show a message to the user
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "No Crosshair Position",
                "Please place the crosshair on the map at your current position before adding an Exiva reading.\n\n"
                "Click on the map to place the crosshair."
            )
            return

        # Get selected direction, distance, and difficulty from monster tracking panel
        panel = self.monster_tracking_panel
        direction = panel.direction_combo.currentData()
        distance = panel.distance_combo.currentData()
        difficulty = panel.difficulty_combo.currentData()

        if direction is None or distance is None or difficulty is None:
            logger.warning("Invalid direction, distance, or difficulty selection")
            return

        # Get auto-restart setting from checkbox
        auto_restart = panel.auto_restart_checkbox.isChecked()

        # Add the reading
        success = self.graphics_view.add_exiva_reading(
            direction,
            distance,
            difficulty,
            auto_restart_on_difficulty_change=auto_restart
        )

        if success:
            # Update session status to show reading count
            session = self.graphics_view.get_current_exiva_session()
            if session:
                reading_count = len(session.readings)
                panel.session_status_label.setText(
                    f"Active hunt: {session.session_id[:8]}... ({reading_count} reading{'s' if reading_count != 1 else ''})"
                )

            # Update dimmed percentage display
            self.update_dimmed_percentage_display()

            # Update possible areas list
            self.update_possible_areas_list()

            logger.info(f"Added Exiva reading: {direction.value} {distance.value} ({difficulty.value})")
        else:
            logger.error("Failed to add Exiva reading")

    def reset_exiva_ui_state(self):
        """Reset Exiva UI controls to initial state."""
        panel = self.monster_tracking_panel
        panel.start_hunt_button.setEnabled(True)
        panel.reset_hunt_button.setEnabled(False)
        panel.direction_combo.setEnabled(False)
        panel.distance_combo.setEnabled(False)
        panel.difficulty_combo.setEnabled(False)
        panel.add_reading_button.setEnabled(False)

        panel.session_status_label.setText("No active hunt session")
        panel.session_status_label.setStyleSheet("font-style: italic; color: #666;")

    def on_difficulty_change_restart(self, old_difficulty: str, new_difficulty: str):
        """Handle automatic session restart due to difficulty change.

        Smoothly restarts the session without blocking dialogs.

        Args:
            old_difficulty: Previous monster difficulty
            new_difficulty: New monster difficulty
        """
        # Update UI to reflect new session - smooth and non-blocking
        session = self.graphics_view.get_current_exiva_session()
        if session:
            panel = self.monster_tracking_panel
            # Show brief notification in status label
            panel.session_status_label.setText(
                f"🔄 Monster changed ({old_difficulty} → {new_difficulty}) - Session restarted"
            )
            panel.session_status_label.setStyleSheet(
                "font-style: italic; color: #FF8C00; font-weight: bold; "
                "padding: 5px; background-color: #FFF3E0; border-radius: 3px;"
            )

            logger.info(f"Monster changed: {old_difficulty} → {new_difficulty} - Session auto-restarted")

            # Update possible areas list with new difficulty
            self.update_possible_areas_list()

            # After 3 seconds, update to normal session status
            QTimer.singleShot(3000, lambda: self._update_session_status_normal())

    def _update_session_status_normal(self):
        """Update session status label to normal state after auto-restart notification."""
        session = self.graphics_view.get_current_exiva_session()
        if session:
            panel = self.monster_tracking_panel
            reading_count = len(session.readings)
            panel.session_status_label.setText(
                f"Active hunt: {session.session_id[:8]}... ({reading_count} reading{'s' if reading_count != 1 else ''})"
            )
            panel.session_status_label.setStyleSheet("font-style: italic; color: #008000;")

            # Update dimmed percentage display
            self.update_dimmed_percentage_display()

            # Update possible areas list
            self.update_possible_areas_list()

    def on_full_dim_restart(self, dimmed_percentage: float):
        """Handle automatic session restart due to 100% dimming.

        Smoothly restarts the session without blocking dialogs.

        Args:
            dimmed_percentage: The percentage of the map that was dimmed
        """
        # Update UI to reflect new session - smooth and non-blocking
        session = self.graphics_view.get_current_exiva_session()
        if session:
            panel = self.monster_tracking_panel
            # Show brief notification in status label
            panel.session_status_label.setText(
                f"🔄 Area 100% dimmed - Session restarted"
            )
            panel.session_status_label.setStyleSheet(
                "font-style: italic; color: #9C27B0; font-weight: bold; "
                "padding: 5px; background-color: #F3E5F5; border-radius: 3px;"
            )

            logger.info(f"Area {dimmed_percentage:.2f}% dimmed - Session auto-restarted")

            # Update possible areas list
            self.update_possible_areas_list()

            # After 3 seconds, update to normal session status
            QTimer.singleShot(3000, lambda: self._update_session_status_normal())

    def update_dimmed_percentage_display(self):
        """Update the dimmed percentage label in the monster tracking panel."""
        panel = self.monster_tracking_panel

        # Get dimmed percentage from overlay
        if self.graphics_view.exiva_overlay_item:
            dimmed_pct = self.graphics_view.exiva_overlay_item.get_dimmed_percentage()

            # Color code based on percentage
            if dimmed_pct >= 100.0:
                color = "#9C27B0"  # Purple - 100% dimmed
                style = "font-style: italic; color: #9C27B0; font-weight: bold; font-size: 11px;"
            elif dimmed_pct >= 90.0:
                color = "#FF6B6B"  # Red - very high
                style = "font-style: italic; color: #FF6B6B; font-weight: bold; font-size: 11px;"
            elif dimmed_pct >= 75.0:
                color = "#FFA500"  # Orange - high
                style = "font-style: italic; color: #FFA500; font-size: 11px;"
            elif dimmed_pct >= 50.0:
                color = "#FFD700"  # Gold - medium
                style = "font-style: italic; color: #FFD700; font-size: 11px;"
            else:
                color = "#666"  # Gray - low
                style = "font-style: italic; color: #666; font-size: 11px;"

            panel.dimmed_percentage_label.setText(f"Dimmed area: {dimmed_pct:.1f}%")
            panel.dimmed_percentage_label.setStyleSheet(style)
        else:
            # No overlay active
            panel.dimmed_percentage_label.setText("Dimmed area: 0.0%")
            panel.dimmed_percentage_label.setStyleSheet("font-style: italic; color: #666; font-size: 11px;")

    def _convert_floor_to_display(self, floor: int) -> int:
        """Convert internal floor number to display floor number.

        Floor 7 is ground level (0), floors above are positive, floors below are negative.
        Internal -> Display:
        7 -> 0 (ground)
        6 -> 1, 5 -> 2, 4 -> 3, 3 -> 4, 2 -> 5, 1 -> 6, 0 -> 7
        8 -> -1, 9 -> -2, 10 -> -3, 11 -> -4, 12 -> -5, 13 -> -6, 14 -> -7, 15 -> -8
        """
        return 7 - floor

    def update_possible_areas_list(self):
        """Update the list of possible areas based on current monster difficulty and dimmed areas."""
        # Clear the list first
        self.possible_areas_list.clear()

        # Check if we have an active Exiva session
        session = self.graphics_view.get_current_exiva_session()
        if not session or not session.is_active:
            # No active session - show empty list
            list_item = QListWidgetItem("(No active hunt)")
            list_item.setForeground(QColor("#999"))
            self.possible_areas_list.addItem(list_item)
            return

        # Check if we have dimmed areas (overlay exists and has readings)
        if not self.graphics_view.exiva_overlay_item or not session.readings:
            # No dimmed areas - show empty list
            list_item = QListWidgetItem("(No readings yet)")
            list_item.setForeground(QColor("#999"))
            self.possible_areas_list.addItem(list_item)
            return

        # Get the current monster difficulty from the session
        current_difficulty = session.last_difficulty

        # Get all areas across all floors
        all_areas = []
        for floor in range(16):  # Floors 0-15
            areas = self.graphics_view.area_data_manager.get_areas_for_floor(floor)
            for area in areas:
                all_areas.append((floor, area))

        # Filter areas based on difficulty and dimmed status
        possible_areas = []

        for floor, area in all_areas:
            # Check if area matches difficulty
            if current_difficulty == MonsterDifficulty.UNKNOWN:
                # Unknown difficulty - show all areas
                difficulty_matches = True
            else:
                # Check if area has the current difficulty
                difficulty_matches = current_difficulty.value in area.difficulty_levels

            if not difficulty_matches:
                continue

            # Check if area is in the undimmed region (where monster could be)
            area_is_possible = self._is_area_in_undimmed_region(area, floor)

            if area_is_possible:
                possible_areas.append((floor, area))

        # Sort by name and then by floor
        possible_areas.sort(key=lambda x: (x[1].name, x[0]))

        # Add to list widget
        if possible_areas:
            for floor, area in possible_areas:
                display_floor = self._convert_floor_to_display(floor)
                list_item = QListWidgetItem(f"{area.name} ({display_floor})")
                # Store floor and area data for click handling
                list_item.setData(Qt.ItemDataRole.UserRole, (floor, area))
                self.possible_areas_list.addItem(list_item)
        else:
            # No possible areas found
            list_item = QListWidgetItem("(No matching areas)")
            list_item.setForeground(QColor("#999"))
            self.possible_areas_list.addItem(list_item)

    def _is_area_in_undimmed_region(self, area: AreaData, floor: int) -> bool:
        """Check if an area has significant overlap with the undimmed region.

        The overlay dims areas where the monster CANNOT be. The undimmed area (intersection path)
        shows where the monster COULD be. We want to show areas that have meaningful overlap
        with the undimmed region.

        NOTE: The undimmed region is a 2D area on the map. We check areas from ALL floors
        against this 2D region, because the monster could be on any floor within that area.

        Args:
            area: The area to check
            floor: The floor the area is on

        Returns:
            True if the area has significant overlap with undimmed region, False otherwise
        """
        # Get the current session
        session = self.graphics_view.get_current_exiva_session()
        if not session:
            return False

        # Get the overlay item
        overlay = self.graphics_view.exiva_overlay_item

        # Check against the overlay (applies to all floors)
        if not overlay:
            return False

        if not overlay.last_intersection_path:
            return False

        # Create a polygon from the area coordinates
        area_polygon = QPolygonF()
        for x, y in area.coordinates:
            area_polygon.append(QPointF(x, y))

        # Create a path from the area polygon
        area_path = QPainterPath()
        area_path.addPolygon(area_polygon)

        # Check if the area intersects with the undimmed region (intersection path)
        # The intersection path represents the undimmed area where monster could be
        intersection = area_path.intersected(overlay.last_intersection_path)

        if intersection.isEmpty():
            # No intersection - area is completely dimmed
            return False

        # Calculate the percentage of the area that's in the undimmed region
        area_bounding = area_path.boundingRect()
        intersection_bounding = intersection.boundingRect()

        if area_bounding.width() <= 0 or area_bounding.height() <= 0:
            return False

        area_size = area_bounding.width() * area_bounding.height()
        intersection_size = intersection_bounding.width() * intersection_bounding.height()

        # Consider an area "possible" if ANY part of it is in the undimmed region
        # Use 1% threshold to catch areas with even small overlap
        overlap_percentage = (intersection_size / area_size) * 100.0 if area_size > 0 else 0.0

        return overlap_percentage >= 1.0

    def on_possible_area_clicked(self, item: QListWidgetItem):
        """Handle click on a possible area in the list.

        Centers the camera on the area's center position and switches to its floor.

        Args:
            item: The clicked list item
        """
        # Get the stored floor and area data
        data = item.data(Qt.ItemDataRole.UserRole)
        if data is None:
            # Placeholder items (like "(No active hunt)") don't have data
            return

        floor, area = data

        # Calculate the center of the area polygon
        if not area.coordinates:
            logger.warning(f"Area {area.name} has no coordinates")
            return

        # Calculate centroid of the polygon
        sum_x = sum(x for x, y in area.coordinates)
        sum_y = sum(y for x, y in area.coordinates)
        num_points = len(area.coordinates)
        center_x = sum_x / num_points
        center_y = sum_y / num_points

        logger.info(f"Centering camera on area '{area.name}' at ({center_x:.1f}, {center_y:.1f}) on floor {floor}")

        # Switch to the area's floor if different
        if floor != self.current_floor:
            self.set_floor(floor)

        # Center camera on the area with zoom 1.0
        center_pos = QPointF(center_x, center_y)
        self.graphics_view.zoom_to_factor(1.0)
        self.graphics_view.centerOn(center_pos)

    # Automated tracking methods
    def start_automated_tracking(self):
        """Start automated monster tracking with screenshot monitoring."""
        try:
            # Import required modules
            from src.models.settings_manager import SettingsManager
            from src.models.dataset_models import DatasetManager
            from src.utils.screenshot_monitor import ScreenshotMonitor
            from src.crosshair_prediction import CrosshairPredictor
            from src.exiva_extractor.exiva_extractor import ExivaExtractor

            # Load settings
            settings_manager = SettingsManager()
            dataset_manager = DatasetManager()

            # Get screenshot directory and hotkey from settings
            screenshot_dir = settings_manager.get_tibia_screenshot_folder()
            hotkey = settings_manager.get_screenshot_hotkey()

            # Get minimap and exiva regions from global settings
            regions = dataset_manager.load_global_regions()
            minimap_region = regions.get('minimap_region')
            exiva_region = regions.get('exiva_region')

            if not minimap_region:
                QMessageBox.warning(
                    self,
                    "Minimap Region Not Set",
                    "Please configure the minimap region in Options/Settings before starting automated tracking."
                )
                return

            # Initialize the crosshair predictor (load DINOv3 model)
            panel = self.monster_tracking_panel
            panel.tracking_status_label.setText("Loading DINOv3 model...")
            panel.tracking_status_label.setStyleSheet(
                "font-style: italic; color: #0066cc; padding: 5px; "
                "background-color: #e6f2ff; border-radius: 3px;"
            )

            try:
                self.crosshair_detector = CrosshairPredictor()
                logger.info("Crosshair predictor initialized successfully")
            except Exception as e:
                logger.error(f"Failed to load DINOv3 model: {e}")
                QMessageBox.critical(
                    self,
                    "Model Loading Error",
                    f"Failed to load the crosshair prediction model:\n{e}\n\n"
                    "Please ensure the feature database exists at:\n"
                    "src/crosshair_prediction/database/feature_db_dinov3_vitb16.pkl\n\n"
                    "And model weights exist at:\n"
                    "src/crosshair_prediction/models/dinov3_vitb16_pretrain_*.pth"
                )
                panel.tracking_status_label.setText("Model loading failed")
                panel.tracking_status_label.setStyleSheet(
                    "font-style: italic; color: #cc0000; padding: 5px; "
                    "background-color: #ffe6e6; border-radius: 3px;"
                )
                return

            # Initialize the Exiva extractor (load EasyOCR model)
            if exiva_region:
                panel.tracking_status_label.setText("Loading Exiva OCR model...")
                try:
                    self.exiva_extractor = ExivaExtractor(use_gpu=False)
                    logger.info("Exiva extractor initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to load Exiva extractor: {e}")
                    QMessageBox.warning(
                        self,
                        "Exiva Extractor Warning",
                        f"Failed to load the Exiva extractor:\n{e}\n\n"
                        "Crosshair detection will work, but Exiva readings won't be extracted automatically."
                    )
                    self.exiva_extractor = None
            else:
                logger.info("Exiva region not configured - skipping Exiva extractor initialization")
                self.exiva_extractor = None

            # Store regions for later use
            self.minimap_region = minimap_region
            self.exiva_region = exiva_region

            # Initialize screenshot monitor
            self.screenshot_monitor = ScreenshotMonitor(screenshot_dir, hotkey)

            # Connect signals
            self.screenshot_monitor.screenshot_detected.connect(self.on_screenshot_detected)
            self.screenshot_monitor.status_update.connect(self.on_tracking_status_update)
            self.screenshot_monitor.error_occurred.connect(self.on_tracking_error)

            # Start monitoring
            self.screenshot_monitor.start()

            # Update UI
            panel.start_tracking_button.setEnabled(False)
            panel.stop_tracking_button.setEnabled(True)
            panel.tracking_status_label.setText(f"Waiting for screenshot hotkey '{hotkey}'...")
            panel.tracking_status_label.setStyleSheet(
                "font-style: italic; color: #008000; padding: 5px; "
                "background-color: #e6ffe6; border-radius: 3px;"
            )

            logger.info(f"Automated tracking started: hotkey='{hotkey}', dir='{screenshot_dir}'")

        except Exception as e:
            logger.error(f"Error starting automated tracking: {e}")
            QMessageBox.critical(
                self,
                "Tracking Error",
                f"Failed to start automated tracking:\n{e}"
            )

    def stop_automated_tracking(self):
        """Stop automated monster tracking."""
        try:
            if hasattr(self, 'screenshot_monitor') and self.screenshot_monitor:
                self.screenshot_monitor.stop_monitoring()
                self.screenshot_monitor = None

            # Update UI
            panel = self.monster_tracking_panel
            panel.start_tracking_button.setEnabled(True)
            panel.stop_tracking_button.setEnabled(False)
            panel.tracking_status_label.setText("Not tracking")
            panel.tracking_status_label.setStyleSheet(
                "font-style: italic; color: #666; padding: 5px; "
                "background-color: #f0f0f0; border-radius: 3px;"
            )
            panel.crosshair_position_label.setText("Not detected")
            panel.exiva_extraction_label.setText("Not detected")
            panel.exiva_extraction_label.setStyleSheet("color: #666; padding: 3px;")

            logger.info("Automated tracking stopped")

        except Exception as e:
            logger.error(f"Error stopping automated tracking: {e}")

    def on_screenshot_detected(self, screenshot_path: str):
        """Handle screenshot detection event.

        Args:
            screenshot_path: Path to the detected screenshot
        """
        try:
            panel = self.monster_tracking_panel
            panel.tracking_status_label.setText("Processing screenshot...")
            panel.tracking_status_label.setStyleSheet(
                "font-style: italic; color: #0066cc; padding: 5px; "
                "background-color: #e6f2ff; border-radius: 3px;"
            )

            logger.info(f"Processing screenshot: {screenshot_path}")

            # Run crosshair prediction (always use floor 7 since that's what we have in the database)
            x, y, floor = self.crosshair_detector.predict_from_screenshot(
                screenshot_path,
                self.minimap_region,
                floor=7
            )

            # Update crosshair position on the map
            self.update_crosshair_from_prediction(x, y, floor)

            # Update UI with crosshair results
            panel.crosshair_position_label.setText(
                f"X: {x:.1f}, Y: {y:.1f}, Floor: {floor}"
            )
            panel.crosshair_position_label.setStyleSheet(
                "color: #008000; padding: 3px; font-weight: bold;"
            )

            logger.info(f"Crosshair detected: x={x:.1f}, y={y:.1f}, floor={floor}")

            # Extract Exiva reading if extractor is available
            exiva_status = ""
            if self.exiva_extractor and self.exiva_region:
                try:
                    exiva_result = self._extract_exiva_from_screenshot(screenshot_path)
                    if exiva_result:
                        # Format exiva display text
                        exiva_text = f"{exiva_result['direction']} {exiva_result['range']} ({exiva_result['difficulty']})"
                        if exiva_result['floor_indication'] != "none":
                            exiva_text += f" [{exiva_result['floor_indication']}]"
                        exiva_text += f" - Confidence: {exiva_result['confidence']:.2f}"

                        # Update exiva extraction label
                        panel.exiva_extraction_label.setText(exiva_text)
                        panel.exiva_extraction_label.setStyleSheet(
                            "color: #008000; padding: 3px; font-weight: bold;"
                        )

                        exiva_status = f" | Exiva: {exiva_result['direction']} {exiva_result['range']}"
                        logger.info(f"Exiva extracted: {exiva_result}")
                    else:
                        panel.exiva_extraction_label.setText("No Exiva reading detected")
                        panel.exiva_extraction_label.setStyleSheet("color: #666; padding: 3px;")
                except Exception as e:
                    logger.error(f"Error extracting Exiva: {e}")
                    panel.exiva_extraction_label.setText(f"Extraction failed: {str(e)[:50]}")
                    panel.exiva_extraction_label.setStyleSheet("color: #cc0000; padding: 3px;")
                    exiva_status = " | Exiva: extraction failed"
            else:
                # Exiva extractor not available
                if not self.exiva_region:
                    panel.exiva_extraction_label.setText("Exiva region not configured")
                else:
                    panel.exiva_extraction_label.setText("Exiva extractor not loaded")
                panel.exiva_extraction_label.setStyleSheet("color: #999; padding: 3px; font-style: italic;")

            # Update status label with combined results
            panel.tracking_status_label.setText(
                f"Crosshair detected at ({x:.1f}, {y:.1f}) on floor {floor}{exiva_status}"
            )
            panel.tracking_status_label.setStyleSheet(
                "font-style: italic; color: #008000; padding: 5px; "
                "background-color: #e6ffe6; border-radius: 3px;"
            )

        except Exception as e:
            logger.error(f"Error processing screenshot: {e}")
            self.on_tracking_error(f"Processing error: {e}")

    def on_tracking_status_update(self, status: str):
        """Handle status update from screenshot monitor.

        Args:
            status: Status message
        """
        panel = self.monster_tracking_panel
        panel.tracking_status_label.setText(status)
        panel.tracking_status_label.setStyleSheet(
            "font-style: italic; color: #0066cc; padding: 5px; "
            "background-color: #e6f2ff; border-radius: 3px;"
        )

    def on_tracking_error(self, error: str):
        """Handle error from screenshot monitor.

        Args:
            error: Error message
        """
        panel = self.monster_tracking_panel
        panel.tracking_status_label.setText(f"Error: {error}")
        panel.tracking_status_label.setStyleSheet(
            "font-style: italic; color: #cc0000; padding: 5px; "
            "background-color: #ffe6e6; border-radius: 3px;"
        )
        logger.error(f"Tracking error: {error}")

    def update_crosshair_from_prediction(self, x: float, y: float, floor: int):
        """Update the crosshair position on the map from ML prediction.

        Args:
            x: X coordinate on the full map
            y: Y coordinate on the full map
            floor: Floor number (1-15)
        """
        try:
            # Switch to the predicted floor if different
            if floor != self.current_floor:
                logger.info(f"Switching to floor {floor}")
                self.set_floor(floor)

            # Place crosshair at the predicted position
            crosshair_pos = QPointF(x, y)
            self.graphics_view.place_crosshairs(crosshair_pos)

            # Center camera on crosshair with zoom 1.0
            self.graphics_view.zoom_to_factor(1.0)
            self.graphics_view.centerOn(crosshair_pos)

            logger.info(f"Crosshair placed at ({x:.1f}, {y:.1f}) on floor {floor}, camera centered with zoom 1.0")

        except Exception as e:
            logger.error(f"Error updating crosshair: {e}")

    def _extract_exiva_from_screenshot(self, screenshot_path: str) -> Optional[Dict[str, str]]:
        """Extract Exiva reading from screenshot and apply it to the overlay.

        Args:
            screenshot_path: Path to the screenshot file

        Returns:
            Dictionary with extracted exiva data or None if extraction failed
        """
        try:
            from PIL import Image
            import tempfile
            import os

            # Load the screenshot
            screenshot = Image.open(screenshot_path)

            # Crop the exiva region
            x, y, width, height = self.exiva_region
            exiva_crop = screenshot.crop((x, y, x + width, y + height))

            # Save to temporary file for OCR processing
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                exiva_crop.save(tmp_path)

            try:
                # Extract exiva data using OCR
                exiva_output = self.exiva_extractor.extract_from_image(tmp_path)

                # Check if we got valid data
                if exiva_output.direction == "none":
                    logger.info("No Exiva reading detected in screenshot")
                    return None

                # Check confidence threshold (minimum 0.3 for direction to be considered valid)
                if exiva_output.confidence < 0.3:
                    logger.warning(f"Exiva extraction confidence too low: {exiva_output.confidence:.2f}")
                    return None

                # Map the extracted data to UI enums
                direction_enum = self._map_direction_to_enum(exiva_output.direction)
                range_enum = self._map_range_to_enum(exiva_output.range)
                difficulty_enum = self._map_difficulty_to_enum(exiva_output.difficulty)

                if direction_enum is None or range_enum is None:
                    logger.warning(f"Failed to map Exiva output: dir={exiva_output.direction}, range={exiva_output.range}")
                    return None

                # Skip if direction is NONE (invalid)
                if direction_enum == ExivaDirection.NONE:
                    logger.info("Exiva direction is NONE - skipping")
                    return None

                # Automatically start Exiva session if not active
                if not (self.graphics_view.current_exiva_session and self.graphics_view.current_exiva_session.is_active):
                    logger.info("No active Exiva session - starting one automatically")
                    self.graphics_view.start_exiva_session()

                    # Update UI to reflect new session
                    panel = self.monster_tracking_panel
                    panel.start_hunt_button.setEnabled(False)
                    panel.reset_hunt_button.setEnabled(True)
                    panel.direction_combo.setEnabled(True)
                    panel.distance_combo.setEnabled(True)
                    panel.difficulty_combo.setEnabled(True)
                    panel.add_reading_button.setEnabled(True)
                    panel.session_status_label.setText(
                        f"Active session on floor {self.current_floor} - 0 readings"
                    )
                    panel.session_status_label.setStyleSheet("color: #008000; font-weight: bold;")

                # Apply the Exiva reading to the overlay
                # Check auto-restart setting
                auto_restart = self.monster_tracking_panel.auto_restart_checkbox.isChecked()

                # Add the reading via the graphics_view
                success = self.graphics_view.add_exiva_reading(
                    direction=direction_enum,
                    distance=range_enum,
                    monster_difficulty=difficulty_enum,
                    auto_restart_on_difficulty_change=auto_restart
                )

                if success:
                    logger.info(f"Exiva reading applied: {direction_enum.value} {range_enum.value} ({difficulty_enum.value})")

                    # Update session status with reading count
                    session = self.graphics_view.get_current_exiva_session()
                    if session:
                        reading_count = len(session.readings)
                        panel = self.monster_tracking_panel
                        panel.session_status_label.setText(
                            f"Active session on floor {self.current_floor} - {reading_count} reading{'s' if reading_count != 1 else ''}"
                        )

                        # Update dimmed percentage if overlay exists
                        if self.graphics_view.exiva_overlay_item:
                            dimmed_pct = self.graphics_view.exiva_overlay_item.get_dimmed_percentage()
                            panel.dimmed_percentage_label.setText(f"Dimmed area: {dimmed_pct:.1f}%")

                        # Update possible areas list
                        self.update_possible_areas_list()
                else:
                    logger.warning("Failed to apply Exiva reading")

                # Return the extracted data for display
                return {
                    'direction': exiva_output.direction,
                    'range': exiva_output.range,
                    'difficulty': exiva_output.difficulty,
                    'floor_indication': exiva_output.floor_indication,
                    'confidence': exiva_output.confidence
                }

            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass

        except Exception as e:
            logger.error(f"Error extracting Exiva from screenshot: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _map_direction_to_enum(self, direction_str: str) -> Optional[ExivaDirection]:
        """Map direction string from extractor to ExivaDirection enum.

        Args:
            direction_str: Direction string (N, S, E, W, NE, NW, SE, SW, or "none")

        Returns:
            ExivaDirection enum or None if invalid
        """
        direction_map = {
            "N": ExivaDirection.NORTH,
            "S": ExivaDirection.SOUTH,
            "E": ExivaDirection.EAST,
            "W": ExivaDirection.WEST,
            "NE": ExivaDirection.NORTHEAST,
            "NW": ExivaDirection.NORTHWEST,
            "SE": ExivaDirection.SOUTHEAST,
            "SW": ExivaDirection.SOUTHWEST,
            "none": ExivaDirection.NONE
        }
        return direction_map.get(direction_str)

    def _map_range_to_enum(self, range_str: str) -> Optional[ExivaRange]:
        """Map range string from extractor to ExivaRange enum.

        Args:
            range_str: Range string ("to the", "far", "very far", or "none")

        Returns:
            ExivaRange enum or None if invalid
        """
        range_map = {
            "to the": ExivaRange.TO_THE,
            "far": ExivaRange.FAR,
            "very far": ExivaRange.VERY_FAR,
            "none": ExivaRange.NONE
        }
        return range_map.get(range_str)

    def _map_difficulty_to_enum(self, difficulty_str: str) -> MonsterDifficulty:
        """Map difficulty string from extractor to MonsterDifficulty enum.

        Args:
            difficulty_str: Difficulty string (trivial, easy, medium, hard, unknown, or "none")

        Returns:
            MonsterDifficulty enum (defaults to UNKNOWN if invalid)
        """
        difficulty_map = {
            "trivial": MonsterDifficulty.TRIVIAL,
            "easy": MonsterDifficulty.EASY,
            "medium": MonsterDifficulty.MEDIUM,
            "hard": MonsterDifficulty.HARD,
            "unknown": MonsterDifficulty.UNKNOWN,
            "none": MonsterDifficulty.UNKNOWN,
            "harmless": MonsterDifficulty.HARMLESS,
            "challenging": MonsterDifficulty.CHALLENGING
        }
        return difficulty_map.get(difficulty_str, MonsterDifficulty.UNKNOWN)

    def show_minimap_dataset_panel(self):
        """Show the minimap dataset creator panel."""
        if self.minimap_dataset_panel is None:
            from src.models.dataset_models import DatasetManager
            from src.ui.dataset_creator_ui import MinimapDatasetCreatorPanel

            dataset_manager = DatasetManager()

            # Create panel with callback to get crosshair position and graphics view reference
            def get_crosshair_pos():
                return self.graphics_view.global_crosshair_position

            def get_current_floor():
                return self.current_floor

            self.minimap_dataset_panel = MinimapDatasetCreatorPanel(
                dataset_manager, get_crosshair_pos, self.graphics_view, get_current_floor, self
            )
            self.right_tab_widget.addTab(self.minimap_dataset_panel, "Minimap Data")

        # Switch to the minimap dataset tab
        for i in range(self.right_tab_widget.count()):
            if self.right_tab_widget.tabText(i) == "Minimap Data":
                self.right_tab_widget.setCurrentIndex(i)
                break

    def show_exiva_dataset_panel(self):
        """Show the Exiva dataset creator panel."""
        if self.exiva_dataset_panel is None:
            from src.models.dataset_models import DatasetManager
            from src.ui.dataset_creator_ui import ExivaDatasetCreator

            dataset_manager = DatasetManager()
            self.exiva_dataset_panel = ExivaDatasetCreator(dataset_manager, self)
            self.right_tab_widget.addTab(self.exiva_dataset_panel, "Exiva Data")

        # Switch to the Exiva dataset tab
        for i in range(self.right_tab_widget.count()):
            if self.right_tab_widget.tabText(i) == "Exiva Data":
                self.right_tab_widget.setCurrentIndex(i)
                break


class MinimapViewerWindow(QMainWindow):
    """Main window for the minimap viewer application."""

    def __init__(self, minimap_dir: str = "processed_minimap"):
        super().__init__()

        self.setWindowTitle("FiendishFinder - Minimap Viewer")
        self.setGeometry(100, 100, 1200, 800)

        # Create stacked widget to hold different views
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Create minimap viewer
        self.minimap_viewer = MinimapViewer(minimap_dir)
        self.stacked_widget.addWidget(self.minimap_viewer)

        # Create options panel (lazy initialization)
        self.options_panel = OptionsPanel(self)
        self.stacked_widget.addWidget(self.options_panel)

        # Set initial view to minimap
        self.stacked_widget.setCurrentWidget(self.minimap_viewer)
        self.current_view = "minimap"  # Track current view: "minimap" or "options"

        self.minimap_viewer.floorChanged.connect(self.on_floor_changed)

        self.status_bar = self.statusBar()
        self.update_status()

        # Dataset creator windows (lazy initialization)
        self.crosshair_creator_window = None
        self.exiva_creator_window = None
        self.dataset_browser_window = None

        # Set up menu bar
        self.setup_menu_bar()

    def setup_menu_bar(self):
        """Set up the menu bar with navigation and tools."""
        menubar = self.menuBar()

        # Map action (direct click, no dropdown)
        map_action = menubar.addAction("&Map")
        map_action.setStatusTip("Show minimap viewer")
        map_action.triggered.connect(self.show_minimap_viewer)

        # Dataset menu (has multiple items, so keep as menu)
        dataset_menu = menubar.addMenu("&Datasets")

        # Minimap dataset creator
        minimap_action = dataset_menu.addAction("Create Minimap Dataset")
        minimap_action.setStatusTip("Show minimap dataset creator panel")
        minimap_action.triggered.connect(self.show_minimap_dataset_panel)

        # Exiva dataset creator
        exiva_action = dataset_menu.addAction("Create Exiva Dataset")
        exiva_action.setStatusTip("Show Exiva spell message dataset creator panel")
        exiva_action.triggered.connect(self.show_exiva_dataset_panel)

        dataset_menu.addSeparator()

        # Browse datasets
        browse_action = dataset_menu.addAction("Browse Datasets")
        browse_action.setStatusTip("Browse and manage existing datasets")
        browse_action.triggered.connect(self.open_dataset_browser)

        # Options action (direct click, no dropdown)
        options_action = menubar.addAction("&Options")
        options_action.setStatusTip("Show application options and settings")
        options_action.triggered.connect(self.show_options_panel)

    def show_options_panel(self):
        """Show the options panel, replacing the minimap viewer."""
        self.stacked_widget.setCurrentWidget(self.options_panel)
        self.current_view = "options"
        self.setWindowTitle("FiendishFinder - Options")
        self.status_bar.showMessage("Options")

    def show_minimap_viewer(self):
        """Show the minimap viewer, replacing the options panel."""
        self.stacked_widget.setCurrentWidget(self.minimap_viewer)
        self.current_view = "minimap"
        self.setWindowTitle("FiendishFinder - Minimap Viewer")
        self.update_status()

    def show_minimap_dataset_panel(self):
        """Show the minimap dataset creator panel in the main viewer."""
        # First, make sure we're showing the minimap viewer
        if self.current_view != "minimap":
            self.show_minimap_viewer()
        self.minimap_viewer.show_minimap_dataset_panel()

    def show_exiva_dataset_panel(self):
        """Show the Exiva dataset creator panel in the main viewer."""
        # First, make sure we're showing the minimap viewer
        if self.current_view != "minimap":
            self.show_minimap_viewer()
        self.minimap_viewer.show_exiva_dataset_panel()

    def open_dataset_browser(self):
        """Open the dataset browser window."""
        if self.dataset_browser_window is None:
            from src.ui.dataset_browser_ui import DatasetBrowserWindow

            # Create callback to set floor in the minimap viewer
            def set_floor(floor: int):
                self.minimap_viewer.set_floor(floor)

            self.dataset_browser_window = DatasetBrowserWindow(
                parent=self,
                graphics_view=self.minimap_viewer.graphics_view,
                set_floor_callback=set_floor
            )

        self.dataset_browser_window.show()
        self.dataset_browser_window.raise_()
        self.dataset_browser_window.activateWindow()

    def on_floor_changed(self, _floor: int):
        """Handle floor change events."""
        self.update_status()

    def update_status(self):
        """Update status bar with current information."""
        camera_info = self.minimap_viewer.get_camera_info()
        available_floors = len(self.minimap_viewer.get_available_floors())

        status_text = (
            f"Floor: {camera_info['floor']:02d} | "
            f"Zoom: {camera_info['zoom_factor']:.1f}x | "
            f"Position: ({camera_info['center_x']:.0f}, {camera_info['center_y']:.0f}) | "
            f"Available Floors: {available_floors}"
        )

        self.status_bar.showMessage(status_text)


def main():
    """Main function to run the minimap viewer application."""
    app = QApplication(sys.argv)

    app.setApplicationName("FiendishFinder Minimap Viewer")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("FiendishFinder")

    window = MinimapViewerWindow()

    minimap_dir = Path("processed_minimap")
    if not minimap_dir.exists():
        QMessageBox.warning(
            window,
            "Directory Not Found",
            f"Minimap directory '{minimap_dir}' not found.\n"
            "Please ensure the processed minimap images are available."
        )

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
