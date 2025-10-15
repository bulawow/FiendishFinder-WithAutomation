#!/usr/bin/env python3
"""
Minimap Dataset Generator for FiendishFinder

Generates minimap crop images from processed floor maps for crosshair detection training.
Uses walkable area detection to ensure valid training positions.
"""

import random
import uuid
import time
from pathlib import Path
from typing import Optional, Tuple, List
from PIL import Image, ImageDraw
import logging

from src.utils.walkable_detector import WalkableDetector
from src.models.dataset_models import DatasetManager, MinimapDatasetEntry

logger = logging.getLogger(__name__)


class MinimapDatasetGenerator:
    """Generates minimap dataset entries from processed floor maps."""
    
    def __init__(self,
                 minimap_dir: str = "processed_minimap",
                 dataset_manager: Optional[DatasetManager] = None,
                 minimap_size: Tuple[int, int] = (106, 106),
                 marker_file: str = "assets/minimap/minimapmarkers.bin",
                 marker_icons_dir: str = "assets/minimap",
                 raw_tiles_dir: str = "raw_minimap"):
        """Initialize the minimap dataset generator.

        Args:
            minimap_dir: Directory containing processed floor images
            dataset_manager: DatasetManager instance for saving entries
            minimap_size: Size of the minimap crop (width, height)
            marker_file: Path to minimapmarkers.bin file
            marker_icons_dir: Directory containing marker icon images
            raw_tiles_dir: Directory containing raw minimap tiles (for calculating bounds)
        """
        self.minimap_dir = Path(minimap_dir)
        self.dataset_manager = dataset_manager or DatasetManager()
        self.minimap_size = minimap_size
        self.walkable_detector = WalkableDetector(tolerance=5)
        self.marker_file = Path(marker_file)
        self.marker_icons_dir = Path(marker_icons_dir)
        self.raw_tiles_dir = Path(raw_tiles_dir)

        # Cache for floor images
        self.floor_images = {}

        # Cache for walkable positions
        self.walkable_positions_cache = {}

        # Cache for markers
        self.markers_by_floor = {}
        self._icon_cache = {}

        # Global bounds for coordinate conversion
        self.global_min_x = None
        self.global_min_y = None

        # Load global bounds
        self.load_global_bounds()

        # Load markers
        self.load_markers()

    def load_global_bounds(self):
        """Load global bounds from raw minimap tiles."""
        try:
            from src.core.minimap_stitcher import MinimapAnalyzer

            analyzer = MinimapAnalyzer(str(self.raw_tiles_dir))
            analyzer.scan_tiles()
            analyzer.build_floor_maps()

            if analyzer.floor_maps:
                self.global_min_x = min(fm.min_x for fm in analyzer.floor_maps.values())
                self.global_min_y = min(fm.min_y for fm in analyzer.floor_maps.values())
                logger.info(f"Loaded global bounds: min_x={self.global_min_x}, min_y={self.global_min_y}")
            else:
                logger.warning("No floor maps found, cannot determine global bounds")
                self.global_min_x = 0
                self.global_min_y = 0
        except Exception as e:
            logger.error(f"Error loading global bounds: {e}")
            self.global_min_x = 0
            self.global_min_y = 0

    def load_markers(self):
        """Load and parse markers from the marker file."""
        try:
            from src.utils.marker_parser import MarkerParser

            if not self.marker_file.exists():
                logger.warning(f"Marker file not found: {self.marker_file}")
                self.markers_by_floor = {}
                return

            parser = MarkerParser(self.marker_file)
            markers = parser.parse_markers()

            # Group markers by floor
            self.markers_by_floor = {}
            for marker in markers:
                if marker.floor not in self.markers_by_floor:
                    self.markers_by_floor[marker.floor] = []
                self.markers_by_floor[marker.floor].append(marker)

            logger.info(f"Loaded {len(markers)} markers across {len(self.markers_by_floor)} floors")
        except Exception as e:
            logger.warning(f"Could not load markers: {e}")
            self.markers_by_floor = {}

    def load_marker_icon(self, icon_filename: str) -> Optional[Image.Image]:
        """Load a marker icon image with caching."""
        if icon_filename in self._icon_cache:
            return self._icon_cache[icon_filename]

        icon_path = self.marker_icons_dir / icon_filename
        if not icon_path.exists():
            logger.warning(f"Marker icon not found: {icon_path}")
            return None

        try:
            icon = Image.open(icon_path)
            if icon.mode != 'RGBA':
                icon = icon.convert('RGBA')
            self._icon_cache[icon_filename] = icon
            return icon
        except Exception as e:
            logger.error(f"Error loading marker icon {icon_filename}: {e}")
            return None

    def load_floor_image(self, floor: int) -> Optional[Image.Image]:
        """Load a floor image from the processed minimap directory.

        Args:
            floor: Floor number

        Returns:
            PIL Image or None if not found
        """
        if floor in self.floor_images:
            return self.floor_images[floor]

        floor_file = self.minimap_dir / f"floor_{floor:02d}.png"
        if not floor_file.exists():
            logger.error(f"Floor image not found: {floor_file}")
            return None

        try:
            image = Image.open(floor_file)
            if image.mode != 'RGB' and image.mode != 'RGBA':
                image = image.convert('RGBA')
            self.floor_images[floor] = image
            return image
        except Exception as e:
            logger.error(f"Error loading floor {floor}: {e}")
            return None
    
    def get_walkable_positions(self, floor: int, sample_rate: int = 2) -> List[Tuple[int, int]]:
        """Get all walkable positions for a floor.

        Args:
            floor: Floor number
            sample_rate: Sampling rate for position detection

        Returns:
            List of (x, y) walkable positions
        """
        # Check cache
        cache_key = (floor, sample_rate)
        if cache_key in self.walkable_positions_cache:
            return self.walkable_positions_cache[cache_key]
        
        # Load floor image
        image = self.load_floor_image(floor)
        if image is None:
            return []
        
        # Get walkable positions (pass floor for floor-specific rules)
        logger.info(f"Detecting walkable positions on floor {floor}...")
        positions = self.walkable_detector.get_walkable_positions(image, sample_rate, floor)
        logger.info(f"Found {len(positions)} walkable positions on floor {floor}")
        
        # Cache the results
        self.walkable_positions_cache[cache_key] = positions
        
        return positions
    
    def draw_crosshair(self, image: Image.Image) -> Image.Image:
        """Draw a Tibia-style crosshair in the center of the image.

        The crosshair is a white (+) symbol with 2-pixel thick arms:
        - Vertical arm: 2 pixels wide, 6 pixels tall
        - Horizontal arm: 6 pixels wide, 2 pixels tall
        - Color: White RGB(255, 255, 255)

        Args:
            image: PIL Image to draw on (will be modified)

        Returns:
            Modified PIL Image with crosshair
        """
        # Get image center (for 106x106, center is at 53, 53 in 0-indexed coords)
        center_x = image.width // 2
        center_y = image.height // 2

        return self.draw_crosshair_at_position(image, center_x, center_y)

    def get_marker_positions_in_region(self, floor: int, left: int, top: int,
                                        right: int, bottom: int) -> set:
        """Get all marker pixel positions within a region.

        Args:
            floor: Floor number
            left, top, right, bottom: Region bounds in PIXEL coordinates (relative to floor image)

        Returns a set of (x, y) tuples representing pixels occupied by markers (in pixel coordinates).
        """
        marker_pixels = set()

        if floor not in self.markers_by_floor:
            return marker_pixels

        if self.global_min_x is None or self.global_min_y is None:
            logger.warning("Global bounds not loaded, cannot convert marker coordinates")
            return marker_pixels

        from src.utils.marker_parser import MarkerParser
        parser = MarkerParser()

        markers = self.markers_by_floor[floor]
        markers_in_region = 0

        for marker in markers:
            # Convert marker world coordinates to pixel coordinates
            marker_pixel_x = marker.x - self.global_min_x
            marker_pixel_y = marker.y - self.global_min_y

            # Check if marker is within the region (using pixel coordinates)
            if not (left <= marker_pixel_x < right and top <= marker_pixel_y < bottom):
                continue

            markers_in_region += 1

            # Get icon filename to determine marker size
            icon_filename = parser.get_icon_filename(marker.icon_id)
            if icon_filename:
                icon = self.load_marker_icon(icon_filename)
                if icon:
                    # Calculate marker bounds in pixel coordinates
                    icon_left = marker_pixel_x - icon.width // 2
                    icon_top = marker_pixel_y - icon.height // 2
                    icon_right = icon_left + icon.width
                    icon_bottom = icon_top + icon.height

                    # Only add pixels that are actually part of the marker (non-transparent)
                    # This handles circular markers correctly instead of treating them as squares
                    for mx in range(max(left, icon_left), min(right, icon_right)):
                        for my in range(max(top, icon_top), min(bottom, icon_bottom)):
                            # Get pixel position within the icon
                            icon_x = mx - icon_left
                            icon_y = my - icon_top

                            # Check if this pixel is within the icon bounds
                            if 0 <= icon_x < icon.width and 0 <= icon_y < icon.height:
                                # Check if the pixel is non-transparent (alpha > threshold)
                                pixel = icon.getpixel((icon_x, icon_y))
                                # pixel is either (R, G, B, A) or (R, G, B) depending on mode
                                if len(pixel) >= 4:  # Has alpha channel
                                    alpha = pixel[3]
                                    if alpha > 50:  # Only consider pixels with significant opacity
                                        marker_pixels.add((mx, my))
                                else:  # No alpha channel, assume opaque
                                    marker_pixels.add((mx, my))
                else:
                    # Default marker size if icon not found (small circle)
                    radius = 3
                    for mx in range(marker_pixel_x - radius, marker_pixel_x + radius + 1):
                        for my in range(marker_pixel_y - radius, marker_pixel_y + radius + 1):
                            if left <= mx < right and top <= my < bottom:
                                # Use circular distance check
                                dist_sq = (mx - marker_pixel_x) ** 2 + (my - marker_pixel_y) ** 2
                                if dist_sq <= radius ** 2:
                                    marker_pixels.add((mx, my))
            else:
                # Default marker size if icon not found (circular)
                radius = 3
                for mx in range(marker_pixel_x - radius, marker_pixel_x + radius + 1):
                    for my in range(marker_pixel_y - radius, marker_pixel_y + radius + 1):
                        if left <= mx < right and top <= my < bottom:
                            # Use circular distance check
                            dist_sq = (mx - marker_pixel_x) ** 2 + (my - marker_pixel_y) ** 2
                            if dist_sq <= radius ** 2:
                                marker_pixels.add((mx, my))

        return marker_pixels

    def draw_crosshair_at_position(self, image: Image.Image, x: int, y: int,
                                   skip_pixels: set = None) -> Image.Image:
        """Draw a Tibia-style crosshair at a specific position on the image.

        The crosshair is a white (+) symbol with 2-pixel thick arms:
        - Vertical arm: 2 pixels wide, 6 pixels tall
        - Horizontal arm: 6 pixels wide, 2 pixels tall
        - Color: White RGB(255, 255, 255)

        Args:
            image: PIL Image to draw on (will be modified)
            x: X coordinate of the crosshair center
            y: Y coordinate of the crosshair center
            skip_pixels: Optional set of (x, y) tuples to skip drawing (for markers)

        Returns:
            Modified PIL Image with crosshair
        """
        # Ensure image is in RGBA mode for drawing
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        # Create a drawing context
        draw = ImageDraw.Draw(image)

        # Crosshair color: white
        crosshair_color = (255, 255, 255, 255)

        # Draw the crosshair pattern based on analysis:
        # Relative positions from center that should be white
        crosshair_pixels = [
            # Vertical arm (x=0,1 relative; y=-2 to +3 relative)
            (0, -2), (1, -2),
            (0, -1), (1, -1),
            (0, 0), (1, 0),
            (0, 1), (1, 1),
            (0, 2), (1, 2),
            (0, 3), (1, 3),
            # Horizontal arm (y=0,1 relative; x=-2 to +3 relative)
            (-2, 0), (-2, 1),
            (-1, 0), (-1, 1),
            # (0, 0), (1, 0), (0, 1), (1, 1) already covered by vertical arm
            (2, 0), (2, 1),
            (3, 0), (3, 1),
        ]

        # Draw each pixel of the crosshair
        for dx, dy in crosshair_pixels:
            pixel_x = x + dx
            pixel_y = y + dy
            # Only draw if within image bounds and not on a marker
            if 0 <= pixel_x < image.width and 0 <= pixel_y < image.height:
                # Skip if this pixel is occupied by a marker
                if skip_pixels and (pixel_x, pixel_y) in skip_pixels:
                    continue
                draw.point((pixel_x, pixel_y), fill=crosshair_color)

        return image

    def overlay_markers_in_region(self, image: Image.Image, floor: int,
                                   left: int, top: int, right: int, bottom: int) -> Image.Image:
        """Overlay markers that fall within the specified region.

        Args:
            image: PIL Image to draw markers on
            floor: Floor number
            left, top, right, bottom: Region bounds in floor coordinates

        Returns:
            Image with markers overlaid
        """
        if floor not in self.markers_by_floor:
            return image

        from src.utils.marker_parser import MarkerParser
        parser = MarkerParser()  # Just for icon mapping

        markers = self.markers_by_floor[floor]

        for marker in markers:
            # Check if marker is within the region
            if not (left <= marker.x < right and top <= marker.y < bottom):
                continue

            # Get icon filename
            icon_filename = parser.get_icon_filename(marker.icon_id)
            if not icon_filename:
                # Draw a simple circle if icon not found
                draw = ImageDraw.Draw(image)
                radius = 3
                # Convert marker position to image coordinates
                marker_x = marker.x - left
                marker_y = marker.y - top
                draw.ellipse(
                    [marker_x - radius, marker_y - radius, marker_x + radius, marker_y + radius],
                    fill=(255, 0, 0, 200),
                    outline=(255, 255, 255, 255)
                )
                continue

            # Load and paste icon
            icon = self.load_marker_icon(icon_filename)
            if icon:
                # Convert marker position to image coordinates
                marker_x = marker.x - left
                marker_y = marker.y - top

                # Center the icon on the marker position
                icon_x = marker_x - icon.width // 2
                icon_y = marker_y - icon.height // 2

                # Paste with alpha channel for transparency
                image.paste(icon, (icon_x, icon_y), icon)

        return image

    def generate_minimap_crop(self, floor: int, center_x: int, center_y: int,
                             add_crosshair: bool = False) -> Optional[Image.Image]:
        """Generate a minimap crop centered at the given position.

        Args:
            floor: Floor number
            center_x: X coordinate of the center (crosshair position)
            center_y: Y coordinate of the center (crosshair position)
            add_crosshair: If True, draw a crosshair in the center of the crop

        Returns:
            Cropped PIL Image or None if invalid
        """
        image = self.load_floor_image(floor)
        if image is None:
            return None

        # Calculate crop bounds
        half_width = self.minimap_size[0] // 2
        half_height = self.minimap_size[1] // 2

        left = center_x - half_width
        top = center_y - half_height
        right = center_x + half_width
        bottom = center_y + half_height

        # Check if crop is within bounds
        if left < 0 or top < 0 or right > image.width or bottom > image.height:
            logger.warning(f"Crop bounds out of image: {left}, {top}, {right}, {bottom}")
            return None

        # Crop the image first
        cropped = image.crop((left, top, right, bottom))

        # Ensure cropped image is in RGBA mode
        if cropped.mode != 'RGBA':
            cropped = cropped.convert('RGBA')

        # If crosshair is requested, draw it but skip pixels where markers are located
        if add_crosshair:
            # Get marker positions in the crop region (in floor coordinates)
            marker_pixels_floor = self.get_marker_positions_in_region(floor, left, top, right, bottom)

            # Convert marker positions to crop-relative coordinates
            marker_pixels_crop = set()
            for mx, my in marker_pixels_floor:
                crop_x = mx - left
                crop_y = my - top
                marker_pixels_crop.add((crop_x, crop_y))

            # Draw crosshair at the center of the crop, skipping marker pixels
            center_of_crop_x = cropped.width // 2
            center_of_crop_y = cropped.height // 2
            cropped = self.draw_crosshair_at_position(cropped, center_of_crop_x, center_of_crop_y,
                                                      skip_pixels=marker_pixels_crop)

        return cropped
    
    def generate_entry_from_position(self, floor: int, player_x: int, player_y: int,
                                     check_duplicates: bool = True) -> Optional[MinimapDatasetEntry]:
        """Generate a dataset entry from a specific player position.

        Args:
            floor: Floor number
            player_x: X coordinate of the player position on the floor map
            player_y: Y coordinate of the player position on the floor map
            check_duplicates: If True, check if this position already exists in dataset

        Returns:
            MinimapDatasetEntry or None if generation failed or duplicate found
        """
        # Calculate crosshair position (snapped to pixel center)
        crosshair_x = float(player_x) + 0.5
        crosshair_y = float(player_y) + 0.5

        # Check for duplicates if requested
        if check_duplicates:
            existing_positions = self.get_existing_positions()
            position_key = (crosshair_x, crosshair_y, floor)

            if position_key in existing_positions:
                logger.warning(f"Position ({player_x}, {player_y}) on floor {floor} already exists in dataset")
                return None

        # Load floor image to check bounds
        image = self.load_floor_image(floor)
        if image is None:
            logger.error(f"Failed to load floor image for floor {floor}")
            return None

        # Check if position is within image bounds
        if player_x < 0 or player_y < 0 or player_x >= image.width or player_y >= image.height:
            logger.error(f"Position ({player_x}, {player_y}) is out of bounds on floor {floor}")
            return None

        # Check if the position is walkable
        if not self.walkable_detector.is_position_walkable(image, player_x, player_y, floor):
            logger.error(f"Position ({player_x}, {player_y}) is not walkable on floor {floor}")
            return None

        # Check if the position allows a full minimap crop
        half_width = self.minimap_size[0] // 2
        half_height = self.minimap_size[1] // 2

        if (player_x - half_width < 0 or player_y - half_height < 0 or
            player_x + half_width > image.width or player_y + half_height > image.height):
            logger.error(f"Position ({player_x}, {player_y}) does not allow full minimap crop on floor {floor}")
            return None

        # Generate minimap crop centered at the player position with crosshair
        minimap_crop = self.generate_minimap_crop(floor, player_x, player_y, add_crosshair=True)
        if minimap_crop is None:
            logger.error(f"Failed to generate minimap crop at ({player_x}, {player_y}) on floor {floor}")
            return None

        # Generate entry ID
        entry_id = str(uuid.uuid4())

        # Save the cropped minimap image
        minimap_filename = f"{entry_id}.png"
        minimap_path = self.dataset_manager.minimap_screenshots_dir / minimap_filename
        minimap_crop.save(str(minimap_path))

        # IMPORTANT: crosshair_x and crosshair_y are the PLAYER POSITION on the FLOOR MAP
        # NOT relative to the crop! This represents where the player/crosshair is on the full map.
        # Snap to pixel center (add 0.5) to match manual crosshair placement behavior
        # When user clicks on map, it snaps to: floor(x) + 0.5, floor(y) + 0.5

        # Create dataset entry
        entry = MinimapDatasetEntry(
            entry_id=entry_id,
            screenshot_path=f"minimap/screenshots/{minimap_filename}",
            crosshair_x=crosshair_x,
            crosshair_y=crosshair_y,
            floor=floor,
            image_width=self.minimap_size[0],
            image_height=self.minimap_size[1],
            notes=f"Player at floor map position ({player_x}, {player_y}), crop centered at player",
            created_timestamp=time.time(),
            modified_timestamp=time.time()
        )

        logger.info(f"Generated entry {entry_id} from player position ({player_x}, {player_y}) on floor {floor}")
        return entry

    def generate_random_entry(self, floor: int, sample_rate: int = 2) -> Optional[MinimapDatasetEntry]:
        """Generate a random dataset entry for a floor.

        Args:
            floor: Floor number
            sample_rate: Sampling rate for walkable position detection

        Returns:
            MinimapDatasetEntry or None if generation failed
        """
        # Get walkable positions
        walkable_positions = self.get_walkable_positions(floor, sample_rate)

        if not walkable_positions:
            logger.error(f"No walkable positions found on floor {floor}")
            return None

        # Filter positions that allow full minimap crop
        image = self.load_floor_image(floor)
        if image is None:
            return None

        half_width = self.minimap_size[0] // 2
        half_height = self.minimap_size[1] // 2

        valid_positions = [
            (x, y) for x, y in walkable_positions
            if (x - half_width >= 0 and y - half_height >= 0 and
                x + half_width <= image.width and y + half_height <= image.height)
        ]

        if not valid_positions:
            logger.error(f"No valid positions for full minimap crop on floor {floor}")
            return None

        # Select random position (this is the player position)
        player_x, player_y = random.choice(valid_positions)

        # Generate minimap crop centered at player position with crosshair
        minimap_crop = self.generate_minimap_crop(floor, player_x, player_y, add_crosshair=True)
        if minimap_crop is None:
            return None

        # Generate entry ID
        entry_id = str(uuid.uuid4())

        # Save the cropped minimap image
        minimap_filename = f"{entry_id}.png"
        minimap_path = self.dataset_manager.minimap_screenshots_dir / minimap_filename
        minimap_crop.save(str(minimap_path))

        # IMPORTANT: crosshair_x and crosshair_y are the PLAYER POSITION on the FLOOR MAP
        # NOT relative to the crop! This represents where the player is on the full map.
        # Snap to pixel center (add 0.5) to match manual crosshair placement behavior
        # When user clicks on map, it snaps to: floor(x) + 0.5, floor(y) + 0.5
        crosshair_x = float(player_x) + 0.5
        crosshair_y = float(player_y) + 0.5

        # Create dataset entry
        entry = MinimapDatasetEntry(
            entry_id=entry_id,
            screenshot_path=f"minimap/screenshots/{minimap_filename}",
            crosshair_x=crosshair_x,
            crosshair_y=crosshair_y,
            floor=floor,
            image_width=self.minimap_size[0],
            image_height=self.minimap_size[1],
            notes=f"Auto-generated: player at floor map position ({player_x}, {player_y})",
            created_timestamp=time.time(),
            modified_timestamp=time.time()
        )

        return entry

    def get_existing_positions(self) -> set:
        """Get all existing crosshair positions (x, y, floor) from the dataset.

        Returns:
            Set of tuples (crosshair_x, crosshair_y, floor) that already exist
        """
        existing_entries = self.dataset_manager.load_minimap_dataset()
        existing_positions = set()

        for entry in existing_entries:
            # Store as (x, y, floor) tuple for duplicate detection
            existing_positions.add((entry.crosshair_x, entry.crosshair_y, entry.floor))

        logger.info(f"Found {len(existing_positions)} existing crosshair positions in dataset")
        return existing_positions

    def generate_multiple_entries(self, floor: int, count: int, sample_rate: int = 2,
                                  progress_callback=None) -> List[MinimapDatasetEntry]:
        """Generate multiple random dataset entries for a floor.

        Ensures no duplicate crosshair positions (x, y, floor) are generated.

        Args:
            floor: Floor number
            count: Number of entries to generate
            sample_rate: Sampling rate for walkable position detection
            progress_callback: Optional callback function(current, total, message) for progress updates

        Returns:
            List of generated MinimapDatasetEntry objects
        """
        entries = []

        # Load existing positions to avoid duplicates
        existing_positions = self.get_existing_positions()
        used_positions_this_batch = set()

        # Get all valid positions for this floor
        walkable_positions = self.get_walkable_positions(floor, sample_rate)
        if not walkable_positions:
            logger.error(f"No walkable positions found on floor {floor}")
            return entries

        # Filter positions that allow full minimap crop
        image = self.load_floor_image(floor)
        if image is None:
            return entries

        half_width = self.minimap_size[0] // 2
        half_height = self.minimap_size[1] // 2

        valid_positions = [
            (x, y) for x, y in walkable_positions
            if (x - half_width >= 0 and y - half_height >= 0 and
                x + half_width <= image.width and y + half_height <= image.height)
        ]

        if not valid_positions:
            logger.error(f"No valid positions for full minimap crop on floor {floor}")
            return entries

        # Filter out positions that already exist in dataset
        # Crosshair positions are stored as float(x) + 0.5, float(y) + 0.5
        available_positions = [
            (x, y) for x, y in valid_positions
            if (float(x) + 0.5, float(y) + 0.5, floor) not in existing_positions
        ]

        if not available_positions:
            logger.warning(f"All valid positions on floor {floor} already exist in dataset!")
            return entries

        logger.info(f"Found {len(available_positions)} available positions (out of {len(valid_positions)} valid positions)")

        # Shuffle to get random order
        random.shuffle(available_positions)

        for i in range(count):
            if i >= len(available_positions):
                logger.warning(f"Only {len(available_positions)} unique positions available, cannot generate {count} entries")
                break

            # Update progress
            if progress_callback and i % 10 == 0:  # Update every 10 entries to avoid UI spam
                progress_callback(i, count, f"Generating entry {i+1}/{count}...")

            logger.info(f"Generating entry {i+1}/{count} for floor {floor}...")

            # Get next unique position
            player_x, player_y = available_positions[i]
            crosshair_x = float(player_x) + 0.5
            crosshair_y = float(player_y) + 0.5

            # Double-check this position hasn't been used
            position_key = (crosshair_x, crosshair_y, floor)
            if position_key in used_positions_this_batch:
                logger.warning(f"Position ({player_x}, {player_y}) on floor {floor} already used in this batch, skipping")
                continue

            # Generate minimap crop centered at player position with crosshair
            minimap_crop = self.generate_minimap_crop(floor, player_x, player_y, add_crosshair=True)
            if minimap_crop is None:
                logger.warning(f"Failed to generate minimap crop at ({player_x}, {player_y})")
                continue

            # Generate entry ID
            entry_id = str(uuid.uuid4())

            # Save the cropped minimap image
            minimap_filename = f"{entry_id}.png"
            minimap_path = self.dataset_manager.minimap_screenshots_dir / minimap_filename
            minimap_crop.save(str(minimap_path))

            # Create dataset entry
            entry = MinimapDatasetEntry(
                entry_id=entry_id,
                screenshot_path=f"minimap/screenshots/{minimap_filename}",
                crosshair_x=crosshair_x,
                crosshair_y=crosshair_y,
                floor=floor,
                image_width=self.minimap_size[0],
                image_height=self.minimap_size[1],
                notes=f"Auto-generated: player at floor map position ({player_x}, {player_y})",
                created_timestamp=time.time(),
                modified_timestamp=time.time()
            )

            entries.append(entry)
            used_positions_this_batch.add(position_key)
            logger.info(f"Generated entry {entry_id} at unique position ({player_x}, {player_y}) on floor {floor}")

        logger.info(f"Successfully generated {len(entries)}/{count} unique entries for floor {floor}")
        return entries
    
    def generate_and_save_entries(self, floor: int, count: int, sample_rate: int = 2,
                                  progress_callback=None, batch_size: int = 1000) -> int:
        """Generate and save multiple dataset entries in batches to avoid memory issues.

        Args:
            floor: Floor number
            count: Number of entries to generate
            sample_rate: Sampling rate for walkable position detection
            progress_callback: Optional callback function(current, total, message) for progress updates
            batch_size: Number of entries to save per batch (default 1000)

        Returns:
            Number of successfully saved entries
        """
        # For small counts, use the old method (generate all then save)
        if count <= batch_size:
            if progress_callback:
                progress_callback(0, count, f"Generating {count} entries for floor {floor}...")

            entries = self.generate_multiple_entries(floor, count, sample_rate, progress_callback)

            if not entries:
                logger.warning("No entries were generated")
                return 0

            # Save all entries in a single batch operation
            if progress_callback:
                progress_callback(len(entries), count, f"Saving {len(entries)} entries to dataset...")

            logger.info(f"Saving {len(entries)} entries in batch...")
            if self.dataset_manager.add_minimap_entries_batch(entries):
                logger.info(f"Successfully saved {len(entries)} entries to dataset")
                return len(entries)
            else:
                logger.error("Failed to save entries batch")
                return 0

        # For large counts, generate and save in batches
        total_saved = 0
        remaining = count

        while remaining > 0:
            current_batch_size = min(batch_size, remaining)

            if progress_callback:
                progress_callback(total_saved, count,
                                f"Generating batch {total_saved // batch_size + 1} ({current_batch_size} entries)...")

            # Generate batch
            entries = self.generate_multiple_entries(floor, current_batch_size, sample_rate,
                                                    lambda c, t, m: progress_callback(total_saved + c, count, m) if progress_callback else None)

            if not entries:
                logger.warning(f"No entries generated in batch starting at {total_saved}")
                break

            # Save batch
            if progress_callback:
                progress_callback(total_saved + len(entries), count,
                                f"Saving batch {total_saved // batch_size + 1} ({len(entries)} entries)...")

            logger.info(f"Saving batch of {len(entries)} entries...")
            if self.dataset_manager.add_minimap_entries_batch(entries):
                total_saved += len(entries)
                logger.info(f"Successfully saved batch. Total: {total_saved}/{count}")
            else:
                logger.error("Failed to save batch")
                break

            remaining -= current_batch_size

            # Clear entries from memory
            entries.clear()

        logger.info(f"Completed: saved {total_saved} out of {count} requested entries")
        return total_saved
    
    def clear_caches(self):
        """Clear all cached data to free memory."""
        # Close any PIL images before clearing
        for floor, img in self.floor_images.items():
            if img is not None:
                img.close()

        self.floor_images.clear()
        self.walkable_positions_cache.clear()
        logger.info("Cleared all caches and freed memory")

