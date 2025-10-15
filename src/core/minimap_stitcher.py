#!/usr/bin/env python3
"""
Tibia Minimap Tile Stitcher

This module provides functionality to stitch together individual minimap image tiles
from the Tibia game into complete floor maps.

File naming convention: Minimap_Color_X_Y_Z.png
- X, Y: World coordinates (e.g., 31744, 30976)
- Z: Floor level (0-15)
- Each tile is 256x256 pixels
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TileInfo:
    """Information about a single minimap tile."""
    filename: str
    x_coord: int
    y_coord: int
    floor: int
    file_path: Path

    @property
    def grid_x(self) -> int:
        return self.x_coord

    @property
    def grid_y(self) -> int:
        return self.y_coord

@dataclass
class FloorMap:
    """Represents a complete floor map with its tiles."""
    floor: int
    tiles: Dict[Tuple[int, int], TileInfo]
    min_x: int
    max_x: int
    min_y: int
    max_y: int

    @property
    def width_tiles(self) -> int:
        return (self.max_x - self.min_x) // 256 + 1

    @property
    def height_tiles(self) -> int:
        return (self.max_y - self.min_y) // 256 + 1

    @property
    def width_pixels(self) -> int:
        return self.width_tiles * 256

    @property
    def height_pixels(self) -> int:
        return self.height_tiles * 256

class SpatialAnalyzer:
    """Analyzes spatial relationships between tiles and calculates positioning."""

    @staticmethod
    def world_to_grid_position(world_x: int, world_y: int, min_x: int, min_y: int) -> Tuple[int, int]:
        grid_x = (world_x - min_x) // 256
        grid_y = (world_y - min_y) // 256
        return grid_x, grid_y

    @staticmethod
    def grid_to_pixel_position(grid_x: int, grid_y: int) -> Tuple[int, int]:
        pixel_x = grid_x * 256
        pixel_y = grid_y * 256
        return pixel_x, pixel_y

    @staticmethod
    def calculate_canvas_size(floor_map: FloorMap) -> Tuple[int, int]:
        return floor_map.width_pixels, floor_map.height_pixels

    @staticmethod
    def get_tile_positions(floor_map: FloorMap) -> Dict[Tuple[int, int], Tuple[int, int]]:
        positions = {}
        for (world_x, world_y) in floor_map.tiles.keys():
            grid_x, grid_y = SpatialAnalyzer.world_to_grid_position(
                world_x, world_y, floor_map.min_x, floor_map.min_y
            )
            pixel_x, pixel_y = SpatialAnalyzer.grid_to_pixel_position(grid_x, grid_y)
            positions[(world_x, world_y)] = (pixel_x, pixel_y)
        return positions

class ImageStitcher:
    """Handles the actual stitching of minimap tiles into complete floor images."""

    def __init__(self, output_dir: str = "processed_minimap", marker_icons_dir: str = "assets/minimap"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.spatial_analyzer = SpatialAnalyzer()
        self.marker_icons_dir = Path(marker_icons_dir)
        self._icon_cache = {}  # Cache for loaded marker icons

    def stitch_floor(self, floor_map: FloorMap,
                    global_min_x: int = None, global_max_x: int = None,
                    global_min_y: int = None, global_max_y: int = None) -> Optional[Image.Image]:
        """Stitch all tiles for a single floor into one complete image.

        Args:
            floor_map: The floor map to stitch
            global_min_x: Optional global minimum X coordinate (for uniform sizing)
            global_max_x: Optional global maximum X coordinate (for uniform sizing)
            global_min_y: Optional global minimum Y coordinate (for uniform sizing)
            global_max_y: Optional global maximum Y coordinate (for uniform sizing)
        """
        logger.info(f"Stitching floor {floor_map.floor} with {len(floor_map.tiles)} tiles")

        # Use global bounds if provided, otherwise use floor-specific bounds
        min_x = global_min_x if global_min_x is not None else floor_map.min_x
        max_x = global_max_x if global_max_x is not None else floor_map.max_x
        min_y = global_min_y if global_min_y is not None else floor_map.min_y
        max_y = global_max_y if global_max_y is not None else floor_map.max_y

        # Calculate canvas size based on bounds
        width_tiles = (max_x - min_x) // 256 + 1
        height_tiles = (max_y - min_y) // 256 + 1
        canvas_width = width_tiles * 256
        canvas_height = height_tiles * 256

        logger.info(f"Canvas size: {canvas_width}x{canvas_height} pixels")

        canvas = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 255))

        expected_tiles = set()
        for x in range(min_x, max_x + 256, 256):
            for y in range(min_y, max_y + 256, 256):
                expected_tiles.add((x, y))

        black_tile = Image.new('RGBA', (256, 256), (0, 0, 0, 255))

        placed_tiles = 0
        missing_tiles = 0

        for world_x, world_y in expected_tiles:
            grid_x, grid_y = self.spatial_analyzer.world_to_grid_position(
                world_x, world_y, min_x, min_y
            )
            pixel_x, pixel_y = self.spatial_analyzer.grid_to_pixel_position(grid_x, grid_y)

            if (world_x, world_y) in floor_map.tiles:
                tile_info = floor_map.tiles[(world_x, world_y)]
                try:
                    tile_image = Image.open(tile_info.file_path)

                    if tile_image.mode != 'RGBA':
                        tile_image = tile_image.convert('RGBA')

                    canvas.paste(tile_image, (pixel_x, pixel_y), tile_image)
                    placed_tiles += 1

                except Exception as e:
                    logger.error(f"Error loading tile {tile_info.filename}: {e}")
                    canvas.paste(black_tile, (pixel_x, pixel_y))
                    missing_tiles += 1
            else:
                canvas.paste(black_tile, (pixel_x, pixel_y))
                missing_tiles += 1

        logger.info(f"Successfully placed {placed_tiles} tiles, filled {missing_tiles} missing tiles with black")

        if placed_tiles == 0:
            logger.error(f"No tiles could be placed for floor {floor_map.floor}")
            return None

        return canvas

    def load_marker_icon(self, icon_filename: str) -> Optional[Image.Image]:
        """Load a marker icon image with caching.

        Args:
            icon_filename: Name of the icon file

        Returns:
            PIL Image or None if not found
        """
        if icon_filename in self._icon_cache:
            return self._icon_cache[icon_filename]

        icon_path = self.marker_icons_dir / icon_filename
        if not icon_path.exists():
            logger.warning(f"Marker icon not found: {icon_path}")
            return None

        try:
            icon = Image.open(icon_path)
            # Load the image data immediately to avoid lazy loading issues
            icon.load()
            logger.debug(f"Loaded icon {icon_filename}: {icon.width}x{icon.height}, mode={icon.mode}")
            if icon.mode != 'RGBA':
                logger.debug(f"Converting {icon_filename} from {icon.mode} to RGBA")
                icon = icon.convert('RGBA')
            self._icon_cache[icon_filename] = icon
            return icon
        except Exception as e:
            logger.error(f"Error loading marker icon {icon_filename}: {e}")
            return None

    def overlay_markers(self, canvas: Image.Image, markers: List,
                       min_x: int, min_y: int) -> Image.Image:
        """Overlay map markers on the canvas.

        Args:
            canvas: The floor image canvas
            markers: List of MapMarker objects for this floor
            min_x: Minimum X coordinate of the canvas
            min_y: Minimum Y coordinate of the canvas

        Returns:
            Canvas with markers overlaid
        """
        if not markers:
            return canvas

        # Import here to avoid circular dependency
        from src.utils.marker_parser import MarkerParser

        parser = MarkerParser()  # Just for icon mapping

        for marker in markers:
            # Convert world coordinates to pixel position on canvas
            # Marker coordinates are absolute world coordinates
            pixel_x = marker.x - min_x
            pixel_y = marker.y - min_y

            # Check if marker is within canvas bounds
            if pixel_x < 0 or pixel_y < 0 or pixel_x >= canvas.width or pixel_y >= canvas.height:
                continue

            # Get icon filename
            icon_filename = parser.get_icon_filename(marker.icon_id)
            if not icon_filename:
                # Draw a simple circle if icon not found
                draw = ImageDraw.Draw(canvas)
                radius = 3
                draw.ellipse(
                    [pixel_x - radius, pixel_y - radius, pixel_x + radius, pixel_y + radius],
                    fill=(255, 0, 0, 200),
                    outline=(255, 255, 255, 255)
                )
                continue

            # Load and paste icon
            icon = self.load_marker_icon(icon_filename)
            if icon:
                # Center the icon on the marker position
                icon_x = pixel_x - icon.width // 2
                icon_y = pixel_y - icon.height // 2

                # Paste with alpha channel for transparency
                canvas.paste(icon, (icon_x, icon_y), icon)
            else:
                logger.warning(f"Failed to load icon: {icon_filename} for marker at ({pixel_x}, {pixel_y})")

        logger.info(f"Overlaid {len(markers)} markers on canvas")
        return canvas

    def save_floor_image(self, floor: int, image: Image.Image, format: str = 'PNG') -> Path:
        """Save a floor image to the output directory."""
        filename = f"floor_{floor:02d}.{format.lower()}"
        output_path = self.output_dir / filename

        if format.upper() == 'JPEG' and image.mode == 'RGBA':
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[-1])
            image = rgb_image

        image.save(output_path, format=format)
        logger.info(f"Saved floor {floor} image to {output_path}")
        return output_path

    def stitch_all_floors(self, floor_maps: Dict[int, FloorMap],
                         format: str = 'PNG', markers_by_floor: Dict[int, List] = None) -> Dict[int, Path]:
        """Stitch all floors and save the resulting images.

        Args:
            floor_maps: Dictionary of floor maps
            format: Output image format
            markers_by_floor: Optional dictionary of markers grouped by floor
        """
        logger.info(f"Stitching {len(floor_maps)} floors")
        results = {}

        # Calculate global bounds across all floors to ensure uniform size
        global_min_x = min(fm.min_x for fm in floor_maps.values())
        global_max_x = max(fm.max_x for fm in floor_maps.values())
        global_min_y = min(fm.min_y for fm in floor_maps.values())
        global_max_y = max(fm.max_y for fm in floor_maps.values())

        logger.info(f"Global bounds: ({global_min_x},{global_min_y}) to ({global_max_x},{global_max_y})")

        for floor in sorted(floor_maps.keys()):
            floor_map = floor_maps[floor]

            floor_image = self.stitch_floor(floor_map, global_min_x, global_max_x,
                                           global_min_y, global_max_y)
            if floor_image is None:
                logger.error(f"Failed to stitch floor {floor}")
                continue

            # Overlay markers if available
            if markers_by_floor and floor in markers_by_floor:
                floor_image = self.overlay_markers(
                    floor_image,
                    markers_by_floor[floor],
                    global_min_x,
                    global_min_y
                )

            output_path = self.save_floor_image(floor, floor_image, format)
            results[floor] = output_path

        logger.info(f"Successfully stitched {len(results)} floors")
        return results

class OverlapHandler:
    """Handles overlapping areas between tiles."""

    @staticmethod
    def detect_overlaps(floor_map: FloorMap) -> List[Tuple[TileInfo, TileInfo]]:
        overlaps = []
        tiles_list = list(floor_map.tiles.values())

        for i, tile1 in enumerate(tiles_list):
            for tile2 in tiles_list[i+1:]:
                if OverlapHandler._tiles_overlap(tile1, tile2):
                    overlaps.append((tile1, tile2))

        return overlaps

    @staticmethod
    def _tiles_overlap(tile1: TileInfo, tile2: TileInfo) -> bool:
        tile1_left = tile1.x_coord
        tile1_right = tile1.x_coord + 256
        tile1_top = tile1.y_coord
        tile1_bottom = tile1.y_coord + 256

        tile2_left = tile2.x_coord
        tile2_right = tile2.x_coord + 256
        tile2_top = tile2.y_coord
        tile2_bottom = tile2.y_coord + 256

        return not (tile1_right <= tile2_left or tile2_right <= tile1_left or
                   tile1_bottom <= tile2_top or tile2_bottom <= tile1_top)

    @staticmethod
    def calculate_overlap_area(tile1: TileInfo, tile2: TileInfo) -> Tuple[int, int, int, int]:
        if not OverlapHandler._tiles_overlap(tile1, tile2):
            return (0, 0, 0, 0)

        left = max(tile1.x_coord, tile2.x_coord)
        right = min(tile1.x_coord + 256, tile2.x_coord + 256)
        top = max(tile1.y_coord, tile2.y_coord)
        bottom = min(tile1.y_coord + 256, tile2.y_coord + 256)

        return (left, top, right - left, bottom - top)

    @staticmethod
    def blend_overlapping_tiles(tile1_image: Image.Image, tile2_image: Image.Image,
                               overlap_area: Tuple[int, int, int, int]) -> Image.Image:
        return tile1_image.copy()

class MinimapStitchingSystem:
    """Main system that coordinates all components for minimap stitching."""

    def __init__(self, raw_minimap_dir: str = "raw_minimap",
                 output_dir: str = "processed_minimap",
                 marker_file: str = "assets/minimap/minimapmarkers.bin"):
        self.analyzer = MinimapAnalyzer(raw_minimap_dir)
        self.stitcher = ImageStitcher(output_dir)
        self.overlap_handler = OverlapHandler()
        self.output_dir = Path(output_dir)
        self.marker_file = marker_file
        self.markers_by_floor = None

    def load_markers(self):
        """Load and parse markers from the marker file."""
        try:
            from src.utils.marker_parser import MarkerParser

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

    def process_all_floors(self, image_format: str = 'PNG', include_markers: bool = True) -> Dict[int, Dict]:
        """Process all floors and generate complete minimap images.

        Args:
            image_format: Output image format (PNG or JPEG)
            include_markers: Whether to overlay map markers on the images
        """
        logger.info("Starting minimap stitching process")

        self.analyzer.scan_tiles()
        self.analyzer.build_floor_maps()

        issues = self.analyzer.validate_tiles()
        if issues:
            logger.warning("Validation issues found:")
            for floor, floor_issues in issues.items():
                for issue in floor_issues:
                    logger.warning(f"Floor {floor}: {issue}")

        # Load markers if requested
        if include_markers:
            self.load_markers()
        else:
            self.markers_by_floor = None

        results = self.stitcher.stitch_all_floors(
            self.analyzer.floor_maps,
            image_format,
            self.markers_by_floor
        )

        # Calculate global bounds for the summary report
        global_min_x = min(fm.min_x for fm in self.analyzer.floor_maps.values())
        global_max_x = max(fm.max_x for fm in self.analyzer.floor_maps.values())
        global_min_y = min(fm.min_y for fm in self.analyzer.floor_maps.values())
        global_max_y = max(fm.max_y for fm in self.analyzer.floor_maps.values())

        summary = self._generate_summary_report(results, global_min_x, global_max_x,
                                                global_min_y, global_max_y)

        logger.info("Minimap stitching process completed")
        return summary

    def process_single_floor(self, floor: int, image_format: str = 'PNG', include_markers: bool = True) -> Optional[Path]:
        """Process a single floor and generate its minimap image.

        Args:
            floor: Floor number to process
            image_format: Output image format (PNG or JPEG)
            include_markers: Whether to overlay map markers on the image
        """
        logger.info(f"Processing single floor: {floor}")

        if not self.analyzer.floor_maps:
            self.analyzer.scan_tiles()
            self.analyzer.build_floor_maps()

        if floor not in self.analyzer.floor_maps:
            logger.error(f"Floor {floor} not found in available floors")
            return None

        # Calculate global bounds across all floors for uniform sizing
        global_min_x = min(fm.min_x for fm in self.analyzer.floor_maps.values())
        global_max_x = max(fm.max_x for fm in self.analyzer.floor_maps.values())
        global_min_y = min(fm.min_y for fm in self.analyzer.floor_maps.values())
        global_max_y = max(fm.max_y for fm in self.analyzer.floor_maps.values())

        floor_map = self.analyzer.floor_maps[floor]
        floor_image = self.stitcher.stitch_floor(floor_map, global_min_x, global_max_x,
                                                 global_min_y, global_max_y)

        if floor_image is None:
            logger.error(f"Failed to stitch floor {floor}")
            return None

        # Overlay markers if requested
        if include_markers:
            if self.markers_by_floor is None:
                self.load_markers()

            if self.markers_by_floor and floor in self.markers_by_floor:
                floor_image = self.stitcher.overlay_markers(
                    floor_image,
                    self.markers_by_floor[floor],
                    global_min_x,
                    global_min_y
                )

        return self.stitcher.save_floor_image(floor, floor_image, image_format)

    def _generate_summary_report(self, results: Dict[int, Path],
                                 global_min_x: int = None, global_max_x: int = None,
                                 global_min_y: int = None, global_max_y: int = None) -> Dict[int, Dict]:
        summary = {}

        # Calculate global dimensions if bounds are provided
        if all(x is not None for x in [global_min_x, global_max_x, global_min_y, global_max_y]):
            global_width_tiles = (global_max_x - global_min_x) // 256 + 1
            global_height_tiles = (global_max_y - global_min_y) // 256 + 1
            global_width_pixels = global_width_tiles * 256
            global_height_pixels = global_height_tiles * 256
        else:
            global_width_pixels = None
            global_height_pixels = None
            global_width_tiles = None
            global_height_tiles = None

        for floor in sorted(self.analyzer.floor_maps.keys()):
            floor_map = self.analyzer.floor_maps[floor]

            # Use global dimensions if available, otherwise use floor-specific
            if global_width_pixels is not None:
                width_pixels = global_width_pixels
                height_pixels = global_height_pixels
                width_tiles = global_width_tiles
                height_tiles = global_height_tiles
            else:
                width_pixels = floor_map.width_pixels
                height_pixels = floor_map.height_pixels
                width_tiles = floor_map.width_tiles
                height_tiles = floor_map.height_tiles

            floor_summary = {
                'floor': floor,
                'tile_count': len(floor_map.tiles),
                'dimensions': {
                    'width_pixels': width_pixels,
                    'height_pixels': height_pixels,
                    'width_tiles': width_tiles,
                    'height_tiles': height_tiles
                },
                'bounds': {
                    'min_x': floor_map.min_x,
                    'max_x': floor_map.max_x,
                    'min_y': floor_map.min_y,
                    'max_y': floor_map.max_y
                },
                'output_file': str(results.get(floor, 'Failed to generate')),
                'success': floor in results
            }

            summary[floor] = floor_summary

        return summary

    def get_available_floors(self) -> List[int]:
        if not self.analyzer.floor_maps:
            self.analyzer.scan_tiles()
            self.analyzer.build_floor_maps()

        return sorted(self.analyzer.floor_maps.keys())

    def save_summary_report(self, summary: Dict[int, Dict], filename: str = "stitching_report.txt") -> Path:
        report_path = self.output_dir / filename

        with open(report_path, 'w') as f:
            f.write("Tibia Minimap Stitching Report\n")
            f.write("=" * 50 + "\n\n")

            successful_floors = sum(1 for floor_data in summary.values() if floor_data['success'])
            total_floors = len(summary)

            f.write(f"Summary: {successful_floors}/{total_floors} floors processed successfully\n\n")

            for floor in sorted(summary.keys()):
                floor_data = summary[floor]
                f.write(f"Floor {floor}:\n")
                f.write(f"  Status: {'SUCCESS' if floor_data['success'] else 'FAILED'}\n")
                f.write(f"  Tiles: {floor_data['tile_count']}\n")
                f.write(f"  Dimensions: {floor_data['dimensions']['width_pixels']}x{floor_data['dimensions']['height_pixels']} pixels\n")
                f.write(f"  Grid: {floor_data['dimensions']['width_tiles']}x{floor_data['dimensions']['height_tiles']} tiles\n")
                f.write(f"  Output: {floor_data['output_file']}\n")
                f.write("\n")

        logger.info(f"Summary report saved to {report_path}")
        return report_path

def main():
    """Main function to run the minimap stitching system."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Stitch Tibia minimap tiles into complete floor maps')
    parser.add_argument('--input-dir', default='raw_minimap',
                       help='Directory containing raw minimap tiles (default: raw_minimap)')
    parser.add_argument('--output-dir', default='processed_minimap',
                       help='Directory to save processed minimap images (default: processed_minimap)')
    parser.add_argument('--format', choices=['PNG', 'JPEG'], default='PNG',
                       help='Output image format (default: PNG)')
    parser.add_argument('--floor', type=int,
                       help='Process only a specific floor (default: process all floors)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        system = MinimapStitchingSystem(args.input_dir, args.output_dir)

        if args.floor is not None:
            result = system.process_single_floor(args.floor, args.format)
            if result:
                print(f"Successfully processed floor {args.floor}: {result}")
            else:
                print(f"Failed to process floor {args.floor}")
                sys.exit(1)
        else:
            summary = system.process_all_floors(args.format)
            report_path = system.save_summary_report(summary)

            successful_floors = sum(1 for floor_data in summary.values() if floor_data['success'])
            total_floors = len(summary)

            print(f"\nProcessing complete!")
            print(f"Successfully processed {successful_floors}/{total_floors} floors")
            print(f"Summary report saved to: {report_path}")

            if successful_floors < total_floors:
                print("\nSome floors failed to process. Check the log for details.")
                sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

class MinimapAnalyzer:
    """Analyzes minimap tiles and builds floor map structures."""

    FILENAME_PATTERN = re.compile(r'Minimap_Color_(\d+)_(\d+)_(\d+)\.png')
    TILE_SIZE = 256

    def __init__(self, raw_minimap_dir: str):
        self.raw_minimap_dir = Path(raw_minimap_dir)
        self.tiles: List[TileInfo] = []
        self.floor_maps: Dict[int, FloorMap] = {}

    def scan_tiles(self) -> None:
        logger.info(f"Scanning tiles in {self.raw_minimap_dir}")

        if not self.raw_minimap_dir.exists():
            raise FileNotFoundError(f"Raw minimap directory not found: {self.raw_minimap_dir}")

        tile_count = 0
        for file_path in self.raw_minimap_dir.glob("*.png"):
            tile_info = self._parse_filename(file_path)
            if tile_info:
                self.tiles.append(tile_info)
                tile_count += 1

        logger.info(f"Found {tile_count} valid minimap tiles")

    def _parse_filename(self, file_path: Path) -> Optional[TileInfo]:
        match = self.FILENAME_PATTERN.match(file_path.name)
        if not match:
            logger.warning(f"Invalid filename format: {file_path.name}")
            return None

        try:
            x_coord = int(match.group(1))
            y_coord = int(match.group(2))
            floor = int(match.group(3))

            return TileInfo(
                filename=file_path.name,
                x_coord=x_coord,
                y_coord=y_coord,
                floor=floor,
                file_path=file_path
            )
        except ValueError as e:
            logger.error(f"Error parsing coordinates from {file_path.name}: {e}")
            return None
    
    def build_floor_maps(self) -> None:
        logger.info("Building floor map structures")

        floors_tiles: Dict[int, List[TileInfo]] = {}
        for tile in self.tiles:
            if tile.floor not in floors_tiles:
                floors_tiles[tile.floor] = []
            floors_tiles[tile.floor].append(tile)

        for floor, tiles in floors_tiles.items():
            tiles_dict = {(tile.x_coord, tile.y_coord): tile for tile in tiles}

            x_coords = [tile.x_coord for tile in tiles]
            y_coords = [tile.y_coord for tile in tiles]

            floor_map = FloorMap(
                floor=floor,
                tiles=tiles_dict,
                min_x=min(x_coords),
                max_x=max(x_coords),
                min_y=min(y_coords),
                max_y=max(y_coords)
            )

            self.floor_maps[floor] = floor_map
            logger.info(f"Floor {floor}: {len(tiles)} tiles, "
                       f"bounds ({floor_map.min_x},{floor_map.min_y}) to ({floor_map.max_x},{floor_map.max_y})")

    def get_floor_summary(self) -> Dict[int, Dict]:
        summary = {}
        for floor, floor_map in self.floor_maps.items():
            summary[floor] = {
                'tile_count': len(floor_map.tiles),
                'bounds': {
                    'min_x': floor_map.min_x,
                    'max_x': floor_map.max_x,
                    'min_y': floor_map.min_y,
                    'max_y': floor_map.max_y
                },
                'dimensions': {
                    'width_tiles': floor_map.width_tiles,
                    'height_tiles': floor_map.height_tiles,
                    'width_pixels': floor_map.width_pixels,
                    'height_pixels': floor_map.height_pixels
                }
            }
        return summary
    
    def validate_tiles(self) -> Dict[int, List[str]]:
        issues = {}

        for floor, floor_map in self.floor_maps.items():
            floor_issues = []

            expected_tiles = set()
            for x in range(floor_map.min_x, floor_map.max_x + 256, 256):
                for y in range(floor_map.min_y, floor_map.max_y + 256, 256):
                    expected_tiles.add((x, y))

            actual_tiles = set(floor_map.tiles.keys())
            missing_tiles = expected_tiles - actual_tiles

            if missing_tiles:
                floor_issues.append(f"Missing {len(missing_tiles)} tiles: {list(missing_tiles)[:5]}...")

            coord_counts = {}
            for coord in actual_tiles:
                coord_counts[coord] = coord_counts.get(coord, 0) + 1

            duplicates = {coord: count for coord, count in coord_counts.items() if count > 1}
            if duplicates:
                floor_issues.append(f"Duplicate coordinates: {duplicates}")

            if floor_issues:
                issues[floor] = floor_issues

        return issues

if __name__ == "__main__":
    main()
