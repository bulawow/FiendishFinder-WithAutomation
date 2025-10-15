"""
Parser for Tibia minimap markers from minimapmarkers.bin file.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MapMarker:
    """Represents a single map marker."""
    x: int  # Absolute X coordinate
    y: int  # Absolute Y coordinate
    floor: int  # Floor ID (0-15)
    icon_id: int  # Icon image ID (0-19)
    description: str  # Marker description text
    
    def __repr__(self):
        return f"MapMarker(x={self.x}, y={self.y}, floor={self.floor}, icon={self.icon_id}, desc='{self.description}')"


class MarkerParser:
    """Parser for Tibia minimapmarkers.bin files."""
    
    # Icon ID to filename mapping (based on OFFICIAL tibiamaps.io documentation)
    # Source: https://tibiamaps.io/guides/map-file-format
    ICON_MAP = {
        0x00: 'checkmark.png',   # Green checkmark ✔
        0x01: 'question.png',    # Blue question mark ❓
        0x02: 'exclamation.png', # Red exclamation mark ❗
        0x03: 'star.png',        # Orange star 🟊
        0x04: 'crossmark.png',   # Bright red crossmark ❌
        0x05: 'cross.png',       # Dark red cross 🕇
        0x06: 'mouth.png',       # Mouth with red lips 👄
        0x07: 'sword.png',       # Spear 🏹 (using sword.png)
        0x08: 'sword.png',       # Sword ⚔
        0x09: 'flag.png',        # Blue flag ⚑
        0x0A: 'lock.png',        # Golden lock 🔒
        0x0B: 'bag.png',         # Brown bag 👛
        0x0C: 'skull.png',       # Skull 💀
        0x0D: 'dollar.png',      # Green dollar sign 💰💲
        0x0E: 'red-up.png',      # Red arrow up ⬆️🔺
        0x0F: 'red-down.png',    # Red arrow down ⬇🔻
        0x10: 'red-right.png',   # Red arrow right ➡️
        0x11: 'red-left.png',    # Red arrow left ⬅️
        0x12: 'up.png',          # Green arrow up ⬆ (NPC positions)
        0x13: 'down.png',        # Green arrow down ⬇
    }
    
    def __init__(self, marker_file: str = "assets/minimap/minimapmarkers.bin"):
        """Initialize the marker parser.
        
        Args:
            marker_file: Path to the minimapmarkers.bin file
        """
        self.marker_file = Path(marker_file)
        self.markers: List[MapMarker] = []
        
    def parse_coordinate(self, data: bytes, offset: int) -> Tuple[int, int]:
        """Parse a coordinate value from the binary data.
        
        Args:
            data: Binary data
            offset: Starting offset in the data
            
        Returns:
            Tuple of (coordinate_value, bytes_consumed)
        """
        # Read bytes until we hit a marker (0x10 or 0x18)
        coord_bytes = []
        i = offset
        
        while i < len(data) and data[i] not in (0x10, 0x18):
            coord_bytes.append(data[i])
            i += 1
        
        if not coord_bytes:
            return 0, 0
        
        # Convert bytes to coordinate using Tibia's formula
        # x = x1 + ((x2 - 1) << 7) + ((x3 - 1) << 14)
        x1 = coord_bytes[0] if len(coord_bytes) > 0 else 0
        x2 = coord_bytes[1] if len(coord_bytes) > 1 else 1
        x3 = coord_bytes[2] if len(coord_bytes) > 2 else 1
        
        coord = x1 + ((x2 - 1) << 7) + ((x3 - 1) << 14)
        
        return coord, len(coord_bytes)
    
    def parse_markers(self) -> List[MapMarker]:
        """Parse all markers from the binary file.
        
        Returns:
            List of MapMarker objects
        """
        if not self.marker_file.exists():
            logger.warning(f"Marker file not found: {self.marker_file}")
            return []
        
        try:
            with open(self.marker_file, 'rb') as f:
                data = f.read()
        except Exception as e:
            logger.error(f"Error reading marker file: {e}")
            return []
        
        markers = []
        i = 0
        
        while i < len(data):
            # Look for marker start (0x0A)
            if data[i] != 0x0A:
                i += 1
                continue
            
            try:
                # Read marker size
                if i + 1 >= len(data):
                    break
                marker_size = data[i + 1]
                
                # Skip to coordinate data block (should be 0x0A)
                if i + 2 >= len(data) or data[i + 2] != 0x0A:
                    i += 1
                    continue
                
                # Read coordinate block size
                if i + 3 >= len(data):
                    break
                coord_block_size = data[i + 3]
                
                # Look for X coordinate marker (0x08)
                if i + 4 >= len(data) or data[i + 4] != 0x08:
                    i += 1
                    continue
                
                # Parse X coordinate
                x, x_bytes = self.parse_coordinate(data, i + 5)
                
                # Skip to Y coordinate (0x10 marker)
                y_start = i + 5 + x_bytes
                if y_start >= len(data) or data[y_start] != 0x10:
                    i += 1
                    continue
                
                # Parse Y coordinate
                y, y_bytes = self.parse_coordinate(data, y_start + 1)
                
                # Skip to floor (0x18 marker)
                floor_start = y_start + 1 + y_bytes
                if floor_start >= len(data) or data[floor_start] != 0x18:
                    i += 1
                    continue
                
                # Read floor
                if floor_start + 1 >= len(data):
                    break
                floor = data[floor_start + 1]
                
                # Skip 0x10 marker
                icon_start = floor_start + 2
                if icon_start >= len(data) or data[icon_start] != 0x10:
                    i += 1
                    continue
                
                # Read icon ID
                if icon_start + 1 >= len(data):
                    break
                icon_id = data[icon_start + 1]
                
                # Skip 0x1A marker
                desc_start = icon_start + 2
                if desc_start >= len(data) or data[desc_start] != 0x1A:
                    i += 1
                    continue
                
                # Read description length
                if desc_start + 1 >= len(data):
                    break
                desc_len = data[desc_start + 1]
                
                # Read description
                desc_bytes_start = desc_start + 2
                if desc_bytes_start + desc_len > len(data):
                    break
                
                try:
                    description = data[desc_bytes_start:desc_bytes_start + desc_len].decode('utf-8', errors='replace')
                except:
                    description = ""
                
                # Create marker
                marker = MapMarker(
                    x=x,
                    y=y,
                    floor=floor,
                    icon_id=icon_id,
                    description=description
                )
                markers.append(marker)
                
                # Move to next marker (skip past 0x20 0x00 end marker)
                i = desc_bytes_start + desc_len + 2
                
            except Exception as e:
                logger.debug(f"Error parsing marker at offset {i}: {e}")
                i += 1
                continue
        
        logger.info(f"Parsed {len(markers)} markers from {self.marker_file}")
        self.markers = markers
        return markers
    
    def get_markers_for_floor(self, floor: int) -> List[MapMarker]:
        """Get all markers for a specific floor.
        
        Args:
            floor: Floor number (0-15)
            
        Returns:
            List of markers on that floor
        """
        return [m for m in self.markers if m.floor == floor]
    
    def get_icon_filename(self, icon_id: int) -> Optional[str]:
        """Get the icon filename for a given icon ID.
        
        Args:
            icon_id: Icon ID from marker
            
        Returns:
            Icon filename or None if not found
        """
        return self.ICON_MAP.get(icon_id)

