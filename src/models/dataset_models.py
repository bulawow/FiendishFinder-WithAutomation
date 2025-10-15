#!/usr/bin/env python3
"""
Dataset Models for FiendishFinder

Data structures for creating and managing training/testing datasets
for crosshair detection and Exiva spell message parsing.
"""

import json
import uuid
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from PIL import Image


class ExivaRange(Enum):
    """Exiva spell distance ranges."""
    NONE = "none"  # Range not specified (offline/not found/no range given)
    TO_THE = "to the"  # 5-100 squares
    FAR = "far"  # 101-250 squares
    VERY_FAR = "very far"  # 251+ squares


class ExivaDirection(Enum):
    """Exiva spell directions (8 compass points)."""
    NORTH = "N"
    NORTHEAST = "NE"
    EAST = "E"
    SOUTHEAST = "SE"
    SOUTH = "S"
    SOUTHWEST = "SW"
    WEST = "W"
    NORTHWEST = "NW"
    NONE = "none"  # No direction (offline/not found)


class MonsterDifficulty(Enum):
    """Monster difficulty levels."""
    NONE = "none"
    UNKNOWN = "unknown"
    HARMLESS = "harmless"
    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    CHALLENGING = "challenging"


class FloorIndication(Enum):
    """Floor level indication from Exiva spell."""
    SAME_FLOOR = "none"  # Same floor
    HIGHER_LEVEL = "higher level"  # Different floor, higher
    LOWER_LEVEL = "lower level"  # Different floor, lower


@dataclass
class MinimapDatasetEntry:
    """Dataset entry for minimap images with crosshair positions."""
    entry_id: str
    screenshot_path: str  # Relative path to minimap image
    crosshair_x: float  # X coordinate of crosshair center
    crosshair_y: float  # Y coordinate of crosshair center
    floor: int  # Floor number where the character is located
    image_width: int  # Width of the minimap image
    image_height: int  # Height of the minimap image
    notes: str  # Optional notes about this entry
    created_timestamp: float
    modified_timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MinimapDatasetEntry':
        """Create MinimapDatasetEntry from dictionary."""
        # Handle legacy entries without floor field
        if 'floor' not in data:
            data['floor'] = 7  # Default to floor 7 for legacy entries
        return cls(**data)


@dataclass
class ExivaDatasetEntry:
    """Dataset entry for Exiva spell message parsing."""
    entry_id: str
    screenshot_path: str  # Relative path to screenshot image

    # Exiva spell data
    range: str  # ExivaRange value
    direction: str  # ExivaDirection value
    difficulty: str  # MonsterDifficulty value
    floor_indication: str  # FloorIndication value

    # Image metadata
    image_width: int
    image_height: int

    # Additional metadata
    notes: str  # Optional notes about this entry
    raw_text: str  # Optional: the actual Exiva text from the screenshot
    created_timestamp: float
    modified_timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExivaDatasetEntry':
        """Create ExivaDatasetEntry from dictionary."""
        return cls(**data)


class DatasetManager:
    """Manager for loading, saving, and managing datasets."""

    def __init__(self, dataset_dir: str = "datasets"):
        """Initialize the dataset manager.

        Args:
            dataset_dir: Directory to store dataset files
        """
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(exist_ok=True)

        # Create subdirectories for each dataset type
        self.minimap_dir = self.dataset_dir / "minimap"
        self.exiva_dir = self.dataset_dir / "exiva"

        # Create separate screenshot directories for each dataset type
        self.minimap_screenshots_dir = self.minimap_dir / "screenshots"
        self.exiva_screenshots_dir = self.exiva_dir / "screenshots"

        self.minimap_dir.mkdir(exist_ok=True)
        self.exiva_dir.mkdir(exist_ok=True)
        self.minimap_screenshots_dir.mkdir(exist_ok=True)
        self.exiva_screenshots_dir.mkdir(exist_ok=True)

        # Dataset files
        self.minimap_dataset_file = self.minimap_dir / "minimap_dataset.jsonl"  # Changed to JSONL
        self.exiva_dataset_file = self.exiva_dir / "exiva_dataset.json"
        self.global_regions_file = self.dataset_dir / "global_regions.json"

        # Legacy JSON file for backward compatibility
        self.minimap_dataset_json_legacy = self.minimap_dir / "minimap_dataset.json"

    # Minimap Dataset Methods

    def load_minimap_dataset(self) -> List[MinimapDatasetEntry]:
        """Load the minimap dataset from JSONL file.

        Also supports loading from legacy JSON format for backward compatibility.
        """
        # Try JSONL format first
        if self.minimap_dataset_file.exists():
            try:
                entries = []
                with open(self.minimap_dataset_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            data = json.loads(line)
                            entries.append(MinimapDatasetEntry.from_dict(data))
                return entries
            except Exception as e:
                print(f"Error loading minimap dataset from JSONL: {e}")
                return []

        # Fall back to legacy JSON format
        if self.minimap_dataset_json_legacy.exists():
            try:
                with open(self.minimap_dataset_json_legacy, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return [MinimapDatasetEntry.from_dict(entry) for entry in data]
            except Exception as e:
                print(f"Error loading minimap dataset from legacy JSON: {e}")
                return []

        return []

    def get_minimap_dataset_count(self) -> int:
        """Get the count of entries in the minimap dataset without loading all data.

        Returns:
            Number of entries in the dataset
        """
        # Try JSONL format first
        if self.minimap_dataset_file.exists():
            try:
                count = 0
                with open(self.minimap_dataset_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():  # Count non-empty lines
                            count += 1
                return count
            except Exception as e:
                print(f"Error counting minimap dataset entries from JSONL: {e}")
                return 0

        # Fall back to legacy JSON format
        if self.minimap_dataset_json_legacy.exists():
            try:
                with open(self.minimap_dataset_json_legacy, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return len(data) if isinstance(data, list) else 0
            except Exception as e:
                print(f"Error counting minimap dataset entries from legacy JSON: {e}")
                return 0

        return 0

    def load_minimap_dataset_paginated(self, offset: int = 0, limit: int = 100) -> List[MinimapDatasetEntry]:
        """Load a paginated subset of the minimap dataset.

        Args:
            offset: Starting index (0-based)
            limit: Maximum number of entries to load

        Returns:
            List of MinimapDatasetEntry objects for the requested page
        """
        # Try JSONL format first
        if self.minimap_dataset_file.exists():
            try:
                entries = []
                current_index = 0
                with open(self.minimap_dataset_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        # Skip lines before offset
                        if current_index < offset:
                            current_index += 1
                            continue

                        # Stop if we've reached the limit
                        if len(entries) >= limit:
                            break

                        data = json.loads(line)
                        entries.append(MinimapDatasetEntry.from_dict(data))
                        current_index += 1

                return entries
            except Exception as e:
                print(f"Error loading minimap dataset page from JSONL: {e}")
                return []

        # Fall back to legacy JSON format
        if self.minimap_dataset_json_legacy.exists():
            try:
                with open(self.minimap_dataset_json_legacy, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Get the requested slice
                page_data = data[offset:offset + limit]
                return [MinimapDatasetEntry.from_dict(entry) for entry in page_data]
            except Exception as e:
                print(f"Error loading minimap dataset page from legacy JSON: {e}")
                return []

        return []

    def get_minimap_positions_for_floor(self, floor: int) -> List[Tuple[float, float]]:
        """Get only the crosshair positions for a specific floor (optimized for coverage overlay).

        Args:
            floor: Floor number to filter by

        Returns:
            List of (crosshair_x, crosshair_y) tuples for the specified floor
        """
        # Try JSONL format first
        if self.minimap_dataset_file.exists():
            try:
                positions = []
                with open(self.minimap_dataset_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        entry_dict = json.loads(line)
                        if entry_dict.get('floor') == floor:
                            positions.append((entry_dict.get('crosshair_x'), entry_dict.get('crosshair_y')))

                return positions
            except Exception as e:
                print(f"Error loading minimap positions for floor {floor} from JSONL: {e}")
                return []

        # Fall back to legacy JSON format
        if self.minimap_dataset_json_legacy.exists():
            try:
                positions = []
                with open(self.minimap_dataset_json_legacy, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Extract only positions for the specified floor
                for entry_dict in data:
                    if entry_dict.get('floor') == floor:
                        positions.append((entry_dict.get('crosshair_x'), entry_dict.get('crosshair_y')))

                return positions
            except Exception as e:
                print(f"Error loading minimap positions for floor {floor} from legacy JSON: {e}")
                return []

        return []

    def save_minimap_dataset(self, entries: List[MinimapDatasetEntry]) -> bool:
        """Save the minimap dataset to JSONL file.

        Note: This completely rewrites the JSONL file with all entries.
        For appending new entries, use add_minimap_entries_batch() instead.
        """
        try:
            with open(self.minimap_dataset_file, 'w', encoding='utf-8') as f:
                for entry in entries:
                    json.dump(entry.to_dict(), f, separators=(',', ':'))
                    f.write('\n')
            return True
        except Exception as e:
            print(f"Error saving minimap dataset: {e}")
            return False

    def add_minimap_entry(self, entry: MinimapDatasetEntry) -> bool:
        """Add a new entry to the minimap dataset by appending to JSONL file."""
        try:
            with open(self.minimap_dataset_file, 'a', encoding='utf-8') as f:
                json.dump(entry.to_dict(), f, separators=(',', ':'))
                f.write('\n')
            return True
        except Exception as e:
            print(f"Error adding minimap entry: {e}")
            return False

    def add_minimap_entries_batch(self, new_entries: List[MinimapDatasetEntry]) -> bool:
        """Add multiple entries to the minimap dataset by appending to JSONL file.

        This is extremely efficient for large datasets because it only appends
        new entries without loading existing data.

        Args:
            new_entries: List of MinimapDatasetEntry objects to add

        Returns:
            True if saved successfully, False otherwise
        """
        if not new_entries:
            return True

        try:
            with open(self.minimap_dataset_file, 'a', encoding='utf-8') as f:
                for entry in new_entries:
                    json.dump(entry.to_dict(), f, separators=(',', ':'))
                    f.write('\n')
            return True
        except Exception as e:
            print(f"Error adding minimap entries batch: {e}")
            return False

    def update_minimap_entry(self, entry_id: str, updated_entry: MinimapDatasetEntry) -> bool:
        """Update an existing minimap dataset entry."""
        entries = self.load_minimap_dataset()
        for i, entry in enumerate(entries):
            if entry.entry_id == entry_id:
                entries[i] = updated_entry
                return self.save_minimap_dataset(entries)
        return False

    def delete_minimap_entry(self, entry_id: str) -> bool:
        """Delete a minimap dataset entry and its associated screenshot file."""
        entries = self.load_minimap_dataset()

        # Find the entry to get the screenshot path before deleting
        entry_to_delete = None
        for entry in entries:
            if entry.entry_id == entry_id:
                entry_to_delete = entry
                break

        # Delete the screenshot file if it exists
        if entry_to_delete:
            screenshot_path = self.get_screenshot_full_path(entry_to_delete.screenshot_path)
            if screenshot_path.exists():
                try:
                    screenshot_path.unlink()
                    print(f"Deleted screenshot: {screenshot_path}")
                except Exception as e:
                    print(f"Error deleting screenshot {screenshot_path}: {e}")

        # Remove entry from list and save
        entries = [e for e in entries if e.entry_id != entry_id]
        return self.save_minimap_dataset(entries)

    # Exiva Dataset Methods
    
    def load_exiva_dataset(self) -> List[ExivaDatasetEntry]:
        """Load the Exiva dataset from JSON file."""
        if not self.exiva_dataset_file.exists():
            return []
        
        try:
            with open(self.exiva_dataset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [ExivaDatasetEntry.from_dict(entry) for entry in data]
        except Exception as e:
            print(f"Error loading Exiva dataset: {e}")
            return []

    def save_exiva_dataset(self, entries: List[ExivaDatasetEntry]) -> bool:
        """Save the Exiva dataset to JSON file."""
        try:
            data = [entry.to_dict() for entry in entries]
            with open(self.exiva_dataset_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving Exiva dataset: {e}")
            return False

    def add_exiva_entry(self, entry: ExivaDatasetEntry) -> bool:
        """Add a new entry to the Exiva dataset."""
        entries = self.load_exiva_dataset()
        entries.append(entry)
        return self.save_exiva_dataset(entries)

    def update_exiva_entry(self, entry_id: str, updated_entry: ExivaDatasetEntry) -> bool:
        """Update an existing Exiva dataset entry."""
        entries = self.load_exiva_dataset()
        for i, entry in enumerate(entries):
            if entry.entry_id == entry_id:
                entries[i] = updated_entry
                return self.save_exiva_dataset(entries)
        return False

    def delete_exiva_entry(self, entry_id: str) -> bool:
        """Delete an Exiva dataset entry and its associated screenshot file."""
        entries = self.load_exiva_dataset()

        # Find the entry to get the screenshot path before deleting
        entry_to_delete = None
        for entry in entries:
            if entry.entry_id == entry_id:
                entry_to_delete = entry
                break

        # Delete the screenshot file if it exists
        if entry_to_delete:
            screenshot_path = self.get_screenshot_full_path(entry_to_delete.screenshot_path)
            if screenshot_path.exists():
                try:
                    screenshot_path.unlink()
                    print(f"Deleted screenshot: {screenshot_path}")
                except Exception as e:
                    print(f"Error deleting screenshot {screenshot_path}: {e}")

        # Remove entry from list and save
        entries = [e for e in entries if e.entry_id != entry_id]
        return self.save_exiva_dataset(entries)

    # Screenshot Management

    def copy_screenshot_to_dataset(self, source_path: str, entry_id: str, dataset_type: str = "exiva",
                                   crop_region: Optional[Tuple[int, int, int, int]] = None) -> Optional[str]:
        """Copy a screenshot to the dataset directory, optionally cropping it first.

        Args:
            source_path: Path to the source screenshot
            entry_id: Unique ID for this entry
            dataset_type: Type of dataset ("minimap" or "exiva")
            crop_region: Optional (x, y, width, height) tuple to crop the image before saving

        Returns:
            Relative path to the copied screenshot, or None on error
        """
        try:
            source = Path(source_path)
            if not source.exists():
                print(f"Source screenshot not found: {source_path}")
                return None

            # Create filename with entry_id and original extension
            extension = source.suffix
            filename = f"{entry_id}{extension}"

            # Determine destination based on dataset type
            if dataset_type == "minimap":
                dest_path = self.minimap_screenshots_dir / filename
                rel_path = f"minimap/screenshots/{filename}"
            else:  # exiva
                dest_path = self.exiva_screenshots_dir / filename
                rel_path = f"exiva/screenshots/{filename}"

            # If crop_region is specified, crop the image before saving
            if crop_region:
                x, y, width, height = crop_region
                img = Image.open(source)
                cropped_img = img.crop((x, y, x + width, y + height))
                cropped_img.save(str(dest_path))
            else:
                # Copy the file without cropping
                import shutil
                shutil.copy2(source, dest_path)

            # Return relative path from dataset_dir
            return rel_path
        except Exception as e:
            print(f"Error copying screenshot: {e}")
            return None

    def get_screenshot_full_path(self, relative_path: str) -> Path:
        """Get the full path to a screenshot from its relative path."""
        return self.dataset_dir / relative_path

    def get_dataset_stats(self) -> Dict[str, int]:
        """Get statistics about the datasets."""
        minimap_entries = self.load_minimap_dataset()
        exiva_entries = self.load_exiva_dataset()

        return {
            'minimap_count': len(minimap_entries),
            'exiva_count': len(exiva_entries),
            'total_count': len(minimap_entries) + len(exiva_entries)
        }

    # Global Region Settings Methods

    def save_global_regions(self, minimap_region: Optional[Tuple[int, int, int, int]],
                           exiva_region: Optional[Tuple[int, int, int, int]]) -> bool:
        """Save global minimap and exiva regions.

        Args:
            minimap_region: (x, y, width, height) for minimap area, or None
            exiva_region: (x, y, width, height) for exiva area, or None

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            data = {
                'minimap_region': minimap_region,
                'exiva_region': exiva_region
            }
            with open(self.global_regions_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving global regions: {e}")
            return False

    def load_global_regions(self) -> Dict[str, Optional[Tuple[int, int, int, int]]]:
        """Load global minimap and exiva regions.

        Returns:
            Dictionary with 'minimap_region' and 'exiva_region' keys
        """
        if not self.global_regions_file.exists():
            return {'minimap_region': None, 'exiva_region': None}

        try:
            with open(self.global_regions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert lists back to tuples if they exist
            minimap_region = tuple(data['minimap_region']) if data.get('minimap_region') else None
            exiva_region = tuple(data['exiva_region']) if data.get('exiva_region') else None

            return {
                'minimap_region': minimap_region,
                'exiva_region': exiva_region
            }
        except Exception as e:
            print(f"Error loading global regions: {e}")
            return {'minimap_region': None, 'exiva_region': None}

