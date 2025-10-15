"""
FiendishFinder - Tibia Minimap Viewer & Dataset Creator

A comprehensive system for viewing and stitching Tibia minimap images,
with dataset creation tools for minimap and Exiva spell parsing.
"""

__version__ = "1.0.0"
__author__ = "FiendishFinder Team"

from src.core.minimap_stitcher import MinimapStitchingSystem
from src.models.dataset_models import (
    DatasetManager,
    MinimapDatasetEntry,
    ExivaDatasetEntry,
    ExivaRange,
    ExivaDirection,
    MonsterDifficulty,
    FloorIndication
)

__all__ = [
    "MinimapStitchingSystem",
    "DatasetManager",
    "MinimapDatasetEntry",
    "ExivaDatasetEntry",
    "ExivaRange",
    "ExivaDirection",
    "MonsterDifficulty",
    "FloorIndication",
]

