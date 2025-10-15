#!/usr/bin/env python3
"""
Exiva Spell Mechanics Utilities

Centralized calculations and constants for Tibia's Exiva spell mechanics,
including direction angles, distance ranges, and coordinate transformations.
"""

import math
from typing import Dict, List, Tuple

# Import canonical enums from dataset_models
from src.models.dataset_models import ExivaRange, ExivaDirection


# ============================================================================
# CONSTANTS
# ============================================================================

# Exiva uses a 2.42:1 diagonal ratio for direction boundaries
# This creates 8 directional sectors with asymmetric boundaries
EXIVA_DIAGONAL_RATIO = 2.42

# Distance ranges in squares (Chebyshev distance)
EXIVA_DISTANCE_RANGES: Dict[ExivaRange, Tuple[float, float]] = {
    ExivaRange.TO_THE: (5.0, 100.0),          # 5-100 squares
    ExivaRange.FAR: (101.0, 250.0),           # 101-250 squares
    ExivaRange.VERY_FAR: (251.0, float('inf')) # 251+ squares
}


# ============================================================================
# ANGLE CALCULATIONS
# ============================================================================

def calculate_primary_angle() -> float:
    """
    Calculate the primary angle for Exiva direction boundaries.
    
    Uses the 2.42:1 diagonal ratio to determine the angle between
    cardinal and intercardinal directions.
    
    Returns:
        Angle in degrees (approximately 22.5°)
    """
    angle_rad = math.atan(1.0 / EXIVA_DIAGONAL_RATIO)
    return math.degrees(angle_rad)


def calculate_boundary_angles() -> List[float]:
    """
    Calculate the 8 boundary angles for Exiva direction sectors.
    
    These angles define the boundaries between the 8 compass directions
    used by the Exiva spell. The angles correspond to the 8 half-winds
    on a 16-wind compass rose.
    
    Returns:
        List of 8 angles in degrees [0-360), ordered as:
        [NNE, ENE, ESE, SSE, SSW, WSW, WNW, NNW]
    """
    primary_angle = calculate_primary_angle()
    
    return [
        primary_angle,                    # north-northeast (NNE) ≈ 22.5°
        90.0 - primary_angle,            # east-northeast (ENE) ≈ 67.5°
        90.0 + primary_angle,            # east-southeast (ESE) ≈ 112.5°
        180.0 - primary_angle,           # south-southeast (SSE) ≈ 157.5°
        180.0 + primary_angle,           # south-southwest (SSW) ≈ 202.5°
        270.0 - primary_angle,           # west-southwest (WSW) ≈ 247.5°
        270.0 + primary_angle,           # west-northwest (WNW) ≈ 292.5°
        360.0 - primary_angle            # north-northwest (NNW) ≈ 337.5°
    ]


# Cache boundary angles (computed once)
_BOUNDARY_ANGLES = calculate_boundary_angles()


def get_direction_angle_ranges() -> Dict[ExivaDirection, Tuple[float, float]]:
    """
    Get the angle range (min, max) for each Exiva direction.
    
    Each direction covers a sector defined by two boundary angles.
    The NORTH direction wraps around 0°.
    
    Returns:
        Dictionary mapping each ExivaDirection to (min_angle, max_angle) tuple
    """
    boundary_angles = _BOUNDARY_ANGLES
    
    return {
        ExivaDirection.NORTH: (boundary_angles[7], boundary_angles[0]),        # NNW to NNE (wraps around 0°)
        ExivaDirection.NORTHEAST: (boundary_angles[0], boundary_angles[1]),    # NNE to ENE
        ExivaDirection.EAST: (boundary_angles[1], boundary_angles[2]),         # ENE to ESE
        ExivaDirection.SOUTHEAST: (boundary_angles[2], boundary_angles[3]),    # ESE to SSE
        ExivaDirection.SOUTH: (boundary_angles[3], boundary_angles[4]),        # SSE to SSW
        ExivaDirection.SOUTHWEST: (boundary_angles[4], boundary_angles[5]),    # SSW to WSW
        ExivaDirection.WEST: (boundary_angles[5], boundary_angles[6]),         # WSW to WNW
        ExivaDirection.NORTHWEST: (boundary_angles[6], boundary_angles[7]),    # WNW to NNW
    }


# Cache direction angle ranges
_DIRECTION_ANGLE_RANGES = get_direction_angle_ranges()


# ============================================================================
# DIRECTION & DISTANCE CALCULATIONS
# ============================================================================

def calculate_angle_between_points(from_x: float, from_y: float, 
                                   to_x: float, to_y: float) -> float:
    """
    Calculate the angle from one point to another.
    
    Args:
        from_x: X coordinate of starting point
        from_y: Y coordinate of starting point
        to_x: X coordinate of target point
        to_y: Y coordinate of target point
    
    Returns:
        Angle in degrees [0-360), where 0° is North, 90° is East
    """
    dx = to_x - from_x
    dy = to_y - from_y
    
    # Calculate angle (note: -dy because Y increases downward in most coordinate systems)
    angle_rad = math.atan2(dx, -dy)
    angle_deg = math.degrees(angle_rad)
    
    # Normalize to 0-360 range
    if angle_deg < 0:
        angle_deg += 360
    
    return angle_deg


def get_direction_from_angle(angle: float) -> ExivaDirection:
    """
    Determine the Exiva direction for a given angle.
    
    Args:
        angle: Angle in degrees [0-360)
    
    Returns:
        The ExivaDirection corresponding to the angle
    """
    # Normalize angle to 0-360 range
    angle = angle % 360
    
    for direction, (min_angle, max_angle) in _DIRECTION_ANGLE_RANGES.items():
        if direction == ExivaDirection.NORTH:
            # Special case: NORTH wraps around 0°
            if angle >= min_angle or angle <= max_angle:
                return direction
        else:
            if min_angle <= angle <= max_angle:
                return direction
    
    # Fallback (should never happen with correct boundary angles)
    return ExivaDirection.NORTH


def calculate_chebyshev_distance(from_x: float, from_y: float,
                                 to_x: float, to_y: float) -> float:
    """
    Calculate Chebyshev distance (square-based distance) between two points.
    
    Tibia uses Chebyshev distance (also called chessboard distance),
    where distance is the maximum of absolute differences in coordinates.
    
    Args:
        from_x: X coordinate of starting point
        from_y: Y coordinate of starting point
        to_x: X coordinate of target point
        to_y: Y coordinate of target point
    
    Returns:
        Distance in squares
    """
    return max(abs(to_x - from_x), abs(to_y - from_y))


def get_range_from_distance(distance: float) -> ExivaRange:
    """
    Determine the Exiva range category for a given distance.
    
    Args:
        distance: Distance in squares (Chebyshev distance)
    
    Returns:
        The ExivaRange corresponding to the distance
    """
    for exiva_range, (min_dist, max_dist) in EXIVA_DISTANCE_RANGES.items():
        if min_dist <= distance <= max_dist:
            return exiva_range
    
    # Fallback for very large distances
    return ExivaRange.VERY_FAR


def is_angle_in_direction(angle: float, direction: ExivaDirection) -> bool:
    """
    Check if an angle falls within a specific Exiva direction sector.
    
    Args:
        angle: Angle in degrees [0-360)
        direction: ExivaDirection to check
    
    Returns:
        True if angle is within the direction's sector
    """
    min_angle, max_angle = _DIRECTION_ANGLE_RANGES[direction]
    
    # Normalize angle
    angle = angle % 360
    
    # Handle NORTH direction which wraps around 0°
    if direction == ExivaDirection.NORTH:
        return angle >= min_angle or angle <= max_angle
    else:
        return min_angle <= angle <= max_angle


def is_distance_in_range(distance: float, exiva_range: ExivaRange) -> bool:
    """
    Check if a distance falls within a specific Exiva range category.
    
    Args:
        distance: Distance in squares
        exiva_range: ExivaRange to check
    
    Returns:
        True if distance is within the range
    """
    min_dist, max_dist = EXIVA_DISTANCE_RANGES[exiva_range]
    return min_dist <= distance <= max_dist


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_exiva_reading(from_x: float, from_y: float, 
                     to_x: float, to_y: float) -> Tuple[ExivaDirection, ExivaRange]:
    """
    Get the complete Exiva reading (direction and range) between two points.
    
    Args:
        from_x: X coordinate of player position
        from_y: Y coordinate of player position
        to_x: X coordinate of target position
        to_y: Y coordinate of target position
    
    Returns:
        Tuple of (ExivaDirection, ExivaRange)
    """
    angle = calculate_angle_between_points(from_x, from_y, to_x, to_y)
    distance = calculate_chebyshev_distance(from_x, from_y, to_x, to_y)
    
    direction = get_direction_from_angle(angle)
    exiva_range = get_range_from_distance(distance)
    
    return direction, exiva_range


def get_boundary_angles() -> List[float]:
    """
    Get the cached boundary angles.
    
    Returns:
        List of 8 boundary angles in degrees
    """
    return _BOUNDARY_ANGLES.copy()


def get_direction_ranges() -> Dict[ExivaDirection, Tuple[float, float]]:
    """
    Get the cached direction angle ranges.
    
    Returns:
        Dictionary mapping directions to angle ranges
    """
    return _DIRECTION_ANGLE_RANGES.copy()

