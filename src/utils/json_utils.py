#!/usr/bin/env python3
"""
JSON Utilities for FiendishFinder

Centralized JSON file I/O operations with consistent error handling and logging.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def load_json(file_path: Path,
             default: Any = None,
             create_if_missing: bool = False,
             custom_logger: Optional[logging.Logger] = None) -> Any:
    """
    Load data from a JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        default: Default value to return if file doesn't exist or loading fails
        create_if_missing: If True, create file with default value if it doesn't exist
        custom_logger: Optional custom logger to use
    
    Returns:
        Loaded data or default value
    """
    log = custom_logger or logger
    
    # Create file with default if requested
    if create_if_missing and not file_path.exists() and default is not None:
        save_json(file_path, default, custom_logger=custom_logger)
        return default
    
    # Return default if file doesn't exist
    if not file_path.exists():
        log.info(f"JSON file not found: {file_path}, using default value")
        return default
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        log.debug(f"Successfully loaded JSON from {file_path}")
        return data
        
    except json.JSONDecodeError as e:
        log.error(f"Invalid JSON in {file_path}: {e}")
        return default
        
    except Exception as e:
        log.error(f"Error loading JSON from {file_path}: {e}")
        return default


def save_json(file_path: Path,
             data: Any,
             indent: int = 2,
             ensure_ascii: bool = False,
             custom_logger: Optional[logging.Logger] = None) -> bool:
    """
    Save data to a JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        data: Data to save (must be JSON-serializable)
        indent: Indentation level for pretty printing
        ensure_ascii: If True, escape non-ASCII characters
        custom_logger: Optional custom logger to use
    
    Returns:
        True if saved successfully, False otherwise
    """
    log = custom_logger or logger
    
    try:
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        
        log.debug(f"Successfully saved JSON to {file_path}")
        return True
        
    except TypeError as e:
        log.error(f"Data is not JSON-serializable: {e}")
        return False
        
    except Exception as e:
        log.error(f"Error saving JSON to {file_path}: {e}")
        return False

