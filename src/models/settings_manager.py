#!/usr/bin/env python3
"""
Settings Manager for FiendishFinder

Manages application settings and configuration persistence.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


class SettingsManager:
    """Manages application settings and configuration."""

    def __init__(self, settings_dir: str = "settings"):
        """Initialize the settings manager.
        
        Args:
            settings_dir: Directory to store settings files
        """
        self.settings_dir = Path(settings_dir)
        self.settings_dir.mkdir(exist_ok=True)
        
        self.settings_file = self.settings_dir / "app_settings.json"
        self.settings: Dict[str, Any] = {}
        
        # Default settings
        self.defaults = {
            'screenshot_hotkey': '.',
            'tibia_screenshot_folder': r'C:\Users\bulaw\AppData\Local\Tibia\packages\Tibia\screenshots'
        }
        
        self.load_settings()
    
    def load_settings(self) -> None:
        """Load settings from JSON file."""
        if not self.settings_file.exists():
            # Use defaults if no settings file exists
            self.settings = self.defaults.copy()
            self.save_settings()
            return
        
        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                loaded_settings = json.load(f)
            
            # Merge with defaults to ensure all keys exist
            self.settings = self.defaults.copy()
            self.settings.update(loaded_settings)
            
        except Exception as e:
            print(f"Error loading settings: {e}")
            self.settings = self.defaults.copy()
    
    def save_settings(self) -> bool:
        """Save settings to JSON file.
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value.
        
        Args:
            key: Setting key
            default: Default value if key doesn't exist
            
        Returns:
            Setting value or default
        """
        return self.settings.get(key, default)
    
    def set(self, key: str, value: Any) -> bool:
        """Set a setting value and save.
        
        Args:
            key: Setting key
            value: Setting value
            
        Returns:
            True if saved successfully, False otherwise
        """
        self.settings[key] = value
        return self.save_settings()
    
    def get_screenshot_hotkey(self) -> str:
        """Get the screenshot hotkey.
        
        Returns:
            Screenshot hotkey string
        """
        return self.get('screenshot_hotkey', '.')
    
    def set_screenshot_hotkey(self, hotkey: str) -> bool:
        """Set the screenshot hotkey.

        Args:
            hotkey: Hotkey string

        Returns:
            True if saved successfully, False otherwise
        """
        return self.set('screenshot_hotkey', hotkey)

    def get_tibia_screenshot_folder(self) -> str:
        """Get the Tibia screenshot folder path.

        Returns:
            Tibia screenshot folder path
        """
        return self.get('tibia_screenshot_folder', r'C:\Users\bulaw\AppData\Local\Tibia\packages\Tibia\screenshots')

    def set_tibia_screenshot_folder(self, folder_path: str) -> bool:
        """Set the Tibia screenshot folder path.

        Args:
            folder_path: Folder path string

        Returns:
            True if saved successfully, False otherwise
        """
        return self.set('tibia_screenshot_folder', folder_path)

