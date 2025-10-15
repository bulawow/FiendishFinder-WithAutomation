"""
Screenshot Monitor Utility

Monitors a directory for new screenshot files and provides utilities
for retrieving the latest screenshot.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Callable
from PyQt6.QtCore import QThread, pyqtSignal
from pynput import keyboard

logger = logging.getLogger(__name__)


class ScreenshotMonitor(QThread):
    """Monitor for screenshot hotkey and directory changes."""
    
    # Signals
    screenshot_detected = pyqtSignal(str)  # Emits path to new screenshot
    status_update = pyqtSignal(str)  # Emits status messages
    error_occurred = pyqtSignal(str)  # Emits error messages
    
    def __init__(self, screenshot_dir: str, hotkey: str = '.', parent=None):
        """Initialize the screenshot monitor.
        
        Args:
            screenshot_dir: Directory to monitor for screenshots
            hotkey: Keyboard key to monitor for screenshot capture
            parent: Parent QObject
        """
        super().__init__(parent)
        self.screenshot_dir = Path(screenshot_dir)
        self.hotkey = hotkey.lower()
        self.is_monitoring = False
        self.keyboard_listener = None
        
        logger.info(f"Screenshot monitor initialized: dir={screenshot_dir}, hotkey={hotkey}")
    
    def run(self):
        """Run the monitoring thread."""
        self.is_monitoring = True
        self.status_update.emit(f"Waiting for screenshot hotkey '{self.hotkey}'...")
        
        # Start keyboard listener
        try:
            with keyboard.Listener(on_press=self._on_key_press) as listener:
                self.keyboard_listener = listener
                while self.is_monitoring:
                    time.sleep(0.1)  # Small sleep to prevent busy waiting
        except Exception as e:
            logger.error(f"Error in keyboard listener: {e}")
            self.error_occurred.emit(f"Keyboard listener error: {e}")
    
    def _on_key_press(self, key):
        """Handle key press events.
        
        Args:
            key: The key that was pressed
        """
        if not self.is_monitoring:
            return
        
        try:
            # Check if the pressed key matches our hotkey
            key_char = None
            if hasattr(key, 'char') and key.char:
                key_char = key.char.lower()
            
            if key_char == self.hotkey:
                logger.info(f"Screenshot hotkey '{self.hotkey}' detected")
                self.status_update.emit("Screenshot hotkey detected! Waiting for file...")
                
                # Wait a moment for the screenshot to be written to disk
                time.sleep(1.0)
                
                # Get the latest screenshot
                latest_screenshot = self.get_latest_screenshot()
                
                if latest_screenshot:
                    logger.info(f"Latest screenshot found: {latest_screenshot}")
                    self.screenshot_detected.emit(str(latest_screenshot))
                else:
                    logger.warning("No screenshot found after hotkey press")
                    self.error_occurred.emit("No screenshot found in directory")
        
        except Exception as e:
            logger.error(f"Error handling key press: {e}")
            self.error_occurred.emit(f"Key press error: {e}")
    
    def get_latest_screenshot(self) -> Optional[Path]:
        """Get the most recently created screenshot file.
        
        Returns:
            Path to the latest screenshot, or None if no screenshots found
        """
        if not self.screenshot_dir.exists():
            logger.error(f"Screenshot directory does not exist: {self.screenshot_dir}")
            return None
        
        try:
            # Find all image files in the directory
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
            screenshot_files = []
            
            for ext in image_extensions:
                screenshot_files.extend(self.screenshot_dir.glob(f'*{ext}'))
            
            if not screenshot_files:
                logger.warning(f"No screenshot files found in {self.screenshot_dir}")
                return None
            
            # Sort by modification time (most recent first)
            screenshot_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            latest = screenshot_files[0]
            logger.info(f"Latest screenshot: {latest.name} (modified: {time.ctime(latest.stat().st_mtime)})")
            
            return latest
        
        except Exception as e:
            logger.error(f"Error getting latest screenshot: {e}")
            return None
    
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        logger.info("Stopping screenshot monitor")
        self.is_monitoring = False
        
        if self.keyboard_listener:
            try:
                self.keyboard_listener.stop()
            except Exception as e:
                logger.error(f"Error stopping keyboard listener: {e}")
        
        self.status_update.emit("Monitoring stopped")
        self.quit()
        self.wait()


def get_latest_screenshot_sync(screenshot_dir: str) -> Optional[Path]:
    """Synchronously get the latest screenshot from a directory.
    
    This is a utility function for one-time screenshot retrieval without
    starting a monitoring thread.
    
    Args:
        screenshot_dir: Directory containing screenshots
        
    Returns:
        Path to the latest screenshot, or None if no screenshots found
    """
    screenshot_path = Path(screenshot_dir)
    
    if not screenshot_path.exists():
        logger.error(f"Screenshot directory does not exist: {screenshot_dir}")
        return None
    
    try:
        # Find all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        screenshot_files = []
        
        for ext in image_extensions:
            screenshot_files.extend(screenshot_path.glob(f'*{ext}'))
        
        if not screenshot_files:
            logger.warning(f"No screenshot files found in {screenshot_dir}")
            return None
        
        # Sort by modification time (most recent first)
        screenshot_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        return screenshot_files[0]
    
    except Exception as e:
        logger.error(f"Error getting latest screenshot: {e}")
        return None

