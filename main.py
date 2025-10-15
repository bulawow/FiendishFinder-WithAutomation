#!/usr/bin/env python3
"""
FiendishFinder - Main Entry Point

A comprehensive system for viewing and stitching Tibia minimap images,
with dataset creation tools for crosshair detection and Exiva spell parsing.

Usage:
    python main.py              # Launch the minimap viewer application
    python main.py --help       # Show help message
"""

import sys
from pathlib import Path


def check_requirements():
    """Check if all required dependencies are installed."""
    missing_deps = []

    try:
        import PyQt6
    except ImportError:
        missing_deps.append("PyQt6")

    try:
        from PIL import Image
    except ImportError:
        missing_deps.append("Pillow")

    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")

    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\n💡 Install them with: pip install -r requirements.txt")
        return False

    return True


def check_minimap_images():
    """Check if processed minimap images are available."""
    minimap_dir = Path("processed_minimap")
    
    if not minimap_dir.exists():
        print(f"⚠️  Warning: Minimap directory '{minimap_dir}' not found.")
        print("💡 Run the minimap stitcher first to generate floor images:")
        print("   python scripts/stitch_minimap.py")
        return False

    floor_files = list(minimap_dir.glob("floor_*.png"))
    if not floor_files:
        print(f"⚠️  Warning: No floor images found in '{minimap_dir}'.")
        print("💡 Run the minimap stitcher first to generate floor images:")
        print("   python scripts/stitch_minimap.py")
        return False

    print(f"✓ Found {len(floor_files)} floor images in '{minimap_dir}'")
    return True


def main():
    """Main entry point for FiendishFinder application."""
    print("=" * 60)
    print("FiendishFinder - Tibia Minimap Viewer & Dataset Creator")
    print("=" * 60)
    print()

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    print("✓ All dependencies installed")
    print()

    # Check minimap images
    images_available = check_minimap_images()
    
    if not images_available:
        response = input("\n❓ Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("\n💡 Generate minimap images first with:")
            print("   python scripts/stitch_minimap.py")
            sys.exit(1)

    print()
    print("🚀 Launching FiendishFinder Minimap Viewer...")
    print()

    # Import and run the application
    try:
        from PyQt6.QtWidgets import QApplication
        from src.ui.minimap_viewer import MinimapViewerWindow

        app = QApplication(sys.argv)
        app.setApplicationName("FiendishFinder")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("FiendishFinder")

        window = MinimapViewerWindow()
        window.show()

        sys.exit(app.exec())

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Make sure you're running from the project root directory")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

