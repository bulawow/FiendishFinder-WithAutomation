#!/usr/bin/env python3
"""Minimap Viewer Launcher"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_requirements():
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
        print("Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nInstall them with: pip install -r requirements.txt")
        return False

    return True

def check_minimap_images():
    minimap_dir = Path("processed_minimap")
    if not minimap_dir.exists():
        print(f"Warning: Minimap directory '{minimap_dir}' not found.")
        print("Run the minimap stitcher first to generate floor images:")
        print("  python main.py")
        return False

    floor_files = list(minimap_dir.glob("floor_*.png"))
    if not floor_files:
        print(f"Warning: No floor images found in '{minimap_dir}'.")
        print("Run the minimap stitcher first to generate floor images:")
        print("  python main.py")
        return False

    print(f"Found {len(floor_files)} floor images in '{minimap_dir}'")
    return True

def run_standalone_viewer():
    print("Starting standalone minimap viewer...")
    from PyQt6.QtWidgets import QApplication
    from src.ui.minimap_viewer import MinimapViewerWindow

    app = QApplication(sys.argv)
    app.setApplicationName("FiendishFinder")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("FiendishFinder")

    window = MinimapViewerWindow()
    window.show()

    sys.exit(app.exec())

def run_demo():
    print("Demo mode not yet implemented.")
    print("Use: python main.py")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(
        description="FiendishFinder Minimap Viewer Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_minimap_viewer.py              # Run standalone viewer
  python run_minimap_viewer.py --demo       # Run demo application
  python run_minimap_viewer.py --check      # Check requirements only
        """
    )

    parser.add_argument("--demo", action="store_true",
                       help="Run the demo application with additional features")
    parser.add_argument("--check", action="store_true",
                       help="Check requirements and minimap availability only")

    args = parser.parse_args()

    print("FiendishFinder - Minimap Viewer Launcher")
    print("=" * 50)

    if not check_requirements():
        sys.exit(1)

    images_available = check_minimap_images()

    if args.check:
        print("\nRequirement check complete.")
        if images_available:
            print("✓ All requirements met. Ready to run minimap viewer.")
        else:
            print("⚠ Minimap images not available. Generate them first.")
        return

    if not images_available:
        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Exiting. Generate minimap images first with: python main.py")
            sys.exit(1)

    print()

    try:
        if args.demo:
            run_demo()
        else:
            run_standalone_viewer()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
