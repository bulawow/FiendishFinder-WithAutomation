#!/usr/bin/env python3
"""FiendishFinder - Tibia Minimap Stitcher"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.minimap_stitcher import MinimapStitchingSystem

def main():
    parser = argparse.ArgumentParser(
        description="FiendishFinder Minimap Stitcher - Process raw minimap tiles into complete floor images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stitch_minimap.py                    # Process all floors with PNG format
  python stitch_minimap.py --floor 7          # Process only floor 7
  python stitch_minimap.py --format JPEG      # Use JPEG format
  python stitch_minimap.py --input raw_tiles  # Use custom input directory
  python stitch_minimap.py --output maps      # Use custom output directory

The script will scan the input directory for minimap tiles following the naming
convention: Minimap_Color_X_Y_Z.png where X,Y are coordinates and Z is the floor.
        """
    )

    parser.add_argument("--floor", type=int, help="Process only the specified floor (0-15)")
    parser.add_argument("--format", choices=['PNG', 'JPEG'], default='PNG',
                       help="Output image format (default: PNG)")
    parser.add_argument("--input", default="raw_minimap",
                       help="Input directory containing raw minimap tiles (default: raw_minimap)")
    parser.add_argument("--output", default="processed_minimap",
                       help="Output directory for processed floor images (default: processed_minimap)")
    parser.add_argument("--markers", default="assets/minimap/minimapmarkers.bin",
                       help="Path to minimapmarkers.bin file (default: assets/minimap/minimapmarkers.bin)")
    parser.add_argument("--no-markers", action="store_true",
                       help="Don't overlay map markers on the output images")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        print("Please ensure the raw minimap tiles are available in the specified directory.")
        sys.exit(1)

    print("FiendishFinder - Tibia Minimap Stitcher")
    print("=" * 50)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {args.output}")
    print(f"Output format:    {args.format}")

    if args.floor is not None:
        print(f"Processing floor: {args.floor}")
    else:
        print("Processing:       All floors")

    print()

    system = MinimapStitchingSystem(str(input_dir), args.output, args.markers)
    floors = system.get_available_floors()
    print(f"Found {len(floors)} floors: {floors}")

    if not floors:
        print("No minimap tiles found in the input directory.")
        print("Please check that the directory contains files matching the pattern:")
        print("  Minimap_Color_X_Y_Z.png")
        sys.exit(1)

    include_markers = not args.no_markers
    if include_markers:
        print(f"Map markers will be overlaid from: {args.markers}")
    else:
        print("Map markers will NOT be overlaid")

    if args.floor is not None:
        if args.floor not in floors:
            print(f"Error: Floor {args.floor} not found in available floors: {floors}")
            sys.exit(1)

        print(f"\nProcessing floor {args.floor}...")
        result = system.process_single_floor(args.floor, args.format, include_markers)

        if result:
            print(f"✓ Successfully processed floor {args.floor}")
            print(f"  Output: {result}")
        else:
            print(f"✗ Failed to process floor {args.floor}")
            sys.exit(1)
    else:
        print("\nProcessing all floors...")
        summary = system.process_all_floors(args.format, include_markers)

        successful = sum(1 for floor_data in summary.values() if floor_data['success'])
        total = len(summary)

        print(f"\nResults: {successful}/{total} floors processed successfully")

        for floor in sorted(summary.keys()):
            floor_data = summary[floor]
            status = "✓" if floor_data['success'] else "✗"
            print(f"  {status} Floor {floor:2d}: {floor_data['tile_count']:2d} tiles -> "
                  f"{floor_data['dimensions']['width_pixels']}x{floor_data['dimensions']['height_pixels']} pixels")

        report_path = system.save_summary_report(summary)
        print(f"\nDetailed report saved to: {report_path}")

        if successful < total:
            print(f"\nWarning: {total - successful} floors failed to process.")
            print("Check the detailed report for more information.")

    print("\nStitching complete!")
    print("You can now run 'python main.py' to view the processed minimap images.")

if __name__ == "__main__":
    main()
