"""
Utility script to migrate existing JSON dataset to JSONL format.

This script converts the legacy minimap_dataset.json file to the new
minimap_dataset.jsonl format for better performance with large datasets.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_json_to_jsonl(
    json_path: str = "datasets/minimap/minimap_dataset.json",
    jsonl_path: Optional[str] = None,
    backup: bool = True
) -> bool:
    """
    Migrate a JSON dataset file to JSONL format.
    
    Args:
        json_path: Path to the existing JSON file
        jsonl_path: Path for the new JSONL file (defaults to same name with .jsonl extension)
        backup: Whether to create a backup of the original JSON file
        
    Returns:
        True if migration was successful, False otherwise
    """
    json_file = Path(json_path)
    
    if not json_file.exists():
        logger.error(f"JSON file not found: {json_path}")
        return False
    
    # Determine output path
    if jsonl_path is None:
        jsonl_file = json_file.with_suffix('.jsonl')
    else:
        jsonl_file = Path(jsonl_path)
    
    # Create backup if requested
    if backup:
        backup_file = json_file.with_suffix('.json.backup')
        try:
            import shutil
            shutil.copy2(json_file, backup_file)
            logger.info(f"Created backup: {backup_file}")
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    try:
        # Load JSON data
        logger.info(f"Loading JSON data from {json_path}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            logger.error("JSON file does not contain a list of entries")
            return False
        
        logger.info(f"Loaded {len(data)} entries")
        
        # Write to JSONL format
        logger.info(f"Writing JSONL data to {jsonl_file}...")
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for i, entry in enumerate(data):
                json.dump(entry, f, separators=(',', ':'))
                f.write('\n')
                
                # Progress indicator for large datasets
                if (i + 1) % 10000 == 0:
                    logger.info(f"  Processed {i + 1}/{len(data)} entries...")
        
        logger.info(f"Successfully migrated {len(data)} entries to {jsonl_file}")
        
        # Verify the JSONL file
        logger.info("Verifying JSONL file...")
        count = 0
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
        
        if count == len(data):
            logger.info(f"Verification successful: {count} entries in JSONL file")
            return True
        else:
            logger.error(f"Verification failed: expected {len(data)} entries, found {count}")
            return False
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in megabytes."""
    return file_path.stat().st_size / (1024 * 1024)


def compare_file_sizes(json_path: str, jsonl_path: str):
    """Compare file sizes between JSON and JSONL formats."""
    json_file = Path(json_path)
    jsonl_file = Path(jsonl_path)
    
    if json_file.exists() and jsonl_file.exists():
        json_size = get_file_size_mb(json_file)
        jsonl_size = get_file_size_mb(jsonl_file)
        
        logger.info(f"\nFile size comparison:")
        logger.info(f"  JSON:  {json_size:.2f} MB")
        logger.info(f"  JSONL: {jsonl_size:.2f} MB")
        logger.info(f"  Difference: {json_size - jsonl_size:.2f} MB ({((json_size - jsonl_size) / json_size * 100):.1f}% reduction)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate JSON dataset to JSONL format")
    parser.add_argument(
        "--input",
        default="datasets/minimap/minimap_dataset.json",
        help="Path to input JSON file"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output JSONL file (defaults to input with .jsonl extension)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create a backup of the original JSON file"
    )
    
    args = parser.parse_args()
    
    success = migrate_json_to_jsonl(
        json_path=args.input,
        jsonl_path=args.output,
        backup=not args.no_backup
    )
    
    if success:
        output_path = args.output if args.output else Path(args.input).with_suffix('.jsonl')
        compare_file_sizes(args.input, str(output_path))
        logger.info("\n✓ Migration completed successfully!")
    else:
        logger.error("\n✗ Migration failed!")
        exit(1)

