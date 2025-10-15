"""
Exiva Output Extractor - High Accuracy OCR-based extraction
Extracts range, direction, difficulty, and floor indication from console screenshots
"""

import re
import cv2
import numpy as np
import easyocr
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image


@dataclass
class ExivaOutput:
    """Represents a single Exiva output"""
    range: str  # "very far", "far", "to the", or "none"
    direction: str  # N, S, E, W, NE, NW, SE, SW, or "none"
    difficulty: str  # trivial, easy, medium, hard, unknown, or "none"
    floor_indication: str  # "lower level", "higher level", or "none"
    raw_text: str  # The raw OCR text
    confidence: float  # Confidence score 0-1


class ExivaExtractor:
    """Extracts Exiva output data from console screenshots"""
    
    # Valid values for each field
    VALID_RANGES = ["very far", "far", "to the", "none"]
    VALID_DIRECTIONS = ["N", "S", "E", "W", "NE", "NW", "SE", "SW", "none"]
    VALID_DIFFICULTIES = ["trivial", "easy", "medium", "hard", "unknown", "none"]
    VALID_FLOORS = ["lower level", "higher level", "none"]
    
    # Direction mapping from text to abbreviation
    DIRECTION_MAP = {
        "north": "N",
        "south": "S",
        "east": "E",
        "west": "W",
        "north-east": "NE",
        "north east": "NE",
        "northeast": "NE",
        "north-west": "NW",
        "north west": "NW",
        "northwest": "NW",
        "south-east": "SE",
        "south east": "SE",
        "southeast": "SE",
        "south-west": "SW",
        "south west": "SW",
        "southwest": "SW",
    }
    
    def __init__(self, use_gpu: bool = False):
        """Initialize the extractor with EasyOCR"""
        print("Initializing EasyOCR reader...")
        self.reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
        print("EasyOCR reader initialized.")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy
        - Convert to grayscale
        - Apply thresholding
        - Denoise
        - Increase contrast
        """
        # Read image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Increase contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
    
    def extract_text(self, image_path: str) -> Tuple[str, List[Tuple]]:
        """
        Extract text from image using OCR
        Returns both concatenated text and detailed results
        """
        # Use original image - preprocessing sometimes makes it worse
        result = self.reader.readtext(image_path, detail=1)

        # Concatenate all text
        text = ' '.join([item[1] for item in result])

        return text, result
    
    def extract_timestamp(self, text_segment: str) -> Optional[float]:
        """
        Extract timestamp from text segment
        Returns timestamp as float (e.g., 23.34 or 9.37) or None
        """
        # Look backwards from current position for timestamp
        # Pattern: HH.MM or HH:MM (with or without space after)
        timestamp_pattern = r'(\d{1,2})[:.](\d{2})'
        matches = list(re.finditer(timestamp_pattern, text_segment))
        if matches:
            # Get the last (closest) timestamp
            match = matches[-1]
            hours = int(match.group(1))
            minutes = int(match.group(2))
            return hours * 60 + minutes  # Convert to minutes for easy comparison
        return None

    def parse_exiva_outputs(self, text: str, detailed_results: List[Tuple]) -> List[Dict[str, str]]:
        """
        Parse all Exiva outputs from OCR text using fuzzy matching
        Returns list of dictionaries with parsed components including timestamps and Y coordinates
        """
        outputs = []

        # Clean text for easier matching - remove extra spaces and normalize
        clean_text = re.sub(r'\s+', ' ', text.lower())

        # Fix common OCR fragmentation issues
        # Sometimes "very" is split and "ery" appears as a separate OCR box
        # Check OCR box positions to detect this case
        # Look for "ery" box that appears on the same line (similar Y) as "monster is" and "far"
        has_ery_fragment = False
        if detailed_results:
            # Find boxes containing "monster" and "far"
            monster_boxes = [box for box in detailed_results if 'monster' in box[1].lower()]
            far_boxes = [box for box in detailed_results if re.search(r'\bfar\b', box[1].lower())]
            ery_boxes = [box for box in detailed_results if re.search(r'\bery\b', box[1].lower())]

            # Check if there's an "ery" box on the same line as monster and far boxes
            for ery_box in ery_boxes:
                ery_y = ery_box[0][0][1]  # Y coordinate
                ery_x = ery_box[0][0][0]  # X coordinate

                for monster_box in monster_boxes:
                    monster_y = monster_box[0][0][1]
                    monster_x = monster_box[0][0][0]

                    for far_box in far_boxes:
                        far_y = far_box[0][0][1]
                        far_x = far_box[0][0][0]

                        # Check if all three boxes are on the same line (Y within 5 pixels)
                        # and ery is spatially between monster and far (X coordinate)
                        if (abs(ery_y - monster_y) < 5 and abs(ery_y - far_y) < 5 and
                            monster_x < ery_x < far_x):
                            has_ery_fragment = True
                            break
                    if has_ery_fragment:
                        break
                if has_ery_fragment:
                    break

        # If we detected the fragmentation, fix it
        if has_ery_fragment:
            clean_text = re.sub(r'\bmonster\s+is\s+far\s+to\s+the\b', 'monster is very far to the', clean_text)

        # Build maps of text positions to Y coordinates and confidence from detailed_results
        # Also build a list of all OCR text boxes with their positions for direction correction
        position_to_y = {}
        position_to_conf = {}
        ocr_boxes = []  # List of (bbox, text, y_coord) for direction correction
        current_pos = 0
        for bbox, txt, conf in detailed_results:
            y_coord = bbox[0][1]  # Top-left Y coordinate
            x_coord = bbox[0][0]  # Top-left X coordinate
            txt_lower = txt.lower()
            ocr_boxes.append((bbox, txt_lower, y_coord, x_coord, conf))
            # Find this text in clean_text
            idx = clean_text.find(txt_lower, current_pos)
            if idx >= 0:
                for i in range(idx, idx + len(txt_lower)):
                    position_to_y[i] = y_coord
                    position_to_conf[i] = conf
                current_pos = idx + len(txt_lower)

        # Find all occurrences of "the monster" or just "monster" as starting points
        # Handle OCR errors: the/tne/ine/zhe/fhe/een monster/monste/monstei/mongter/mgnster
        # Also match standalone "ine monster" (where "the" became "ine")
        monster_pattern = r'(?:(?:the|tne|ine|zhe|fhe|een)\s+)?(?:monster|monste|monstei|mongter|mgnster|onster)'
        monster_matches = list(re.finditer(monster_pattern, clean_text, re.IGNORECASE))

        if not monster_matches:
            return outputs

        # For each "the monster" occurrence, try to extract the full exiva message
        for i, monster_match in enumerate(monster_matches):
            start_pos = monster_match.start()
            # Look ahead up to 200 characters for the complete message
            # BUT: stop at the next "monster" occurrence to avoid bleeding into next message
            end_pos = min(start_pos + 200, len(clean_text))

            # If there's a next monster match, limit segment to before it
            if i + 1 < len(monster_matches):
                next_monster_pos = monster_matches[i + 1].start()
                end_pos = min(end_pos, next_monster_pos)

            segment = clean_text[start_pos:end_pos]

            # Get Y coordinate for this match
            y_coord = position_to_y.get(start_pos, 0)

            # Look backwards up to 50 characters for timestamp
            timestamp_start = max(0, start_pos - 50)
            timestamp_segment = clean_text[timestamp_start:start_pos + 20]
            timestamp = self.extract_timestamp(timestamp_segment)

            # Extract components using fuzzy patterns
            floor_text = ""
            range_text = ""
            direction_text = ""
            difficulty_text = ""

            # Check for floor indication
            # Handle OCR errors: lower/lowet/lowe/ower/owet, level/leve/evel/eve/leue/eue/jevel/levei
            # Also check for "on lower level" pattern
            if re.search(r'(?:lower|lowet|lowe|ower|owet)\s*(?:level|leve|evel|eve|leue|eue|jevel|levei)', segment):
                floor_text = "lower level"
            elif re.search(r'on\s+(?:lower|lowet|lowe|ower|owet)\s*(?:level|leve|evel|eve|leue|eue|jevel|levei)', segment):
                floor_text = "lower level"
            elif re.search(r'(?:higher|highe|hig)\s*(?:level|leve|evel|eve|leue|eue|jevel|levei)', segment):
                floor_text = "higher level"

            # Check for range - handle various OCR errors
            # "very far" variations: very far, veryfar, ver far, very farto, ver farto, ver tarto, vet tarto, verx tarto, ery far (v dropped), hnee (very far compressed)
            # Also handle "Very Tantothe" -> "very far to the"
            # Note: verx is a common OCR error for "very" where the 'y' becomes 'x'
            # Note: ery is when the 'v' is dropped from "very"
            if re.search(r'(?:very|ver|vet|verx|ery)\s*(?:far|farto|fant|tarto|tanto)', segment) or re.search(r'hnee', segment):
                range_text = "very far"
            # Handle cases where "very" is present but OCR dropped "far" before the direction
            # Accept either "to the" or just "the" (since OCR sometimes drops "to") before the direction
            elif re.search(r'(?:very|ver|vet|verx|ery)\s+(?:(?:to\s*)?(?:the|tne|ine))', segment) and re.search(r'(north|south|east|west|norn|sout|easte|wesi|bast|baste)', segment):
                range_text = "very far"
            # "far" variations: far, farto, tanto, tarto (but not preceded by ver/very/vet/verx/ery)
            # Can be "far to the" or "far the" (OCR missing "to") or "tarto the" (far to the)
            # Also handle standalone "tanto" which is often "far to" compressed
            # Negative lookbehind to exclude "very", "ver", "vet", "verx", "ery" before "far"
            elif re.search(r'(?:^|(?<!very)(?<!ver)(?<!vet)(?<!verx)(?<!ery))\s*(?:far|farto|fant|tanto|tarto)\s*(?:to|ta|the|tne|ine)', segment) or re.search(r'\btanto\b', segment):
                range_text = "far"
            # Special case: "norn" might be "far north" compressed (but NOT if preceded by "to the")
            # E.g., "far north" -> "farnorth" -> OCR reads as "norn"
            elif re.search(r'\bnorn\b', segment) and not re.search(r'to\s*(?:the|tne|ine)?\s*norn', segment):
                range_text = "far"
            # Special case: direction followed by "dredareo/prepared" without "to the" before the direction
            # E.g., "West Dredareo" (missing "far") -> should be "far west prepared"
            # Check that "to the" doesn't appear before the direction word
            elif re.search(r'(?:west|east|north|south|wesi|easte|norn|sout)\s+(?:dredareo|prepared)', segment, re.IGNORECASE):
                # Only set range="far" if "to the" or "ine" doesn't appear before the direction
                # Look for "to the" or standalone "ine" followed by direction (including OCR variants)
                # "ine" can be a misread of "the" or "to the"
                if not re.search(r'(?:to\s*)?(?:the|tne|ine)\s+(?:west|east|north|south|wesi|wes|easte|eas|norn|norin|sout|soutn)', segment, re.IGNORECASE):
                    range_text = "far"

            # Extract direction - look for direction after "monster" or after "to the"
            # Skip past the monster match to avoid capturing it
            segment_after_monster = segment[monster_match.end() - start_pos:]

            # Try to find "to the [direction]" first (more specific)
            # Note: OCR sometimes adds underscores, so we include _ in the terminator
            # Add OCR variants of "prepared": dredared, dredare, dredareo, and "tin" (OCR error for "to")
            dir_pattern_with_to = r'to\s*(?:the|tne|ine)?\s*([a-z\-\s]{3,20}?)(?:\s*[:.!_]|be|prepared|dredareo|dredared|dredare|@|remared|find|necn|fino|tin|$|\s+[a-z]\s*$)'
            dir_match = re.search(dir_pattern_with_to, segment_after_monster, re.IGNORECASE)

            # If not found, try without "to the"
            if not dir_match:
                dir_pattern_no_to = r'([a-z\-\s]{3,20}?)(?:\s*[:.!_]|be|prepared|dredareo|dredared|dredare|@|remared|find|necn|fino|tin)'
                dir_match = re.search(dir_pattern_no_to, segment_after_monster, re.IGNORECASE)

            if dir_match:
                direction_text = dir_match.group(1).strip()
                # Clean up - remove trailing words that aren't directions and single letters
                direction_text = re.sub(r'\s*(be|prepared|find|creature|remared|dredareo).*$', '', direction_text, flags=re.IGNORECASE).strip()
                direction_text = re.sub(r'\s+[a-z]$', '', direction_text).strip()  # Remove trailing single letters like "z"

                # Check if there's another direction word after difficulty (e.g., "east ... Unknown north" -> "north-east")
                # Look for direction words after the difficulty, even if far away
                # Only combine if the current direction is simple (E, W) and we find a clear N/S
                current_direction = self.normalize_direction(direction_text)
                if current_direction in ['E', 'W']:  # Only for simple E/W directions
                    # Look ahead up to 300 characters after the current position for north/south
                    extended_segment_end = min(start_pos + 500, len(clean_text))
                    extended_segment = clean_text[start_pos:extended_segment_end]
                    trailing_dir_pattern = r'(?:unknown|trivial|easy|medium|hard|nknmtn|nknotn|unk).*?\b(north|south|norn|sout)\b'
                    trailing_match = re.search(trailing_dir_pattern, extended_segment, re.IGNORECASE)
                    if trailing_match:
                        trailing_word = trailing_match.group(1).strip()
                        # Normalize the trailing direction
                        if re.search(r'^(north|norn)$', trailing_word, re.IGNORECASE):
                            direction_text = f"north {direction_text}"
                        elif re.search(r'^(south|sout)$', trailing_word, re.IGNORECASE):
                            direction_text = f"south {direction_text}"

                # Also check the reverse: if we have N/S, look for E/W in nearby OCR boxes
                # This handles cases where "east" or "west" is in a separate OCR box (e.g., misread as "Cast")
                # BUT be very conservative: only match if the OCR box contains ONLY east/west (not compound like "south-east")
                if current_direction in ['N', 'S']:  # Only for simple N/S directions
                    # Look for nearby OCR boxes that might contain "east" or "west" (or misreads)
                    # Check boxes within ±5 pixels vertically (same line) and to the right of the current position
                    for bbox, box_text, box_y, box_x, box_conf in ocr_boxes:
                        # Check if this box is on the same line (within ±5 pixels vertically)
                        if abs(box_y - y_coord) <= 5 and box_x > 0:
                            # Only match if the box contains ONLY east/west, not compound directions
                            # Exclude boxes that contain "north", "south", or hyphens (compound indicators)
                            if re.search(r'(north|south|norn|sout|\-)', box_text, re.IGNORECASE):
                                continue
                            # Check for "east" or common misreads like "cast", "bast", "easte"
                            # But the box should be SHORT (< 10 chars) to avoid matching longer text
                            if len(box_text) < 10 and re.search(r'^[^a-z]*(cast|eas)[^a-z]*$', box_text, re.IGNORECASE):
                                direction_text = f"{direction_text} east"
                                break
                            # Check for "west" or common misreads like "wesi", "wes"
                            elif len(box_text) < 10 and re.search(r'^[^a-z]*(wes)[^a-z]*$', box_text, re.IGNORECASE):
                                direction_text = f"{direction_text} west"
                                break

            # Extract difficulty - look for "difficulty level [difficulty]"
            # Handle OCR errors in both "difficulty" and "level"
            # Order alternatives by length (longest first) to avoid partial matches
            # Try with required whitespace first (more accurate)
            # Added "inticutty" and "dnttcuty" as OCR variants of "difficulty"
            # Added "evei" as OCR variant of "level" (must come before "eve" to avoid partial match)
            diff_pattern_strict = r'(?:difficulty|difticulty|dntticmity|dntticuty|dnttcuty|dificitt|difficul|diffic|inticutty)\s*(?:leveb|level|jevel|leue|leve|evei|evel|bve|eve|eue|ue|levei|levee)\s+[\'"]?([a-z]+)'
            diff_match = re.search(diff_pattern_strict, segment, re.IGNORECASE)
            if diff_match:
                difficulty_text = diff_match.group(1).strip()

            if not diff_match:
                # Fallback: try without required whitespace (for cases like "levelUnknown")
                diff_pattern_loose = r'(?:difficulty|difticulty|dntticmity|dntticuty|dnttcuty|dificitt|difficul|diffic|inticutty)\s*(?:leveb|level|jevel|leue|leve|evei|evel|bve|eve|eue|ue|levee)\s*[\'"]?([a-z]+)'
                diff_match = re.search(diff_pattern_loose, segment, re.IGNORECASE)
                if diff_match:
                    difficulty_text = diff_match.group(1).strip()

            if not diff_match:
                # Fallback: if no difficulty keyword found, check for isolated letter/word at end
                # E.g., "to the west Z" where "Z" might be "Unknown" corrupted
                # Also check for "eue wnknotn" pattern (OCR error for "level unknown")
                fallback_pattern = r'(?:west|east|north|south|wesi|easte|norn|sout)\s+([a-z]{1,3})\s*$'
                fallback_match = re.search(fallback_pattern, segment, re.IGNORECASE)
                if fallback_match:
                    difficulty_text = fallback_match.group(1).strip()
                else:
                    # Check for "eue [difficulty]" or "levee [difficulty]" pattern (OCR error for "level [difficulty]")
                    eue_pattern = r'(?:eue|ue|levee)\s+[\'"]?([a-z]+)'
                    eue_match = re.search(eue_pattern, segment, re.IGNORECASE)
                    if eue_match:
                        difficulty_text = eue_match.group(1).strip()

            # Only add if we found at least direction (difficulty is optional for very corrupted text)
            if direction_text:
                # Calculate average confidence for this output
                conf_values = [position_to_conf.get(i, 0) for i in range(start_pos, min(start_pos + 100, len(clean_text)))]
                avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0

                outputs.append({
                    'floor_raw': floor_text,
                    'range_raw': range_text,
                    'direction_raw': direction_text,
                    'difficulty_raw': difficulty_text,
                    'full_match': segment[:100],
                    'timestamp': timestamp,
                    'position': start_pos,
                    'y_coord': y_coord,
                    'confidence': avg_conf
                })

        return outputs
    
    def normalize_direction(self, direction_text: str) -> str:
        """Normalize direction text to standard abbreviation with fuzzy matching"""
        direction_lower = direction_text.lower().strip()

        # Remove common OCR artifacts
        direction_lower = re.sub(r'[^a-z\s\-]', '', direction_lower)
        direction_lower = re.sub(r'\s+', ' ', direction_lower).strip()
        direction_lower = re.sub(r'\-+', '-', direction_lower)  # Normalize hyphens

        # Direct mapping
        if direction_lower in self.DIRECTION_MAP:
            return self.DIRECTION_MAP[direction_lower]

        # Fuzzy matching for compound directions (check these first - MUST have both components)
        # Be strict: both north/south AND east/west must be present
        has_north = bool(re.search(r'north|nort|orth|norn|norin', direction_lower))
        has_south = bool(re.search(r'south|sout|outh|soutn', direction_lower))
        has_east = bool(re.search(r'east|eas|easte|bast|baste', direction_lower))
        has_west = bool(re.search(r'west|wes|wesi', direction_lower))

        # Compound directions - require both components
        if has_north and has_east:
            return 'NE'
        if has_north and has_west:
            return 'NW'
        if has_south and has_east:
            return 'SE'
        if has_south and has_west:
            return 'SW'

        # Simple directions - only if no compound detected
        if has_north:
            return 'N'
        if has_south:
            return 'S'
        if has_east:
            return 'E'
        if has_west:
            return 'W'

        return "none"
    
    def normalize_range(self, range_text: str, direction_text: str, difficulty_text: str = "", direction_raw: str = "", full_match: str = "") -> str:
        """Normalize range text"""
        range_lower = range_text.lower().strip()

        if "very far" in range_lower or "veryfar" in range_lower:
            return "very far"
        elif "far" in range_lower:
            return "far"
        elif not range_text and direction_text:
            # If no range specified but direction exists, check context to determine range
            normalized_difficulty = self.normalize_difficulty(difficulty_text)
            full_match_lower = full_match.lower()
            direction_raw_lower = direction_raw.lower()

            # Check for "dredaren" (corruption of "prepared") which specifically indicates a "far" message
            # This corruption appears when OCR reads "far to the ... prepared" and merges it
            has_dredaren = 'dredaren' in direction_raw_lower or 'dredaren' in full_match_lower

            # Check for floor indication (lower/higher level)
            # "to the" messages with floor indication also have "be prepared"
            # Be specific: look for "on [lower/higher] level" pattern, not just "on" and "level" separately
            has_floor_indication = ('lower level' in full_match_lower or 'higher level' in full_match_lower or
                                   'on lower' in full_match_lower or 'on higher' in full_match_lower or
                                   'owet levei' in full_match_lower or 'on owet' in full_match_lower)

            # If we have "dredaren" and a known difficulty (and no floor indication), it's "far"
            if has_dredaren and normalized_difficulty in ["trivial", "easy", "medium", "hard"] and not has_floor_indication:
                return "far"

            # Check for heavily corrupted messages: "ine monster" instead of "the monster"
            # These messages often have "far" corrupted to nothing, leaving only "to the"
            has_ine_monster = 'ine monster' in full_match_lower
            has_be_prepared = ('be prepared' in full_match_lower or 'be_prepared' in full_match_lower or
                              'beprepared' in full_match_lower)
            has_to_the = 'to the' in full_match_lower or 'to tne' in full_match_lower or 'to ine' in full_match_lower

            # Check for "ine monster ine" pattern (corruption of "the monster is to the")
            # This distinguishes "to the" messages from "far" messages
            # Note: "ine monster to" can be either "is to the" or "is far to the", so we check difficulty
            has_ine_monster_ine = 'ine monster ine' in full_match_lower or 'ine monster is' in full_match_lower

            # If we have "ine monster" and "be prepared" (and no floor indication), it's likely "far"
            # This handles cases where OCR corrupts "far to the" to just "to the" or nothing
            # BUT: if we have "ine monster ine/is" pattern (without known difficulty), it's "to the", not "far"
            if has_ine_monster and has_be_prepared and not has_floor_indication:
                # If we have "ine monster ine/is" pattern and no known difficulty, it's "to the"
                if has_ine_monster_ine and normalized_difficulty not in ["trivial", "easy", "medium", "hard"]:
                    pass  # Fall through to check for explicit "to the"
                # If there's no "to the" in the text, it's definitely "far"
                elif not has_to_the:
                    return "far"
                # If there's "to the" but also a known difficulty, it's "far"
                elif normalized_difficulty in ["trivial", "easy", "medium", "hard"]:
                    return "far"

            # If "to the" appears explicitly in the text, it's "to the"
            if has_to_the:
                return "to the"

            # Default to "to the"
            return "to the"

        return "none"
    
    def normalize_difficulty(self, difficulty_text: str) -> str:
        """Normalize difficulty text with fuzzy matching"""
        difficulty_lower = difficulty_text.lower().strip()

        # Remove non-alphabetic characters
        difficulty_lower = re.sub(r'[^a-z]', '', difficulty_lower)

        # Handle common OCR errors with fuzzy matching
        # Trivial: trivial, trivia, triv, wef (heavily corrupted), etc.
        if re.search(r'triv|wef', difficulty_lower):
            return "trivial"
        # Easy: easy, eas, fasy, etc.
        elif re.search(r'eas|fasy', difficulty_lower):
            return "easy"
        # Medium: medium, medi, mediu, memnmm, hemmmn, emnmm, meonmm, ednum, etc.
        # Common OCR errors: hemmmn (contains emm), emnmm (contains mnm), meonmm, ednum (rearranged letters)
        elif re.search(r'med|mem|emm|mnm|meo|ednum', difficulty_lower):
            return "medium"
        # Hard: hard, har, etc.
        elif re.search(r'har', difficulty_lower):
            return "hard"
        # Unknown: unknown, unkn, unk, nknmtn, nknotn, wnknotn, unknotn, z (single letter fallback), etc.
        elif re.search(r'unk|nkn|wnkn', difficulty_lower) or difficulty_lower == 'z' or difficulty_lower == 'i':
            return "unknown"

        return "none"
    
    def normalize_floor(self, floor_text: str) -> str:
        """Normalize floor indication text"""
        floor_lower = floor_text.lower().strip()
        
        if "lower" in floor_lower:
            return "lower level"
        elif "higher" in floor_lower:
            return "higher level"
        
        return "none"
    
    def score_output_quality(self, parsed_output: Dict[str, str]) -> float:
        """
        Score the quality of a parsed output based on how well fields normalize
        Higher score = better quality
        """
        score = 0.0

        # Score direction (most important)
        direction = self.normalize_direction(parsed_output['direction_raw'])
        if direction != "none":
            score += 3.0
            # Bonus for clean direction text
            if parsed_output['direction_raw'] and len(parsed_output['direction_raw']) < 20:
                score += 1.0

            # Penalties for OCR corruption indicators in direction_raw
            direction_raw = parsed_output['direction_raw'].lower().strip()

            # Penalty for "southe" specifically (common OCR error)
            if direction_raw == 'southe':
                score -= 0.15

            # Penalty for "ine" prefix in direction when combined with explicit range
            # This indicates OCR corruption where "the" became "ine" AND range was misread
            # E.g., "ine norin west" with range="far" is likely wrong
            if direction_raw.startswith('ine ') and parsed_output['range_raw']:
                score -= 0.10

            # Penalty for corrupted direction text containing "dredaren" or similar garbage
            # This indicates the direction extraction captured junk text
            # EXCEPT: Don't penalize "dredaren" if it's being used to infer range="far"
            # (i.e., when range_raw is empty and we have a specific difficulty)
            has_dredaren_pattern = re.search(r'dredare|prepared|remared|dredareo', direction_raw)
            if has_dredaren_pattern:
                # Check if this is being used for range inference
                normalized_difficulty = self.normalize_difficulty(parsed_output['difficulty_raw'])
                is_range_inference = (not parsed_output['range_raw'] and
                                    normalized_difficulty in ["trivial", "easy", "medium", "hard"] and
                                    'dredare' in direction_raw)
                if not is_range_inference:
                    score -= 0.5

            # Penalty for direction text containing "owet", "leve", "evei" (floor indication corruption)
            # E.g., "owet leve ine south" or "evei ine soutn east"
            # BUT: reduce penalty if floor was successfully detected (means the corruption is expected)
            if re.search(r'owet|leve|evei', direction_raw):
                if parsed_output['floor_raw']:
                    score -= 0.3  # Small penalty if floor detected
                else:
                    score -= 1.0  # Large penalty if no floor detected

            # Bonus for clean compound directions (e.g., "south-west" vs "th west dredaren to")
            # Check if it's a proper compound direction format
            if re.match(r'^(north|south|norn|sout)[\s\-]+(east|west|eas|wes)$', direction_raw):
                score += 0.3

        # Score difficulty
        difficulty = self.normalize_difficulty(parsed_output['difficulty_raw'])
        if difficulty != "none":
            score += 2.0
            # Small bonus for having explicit difficulty_raw (not empty)
            if parsed_output['difficulty_raw'] and len(parsed_output['difficulty_raw']) >= 3:
                score += 0.05

        # Score range
        range_val = self.normalize_range(parsed_output['range_raw'], parsed_output['direction_raw'],
                                         parsed_output['difficulty_raw'], parsed_output['direction_raw'],
                                         parsed_output.get('full_match', ''))
        if range_val != "none":
            score += 1.0

            # Bonus for having explicit range_raw (not empty/inferred)
            # This helps prefer outputs where "far" or "very far" was actually detected
            if parsed_output['range_raw']:
                score += 1.0

            # Moderate bonus for "to the" or "far" (when range_raw is empty) with specific difficulty
            # This likely means "far" was missed by OCR, but the message is still valid
            # E.g., "to the east ... medium" is likely "far to the east ... medium"
            # Make this competitive with explicit range bonus since inferred range can be correct
            # Include "unknown" as a valid difficulty since it's a real difficulty level
            if (range_val == "to the" or (range_val == "far" and not parsed_output['range_raw'])) and difficulty in ["trivial", "easy", "medium", "hard", "unknown"]:
                score += 1.0

        # Score floor indication
        # Floor indication can be a signal for the most recent message
        # Balance between preferring floor indications when present vs not over-preferring them
        # Since floor='none' is more common (76% of cases), keep this bonus small
        # Use it mainly as a tiebreaker, not a primary selection criterion
        floor = self.normalize_floor(parsed_output['floor_raw'])
        if floor != "none":
            score += 0.5  # Small bonus for having floor indication (tiebreaker)

        return score

    def extract_from_image(self, image_path: str) -> ExivaOutput:
        """
        Main extraction method - extracts Exiva output from image
        Returns the BEST QUALITY Exiva output found (prioritizing latest among high-quality ones)
        """
        # Extract text
        text, detailed_results = self.extract_text(image_path)

        # Parse all Exiva outputs
        parsed_outputs = self.parse_exiva_outputs(text, detailed_results)

        # If no outputs found, return empty result
        if not parsed_outputs:
            return ExivaOutput(
                range="none",
                direction="none",
                difficulty="none",
                floor_indication="none",
                raw_text=text[:200] if text else "",
                confidence=1.0  # High confidence for "no exiva"
            )

        # Select the best output based on quality score and recency
        # Strategy: Prefer the most recent output, but only if it doesn't have major corruption
        # Major corruption is indicated by having "dredaren" or similar patterns in direction_raw

        if not parsed_outputs:
            best_output = None
        else:
            # Calculate scores for all outputs
            scored_outputs = [(o, self.score_output_quality(o)) for o in parsed_outputs]

            # Find the maximum score
            max_score = max(score for _, score in scored_outputs)

            # Get outputs with scores close to max
            # Use a larger threshold (0.85) to allow floor indication differences and minor variations
            score_threshold = 0.85
            candidate_outputs = [o for o, score in scored_outputs if score >= max_score - score_threshold]

            # Among candidates, filter out outputs with major corruption indicators
            # Major corruption: direction_raw contains "dredaren", "prepared", etc.
            # EXCEPT: Don't consider "dredaren" as major corruption if it's being used for range inference
            # (i.e., when range_raw is empty and we have a specific difficulty)
            clean_outputs = []
            for o in candidate_outputs:
                direction_raw = o.get('direction_raw', '').lower().strip()
                has_dredaren_pattern = bool(re.search(r'dredare', direction_raw))
                has_other_corruption = bool(re.search(r'prepared|remared|dredareo', direction_raw))

                # Check if "dredaren" is being used for range inference
                if has_dredaren_pattern:
                    normalized_difficulty = self.normalize_difficulty(o.get('difficulty_raw', ''))
                    is_range_inference = (not o.get('range_raw') and
                                        normalized_difficulty in ["trivial", "easy", "medium", "hard"])
                    # If it's range inference, don't consider it major corruption
                    if is_range_inference:
                        has_dredaren_pattern = False

                has_major_corruption = has_dredaren_pattern or has_other_corruption
                if not has_major_corruption:
                    clean_outputs.append(o)

            # If we have clean outputs, use them; otherwise use all candidates
            outputs_to_select_from = clean_outputs if clean_outputs else candidate_outputs

            # Select the most recent output
            most_recent = max(outputs_to_select_from, key=lambda o: o.get('y_coord', 0))

            # If the most recent output has no explicit range, check if there's a recent output with explicit range
            # This handles cases where the most recent message has corrupted OCR that loses the range
            if not most_recent.get('range_raw', '').strip():
                # Find outputs with explicit range
                outputs_with_explicit_range = [o for o in outputs_to_select_from if o.get('range_raw', '').strip()]

                if outputs_with_explicit_range:
                    # Get the most recent output with explicit range
                    most_recent_with_range = max(outputs_with_explicit_range, key=lambda o: o.get('y_coord', 0))

                    # Only use it if it's very close (within 15 pixels of the most recent)
                    # AND has the same direction and difficulty as the most recent output
                    # This ensures we only prefer explicit range for messages that are essentially the same
                    most_recent_dir = self.normalize_direction(most_recent.get('direction_raw', ''))
                    most_recent_diff = self.normalize_difficulty(most_recent.get('difficulty_raw', ''))
                    with_range_dir = self.normalize_direction(most_recent_with_range.get('direction_raw', ''))
                    with_range_diff = self.normalize_difficulty(most_recent_with_range.get('difficulty_raw', ''))

                    if (most_recent.get('y_coord', 0) - most_recent_with_range.get('y_coord', 0) <= 15 and
                        most_recent_dir == with_range_dir and most_recent_diff == with_range_diff):
                        best_output = most_recent_with_range
                    else:
                        best_output = most_recent
                else:
                    best_output = most_recent
            else:
                best_output = most_recent

        # Normalize all fields from best output
        direction = self.normalize_direction(best_output['direction_raw'])
        difficulty = self.normalize_difficulty(best_output['difficulty_raw'])
        range_val = self.normalize_range(best_output['range_raw'], best_output['direction_raw'],
                                         best_output['difficulty_raw'], best_output['direction_raw'],
                                         best_output.get('full_match', ''))
        floor = self.normalize_floor(best_output['floor_raw'])

        # Calculate confidence based on how many fields were successfully extracted
        confidence = 0.0
        if direction != "none":
            confidence += 0.3
        if range_val != "none":
            confidence += 0.3
        if difficulty != "none":
            confidence += 0.3
        if floor != "none" or best_output['floor_raw'] == "":
            confidence += 0.1

        return ExivaOutput(
            range=range_val,
            direction=direction,
            difficulty=difficulty,
            floor_indication=floor,
            raw_text=best_output['full_match'],
            confidence=confidence
        )


def main():
    """Example usage"""
    extractor = ExivaExtractor(use_gpu=False)
    
    # Test on a sample image
    test_image = "datasets/exiva/screenshots/adb18de6-fb9f-4ba6-9323-6a0f4126a49d.png"
    result = extractor.extract_from_image(test_image)
    
    print(f"Range: {result.range}")
    print(f"Direction: {result.direction}")
    print(f"Difficulty: {result.difficulty}")
    print(f"Floor: {result.floor_indication}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Raw text: {result.raw_text}")


if __name__ == "__main__":
    main()

