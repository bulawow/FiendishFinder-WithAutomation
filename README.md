<div align="center">

# 🎯 FiendishFinder

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/PyQt6-GUI-41CD52?style=for-the-badge&logo=qt&logoColor=white)](https://www.riverbankcomputing.com/software/pyqt/)
[![DINOv3](https://img.shields.io/badge/DINOv3-Vision_Transformer-FF6F00?style=for-the-badge&logo=pytorch&logoColor=white)](https://github.com/facebookresearch/dinov3)


</div>

---

## 🚀 Key Features

### 🗺️ **Intelligent Minimap Stitching**
- Automatically combines fragmented 256×256px minimap tiles into seamless floor maps
- Spatial coordinate transformation system for accurate tile placement
- Marker overlay system with icon rendering
- Supports 16 floors with automatic bounds detection

### 🎯 **DINOv3-Powered Position Prediction**
- **5.11px median error** using Vision Transformer feature matching
- 768-dimensional feature extraction from minimap crops
- Cosine similarity matching against 191,000+ position database
- Weighted interpolation with temperature scaling (T=0.01)
- Real-time prediction (~100-200ms per screenshot)

### 📊 **Automated Dataset Generation**
- Walkable area detection using RGB color analysis
- Automatic position sampling with deduplication
- JSONL-based storage with metadata tracking
- Supports multiple zoom levels (4 levels, 106×106px crops)
- Batch generation with progress tracking

### 🖥️ **Professional PyQt6 Interface**
- Interactive minimap viewer with smooth zoom/pan
- Real-time overlay rendering (Exiva spell visualization)
- Area editing tools with polygon selection
- Monster tracking panel with automatic session management
- Dataset creation and browsing tools
- Hotkey-based screenshot monitoring

---

## 🎬 Demo

### Minimap Stitching
*Combines 1000+ fragmented tiles into complete floor maps*

### Position Prediction
*DINOv3 feature matching achieves 5.11px median error*

### Exiva Spell Overlay
*Real-time geometric visualization of spell mechanics*

### Interactive Area Editing
*Professional GUI with polygon editing and property management*

---

## 💻 Installation

### Prerequisites
- **Python 3.8+** (3.10+ recommended)
- **2GB RAM** minimum (4GB+ recommended for DINOv3 models)
- **Windows/Linux/macOS** (tested on Windows 11)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/FiendishFinder.git
cd FiendishFinder

# Install dependencies
pip install -r requirements.txt

# Generate minimap floor images (required on first run)
python scripts/stitch_minimap.py

# Launch the application
python main.py
```

### Dependencies
```
Pillow>=8.0.0          # Image processing
numpy>=1.19.0          # Numerical computations
PyQt6>=6.0.0           # GUI framework
psutil>=5.8.0          # System monitoring
pynput>=1.7.0          # Keyboard/mouse input
```

---

## 🏃 Quick Start

### 1. **Stitch Minimap Tiles**
```bash
python scripts/stitch_minimap.py
```
Combines raw minimap tiles from `raw_minimap/` into complete floor images in `processed_minimap/`.

### 2. **Launch the Viewer**
```bash
python main.py
```
Opens the interactive minimap viewer with all features enabled.

### 3. **Place a Crosshair**
- Click anywhere on the map to place a crosshair
- Use mouse wheel to zoom in/out
- Middle-click and drag to pan

### 4. **Start Monster Tracking**
- Navigate to the **Monster Tracking** tab
- Click **Start Hunt Session**
- Select direction, distance, and difficulty
- Click **Add Reading** to visualize Exiva spell overlays

### 5. **Generate Training Data**
- Navigate to the **Dataset Creator** tab
- Place crosshair at desired position
- Click **Generate Entry** to create training samples
- Or use **Auto-Generate** for batch creation

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     FiendishFinder                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Minimap    │  │  Crosshair   │  │    Exiva     │     │
│  │  Stitching   │  │  Prediction  │  │   Analysis   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                  │                  │            │
│         ▼                  ▼                  ▼            │
│  ┌──────────────────────────────────────────────────┐     │
│  │          PyQt6 Interactive Viewer                │     │
│  │  • Zoom/Pan  • Overlays  • Area Editing          │     │
│  └──────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### **1. Minimap Stitching Pipeline** (`src/core/`)
- **TileParser**: Extracts coordinates from filenames (`Minimap_Color_X_Y_Z.png`)
- **SpatialAnalyzer**: Converts world coordinates to grid/pixel positions
- **ImageStitcher**: Combines tiles with marker overlay support
- **MinimapStitchingSystem**: Orchestrates the entire stitching process

#### **2. Crosshair Prediction System** (`src/crosshair_prediction/`)
- **FeatureExtractor**: DINOv3 ViT-B/16 for 768-dim feature extraction
- **FeatureDatabase**: Efficient similarity search with floor-based indexing
- **CrosshairMatcher**: Top-k retrieval with weighted interpolation
- **CrosshairPredictor**: Production interface for real-time prediction

#### **3. Exiva Analysis Engine** (`src/exiva_extractor/`)
- **ExivaExtractor**: OCR-based text extraction with EasyOCR
- **Fuzzy Matching**: Handles OCR errors with Levenshtein distance
- **Mechanics Implementation**: Authentic Tibia spell calculations
- **Confidence Scoring**: Multi-factor validation for extraction quality

#### **4. Interactive UI** (`src/ui/`)
- **MinimapViewer**: Main viewer with zoom/pan/crosshair controls
- **MonsterTrackingPanel**: Exiva session management and overlay controls
- **AreaPropertiesPanel**: Polygon editing with property management
- **DatasetCreatorPanel**: Automated dataset generation tools

#### **5. Utilities** (`src/utils/`)
- **WalkableDetector**: RGB-based terrain classification
- **MinimapDatasetGenerator**: Automated training data creation
- **ScreenshotMonitor**: Hotkey-based screenshot processing
- **ExivaMechanics**: Geometric calculations for spell mechanics

---

## Technical Deep Dive

### DINOv3 Position Prediction

**Challenge**: Predict player position from a 106×106px minimap crop with sub-pixel accuracy.

**Solution**: Feature matching using self-supervised vision transformers.

#### Algorithm
1. **Feature Extraction**
   - Load DINOv3 ViT-B/16 model
   - Extract 768-dimensional feature vector from minimap crop
   - L2 normalization for cosine similarity

2. **Database Query**
   - Search pre-built database of 191,000+ positions (floor 7)
   - Compute cosine similarity: `sim = query · database_features`
   - Retrieve top-k matches (k=10)

3. **Weighted Interpolation**
   - Apply temperature scaling: `weights = softmax(similarities / T)` where T=0.01
   - Interpolate positions: `x_pred = Σ(weights[i] × x[i])`
   - Return weighted average of top matches

#### Performance
- **Median Error**: 5.11 pixels
- **Mean Error**: 7.23 pixels
- **Inference Time**: 100-200ms per prediction
- **Database Size**: ~191,000 positions (floor 7)

**Why DINOv3?**
- Self-supervised learning captures visual patterns without labels
- Robust to variations in minimap appearance
- Better than previous deep learning approach (13px error)

---

#### Floor Detection
- **Same floor**: Standard distance ranges
- **Different floor (0-4 squares)**: "above/below you"
- **Different floor (5-100 squares)**: "on higher/lower level to [direction]"

#### Overlay Visualization
- Renders geometric regions for each reading
- Intersection of multiple readings narrows down position
- Color-coded by monster difficulty
- Transparency based on reading age (dimming over time)

---

### Dataset Generation Pipeline

**Challenge**: Create high-quality training data for position prediction models.

**Solution**: Automated sampling with walkable area detection and deduplication.

#### Process
1. **Walkable Detection**
   - Analyze floor map RGB values
   - Classify tiles as walkable/non-walkable
   - Floor-specific rules (e.g., RGB(51,102,153) walkable on floors 8-15 only)

2. **Position Sampling**
   - Random sampling from walkable areas
   - Deduplication by (x, y, floor) tuple
   - Configurable sample count per floor

3. **Crop Generation**
   - Extract 106×106px crop centered at position
   - No scaling/resizing (perfect quality)

4. **Metadata Storage**
   - JSONL format for efficient streaming
   - Schema: `{id, floor, player_x, player_y, crosshair_x, crosshair_y, zoom_level, image_path}`
   - Automatic validation and integrity checks

---

## 🔧 Challenges & Solutions

### **Challenge 1: Deep Learning Model Accuracy**
**Problem**: Initial deep learning approach for position prediction had 13px median error - too inaccurate for practical use.

**Solution**: Switched from pure deep learning to DINOv3 feature matching with weighted interpolation. Instead of training a model to predict coordinates directly, I extract features from minimap crops and match them against a pre-built database of 191,000+ positions. This reduced error to 5.11px (62% improvement).

**Key Learning**: Sometimes simpler approaches (feature matching) outperform complex ones (end-to-end deep learning), especially with limited training data.

---

### **Challenge 2: PyQt6 Performance with Large Images**
**Problem**: Rendering full floor maps (10,000+ x 10,000+ pixels) caused severe lag and memory issues when zooming/panning.

**Solution**: Implemented level-of-detail (LOD) rendering system. Only render visible portions of the map at current zoom level. Added caching for frequently accessed regions and lazy loading for floor transitions.

**Key Learning**: Performance optimization often requires trading memory for speed. Profiling revealed the bottleneck wasn't the algorithm but unnecessary rendering.

---

### **Challenge 3: OCR Accuracy with Game Screenshots**
**Problem**: EasyOCR struggled with Tibia's font, especially with similar-looking characters (0/O, 1/I, 5/S), leading to failed spell extractions.

**Solution**: Added fuzzy text matching with Levenshtein distance to handle OCR errors. Pre-defined valid monster names, directions, and distance ranges, then match OCR output against these with confidence scoring. Reject extractions below 80% confidence.

**Key Learning**: Real-world OCR is messy. Building robust systems requires handling errors gracefully and validating outputs against known constraints.

---

### **Challenge 4: Dataset Deduplication**
**Problem**: Random position sampling generated duplicate entries, wasting storage and training time. Simple coordinate comparison missed near-duplicates.

**Solution**: Implemented deduplication based on (x, y, floor) tuple hashing. Track generated positions in a set before creating crops. Added validation to ensure minimum distance between samples (configurable threshold).

**Key Learning**: Data quality matters more than quantity. 10,000 unique samples beat 50,000 samples with duplicates.

---

## 📦 Project Structure

```
FiendishFinder/
├── src/
│   ├── core/                    # Core stitching and spatial analysis
│   │   ├── minimap_stitcher.py  # Tile parsing and stitching logic
│   │   └── __init__.py
│   ├── ui/                      # PyQt6 GUI components
│   │   ├── minimap_viewer.py    # Main viewer window
│   │   ├── dataset_creator_ui.py
│   │   ├── dataset_browser_ui.py
│   │   └── __init__.py
│   ├── models/                  # Data models and settings
│   │   ├── dataset_models.py    # JSONL dataset management
│   │   ├── settings_manager.py  # Configuration persistence
│   │   └── __init__.py
│   ├── crosshair_prediction/    # DINOv3-based position prediction
│   │   ├── feature_extractor.py # DINOv3 feature extraction
│   │   ├── feature_database.py  # Similarity search database
│   │   ├── matcher.py           # Position matching logic
│   │   ├── inference.py         # Production interface
│   │   └── README.md
│   ├── exiva_extractor/         # OCR-based spell analysis
│   │   ├── exiva_extractor.py   # OCR and fuzzy matching
│   │   └── __init__.py
│   └── utils/                   # Utilities
│       ├── exiva_mechanics.py   # Spell mechanics calculations
│       ├── walkable_detector.py # Terrain classification
│       ├── minimap_dataset_generator.py
│       ├── screenshot_monitor.py
│       └── __init__.py
├── scripts/                     # Standalone utility scripts
│   ├── stitch_minimap.py        # Minimap stitching script
│   └── run_minimap_viewer.py
├── datasets/                    # Training data and datasets
│   ├── minimap/                 # Minimap dataset (JSONL + images)
│   ├── exiva/                   # Exiva dataset (JSONL + screenshots)
│   └── global_regions.json      # Global region definitions
├── processed_minimap/           # Generated floor maps (PNG)
├── raw_minimap/                 # Raw minimap tiles (input)
├── settings/                    # Application configuration
│   ├── app_settings.json        # User settings
│   └── non_walkable_colors.json # Walkable detection rules
├── area_data/                   # Area definitions
│   ├── areas.json               # Area metadata
│   └── settings.json            # Area settings
├── assets/                      # Static assets
│   └── minimap/                 # Minimap marker icons
├── requirements.txt             # Python dependencies
├── main.py                      # Application entry point
└── README.md                    # This file
```

