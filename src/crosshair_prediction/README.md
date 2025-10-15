# Crosshair Position Prediction - Production

This module provides production-ready crosshair position prediction using DINOv3 feature matching.

## Overview

The crosshair prediction system uses **DINOv3 ViT-B/16** (Vision Transformer Base) to extract visual features from minimap crops and match them against a pre-built database of positions. This approach achieves **5.11px median error** - significantly better than the previous deep learning approach (13px).

## Architecture

```
src/crosshair_prediction/
├── __init__.py              # Module exports
├── feature_extractor.py     # DINOv3 feature extraction
├── feature_database.py      # Database loading and querying
├── matcher.py               # Position matching logic
├── inference.py             # Production interface (CrosshairPredictor)
└── README.md                # This file
```

## Key Components

### 1. FeatureExtractor
Extracts 768-dimensional visual features from minimap crops using DINOv3 ViT-B/16.

### 2. FeatureDatabase
Loads and queries the pre-built feature database containing ~191,000 positions for floor 7.

### 3. CrosshairMatcher
Combines feature extraction and database querying to predict positions using:
- Cosine similarity matching
- Top-k retrieval (k=10)
- Weighted interpolation with temperature=0.01

### 4. CrosshairPredictor
Production interface used by the automatic monster tracking system.

## Usage

### Basic Usage

```python
from src.crosshair_prediction import CrosshairPredictor

# Initialize predictor (auto-detects best database)
predictor = CrosshairPredictor()

# Predict from screenshot
x, y, floor = predictor.predict_from_screenshot(
    screenshot_path=Path("screenshot.png"),
    minimap_region=(1604, 35, 106, 106),
    floor=7
)

# Predict from minimap crop
from PIL import Image
crop = Image.open("minimap_crop.png")
x, y, floor, confidence = predictor.predict_from_crop(crop, floor=7)
```

### Integration with Monster Tracking

The `CrosshairPredictor` is automatically used by the monster tracking system in `src/ui/minimap_viewer.py`:

```python
from src.crosshair_prediction import CrosshairPredictor

# Initialize in start_automated_tracking()
self.crosshair_detector = CrosshairPredictor()

# Use in process_screenshot()
x, y, floor = self.crosshair_detector.predict_from_screenshot(
    screenshot_path,
    self.minimap_region
)
```

## Performance

**DINOv3 ViT-B/16 Results (300 test samples):**
- **Median Error**: 5.11 pixels 🏆
- **Mean Error**: 5.63 pixels
- **Std Error**: 3.95 pixels
- **Accuracy < 10px**: 87.3%
- **Accuracy < 5px**: 48.7%
- **99th Percentile**: 16.69 pixels

**Comparison with Previous Approach:**
- **61% better median error** (5.11px vs 13.00px)
- **74.6% higher accuracy** at <10px threshold (87.3% vs 50%)
- **No training required** - uses pre-trained DINOv3 weights

## Requirements

### Model Weights
The DINOv3 model weights are located in:
```
src/crosshair_prediction/models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

### Feature Database
The pre-built feature database is located in:
```
src/crosshair_prediction/database/feature_db_dinov3_vitb16.pkl (560 MB)
```

Contains:
- 190,999 positions for floor 7
- 768-dimensional features
- 5-pixel grid spacing

### Dependencies
- PyTorch
- torchvision
- Pillow
- numpy
- Internet connection (for downloading DINOv3 from torch.hub if local weights not found)

## Technical Details

### Feature Extraction
- **Model**: DINOv3 ViT-B/16
- **Input Size**: 112x112 pixels (resized from 106x106)
- **Patch Size**: 16x16
- **Feature Dimension**: 768
- **Normalization**: ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Matching Algorithm
1. Extract 768-dim features from query minimap crop
2. Compute cosine similarity with all database features for the floor
3. Retrieve top-10 most similar positions
4. Apply softmax with temperature=0.01 to compute weights
5. Return weighted average of top-10 positions

### Inference Speed
- **Feature Extraction**: ~65ms per crop (CPU)
- **Database Query**: ~15ms per query
- **Total Time**: ~80ms per prediction
- **Throughput**: ~12-15 predictions/second

## Production Implementation

The `src/crosshair_prediction/` folder contains the **production implementation**:
- Simplified interface optimized for automatic monster tracking
- Self-contained with models and database included
- Automatic model downloading from torch.hub if local weights missing
- No external dependencies on testing folders
- Clean, maintainable codebase

## Troubleshooting

### "Database not found" Error
Make sure the database exists:
```bash
ls src/crosshair_prediction/database/feature_db_dinov3_vitb16.pkl
```

### "Model loading failed" Error
Check if model weights exist locally:
```bash
ls src/crosshair_prediction/models/dinov3_vitb16_pretrain_*.pth
```

If weights are missing, the system will automatically download them from torch.hub when first used.

### "No internet connection" Error
If you're offline and don't have local weights, you need to either:
1. Connect to the internet to download weights, or
2. Manually place the weights in `src/crosshair_prediction/models/`

## Future Improvements

- **GPU Acceleration**: 5-10x faster inference
- **Multi-floor Support**: Build databases for all floors
- **FAISS Integration**: Faster similarity search for large databases
- **Model Quantization**: Reduce memory footprint
- **Batch Prediction**: Process multiple screenshots efficiently

## References

- **DINOv3 Paper**: https://arxiv.org/abs/2304.07193
- **DINOv3 Repository**: https://github.com/facebookresearch/dinov3
- **Test Results**: See `crosshair_matcher/DINOV3_RESULTS.md`

