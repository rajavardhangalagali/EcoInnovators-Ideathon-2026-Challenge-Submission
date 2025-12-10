# Solar Panel Detection System

Government-ready solar panel detection system using YOLOv8 with automated satellite imagery analysis. Detects solar panels from geographic coordinates with VERIFIABLE/NOT_VERIFIABLE quality control status.

## 🎯 Features

- **Multi-format Input Support**: CSV, JSON, and Excel (.xlsx/.xls) files with sample_id, lat, lon coordinates
- **Automated Satellite Imagery**: Free ESRI ArcGIS World Imagery (no API key required, 30cm-1m resolution)
- **Advanced Detection**: YOLOv8n model trained on 12,346 diverse images (blue/teal/black panels, oblique angles, various roof types)
- **Quality Control**: Automated QC status (VERIFIABLE/NOT_VERIFIABLE) with reason codes
- **Buffer Zone Analysis**: 20-meter buffer radius (~1200 sq ft) around detected panels
- **Structured JSON Output**: Government-compliant format with all required fields
- **CLI & Web Interface**: Command-line tools and Flask web application

## 📋 System Requirements

- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB (16GB recommended)
- **GPU**: CUDA-compatible GPU recommended (CPU inference supported)
- **OS**: Windows, Linux, or macOS

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/rajavardhangalagali/EcoInnovators-Ideathon-2026-Challenge-Submission.git
cd EcoInnovators-Ideathon-2026-Challenge-Submission

# Create virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Check if model exists
python cli.py --help
```

### 3. Run Inference (Quick Test)

```bash
# Single site analysis
python cli.py analyze-single -s SITE_001 --lat 37.7749 --lon -122.4194

# Batch processing from Excel file (auto-fetches satellite images)
python cli.py analyze-batch -i data/sample_data/sample_sites.xlsx -o outputs/results
```

## 📖 Usage Guide

### CLI Interface

#### **Single Site Analysis**

```bash
python cli.py analyze-single -s <sample_id> --lat <latitude> --lon <longitude> [-o <output.json>]

# Example:
python cli.py analyze-single -s SITE_001 --lat 37.7749 --lon -122.4194 -o results/site_001.json
```

#### **Batch Processing**

```bash
python cli.py analyze-batch -i <input_file> -o <output_dir>

# From Excel file (auto-fetches satellite images from ESRI):
python cli.py analyze-batch -i sites.xlsx -o outputs/results

# From CSV file:
python cli.py analyze-batch -i sites.csv -o outputs/results

# From JSON file:
python cli.py analyze-batch -i sites.json -o outputs/results
```

**Input File Format** (Excel/CSV/JSON):
```
| sample_id | lat      | lon        |
|-----------|----------|------------|
| SITE_001  | 37.7749  | -122.4194  |
| SITE_002  | 34.0522  | -118.2437  |
```

#### **Model Training**

```bash
python cli.py train-model \
  --train-dir data/train/images \
  --val-dir data/valid/images \
  --epochs 100 \
  --batch-size 16 \
  --image-size 640 \
  --base-model yolov8n.pt

# Continue training from checkpoint:
python cli.py train-model \
  --train-dir data/train/images \
  --val-dir data/valid/images \
  --epochs 50 \
  --base-model models/yolov8_solar/weights/best.pt
```

### Web Interface (Flask App)

```bash
# Start the web server
python app.py

# Open browser to http://localhost:5000
```

**Web Interface Features:**
- Upload satellite images or auto-fetch from coordinates
- Real-time detection visualization
- Interactive bounding box display
- Confidence scores and QC status
- Download results as JSON

### Python API

```python
from src.main_pipeline import SolarDetectionPipeline

# Initialize pipeline
pipeline = SolarDetectionPipeline(config_path="configs/config.yaml")

# Single site analysis
result = pipeline.analyze_site(
    sample_id="SITE_001",
    lat=37.7749,
    lon=-122.4194
)

print(result)
# Output:
# {
#   "sample_id": "SITE_001",
#   "lat": 37.7749,
#   "lon": -122.4194,
#   "has_solar": true,
#   "confidence": 0.87,
#   "pv_area_sqm_est": 45.3,
#   "buffer_radius_sqft": 1200,
#   "qc_status": "VERIFIABLE",
#   "bbox_or_mask": [[x1, y1, x2, y2], ...],
#   "timestamp": "2025-12-07T10:30:00Z"
# }

# Batch processing
results = pipeline.process_batch(
    input_file="sites.xlsx",
    output_file="predictions.json"
)
```

## 📁 Project Structure

```
solar/
├── app.py                          # Flask web application
├── cli.py                          # Command-line interface
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker container configuration
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
│
├── configs/
│   └── config.yaml                 # Main configuration file
│
├── src/
│   ├── main_pipeline.py            # Main detection pipeline
│   ├── detection.py                # YOLOv8 detection logic
│   ├── satellite_fetcher.py        # ESRI satellite imagery fetching
│   ├── data_processing.py          # Input/output processing
│   ├── quantification.py           # Area and capacity estimation
│   ├── quality_control.py          # QC status determination
│   ├── visualization.py            # Overlay image generation
│   └── image_verification.py       # Satellite image verification
│
├── models/
│   └── yolov8_solar/
│       └── weights/
│           └── best.pt             # Trained model (6 MB)
│
├── data/
│   ├── sample_data/
│   │   ├── sample_sites.csv        # Example input (CSV)
│   │   ├── sample_sites.xlsx       # Example input (Excel)
│   │   ├── sample_sites.json       # Example input (JSON)
│   │   ├── sample_predictions.json # Example output
│   │   └── README.md               # Sample data documentation
│   ├── input/
│   │   └── satellite_images/       # Auto-downloaded satellite images
│   └── solar/
│       └── merged_solar_dataset/   # Training dataset (excluded from git)
│
├── outputs/
│   ├── overlays/                   # Detection overlay images
│   └── results/                    # Batch processing results
│
└── docs/
    ├── MODEL_CARD.md               # Model documentation (3 pages)
    ├── DELIVERABLES_CHECKLIST.md   # Submission checklist
    ├── DOCKER_INSTRUCTIONS.md      # Docker setup guide
    └── training_logs/
        ├── training_metrics.csv    # Complete training metrics
        └── README.md               # Training logs documentation
```

## 🔧 Configuration

Edit `configs/config.yaml` to customize:

```yaml
detection:
  confidence_threshold: 0.25        # Detection confidence threshold
  iou_threshold: 0.45               # Non-max suppression IOU
  min_confidence_has_solar: 0.60    # Minimum confidence for "has_solar=true"
  min_panel_area_sqm: 1.0           # Minimum panel area (square meters)

satellite:
  provider: "ESRI"                  # Satellite imagery provider
  zoom: 19                          # Zoom level (higher = more detail)
  image_size: [1024, 1024]          # Image dimensions

quality_control:
  max_cloud_fraction: 0.4           # Maximum cloud coverage allowed
  min_roof_visibility: 0.6          # Minimum roof visibility required
  min_sharpness: 50.0               # Minimum image sharpness

geospatial:
  buffer_radius_meters: 20          # Buffer zone around panels (~1200 sq ft)
```

See `CONFIG_GUIDE.md` for detailed configuration documentation.

## 📊 Output Format

**JSON Output Schema** (Government-Compliant):

```json
{
  "sample_id": "SITE_001",
  "lat": 37.7749,
  "lon": -122.4194,
  "has_solar": true,
  "confidence": 0.87,
  "pv_area_sqm_est": 45.3,
  "buffer_radius_sqft": 1200,
  "panel_count_est": 2,
  "capacity_kw_est": 8.6,
  "qc_status": "VERIFIABLE",
  "qc_notes": [],
  "bbox_or_mask": "outputs/overlays/SITE_001_overlay.png",
  "image_metadata": {
    "source": "DATASET",
    "width": 640,
    "height": 640,
    "capture_date": "2025-12-08",
    "analysis_timestamp": "2025-12-08T14:38:23Z",
    "model_version": "YOLOv8_solar_v1.0"
  },
  "detailed_results": {
    "detection": {
      "boxes": [[100, 150, 200, 250], [300, 350, 400, 450]],
      "confidences": [0.87, 0.82],
      "mean_confidence": 0.845
    },
    "quantification": {
      "individual_panel_areas": [22.5, 22.8],
      "average_panel_area": 22.65,
      "quantification_assumptions": {
        "watts_per_sqm": 190,
        "meters_per_pixel": 0.3,
        "min_panel_area_sqm": 1.0
      }
    },
    "quality_control": {
      "qc_confidence_score": 0.85,
      "image_quality_metrics": {
        "sharpness": 125.4,
        "brightness": 0.62,
        "contrast": 0.38,
        "cloud_fraction": 0.05,
        "resolution_score": 1.0
      },
      "detection_quality_metrics": {
        "max_confidence": 0.87,
        "mean_confidence": 0.845,
        "panel_count": 2,
        "total_area_sqm": 45.3
      }
    }
  }
}
```

**Key Fields:**
- `has_solar`: `true` if confidence ≥ 0.60 and area ≥ 1.0 sqm
- `confidence`: Maximum detection confidence (0.0-1.0)
- `pv_area_sqm_est`: Total estimated panel area in square meters
- `panel_count_est`: Estimated number of individual panels
- `capacity_kw_est`: Estimated capacity (area × 190 W/m² ÷ 1000)
- `buffer_radius_sqft`: 20m buffer = 1200 sq ft

**QC Status Values:**
- `VERIFIABLE`: High-quality detection with clear visibility
- `NOT_VERIFIABLE`: Low-quality image or uncertain detection

**QC Notes** (when NOT_VERIFIABLE):
- `low-resolution imagery`
- `heavy cloud/occlusion`
- `partial roof view`
- `conflicting signals`
- `poor image quality`

## 🐳 Docker Deployment

### Build Docker Image

```bash
# Build the image
docker build -t solar-panel-detector:v1.0 .

# Tag for Docker Hub
docker tag solar-panel-detector:v1.0 <your-dockerhub-username>/solar-panel-detector:v1.0

# Push to Docker Hub
docker push <your-dockerhub-username>/solar-panel-detector:v1.0
```

### Run Docker Container

```bash
# Run batch processing
docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs \
  solar-panel-detector:v1.0 \
  python cli.py analyze-batch -i /app/data/sample_data/sample_sites.xlsx -o /app/outputs/results

# Run single site analysis
docker run solar-panel-detector:v1.0 \
  python cli.py analyze-single -s SITE_001 --lat 37.7749 --lon -122.4194

# Run web interface
docker run -p 5000:5000 solar-panel-detector:v1.0 python app.py
```

**Docker Hub Image**: `<your-dockerhub-username>/solar-panel-detector:v1.0`

## 📈 Model Performance

**Training Dataset**: 12,346 images (merged_solar_dataset)
- Training: 11,480 images (93%)
- Validation: 866 images (7%)
- Total instances: 9,821 train / 3,114 validation

**Data Sources**:
- Solar_Panel.v1 (original): 529 images
- Custom_Workflow_bbox: 9,360 images (converted from polygon annotations)
- LSGI547_bbox: 1,753 images (converted from polygon annotations)
- Combined: Blue/teal/black panels, oblique angles, various roof types

**Training Configuration**:
- Model: YOLOv8n (3M parameters, 8.1 MB)
- Epochs: 100 (fully trained)
- Batch Size: 16
- Image Size: 640×640
- Augmentation: HSV, rotation, shear, perspective, mosaic, mixup, copy_paste
- Optimizer: SGD with lr0=0.01, momentum=0.937

**Final Metrics** (100 epochs):
- **mAP50: 0.812** (81.2% mean average precision at 0.5 IOU)
- **mAP50-95: 0.496** (49.6% mean average precision at 0.5-0.95 IOU)
- **Precision: 0.790** (79.0%)
- **Recall: 0.760** (76.0%)
- **F1 Score: 0.775** (77.5%)

**Performance Notes**:
- Detects blue, teal, and black solar panels
- Handles oblique viewing angles (up to ~60° from nadir)
- Works with varying roof types (flat, sloped, complex)
- Robust to partial occlusion and lighting variations

**Training Logs**: See `docs/training_logs/training_metrics.csv` and `MODEL_CARD.md`

## 🧪 Testing

```bash
# Test single inference
python test_inference.py

# Test batch processing with sample data (auto-fetches satellite images)
python cli.py analyze-batch -i data/sample_data/sample_sites.xlsx -o outputs/test_batch

# Verify Docker build
docker build -t solar-test .
docker run solar-test python cli.py --help
```

## 📝 Model Card

See `MODEL_CARD.md` for comprehensive documentation including:
- Data sources and preprocessing
- Model architecture and training details
- Assumptions and limitations
- Known biases and failure modes
- Retraining guidance
- Performance benchmarks

## 🔄 Retraining the Model

```bash
# Prepare your dataset in YOLO format:
# data/
#   train/
#     images/
#     labels/
#   valid/
#     images/
#     labels/

# Train from scratch
python cli.py train-model \
  --train-dir data/train/images \
  --val-dir data/valid/images \
  --epochs 100 \
  --batch-size 16 \
  --base-model yolov8n.pt

# Fine-tune existing model
python cli.py train-model \
  --train-dir data/new_train/images \
  --val-dir data/new_valid/images \
  --epochs 50 \
  --base-model models/yolov8_solar/weights/best.pt
```

See `TRAINING_GUIDE.md` for detailed retraining instructions.

## 🐛 Troubleshooting

### Common Issues

**1. Module Import Errors**
```bash
# Ensure virtual environment is activated
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**2. CUDA/GPU Issues**
```bash
# Verify PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"

# For CPU-only inference, PyTorch CPU version is sufficient
```

**3. Satellite Image Fetch Failures**
- Check internet connection
- ESRI service may have rate limits (add delays between requests)
- Verify coordinates are valid (lat: -90 to 90, lon: -180 to 180)

**4. Low Detection Performance**
- Check image quality (cloud coverage, resolution)
- Adjust `confidence_threshold` in `config.yaml`
- Verify model file exists at `models/yolov8_solar/weights/best.pt`

## 📦 Dependencies

Core dependencies (see `requirements.txt` for full list):
- `torch>=2.0.0` - PyTorch deep learning framework
- `ultralytics>=8.0.0` - YOLOv8 implementation
- `opencv-python>=4.8.0` - Image processing
- `pandas>=2.0.0` - Data manipulation
- `openpyxl>=3.1.0` - Excel file support
- `flask>=2.3.0` - Web application framework
- `geopy>=2.3.0` - Geospatial calculations

## 🤝 Support

For issues or questions:
1. Check `docs/` directory for additional documentation
2. Review training logs in `runs/detect/train/results.csv`
3. Verify configuration in `configs/config.yaml`

## 📄 License



## 👥 Authors

[Rajavardhan S G]

---

**Version**: 1.0  
**Last Updated**: December 10, 2025  
**Model File**: `models/yolov8_solar/weights/best.pt` (6 MB)  
**Training Dataset**: 12,346 images (merged_solar_dataset)  
**Model Performance**: mAP50 = 81.2%, Precision = 79.0%, Recall = 76.0%
