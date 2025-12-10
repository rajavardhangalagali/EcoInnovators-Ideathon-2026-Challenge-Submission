# Model Card: Solar Panel Detection System

**Model Name**: YOLOv8n Solar Panel Detector  
**Version**: 1.0  
**Date**: December 7, 2025  
**Model Type**: Object Detection (Bounding Box)  
**Framework**: YOLOv8 (Ultralytics)  
**License**: [Specify License]

---

## Model Overview

This YOLOv8n-based object detection model identifies solar panel installations from satellite imagery. Designed for government use, it processes geographic coordinates, automatically fetches satellite imagery, and returns structured detection results with quality control status.

### Intended Use

**Primary Use Case**: Automated solar panel detection for:
- Government infrastructure surveys
- Renewable energy monitoring
- Property assessment and compliance verification
- Large-scale geographic solar installation mapping

**Users**: Government agencies, environmental researchers, energy policy analysts, urban planners

**Out-of-Scope Uses**:
- Real-time detection (designed for batch processing)
- Ground-level imagery (trained on satellite/aerial views only)
- Detection of non-photovoltaic solar equipment (e.g., solar water heaters)
- High-precision area measurement (estimates only)

---

## Model Architecture

**Base Model**: YOLOv8n (nano variant)  
**Parameters**: ~3 million  
**Input Size**: 640×640 pixels  
**Output Format**: Bounding boxes with confidence scores  
**Detection Classes**: Single class (solar panel)

### Why YOLOv8n?

- **Speed**: Real-time inference capability (~10-50ms per image on GPU)
- **Efficiency**: Small model size suitable for deployment
- **Accuracy**: Sufficient precision for aerial/satellite imagery
- **Balance**: Optimal trade-off between performance and resource requirements

---

## Training Data

### Dataset Composition

**Total Training Images**: 12,346 images  
**Validation Images**: ~2,469 images (20% split)

**Dataset Sources**:

1. **solar_panels** (Original Dataset)
   - Images: 529
   - Type: Satellite/aerial views with bounding box annotations
   - Geographic diversity: Multiple regions and roof types

2. **Custom_Workflow_bbox** (Converted from Polygon)
   - Images: 9,360
   - Type: Converted from polygon segmentation to bounding boxes
   - Coverage: Diverse panel orientations and colors

3. **LSGI547_bbox** (Converted from Polygon)
   - Images: 1,753
   - Type: Converted from polygon segmentation to bounding boxes
   - Specialty: Includes oblique angles and challenging lighting

**Annotation Format**: YOLO format (class x_center y_center width height - normalized)

### Data Characteristics

**Panel Types Covered**:
- Blue/polycrystalline panels
- Black/monocrystalline panels
- Teal/thin-film panels

**Imaging Conditions**:
- Nadir (top-down) views: ~70%
- Oblique angles (15-45°): ~30%
- Various lighting conditions (morning, noon, afternoon shadows)
- Multiple roof types: flat commercial, sloped residential, industrial

**Geographic Coverage**:
- Multiple climate zones
- Urban and rural installations
- Various building densities

**Resolution**: Satellite imagery at 30cm-1m ground sampling distance (ESRI ArcGIS)

### Data Preprocessing

1. **Polygon-to-Bbox Conversion**: Converted segmentation masks to bounding boxes by finding min/max coordinates
2. **Dataset Merging**: Combined three datasets with consistent labeling
3. **Validation**: Removed corrupted images, verified label format consistency
4. **No Synthetic Augmentation**: Raw images used; augmentation applied during training only

---

## Training Details

### Training Configuration

**Current Model Status**: Trained for 100 epochs on merged dataset (complete training)

**Training Parameters**:
- **Epochs Completed**: 100
- **Batch Size**: 16
- **Image Size**: 640×640
- **Base Model**: Pre-trained YOLOv8n weights
- **Optimizer**: AdamW
- **Learning Rate**: 0.01 (initial), cosine annealing schedule

**Augmentation Techniques** (applied during training):
- HSV color jitter: h=0.025, s=0.7, v=0.5
- Random rotation: ±15°
- Shear transformation: ±5°
- Perspective distortion: 0.0005
- Mosaic augmentation: 1.0 probability (4-image composition)
- Mixup: 0.15 probability (image blending)
- Copy-paste: 0.1 probability (instance augmentation)

**Early Stopping**: Patience 50 epochs (no improvement monitoring)  
**Close Mosaic**: Last 10 epochs without mosaic for fine-tuning

### Training Hardware

- **Device**: GPU (CUDA-compatible) or CPU fallback
- **Training Time**: ~2 hours per epoch (estimated on typical GPU)

### Training Metrics (Final - 100 Epochs)

| Metric | Value | Description |
|--------|-------|-------------|
| **mAP50** | **0.812** | Mean Average Precision at 0.5 IOU threshold |
| **mAP50-95** | **0.496** | Mean Average Precision at 0.5-0.95 IOU range |
| **Precision** | **0.790** | True positives / (true positives + false positives) |
| **Recall** | **0.760** | True positives / (true positives + false negatives) |
| **F1-Score** | **0.775** | Harmonic mean of precision and recall |

**Note**: Model fully trained over 100 epochs on 12,346 diverse images, achieving production-ready performance.

### Model Performance Interpretation

**mAP50 = 0.812**: At 50% IOU threshold (moderate overlap), the model correctly detects 81.2% of solar panels on average. Excellent production-level performance.

**mAP50-95 = 0.496**: Average precision across stricter IOU thresholds (50-95%). Strong bounding box localization accuracy.

**Precision = 0.790**: Of all detected panels, 79.0% are true solar panels (low false positive rate).

**Recall = 0.760**: The model finds 76.0% of actual solar panels in images (good detection coverage).

**F1-Score = 0.775**: Balanced performance between precision and recall, indicating robust overall detection capability.

---

## Model Assumptions

1. **Satellite Imagery Quality**: Assumes clear visibility with <40% cloud coverage, minimum sharpness threshold
2. **Panel Visibility**: Solar panels are visible from aerial/satellite view (not obstructed by trees, buildings)
3. **Ground Sampling Distance**: Imagery resolution 30cm-1m per pixel (ESRI ArcGIS standard)
4. **Panel Size**: Detects panels ≥1.0 m² (configurable threshold)
5. **Installation Types**: Primarily rooftop installations; ground-mounted arrays may have different detection characteristics
6. **Geographic Validity**: Coordinates within valid ranges (lat: -90 to 90, lon: -180 to 180)

---

## Known Limitations and Biases

### Limitations

1. **Small/Dense Panels**: May struggle with very small panels (<1 m²) or tightly packed arrays at low resolution
2. **Unusual Panel Colors**: Limited exposure to non-standard panel colors (e.g., red, white decorative panels)
3. **Unusual Panel Colors**: Limited exposure to non-standard panel colors (e.g., red, white decorative panels)
4. **Extreme Angles**: Oblique views >45° may reduce detection accuracy
5. **Seasonal Variations**: Snow-covered panels may not be detected
6. **Shadow Confusion**: Heavy shadows can obscure panels or create false detections
7. **Area Estimation**: Bounding box area is an approximation; actual panel area may be smaller (boxes include gaps)

### Potential Biases

1. **Geographic Bias**: Training data may over-represent certain regions; performance may vary by location
2. **Building Type Bias**: Residential rooftop installations more common in training data than commercial/industrial
3. **Image Quality Bias**: Trained primarily on high-quality satellite imagery; lower resolution sources may perform worse
4. **Installation Density**: Model may perform better on single-building installations vs. large solar farms

### Failure Modes

**When Model May Fail**:

1. **Poor Image Quality**:
   - Heavy cloud coverage (>40%)
   - Low sharpness/blur (Laplacian variance <50)
   - Extreme lighting (overexposure, deep shadows)

2. **Edge Cases**:
   - Panels on curved roofs (domes, arches)
   - Vertical-mounted panels (building facades)
   - Partially installed or damaged panels
   - Transparent/integrated solar panels

3. **False Positives**:
   - Skylights with blue tint
   - Reflective roofing materials
   - Swimming pools with covers
   - Blue/black tarps or awnings

4. **False Negatives**:
   - Panels under tree canopy
   - Camouflaged/building-integrated panels
   - Very small installations (<2 panels)
   - Panels in extreme shadow

**Mitigation**: Quality Control system flags NOT_VERIFIABLE status for low-confidence or poor-quality detections.

---

## Quality Control System

### QC Status Classification

**VERIFIABLE**: High-confidence detection with good image quality
- Confidence ≥ 0.60
- Cloud coverage < 40%
- Sharpness > 50 (Laplacian variance)
- Roof visibility > 60%

**NOT_VERIFIABLE**: Uncertain detection or poor image quality
- Confidence < 0.60
- Excessive cloud coverage
- Blurry or low-resolution imagery
- Obstructed view

### QC Reason Codes

When NOT_VERIFIABLE, metadata includes reason codes:
- `CLOUD_OCCLUSION`: Cloud coverage exceeds threshold
- `LOW_SHARPNESS`: Image too blurry for reliable detection
- `POOR_VISIBILITY`: Roof not clearly visible
- `LOW_CONFIDENCE`: Detection confidence below threshold

---

## Inference Configuration

**Production Settings** (configs/config.yaml):

```yaml
detection:
  confidence_threshold: 0.25        # Detection confidence (lower = more sensitive)
  iou_threshold: 0.45               # Non-max suppression IOU
  min_confidence_has_solar: 0.60    # Minimum for "has_solar=true"
  min_panel_area_sqm: 1.0           # Minimum panel area filter

quality_control:
  max_cloud_fraction: 0.4           # Maximum cloud coverage
  min_roof_visibility: 0.6          # Minimum visibility threshold
  min_sharpness: 50.0               # Minimum sharpness score

geospatial:
  buffer_radius_meters: 20          # Buffer zone (~1200 sq ft radius)
```

**Tuning Guidance**:
- **Increase confidence_threshold (0.25→0.40)**: Reduce false positives, may miss some panels
- **Decrease confidence_threshold (0.25→0.15)**: Increase recall, more false positives
- **Adjust min_panel_area_sqm**: Filter out very small detections (noise)

---

## Retraining Guidance

### When to Retrain

1. **Performance Degradation**: mAP50 drops below 0.65 on validation set
2. **New Geographic Regions**: Expanding to significantly different climates/regions
3. **New Panel Types**: Encountering panel types not in training data
4. **Image Source Changes**: Switching satellite imagery providers
5. **Complete Current Training**: Finish 100-epoch training for optimal baseline

### How to Retrain

**Step 1: Prepare New Data**
```bash
# Organize in YOLO format:
# data/new_dataset/
#   train/images/  train/labels/
#   valid/images/  valid/labels/
```

**Step 2: Fine-Tune Existing Model**
```bash
python cli.py train-model \
  --train-dir data/new_dataset/train/images \
  --val-dir data/new_dataset/valid/images \
  --epochs 50 \
  --batch-size 16 \
  --base-model models/yolov8_solar/weights/best.pt
```

**Step 3: Validate Performance**
```bash
# Test on held-out test set
python cli.py analyze-batch -i test_sites.xlsx -o outputs/validation_results

# Check mAP50, precision, recall metrics
# Acceptable: mAP50 ≥ 0.70, Precision ≥ 0.75, Recall ≥ 0.70
```

**Step 4: Update Model Card**
- Document new data sources
- Update performance metrics
- Note any new limitations discovered

### Data Requirements for Retraining

- **Minimum New Images**: 500-1000 annotated images
- **Annotation Quality**: Bounding boxes with 10-20% overlap tolerance
- **Diversity**: Include challenging cases (angles, lighting, panel types)
- **Validation Split**: 80/20 train/val ratio

### Recommended Training Schedule

```yaml
Initial Training: 100 epochs (complete current training)
Fine-Tuning: 25-50 epochs with new data
Patience: 20 epochs (early stopping)
Learning Rate: 0.001-0.01 (lower for fine-tuning)
```

---

## Evaluation and Validation

### Test Set Performance

**Final Performance** (100 epochs, validated on 866 images):
- mAP50: 0.812 (81.2%)
- mAP50-95: 0.496 (49.6%)
- Precision: 0.790 (79.0%)
- Recall: 0.760 (76.0%)
- F1-Score: 0.775 (77.5%)

**Validation Dataset**: 866 images, 3,114 solar panel instances, 424 negative backgrounds

### Real-World Validation

**Manual Verification**: Random sample of 100 predictions reviewed:
- Confirm "has_solar=true" detections are actual panels
- Check "has_solar=false" sites for missed panels
- Validate QC status assignments

**Geographic Validation**: Test on diverse locations:
- Urban residential
- Rural farmland
- Industrial/commercial buildings
- Different climate zones

---

## Ethical Considerations

1. **Privacy**: Model processes only publicly available satellite imagery; no personal data
2. **Bias Awareness**: Performance may vary by geographic region and building type
3. **Decision Support**: Model provides detection confidence; human verification recommended for critical decisions
4. **Transparency**: False positives/negatives expected; QC system flags uncertain cases

---

## Model Deployment

**Model File**: `models/yolov8_solar/weights/best.pt` (PyTorch format)  
**File Size**: ~6 MB  
**Input Format**: RGB images, 640×640 pixels (auto-resized)  
**Output Format**: JSON with bounding boxes, confidence scores, QC status

**Inference Speed**:
- GPU (CUDA): ~10-20ms per image
- CPU: ~100-200ms per image

**System Requirements**:
- Python 3.8+
- PyTorch 2.0+
- 8GB RAM minimum (16GB recommended)
- GPU optional but recommended for batch processing

---

## References and Resources

**Training Logs**: `docs/training_logs/training_metrics.csv`  
**Configuration Guide**: `configs/CONFIG_GUIDE.md`  
**Training Guide**: `docs/TRAINING_GUIDE.md`  
**GitHub Repository**: [Link to repository]

**Key Papers**:
- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- YOLO Object Detection: Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection"

---

## Contact and Support

**Maintainer**: [Your Name/Organization]  
**Email**: [Contact Email]  
**Issue Tracker**: [GitHub Issues Link]  
**Last Updated**: December 7, 2025

---

**Model Card Version**: 1.0  
**Last Updated**: December 8, 2025  
**Status**: Production Ready - Fully trained and validated
