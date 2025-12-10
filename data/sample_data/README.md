# Sample Data and Prediction Files

This directory contains sample input files and example prediction outputs following the government-required schema.

## Sample Input Files

### sample_sites.csv
CSV format input file with sample coordinates. Contains 5 example sites.

**Format**:
```csv
sample_id,lat,lon
SITE_001,37.7749,-122.4194
SITE_002,34.0522,-118.2437
...
```

### sample_sites.xlsx
Excel format (same data as CSV). This is the **primary government-required format**.

**Columns**:
- `sample_id`: Unique site identifier
- `lat`: Latitude (-90 to 90)
- `lon`: Longitude (-180 to 180)

### sample_sites.json
JSON format (same data as CSV/Excel). Alternative input format.

**Format**:
```json
[
  {
    "sample_id": "SITE_001",
    "lat": 37.7749,
    "lon": -122.4194
  },
  ...
]
```

## Sample Output Files

### sample_predictions.json
Example prediction output following the complete government-required schema.

**Schema** (per site):
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
    "detection": {...},
    "quantification": {...},
    "quality_control": {...}
  }
}
```

## Field Descriptions

### Core Fields (Government-Required)

- **sample_id** (string): Unique identifier from input file
- **lat** (float): Latitude coordinate (-90 to 90)
- **lon** (float): Longitude coordinate (-180 to 180)
- **has_solar** (boolean): `true` if solar panels detected with confidence ≥ 0.60, `false` otherwise
- **confidence** (float): Maximum detection confidence (0.0-1.0)
- **pv_area_sqm_est** (float): Estimated total PV area in square meters
- **buffer_radius_sqft** (float): Buffer zone radius in square feet (1200 sq ft ≈ 20m radius)
- **panel_count_est** (integer): Estimated number of individual panels detected
- **capacity_kw_est** (float): Estimated solar capacity in kilowatts (area × 190 W/m² ÷ 1000)
- **qc_status** (string): Quality control status - `VERIFIABLE` or `NOT_VERIFIABLE`
- **qc_notes** (array): Array of QC issue descriptions (if NOT_VERIFIABLE)
- **bbox_or_mask** (string): Path to overlay image with detection visualization

### Metadata Fields

- **image_metadata**: Source, dimensions, timestamps, model version
- **detailed_results**: Complete detection, quantification, and QC metrics

### QC Notes

When `qc_status = "NOT_VERIFIABLE"`, possible notes:
- `low-resolution imagery`
- `heavy cloud/occlusion`
- `partial roof view`
- `conflicting signals`
- `poor image quality`
- `image_location_mismatch`
- `extreme shadows on sloped roof`
- `panel orientation unclear`

### Image Quality Metrics

- **sharpness**: Laplacian variance (higher = sharper)
- **brightness**: Mean intensity (0.0-1.0)
- **contrast**: Standard deviation of intensity (0.0-1.0)
- **cloud_fraction**: Estimated cloud coverage (0.0-1.0)
- **noise_level**: Estimated noise (lower = cleaner)
- **resolution_score**: Relative to 1024px baseline

### Detection Quality Metrics

- **max_confidence**: Highest detection confidence
- **mean_confidence**: Average confidence across all detections
- **panel_count**: Number of detected panels
- **total_area_sqm**: Total estimated PV area

## Example Scenarios in sample_predictions.json

**SITE_001**: High-confidence detection (0.87), 2 panels, 8.6 kW, VERIFIABLE  
**SITE_002**: No solar panels detected, VERIFIABLE quality  
**SITE_003**: Moderate-confidence detection (0.72), 1 panel, 5.4 kW, VERIFIABLE  
**SITE_004**: Poor image quality, NOT_VERIFIABLE (cloud occlusion)  
**SITE_005**: Very high confidence (0.91), 3 panels, 12.9 kW, excellent quality  

## Using Sample Files

### Run Sample Prediction
```bash
# From Excel file (auto-fetches satellite images from ESRI)
python cli.py analyze-batch -i data/sample_data/sample_sites.xlsx -o outputs/my_predictions

# From CSV file
python cli.py analyze-batch -i data/sample_data/sample_sites.csv -o outputs/my_predictions

# From JSON file
python cli.py analyze-batch -i data/sample_data/sample_sites.json -o outputs/my_predictions
```

### Test with Sample Sites
```bash
# Single site analysis
python cli.py analyze-single -s SITE_001 --lat 37.7749 --lon -122.4194

# Web interface
python app.py
# Then upload sample_sites.xlsx or enter coordinates manually
```

## Government Compliance

These sample files demonstrate **100% compliance** with government requirements:

✅ **Input**: Excel (.xlsx) with sample_id, lat, lon  
✅ **Output**: JSON with all required fields (has_solar, confidence, area, buffer, QC status)  
✅ **Buffer Zone**: 1200 sq ft radius (~20 meters)  
✅ **QC Status**: VERIFIABLE / NOT_VERIFIABLE with descriptive notes  
✅ **Panel Count**: Estimated number of panels detected  
✅ **Capacity**: Estimated kW capacity (190 W/m² conversion)  
✅ **Overlay Images**: Detection visualization saved to outputs/overlays/  
✅ **Metadata**: Complete quality metrics and model information  
✅ **Auto-Fetch**: Satellite images automatically downloaded from ESRI (free, no API key)  

## Creating Your Own Input Files

### Excel Format (Recommended)
1. Create Excel file with columns: `sample_id`, `lat`, `lon`
2. Add your site data (one row per site)
3. Save as `.xlsx`
4. Run: `python cli.py analyze-batch -i your_sites.xlsx -o outputs/results`
5. System will auto-fetch satellite images from ESRI

### CSV Format
```csv
sample_id,lat,lon
MY_SITE_001,37.7749,-122.4194
MY_SITE_002,34.0522,-118.2437
```

### JSON Format
```json
[
  {"sample_id": "MY_SITE_001", "lat": 37.7749, "lon": -122.4194},
  {"sample_id": "MY_SITE_002", "lat": 34.0522, "lon": -118.2437}
]
```

---

**Note**: Sample coordinates are for demonstration purposes. The system automatically fetches current satellite imagery for any valid coordinates.

**Last Updated**: December 10, 2025
