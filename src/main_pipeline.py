"""
Main solar panel analysis pipeline.
Integrates detection, quantification, QC, and output generation.
"""

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union
from pathlib import Path

from data_processing import CoordinateProcessor, DataLoader, ImageFetcher, load_config
from detection import SolarPanelDetector
from quantification import SolarQuantifier
from quality_control import QualityController, QCStatus
from visualization import VisualizationGenerator, ReportGenerator
# Production-ready features (optional imports)
try:
    from satellite_fetcher import SatelliteImageFetcher, SatelliteProvider, create_satellite_fetcher
    from image_verification import ImageVerifier, create_image_verifier
    SATELLITE_AVAILABLE = True
except ImportError:
    SATELLITE_AVAILABLE = False
    SatelliteImageFetcher = None
    ImageVerifier = None


class SolarAnalysisPipeline:
    """Main pipeline for solar panel analysis."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize the analysis pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize components
        self.coord_processor = CoordinateProcessor(
            buffer_meters=self.config['detection']['buffer_meters']
        )
        
        self.detector = SolarPanelDetector(
            model_path=self.config['model']['yolo_model_path'],
            device=self.config['model']['device'],
            confidence_threshold=self.config['model']['confidence_threshold'],
            iou_threshold=self.config['model']['iou_threshold']
        )
        
        self.quantifier = SolarQuantifier(
            watts_per_sqm=self.config['quantification']['watts_per_sqm'],
            meters_per_pixel=self.config['quantification']['meters_per_pixel'],
            min_panel_area_sqm=self.config['quantification']['min_panel_area_sqm']
        )
        
        self.qc_controller = QualityController(self.config)
        self.visualizer = VisualizationGenerator(self.config)
        self.report_generator = ReportGenerator()
        
        # Data loader and image fetcher (initialized per run)
        self.data_loader = DataLoader()
        self.image_fetcher = None
        
        # Production-ready: Satellite image fetching and verification (optional)
        self.satellite_fetcher = None
        self.image_verifier = None
        if SATELLITE_AVAILABLE and self.config.get('production', {}).get('enable_satellite_verification', False):
            provider = self.config['production'].get('satellite_provider', 'google_maps')
            api_key = self.config['production'].get('satellite_api_key')
            self.satellite_fetcher = create_satellite_fetcher(
                provider_name=provider,
                api_key=api_key
            )
            similarity_threshold = self.config['production'].get('verification_threshold', 0.7)
            verification_method = self.config['production'].get('verification_method', 'ssim')
            self.image_verifier = create_image_verifier(
                similarity_threshold=similarity_threshold,
                method=verification_method
            )
    
    def analyze_single_site(self, sample_id: Union[str, int], 
                           lat: float, lon: float,
                           image_path: Optional[str] = None,
                           image_dir: Optional[str] = None,
                           verify_with_satellite: bool = False) -> Dict:
        """
        Analyze a single site for solar panels.
        
        Production-ready feature: If verify_with_satellite=True, fetches satellite
        image at (lat, lon) and verifies uploaded image matches before proceeding.
        
        Args:
            sample_id: Unique identifier for the site
            lat: Latitude coordinate
            lon: Longitude coordinate
            image_path: Direct path to uploaded image (overrides image_dir lookup)
            image_dir: Directory to search for image by sample_id
            verify_with_satellite: If True, verify image matches satellite at coordinates
            
        Returns:
            Complete analysis result dictionary
        """
        # Production-ready: Satellite image verification
        verification_result = None
        satellite_metadata = None
        if verify_with_satellite and self.satellite_fetcher and self.image_verifier:
            if not image_path:
                raise ValueError("image_path required for satellite verification")
            
            try:
                # Fetch satellite image at coordinates
                satellite_img, satellite_metadata = self.satellite_fetcher.fetch_image(
                    lat=lat, lon=lon, zoom=20, width=640, height=640
                )
                
                # Load uploaded image
                import cv2
                uploaded_img = cv2.imread(image_path)
                if uploaded_img is None:
                    raise ValueError(f"Could not load image: {image_path}")
                uploaded_img = cv2.cvtColor(uploaded_img, cv2.COLOR_BGR2RGB)
                
                # Verify images match
                verification_result = self.image_verifier.verify_image_match(
                    uploaded_image=uploaded_img,
                    satellite_image=satellite_img
                )
                
                # If mismatch, return early with error
                if not verification_result.get('is_match', False):
                    return self._create_verification_error_result(
                        sample_id, lat, lon, verification_result, satellite_metadata
                    )
                
            except Exception as e:
                # If satellite fetch fails, log but continue (graceful degradation)
                print(f"Warning: Satellite verification failed: {e}")
                verification_result = {
                    "status": "VERIFICATION_SKIPPED",
                    "error": str(e)
                }
        
        # Initialize image fetcher if needed
        if image_dir and not self.image_fetcher:
            self.image_fetcher = ImageFetcher(image_dir)
        
        # Get image path - auto-fetch from satellite if not found
        if not image_path:
            if self.image_fetcher:
                image_path = self.image_fetcher.get_image_path(sample_id)
                if image_path:
                    image_path = str(image_path)  # Convert Path to string
                else:
                    # Auto-fetch satellite image if not found
                    print(f"  Image not found for {sample_id}, fetching from satellite...")
                    if not SATELLITE_AVAILABLE:
                        raise FileNotFoundError(f"No image found for sample_id: {sample_id} and satellite fetcher not available")
                    
                    # Initialize satellite fetcher if not already done
                    if not self.satellite_fetcher:
                        provider = self.config.get('production', {}).get('satellite_provider', 'esri')
                        api_key = self.config.get('production', {}).get('satellite_api_key')
                        self.satellite_fetcher = create_satellite_fetcher(
                            provider_name=provider,
                            api_key=api_key
                        )
                    
                    # Fetch and save satellite image
                    satellite_img, satellite_metadata = self.satellite_fetcher.fetch_image(
                        lat=lat, lon=lon, zoom=19, width=640, height=640
                    )
                    
                    # Save to image directory
                    self.image_fetcher.image_dir.mkdir(parents=True, exist_ok=True)
                    image_path = self.image_fetcher.image_dir / f"{sample_id}.png"
                    from PIL import Image as PILImage
                    PILImage.fromarray(satellite_img).save(image_path)
                    print(f"  ✓ Fetched and saved to {image_path.name}")
                    image_path = str(image_path)  # Convert Path to string
            else:
                raise ValueError("Either image_path or image_dir must be provided")
        
        # Get image metadata
        if self.image_fetcher:
            image_metadata = self.image_fetcher.get_image_metadata(Path(image_path))
        else:
            image_metadata = {"source": "PROVIDED_PATH"}
        
        # Add verification result to metadata if available
        if verification_result:
            image_metadata["image_verification"] = verification_result
            if satellite_metadata:
                image_metadata["satellite_source"] = satellite_metadata.get("source", "Unknown")
        
        # Run detection
        detection_results = self.detector.detect(image_path)
        detection_results['sample_id'] = sample_id
        
        # Run quantification
        quantification_results = self.quantifier.quantify(
            detection_results, image_metadata
        )
        
        # Run quality control
        qc_results = self.qc_controller.assess_quality(
            image_path, detection_results, image_metadata
        )
        
        # Generate overlay image if configured
        overlay_path = None
        if self.config['output']['save_overlays']:
            overlay_dir = Path("outputs/overlays")
            overlay_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = overlay_dir / f"{sample_id}_overlay.{self.config['output']['overlay_format']}"
            
            self.visualizer.create_overlay_image(
                image_path,
                detection_results,
                quantification_results.__dict__,
                qc_results.__dict__,
                str(overlay_path)
            )
        
        # Create structured JSON output
        result = self._create_structured_output(
            sample_id, lat, lon, 
            detection_results, quantification_results, qc_results,
            image_metadata, overlay_path
        )
        
        return result
    
    def analyze_batch(self, input_file: str, 
                     image_dir: str,
                     output_dir: str = "outputs/results") -> List[Dict]:
        """
        Analyze multiple sites from input file.
        
        Args:
            input_file: Path to CSV/JSON file with site data
            image_dir: Directory containing site images
            output_dir: Directory for output files
            
        Returns:
            List of analysis results
        """
        # Load sites
        sites_df = self.data_loader.load_sites(input_file)
        sites_df = self.data_loader.validate_coordinates(sites_df)
        
        # Initialize image fetcher
        self.image_fetcher = ImageFetcher(image_dir)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        print(f"Processing {len(sites_df)} sites...")
        
        for idx, row in sites_df.iterrows():
            sample_id = row['sample_id']
            lat = row['lat']
            lon = row['lon']
            
            try:
                print(f"Processing site {idx+1}/{len(sites_df)}: {sample_id}")
                
                # Analyze site
                result = self.analyze_single_site(sample_id, lat, lon, image_dir=image_dir)
                results.append(result)
                
                # Save individual JSON if configured
                if self.config['output']['save_individual_jsons']:
                    json_path = output_path / f"{sample_id}.json"
                    with open(json_path, 'w') as f:
                        json.dump(result, f, indent=2, default=str)
                
            except Exception as e:
                print(f"Error processing {sample_id}: {e}")
                # Create error result
                error_result = self._create_error_result(sample_id, lat, lon, str(e))
                results.append(error_result)
        
        # Save batch results - ensure it's always saved
        batch_file = output_path / self.config['output']['batch_results_file']
        try:
            # Ensure directory exists
            batch_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write results with proper encoding
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str, ensure_ascii=False)
            
            file_size = batch_file.stat().st_size
            print(f"\n{'='*60}")
            print(f"Batch processing complete!")
            print(f"Results saved to: {batch_file}")
            print(f"Total results: {len(results)}")
            print(f"File size: {file_size} bytes")
            print(f"{'='*60}")
            
            if file_size == 0 and len(results) > 0:
                print("WARNING: batch_results.json is empty but results exist! Retrying...")
                # Retry save
                with open(batch_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, default=str, ensure_ascii=False)
                    
        except Exception as e:
            print(f"ERROR saving batch results: {e}")
            import traceback
            traceback.print_exc()
            # Try to save as backup
            try:
                backup_file = output_path / 'batch_results_backup.json'
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, default=str, ensure_ascii=False)
                print(f"Saved backup to {backup_file}")
            except:
                print("Failed to save backup as well!")
        
        return results
    
    def _create_structured_output(self, sample_id: Union[str, int], 
                                lat: float, lon: float,
                                detection_results: Dict, 
                                quantification_results, 
                                qc_results,
                                image_metadata: Dict,
                                overlay_path: Optional[Path] = None) -> Dict:
        """Create structured JSON output as specified in requirements."""
        
        # Extract values from result objects
        has_solar = detection_results.get('has_solar', False)
        confidence = detection_results.get('max_confidence', 0.0)
        
        # Build structured output
        output = {
            "sample_id": sample_id,
            "lat": lat,
            "lon": lon,
            "has_solar": has_solar,
            "confidence": round(confidence, 3),
            "pv_area_sqm_est": quantification_results.total_area_sqm,
            "buffer_radius_sqft": 1200,  # Standard buffer zone as per requirements (20m ≈ 1200 sq.ft radius)
            "panel_count_est": quantification_results.panel_count,
            "capacity_kw_est": quantification_results.capacity_kw,
            "qc_status": qc_results.status.value,
            "qc_notes": qc_results.reason_codes,
            "bbox_or_mask": str(overlay_path) if overlay_path else None,
            "image_metadata": {
                **image_metadata,
                "analysis_timestamp": datetime.now().isoformat(),
                "model_version": "YOLOv8_solar_v1.0"
            },
            "detailed_results": {
                "detection": {
                    "boxes": detection_results.get('boxes', []),
                    "confidences": detection_results.get('confidences', []),
                    "mean_confidence": float(np.mean(detection_results.get('confidences', [0.0]))),
                    "image_shape": detection_results.get('image_shape')
                },
                "quantification": {
                    "individual_panel_areas": quantification_results.individual_panel_areas,
                    "average_panel_area": quantification_results.average_panel_area,
                    "quantification_assumptions": self.quantifier.get_assumptions()
                },
                "quality_control": {
                    "qc_confidence_score": qc_results.confidence_score,
                    "image_quality_metrics": qc_results.image_quality_metrics,
                    "detection_quality_metrics": qc_results.detection_quality_metrics
                }
            }
        }
        
        return output
    
    def _create_error_result(self, sample_id: Union[str, int], 
                           lat: float, lon: float, 
                           error_message: str) -> Dict:
        """Create error result structure."""
        return {
            "sample_id": sample_id,
            "lat": lat,
            "lon": lon,
            "has_solar": False,
            "confidence": 0.0,
            "pv_area_sqm_est": 0.0,
            "buffer_radius_sqft": 1200,
            "panel_count_est": 0,
            "capacity_kw_est": 0.0,
            "qc_status": "NOT_VERIFIABLE",
            "qc_notes": ["processing_error"],
            "bbox_or_mask": None,
            "image_metadata": {
                "error": error_message,
                "analysis_timestamp": datetime.now().isoformat()
            },
            "detailed_results": {
                "error_details": error_message
            }
        }
    
    def _create_verification_error_result(self, sample_id: Union[str, int],
                                        lat: float, lon: float,
                                        verification_result: Dict,
                                        satellite_metadata: Dict) -> Dict:
        """Create error result for image verification mismatch."""
        return {
            "sample_id": sample_id,
            "lat": lat,
            "lon": lon,
            "has_solar": None,
            "confidence": 0.0,
            "pv_area_sqm_est": 0.0,
            "buffer_radius_sqft": 1200,
            "panel_count_est": 0,
            "capacity_kw_est": 0.0,
            "qc_status": "NOT_VERIFIABLE",
            "qc_notes": ["image_location_mismatch"],
            "bbox_or_mask": None,
            "image_metadata": {
                "source": "USER_UPLOAD",
                "satellite_source": satellite_metadata.get("source", "Unknown"),
                "image_verification": verification_result,
                "analysis_timestamp": datetime.now().isoformat()
            },
            "error": "Image verification failed: uploaded image does not match satellite image at given coordinates",
            "detailed_results": {
                "verification": verification_result,
                "satellite_metadata": satellite_metadata
            }
        }
    
    def get_pipeline_info(self) -> Dict:
        """Get information about the pipeline configuration."""
        return {
            "pipeline_version": "1.0.0",
            "model_info": self.detector.get_model_info(),
            "quantification_assumptions": self.quantifier.get_assumptions(),
            "qc_thresholds": {
                "min_resolution": self.qc_controller.min_resolution,
                "max_cloud_fraction": self.qc_controller.max_cloud_fraction,
                "min_confidence_has_solar": self.qc_controller.min_confidence_has_solar,
                "min_confidence_no_solar": self.qc_controller.min_confidence_no_solar
            },
            "configuration": self.config
        }


def create_sample_input_file(output_path: str = "data/input/sample_sites.csv") -> None:
    """Create a sample input file for testing."""
    import pandas as pd
    
    # Sample data (replace with actual coordinates)
    sample_data = [
        {"sample_id": 1001, "lat": 37.4419, "lon": -122.1430},  # Palo Alto area
        {"sample_id": 1002, "lat": 37.4517, "lon": -122.1481},  # Stanford area
        {"sample_id": 1003, "lat": 37.4462, "lon": -122.1609},  # Menlo Park area
        {"sample_id": 1004, "lat": 37.4337, "lon": -122.1145},  # Mountain View area
        {"sample_id": 1005, "lat": 37.4394, "lon": -122.0780},  # Sunnyvale area
    ]
    
    df = pd.DataFrame(sample_data)
    
    # Create directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Sample input file created at {output_path}")


if __name__ == "__main__":
    # Create sample input file
    create_sample_input_file()
    
    # Initialize pipeline
    pipeline = SolarAnalysisPipeline()
    
    # Print pipeline info
    info = pipeline.get_pipeline_info()
    print("Solar Panel Analysis Pipeline Initialized")
    print(f"Model: {info['model_info']['model_type']}")
    print(f"Confidence Threshold: {info['model_info']['confidence_threshold']}")
    print(f"Quantification Assumption: {info['quantification_assumptions']['watts_per_sqm']} Wp/m²")