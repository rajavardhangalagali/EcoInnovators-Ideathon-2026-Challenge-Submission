"""
Core data processing utilities for solar panel detection.
Handles coordinate processing, GPS buffer search, and image fetching.
"""

import math
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from geopy.distance import geodesic
from pathlib import Path
import json


class CoordinateProcessor:
    """Handles coordinate transformations and buffer search around GPS points."""
    
    def __init__(self, buffer_meters: float = 20.0):
        """
        Initialize coordinate processor.
        
        Args:
            buffer_meters: Buffer distance in meters for GPS jitter handling
        """
        self.buffer_meters = buffer_meters
    
    def degrees_per_meter(self, latitude: float) -> Tuple[float, float]:
        """
        Convert meters to degrees at given latitude.
        
        Args:
            latitude: Latitude in degrees
            
        Returns:
            Tuple of (lat_degrees_per_meter, lon_degrees_per_meter)
        """
        # Approximate conversion (more accurate for small distances)
        lat_deg_per_m = 1 / 111111.0  # ~1 degree = 111km
        lon_deg_per_m = 1 / (111111.0 * math.cos(math.radians(latitude)))
        
        return lat_deg_per_m, lon_deg_per_m
    
    def generate_buffer_grid(self, lat: float, lon: float, 
                           grid_points: int = 5) -> List[Tuple[float, float]]:
        """
        Generate a grid of coordinates around the input point within buffer radius.
        
        Args:
            lat: Central latitude
            lon: Central longitude 
            grid_points: Number of points per dimension in the grid
            
        Returns:
            List of (lat, lon) tuples within buffer
        """
        lat_deg_per_m, lon_deg_per_m = self.degrees_per_meter(lat)
        
        # Buffer in degrees
        lat_buffer = self.buffer_meters * lat_deg_per_m
        lon_buffer = self.buffer_meters * lon_deg_per_m
        
        # Generate grid
        lat_range = np.linspace(lat - lat_buffer, lat + lat_buffer, grid_points)
        lon_range = np.linspace(lon - lon_buffer, lon + lon_buffer, grid_points)
        
        grid_points_list = []
        for lat_point in lat_range:
            for lon_point in lon_range:
                # Check if point is within circular buffer
                distance = geodesic((lat, lon), (lat_point, lon_point)).meters
                if distance <= self.buffer_meters:
                    grid_points_list.append((lat_point, lon_point))
        
        return grid_points_list
    
    def find_best_coordinate(self, lat: float, lon: float, 
                           confidence_scores: List[float]) -> Tuple[float, float, float]:
        """
        Find the coordinate with highest confidence within buffer.
        
        Args:
            lat: Central latitude
            lon: Central longitude
            confidence_scores: Confidence scores for each grid point
            
        Returns:
            Tuple of (best_lat, best_lon, best_confidence)
        """
        grid_coords = self.generate_buffer_grid(lat, lon)
        
        if len(confidence_scores) != len(grid_coords):
            # If no confidence scores provided, return center
            return lat, lon, 0.0
        
        # Find best coordinate
        best_idx = np.argmax(confidence_scores)
        best_lat, best_lon = grid_coords[best_idx]
        best_confidence = confidence_scores[best_idx]
        
        return best_lat, best_lon, best_confidence


class DataLoader:
    """Handles loading and parsing input data (CSV/JSON)."""
    
    def load_sites(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load sites from CSV, JSON, or Excel (.xlsx/.xls) file.
        
        Args:
            file_path: Path to input file
            
        Returns:
            DataFrame with columns: sample_id, lat, lon
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.json':
            df = pd.read_json(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, engine='openpyxl')
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}. Supported formats: .csv, .json, .xlsx, .xls")
        
        # Validate required columns
        required_cols = ['sample_id', 'lat', 'lon']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    
    def validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate coordinate values are within valid ranges.
        
        Args:
            df: DataFrame with lat, lon columns
            
        Returns:
            Filtered DataFrame with valid coordinates
        """
        # Filter valid latitude (-90 to 90)
        valid_lat = (df['lat'] >= -90) & (df['lat'] <= 90)
        
        # Filter valid longitude (-180 to 180)  
        valid_lon = (df['lon'] >= -180) & (df['lon'] <= 180)
        
        valid_coords = valid_lat & valid_lon
        
        if not valid_coords.all():
            invalid_count = (~valid_coords).sum()
            print(f"Warning: Filtered out {invalid_count} sites with invalid coordinates")
        
        return df[valid_coords].copy()


class ImageFetcher:
    """Handles image fetching and management for dataset images."""
    
    def __init__(self, image_dir: Union[str, Path]):
        """
        Initialize image fetcher.
        
        Args:
            image_dir: Directory containing dataset images
        """
        self.image_dir = Path(image_dir)
    
    def get_image_path(self, sample_id: Union[str, int]) -> Optional[Path]:
        """
        Get image path for given sample ID.
        
        Args:
            sample_id: Sample identifier
            
        Returns:
            Path to image file if exists, None otherwise
        """
        # Common image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        
        for ext in extensions:
            img_path = self.image_dir / f"{sample_id}{ext}"
            if img_path.exists():
                return img_path
        
        return None
    
    def get_image_metadata(self, image_path: Path) -> Dict:
        """
        Extract basic metadata from image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with image metadata including capture_date if available
        """
        if not image_path.exists():
            return {}
        
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
            import os
            from datetime import datetime
            
            img = Image.open(image_path)
            stat = os.stat(image_path)
            
            metadata = {
                "source": "DATASET",
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "file_size_bytes": stat.st_size,
                "modified_date": stat.st_mtime
            }
            
            # Extract EXIF data if available (multiple methods for compatibility)
            capture_date = None
            
            # Method 1: Try PIL ExifTags
            try:
                if hasattr(img, 'getexif') and img.getexif() is not None:
                    exif = img.getexif()
                    # Look for DateTime (tag 306) or DateTimeOriginal (tag 36867)
                    for tag_id, value in exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if tag in ['DateTime', 'DateTimeOriginal']:
                            capture_date = str(value)
                            break
            except Exception:
                pass
            
            # Method 2: Try _getexif (older PIL versions)
            if not capture_date:
                try:
                    if hasattr(img, '_getexif') and img._getexif() is not None:
                        exif = img._getexif()
                        if exif and 306 in exif:  # DateTime tag
                            capture_date = str(exif[306])
                except Exception:
                    pass
            
            # Method 3: Use file modification time as fallback
            if not capture_date:
                try:
                    mtime = stat.st_mtime
                    capture_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
                except Exception:
                    pass
            
            if capture_date:
                metadata["capture_date"] = capture_date
            
            return metadata
            
        except Exception as e:
            print(f"Warning: Could not extract metadata from {image_path}: {e}")
            return {"source": "DATASET", "error": str(e)}


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    import yaml
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        # Return default config
        return {
            'detection': {'buffer_meters': 20},
            'quantification': {'watts_per_sqm': 190, 'meters_per_pixel': 0.3},
            'model': {'confidence_threshold': 0.5}
        }