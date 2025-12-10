"""
Satellite/Map image fetching module for production-ready verification.
Fetches satellite images from various providers using coordinates.
"""

import requests
import numpy as np
from PIL import Image
import io
from typing import Optional, Dict, Tuple
from pathlib import Path
import os
from enum import Enum


class SatelliteProvider(Enum):
    """Supported satellite/map image providers."""
    ESRI = "esri"
    GOOGLE_MAPS = "google_maps"
    MAPBOX = "mapbox"
    OPENSTREETMAP = "openstreetmap"
    PLANET = "planet"


class SatelliteImageFetcher:
    """Fetches satellite/map images from various providers."""
    
    def __init__(self, provider: SatelliteProvider = SatelliteProvider.GOOGLE_MAPS,
                 api_key: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize satellite image fetcher.
        
        Args:
            provider: Satellite/map provider to use
            api_key: API key for the provider (or from env var)
            cache_dir: Directory to cache fetched images
        """
        self.provider = provider
        self.api_key = api_key or self._get_api_key_from_env()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variable."""
        env_var_map = {
            SatelliteProvider.GOOGLE_MAPS: "GOOGLE_MAPS_API_KEY",
            SatelliteProvider.MAPBOX: "MAPBOX_API_KEY",
            SatelliteProvider.OPENSTREETMAP: None,  # No API key needed
            SatelliteProvider.PLANET: "PLANET_API_KEY"
        }
        
        env_var = env_var_map.get(self.provider)
        if env_var:
            return os.getenv(env_var)
        return None
    
    def fetch_image(self, lat: float, lon: float, 
                   zoom: int = 20,
                   width: int = 640,
                   height: int = 640,
                   use_cache: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Fetch satellite image at given coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            zoom: Zoom level (1-20, higher = more detail)
            width: Image width in pixels
            height: Image height in pixels
            use_cache: Whether to use cached images if available
            
        Returns:
            Tuple of (image_array, metadata_dict)
        """
        # Check cache first
        if use_cache and self.cache_dir:
            cache_path = self._get_cache_path(lat, lon, zoom, width, height)
            if cache_path.exists():
                img = Image.open(cache_path)
                return np.array(img), {"source": "cache", "cache_path": str(cache_path)}
        
        # Fetch from provider
        if self.provider == SatelliteProvider.ESRI:
            image_array, metadata = self._fetch_esri(lat, lon, zoom, width, height)
        elif self.provider == SatelliteProvider.GOOGLE_MAPS:
            image_array, metadata = self._fetch_google_maps(lat, lon, zoom, width, height)
        elif self.provider == SatelliteProvider.MAPBOX:
            image_array, metadata = self._fetch_mapbox(lat, lon, zoom, width, height)
        elif self.provider == SatelliteProvider.OPENSTREETMAP:
            image_array, metadata = self._fetch_openstreetmap(lat, lon, zoom, width, height)
        elif self.provider == SatelliteProvider.PLANET:
            image_array, metadata = self._fetch_planet(lat, lon, zoom, width, height)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        # Cache the image
        if use_cache and self.cache_dir:
            cache_path = self._get_cache_path(lat, lon, zoom, width, height)
            Image.fromarray(image_array).save(cache_path)
            metadata["cache_path"] = str(cache_path)
        
        return image_array, metadata
    
    def _fetch_google_maps(self, lat: float, lon: float, 
                         zoom: int, width: int, height: int) -> Tuple[np.ndarray, Dict]:
        """
        Fetch image from Google Maps Static API.
        
        API Documentation: https://developers.google.com/maps/documentation/maps-static
        """
        if not self.api_key:
            raise ValueError("Google Maps API key required. Set GOOGLE_MAPS_API_KEY env var.")
        
        # Google Maps Static API endpoint
        url = "https://maps.googleapis.com/maps/api/staticmap"
        
        params = {
            "center": f"{lat},{lon}",
            "zoom": zoom,
            "size": f"{width}x{height}",
            "maptype": "satellite",  # Use satellite imagery
            "key": self.api_key
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Convert to numpy array
        img = Image.open(io.BytesIO(response.content))
        image_array = np.array(img)
        
        metadata = {
            "source": "Google Maps",
            "provider": "google_maps",
            "zoom": zoom,
            "width": width,
            "height": height,
            "coordinates": {"lat": lat, "lon": lon}
        }
        
        return image_array, metadata
    
    def _fetch_mapbox(self, lat: float, lon: float,
                     zoom: int, width: int, height: int) -> Tuple[np.ndarray, Dict]:
        """
        Fetch image from Mapbox Static Images API.
        
        API Documentation: https://docs.mapbox.com/api/maps/static-images/
        """
        if not self.api_key:
            raise ValueError("Mapbox API key required. Set MAPBOX_API_KEY env var.")
        
        # Mapbox uses style ID and overlay format
        # Format: mapbox.satellite/{lon},{lat},{zoom}/{width}x{height}
        url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},{zoom}/{width}x{height}"
        
        params = {
            "access_token": self.api_key
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        img = Image.open(io.BytesIO(response.content))
        image_array = np.array(img)
        
        metadata = {
            "source": "Mapbox",
            "provider": "mapbox",
            "zoom": zoom,
            "width": width,
            "height": height,
            "coordinates": {"lat": lat, "lon": lon}
        }
        
        return image_array, metadata
    
    def _fetch_esri(self, lat: float, lon: float,
                   zoom: int, width: int, height: int) -> Tuple[np.ndarray, Dict]:
        """
        Fetch image from ESRI ArcGIS World Imagery service.
        Completely FREE - no API key, no signup, no credit card required.
        
        Service: https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer
        """
        import math
        
        # Convert lat/lon to tile coordinates
        n = 2.0 ** zoom
        tile_x = int((lon + 180.0) / 360.0 * n)
        tile_y = int((1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
        
        # ESRI World Imagery tile service endpoint
        url = f"https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{tile_y}/{tile_x}"
        
        # No authentication needed - ESRI provides this as a free public service
        response = requests.get(url, headers={"User-Agent": "SolarPanelDetection/1.0"})
        response.raise_for_status()
        
        # Convert to numpy array
        img = Image.open(io.BytesIO(response.content))
        
        # Resize to requested dimensions (ESRI tiles are 256x256)
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        image_array = np.array(img)
        
        metadata = {
            "source": "ESRI ArcGIS World Imagery",
            "provider": "esri",
            "zoom": zoom,
            "width": width,
            "height": height,
            "coordinates": {"lat": lat, "lon": lon},
            "tile": {"x": tile_x, "y": tile_y}
        }
        
        return image_array, metadata
    
    def _fetch_openstreetmap(self, lat: float, lon: float,
                           zoom: int, width: int, height: int) -> Tuple[np.ndarray, Dict]:
        """
        Fetch image from OpenStreetMap (free, no API key needed).
        Note: Lower quality than paid providers.
        """
        # OpenStreetMap tile server
        # Convert lat/lon to tile coordinates
        import math
        
        n = 2.0 ** zoom
        tile_x = int((lon + 180.0) / 360.0 * n)
        tile_y = int((1.0 - math.log(math.tan(math.radians(lat)) + 
                                   1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
        
        # Fetch tile (this is simplified - full implementation would fetch multiple tiles)
        url = f"https://tile.openstreetmap.org/{zoom}/{tile_x}/{tile_y}.png"
        
        response = requests.get(url, headers={"User-Agent": "SolarPanelDetection/1.0"})
        response.raise_for_status()
        
        img = Image.open(io.BytesIO(response.content))
        # Resize to requested dimensions
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        image_array = np.array(img)
        
        metadata = {
            "source": "OpenStreetMap",
            "provider": "openstreetmap",
            "zoom": zoom,
            "width": width,
            "height": height,
            "coordinates": {"lat": lat, "lon": lon}
        }
        
        return image_array, metadata
    
    def _fetch_planet(self, lat: float, lon: float,
                     zoom: int, width: int, height: int) -> Tuple[np.ndarray, Dict]:
        """
        Fetch image from Planet Labs API (high quality, paid).
        
        API Documentation: https://developers.planet.com/
        """
        if not self.api_key:
            raise ValueError("Planet API key required. Set PLANET_API_KEY env var.")
        
        # Planet Labs API implementation
        # This is a placeholder - actual implementation would use Planet's API
        raise NotImplementedError("Planet Labs API integration not yet implemented")
    
    def _get_cache_path(self, lat: float, lon: float, 
                       zoom: int, width: int, height: int) -> Path:
        """Generate cache file path for given parameters."""
        cache_key = f"{lat:.6f}_{lon:.6f}_{zoom}_{width}x{height}.png"
        return self.cache_dir / cache_key


def create_satellite_fetcher(provider_name: str = "google_maps",
                            api_key: Optional[str] = None,
                            cache_dir: Optional[str] = "cache/satellite_images") -> SatelliteImageFetcher:
    """
    Factory function to create satellite image fetcher.
    
    Args:
        provider_name: Name of provider ("google_maps", "mapbox", "openstreetmap")
        api_key: API key (or from env var)
        cache_dir: Cache directory path
        
    Returns:
        SatelliteImageFetcher instance
    """
    provider_map = {
        "esri": SatelliteProvider.ESRI,
        "google_maps": SatelliteProvider.GOOGLE_MAPS,
        "mapbox": SatelliteProvider.MAPBOX,
        "openstreetmap": SatelliteProvider.OPENSTREETMAP,
        "planet": SatelliteProvider.PLANET
    }
    
    provider = provider_map.get(provider_name.lower())
    if not provider:
        raise ValueError(f"Unknown provider: {provider_name}. Choose from: {list(provider_map.keys())}")
    
    return SatelliteImageFetcher(provider=provider, api_key=api_key, cache_dir=cache_dir)

