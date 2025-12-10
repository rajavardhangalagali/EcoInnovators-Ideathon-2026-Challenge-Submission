"""
Quantification engine for solar panel analysis.
Calculates panel count, area estimation, and capacity estimates.
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class QuantificationResult:
    """Results from solar panel quantification."""
    panel_count: int
    total_area_sqm: float
    capacity_kw: float
    individual_panel_areas: List[float]
    average_panel_area: float
    confidence_weighted_area: float


class SolarQuantifier:
    """Quantifies solar panel installations from detection results."""
    
    def __init__(self, 
                 watts_per_sqm: float = 190.0,
                 meters_per_pixel: float = 0.3,
                 min_panel_area_sqm: float = 1.0,
                 typical_panel_area_sqm: float = 2.0):
        """
        Initialize solar quantifier.
        
        Args:
            watts_per_sqm: Solar panel efficiency (Wp/m²). Standard assumption: 190 Wp/m²
            meters_per_pixel: Pixel to meter conversion factor
            min_panel_area_sqm: Minimum area to count as a valid panel
            typical_panel_area_sqm: Typical residential solar panel area for validation
        """
        self.watts_per_sqm = watts_per_sqm
        self.meters_per_pixel = meters_per_pixel
        self.min_panel_area_sqm = min_panel_area_sqm
        self.typical_panel_area_sqm = typical_panel_area_sqm
        
    def calculate_box_area_sqm(self, box: List[float], image_shape: Tuple[int, int]) -> float:
        """
        Calculate area of bounding box in square meters.
        
        Args:
            box: Bounding box coordinates [x1, y1, x2, y2] in pixels
            image_shape: Image shape (height, width)
            
        Returns:
            Area in square meters
        """
        x1, y1, x2, y2 = box
        
        # Calculate area in pixels
        width_pixels = abs(x2 - x1)
        height_pixels = abs(y2 - y1)
        area_pixels = width_pixels * height_pixels
        
        # Convert to square meters
        area_sqm = area_pixels * (self.meters_per_pixel ** 2)
        
        return area_sqm
    
    def estimate_panel_area_from_boxes(self, boxes: List[List[float]], 
                                     confidences: List[float],
                                     image_shape: Tuple[int, int]) -> Tuple[float, List[float]]:
        """
        Estimate total panel area from bounding boxes.
        
        Args:
            boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
            confidences: Confidence scores for each box
            image_shape: Image shape (height, width)
            
        Returns:
            Tuple of (total_area_sqm, individual_areas_sqm)
        """
        if not boxes:
            return 0.0, []
        
        individual_areas = []
        
        for box, confidence in zip(boxes, confidences):
            area_sqm = self.calculate_box_area_sqm(box, image_shape)
            
            # Apply confidence weighting
            weighted_area = area_sqm * confidence
            
            # Filter out unrealistically small areas
            if weighted_area >= self.min_panel_area_sqm:
                individual_areas.append(weighted_area)
        
        total_area = sum(individual_areas)
        return total_area, individual_areas
    
    def validate_panel_estimates(self, panel_count: int, 
                               total_area_sqm: float,
                               individual_areas: List[float]) -> Dict[str, bool]:
        """
        Validate panel estimates for reasonableness.
        
        Args:
            panel_count: Number of detected panels
            total_area_sqm: Total estimated area
            individual_areas: Individual panel areas
            
        Returns:
            Dictionary of validation results
        """
        validations = {
            'reasonable_count': 1 <= panel_count <= 100,  # Reasonable for residential
            'reasonable_total_area': 5.0 <= total_area_sqm <= 500.0,  # 5-500 m²
            'consistent_panel_sizes': True,
            'no_oversized_panels': True
        }
        
        if individual_areas:
            avg_area = np.mean(individual_areas)
            std_area = np.std(individual_areas)
            
            # Check for consistent panel sizes (CV < 0.5)
            if avg_area > 0:
                cv = std_area / avg_area
                validations['consistent_panel_sizes'] = cv < 0.5
            
            # Check for unrealistically large panels (>10x typical)
            max_reasonable_area = self.typical_panel_area_sqm * 10
            validations['no_oversized_panels'] = all(area <= max_reasonable_area 
                                                   for area in individual_areas)
        
        return validations
    
    def calculate_capacity_kw(self, area_sqm: float) -> float:
        """
        Calculate estimated capacity in kW.
        
        Args:
            area_sqm: Total panel area in square meters
            
        Returns:
            Estimated capacity in kW
        """
        # Convert Wp/m² to kW/m²
        kw_per_sqm = self.watts_per_sqm / 1000.0
        capacity_kw = area_sqm * kw_per_sqm
        
        return round(capacity_kw, 2)
    
    def quantify(self, detection_results: Dict, 
                image_metadata: Optional[Dict] = None) -> QuantificationResult:
        """
        Perform complete quantification analysis.
        
        Args:
            detection_results: Results from SolarPanelDetector.detect()
            image_metadata: Optional metadata containing scale information
            
        Returns:
            QuantificationResult with all calculated metrics
        """
        # Extract detection data
        boxes = detection_results.get('boxes', [])
        confidences = detection_results.get('confidences', [])
        image_shape = detection_results.get('image_shape')
        
        # Update meters per pixel if provided in metadata
        if image_metadata and 'meters_per_pixel' in image_metadata:
            self.meters_per_pixel = image_metadata['meters_per_pixel']
        
        # Calculate panel count
        panel_count = len(boxes)
        
        if panel_count == 0:
            return QuantificationResult(
                panel_count=0,
                total_area_sqm=0.0,
                capacity_kw=0.0,
                individual_panel_areas=[],
                average_panel_area=0.0,
                confidence_weighted_area=0.0
            )
        
        # Estimate areas
        total_area, individual_areas = self.estimate_panel_area_from_boxes(
            boxes, confidences, image_shape
        )
        
        # Calculate capacity
        capacity_kw = self.calculate_capacity_kw(total_area)
        
        # Calculate statistics
        average_panel_area = np.mean(individual_areas) if individual_areas else 0.0
        
        # Confidence-weighted area (different from total for validation)
        confidence_weighted_area = sum(area * conf for area, conf 
                                     in zip(individual_areas, confidences[:len(individual_areas)]))
        if len(individual_areas) > 0:
            confidence_weighted_area /= len(individual_areas)
        else:
            confidence_weighted_area = 0.0
        
        return QuantificationResult(
            panel_count=panel_count,
            total_area_sqm=round(total_area, 2),
            capacity_kw=capacity_kw,
            individual_panel_areas=individual_areas,
            average_panel_area=round(average_panel_area, 2),
            confidence_weighted_area=round(confidence_weighted_area, 2)
        )
    
    def get_assumptions(self) -> Dict:
        """
        Get the assumptions used in quantification.
        
        Returns:
            Dictionary of assumptions and parameters
        """
        return {
            'watts_per_sqm': self.watts_per_sqm,
            'meters_per_pixel': self.meters_per_pixel,
            'min_panel_area_sqm': self.min_panel_area_sqm,
            'typical_panel_area_sqm': self.typical_panel_area_sqm,
            'assumptions': {
                'efficiency_source': 'Standard residential solar panel efficiency',
                'scale_method': 'Pixel-to-meter conversion from image metadata',
                'area_method': 'Bounding box area estimation',
                'capacity_formula': 'area_sqm × watts_per_sqm / 1000'
            }
        }


class AdvancedQuantifier(SolarQuantifier):
    """Advanced quantification with segmentation mask support."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def calculate_mask_area_sqm(self, mask: np.ndarray, 
                              image_shape: Tuple[int, int]) -> float:
        """
        Calculate area from segmentation mask.
        
        Args:
            mask: Binary mask of solar panels
            image_shape: Image shape (height, width)
            
        Returns:
            Area in square meters
        """
        # Count positive pixels
        panel_pixels = np.sum(mask > 0)
        
        # Convert to square meters
        area_sqm = panel_pixels * (self.meters_per_pixel ** 2)
        
        return area_sqm
    
    def estimate_panel_count_from_mask(self, mask: np.ndarray) -> int:
        """
        Estimate panel count from segmentation mask using connected components.
        
        Args:
            mask: Binary segmentation mask
            
        Returns:
            Estimated number of individual panels
        """
        import cv2
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        
        # Subtract 1 for background
        panel_count = max(0, num_labels - 1)
        
        return panel_count
    
    def quantify_from_mask(self, mask: np.ndarray, 
                          image_shape: Tuple[int, int],
                          image_metadata: Optional[Dict] = None) -> QuantificationResult:
        """
        Quantify from segmentation mask (more accurate than bounding boxes).
        
        Args:
            mask: Binary segmentation mask
            image_shape: Image shape (height, width)
            image_metadata: Optional metadata
            
        Returns:
            QuantificationResult based on mask analysis
        """
        # Update scale if provided
        if image_metadata and 'meters_per_pixel' in image_metadata:
            self.meters_per_pixel = image_metadata['meters_per_pixel']
        
        # Calculate total area
        total_area = self.calculate_mask_area_sqm(mask, image_shape)
        
        # Estimate panel count
        panel_count = self.estimate_panel_count_from_mask(mask)
        
        # Calculate capacity
        capacity_kw = self.calculate_capacity_kw(total_area)
        
        # Estimate individual panel areas (approximation)
        if panel_count > 0:
            avg_panel_area = total_area / panel_count
            individual_areas = [avg_panel_area] * panel_count
        else:
            avg_panel_area = 0.0
            individual_areas = []
        
        return QuantificationResult(
            panel_count=panel_count,
            total_area_sqm=round(total_area, 2),
            capacity_kw=capacity_kw,
            individual_panel_areas=individual_areas,
            average_panel_area=round(avg_panel_area, 2),
            confidence_weighted_area=round(total_area, 2)  # Full area for masks
        )