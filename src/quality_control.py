"""
Quality control and verification system for solar panel detection.
Determines VERIFIABLE/NOT_VERIFIABLE status with reason codes.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
from dataclasses import dataclass


class QCStatus(Enum):
    """Quality control status enumeration."""
    VERIFIABLE = "VERIFIABLE"
    NOT_VERIFIABLE = "NOT_VERIFIABLE"


@dataclass
class QCResult:
    """Quality control analysis result."""
    status: QCStatus
    confidence_score: float
    reason_codes: List[str]
    image_quality_metrics: Dict[str, float]
    detection_quality_metrics: Dict[str, float]


class ImageQualityAnalyzer:
    """Analyzes image quality metrics for QC assessment."""
    
    def __init__(self):
        """Initialize image quality analyzer."""
        pass
    
    def calculate_sharpness(self, image: np.ndarray) -> float:
        """
        Calculate image sharpness using Laplacian variance.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Sharpness score (higher = sharper)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Calculate Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(laplacian_var)
    
    def calculate_brightness_contrast(self, image: np.ndarray) -> Tuple[float, float]:
        """
        Calculate brightness and contrast metrics.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (brightness, contrast) scores
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Brightness as mean intensity
        brightness = np.mean(gray) / 255.0
        
        # Contrast as standard deviation
        contrast = np.std(gray) / 255.0
        
        return float(brightness), float(contrast)
    
    def detect_cloud_occlusion(self, image: np.ndarray) -> float:
        """
        Estimate cloud/occlusion coverage using brightness and texture analysis.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Estimated cloud fraction (0.0 to 1.0)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Threshold for very bright regions (potential clouds)
        bright_threshold = np.percentile(gray, 90)
        cloud_mask = gray > bright_threshold
        
        # Calculate cloud fraction
        cloud_fraction = np.sum(cloud_mask) / gray.size
        
        return float(min(cloud_fraction, 1.0))
    
    def calculate_noise_level(self, image: np.ndarray) -> float:
        """
        Estimate noise level in the image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Noise level estimate
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Use high-pass filter to estimate noise
        kernel = np.array([[-1, -1, -1], 
                          [-1,  8, -1], 
                          [-1, -1, -1]])
        
        filtered = cv2.filter2D(gray, -1, kernel)
        noise_level = np.std(filtered) / 255.0
        
        return float(noise_level)
    
    def analyze_image_quality(self, image: Union[np.ndarray, str]) -> Dict[str, float]:
        """
        Comprehensive image quality analysis.
        
        Args:
            image: Image as numpy array or path to image file
            
        Returns:
            Dictionary of quality metrics
        """
        # Load image if path provided
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image from {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image
        
        # Calculate all metrics
        sharpness = self.calculate_sharpness(img)
        brightness, contrast = self.calculate_brightness_contrast(img)
        cloud_fraction = self.detect_cloud_occlusion(img)
        noise_level = self.calculate_noise_level(img)
        
        # Image resolution metrics
        height, width = img.shape[:2]
        resolution_score = min(height, width) / 1024.0  # Normalized to 1024px
        
        return {
            'sharpness': sharpness,
            'brightness': brightness,
            'contrast': contrast,
            'cloud_fraction': cloud_fraction,
            'noise_level': noise_level,
            'resolution_score': resolution_score,
            'width': width,
            'height': height
        }


class DetectionQualityAnalyzer:
    """Analyzes detection results for quality assessment."""
    
    def analyze_detection_quality(self, detection_results: Dict) -> Dict[str, float]:
        """
        Analyze quality of detection results.
        
        Args:
            detection_results: Results from SolarPanelDetector
            
        Returns:
            Dictionary of detection quality metrics
        """
        boxes = detection_results.get('boxes', [])
        confidences = detection_results.get('confidences', [])
        max_confidence = detection_results.get('max_confidence', 0.0)
        panel_count = detection_results.get('panel_count', 0)
        
        metrics = {
            'max_confidence': max_confidence,
            'mean_confidence': 0.0,
            'confidence_variance': 0.0,
            'detection_consistency': 0.0,
            'spatial_distribution': 0.0,
            'panel_count': panel_count
        }
        
        if not confidences:
            return metrics
        
        # Confidence statistics
        metrics['mean_confidence'] = float(np.mean(confidences))
        metrics['confidence_variance'] = float(np.var(confidences))
        
        # Detection consistency (low variance is good)
        if len(confidences) > 1:
            metrics['detection_consistency'] = 1.0 - min(metrics['confidence_variance'], 1.0)
        else:
            metrics['detection_consistency'] = 1.0
        
        # Spatial distribution analysis
        if boxes:
            metrics['spatial_distribution'] = self._analyze_spatial_distribution(boxes)
        
        return metrics
    
    def _analyze_spatial_distribution(self, boxes: List[List[float]]) -> float:
        """
        Analyze spatial distribution of detected panels.
        
        Args:
            boxes: List of bounding boxes
            
        Returns:
            Spatial distribution score (0-1, higher = more organized)
        """
        if len(boxes) < 2:
            return 1.0
        
        # Calculate centers of boxes
        centers = []
        for box in boxes:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append([center_x, center_y])
        
        centers = np.array(centers)
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.linalg.norm(centers[i] - centers[j])
                distances.append(dist)
        
        if not distances:
            return 1.0
        
        # Regular spacing indicates organized installation
        distance_cv = np.std(distances) / np.mean(distances) if np.mean(distances) > 0 else 1.0
        organization_score = max(0.0, 1.0 - distance_cv)
        
        return float(organization_score)


class QualityController:
    """Main quality control system for solar panel detection."""
    
    def __init__(self, config: Dict):
        """
        Initialize quality controller.
        
        Args:
            config: Configuration dictionary with thresholds
        """
        self.config = config
        self.qc_config = config.get('qc', {})
        self.detection_config = config.get('detection', {})
        self.reason_codes = config.get('reason_codes', {})
        
        # Initialize analyzers
        self.image_analyzer = ImageQualityAnalyzer()
        self.detection_analyzer = DetectionQualityAnalyzer()
        
        # QC thresholds
        self.min_resolution = self.qc_config.get('min_resolution', 512)
        self.max_cloud_fraction = self.qc_config.get('max_cloud_fraction', 0.4)
        self.min_roof_visibility = self.qc_config.get('min_roof_visibility', 0.6)
        self.min_confidence_has_solar = self.detection_config.get('min_confidence_has_solar', 0.8)
        self.min_confidence_no_solar = self.detection_config.get('min_confidence_no_solar', 0.85)
    
    def assess_quality(self, image: Union[np.ndarray, str], 
                      detection_results: Dict,
                      image_metadata: Optional[Dict] = None) -> QCResult:
        """
        Perform comprehensive quality control assessment.
        
        Args:
            image: Input image (numpy array or path)
            detection_results: Detection results from SolarPanelDetector
            image_metadata: Optional image metadata
            
        Returns:
            QCResult with status, confidence, and reason codes
        """
        # Analyze image quality
        image_metrics = self.image_analyzer.analyze_image_quality(image)
        
        # Analyze detection quality  
        detection_metrics = self.detection_analyzer.analyze_detection_quality(detection_results)
        
        # Determine QC status and reason codes
        status, reason_codes, qc_confidence = self._determine_qc_status(
            image_metrics, detection_metrics, detection_results
        )
        
        return QCResult(
            status=status,
            confidence_score=qc_confidence,
            reason_codes=reason_codes,
            image_quality_metrics=image_metrics,
            detection_quality_metrics=detection_metrics
        )
    
    def _determine_qc_status(self, image_metrics: Dict, 
                           detection_metrics: Dict, 
                           detection_results: Dict) -> Tuple[QCStatus, List[str], float]:
        """
        Determine QC status based on metrics.
        
        Returns:
            Tuple of (status, reason_codes, confidence_score)
        """
        reason_codes = []
        verifiable_points = 0
        total_points = 0
        
        has_solar = detection_results.get('has_solar', False)
        max_confidence = detection_results.get('max_confidence', 0.0)
        
        # Image quality checks
        total_points += 4
        
        # 1. Resolution check
        min_dimension = min(image_metrics['width'], image_metrics['height'])
        if min_dimension >= self.min_resolution:
            verifiable_points += 1
            if has_solar:
                reason_codes.append("clear roof view")
        else:
            reason_codes.append("low-resolution imagery")
        
        # 2. Cloud/occlusion check
        if image_metrics['cloud_fraction'] <= self.max_cloud_fraction:
            verifiable_points += 1
        else:
            reason_codes.append("heavy cloud/occlusion")
        
        # 3. Sharpness check
        if image_metrics['sharpness'] > 100:  # Reasonable threshold
            verifiable_points += 1
        else:
            reason_codes.append("poor image quality")
        
        # 4. Contrast check
        if image_metrics['contrast'] > 0.1:  # Sufficient contrast
            verifiable_points += 1
        else:
            reason_codes.append("poor image quality")
        
        # Detection quality checks
        total_points += 3
        
        # 5. Confidence check
        if has_solar and max_confidence >= self.min_confidence_has_solar:
            verifiable_points += 1
            reason_codes.append("high confidence detection")
        elif not has_solar and max_confidence >= self.min_confidence_no_solar:
            verifiable_points += 1
        else:
            reason_codes.append("conflicting signals")
        
        # 6. Detection consistency
        if detection_metrics['detection_consistency'] > 0.7:
            verifiable_points += 1
        else:
            if has_solar:
                reason_codes.append("conflicting signals")
        
        # 7. Spatial organization (for solar installations)
        if has_solar:
            if detection_metrics['spatial_distribution'] > 0.6:
                verifiable_points += 1
                reason_codes.append("distinct module grid")
            else:
                reason_codes.append("poor spatial organization")
        else:
            verifiable_points += 1  # N/A for no-solar cases
        
        # Additional positive indicators for solar installations
        if has_solar:
            panel_count = detection_results.get('panel_count', 0)
            if panel_count >= 4:  # Reasonable installation size
                reason_codes.append("rectilinear array")
            
            # Check for regular patterns (mock implementation)
            if detection_metrics.get('spatial_distribution', 0) > 0.8:
                reason_codes.append("racking shadows")
        
        # Calculate confidence score
        qc_confidence = verifiable_points / total_points
        
        # Determine final status
        if qc_confidence >= 0.7:  # At least 70% of checks pass
            status = QCStatus.VERIFIABLE
        else:
            status = QCStatus.NOT_VERIFIABLE
        
        # Filter reason codes to remove duplicates and keep most relevant
        reason_codes = list(set(reason_codes))
        
        # Prioritize reason codes
        if status == QCStatus.VERIFIABLE:
            positive_codes = [code for code in reason_codes 
                            if code in self.reason_codes.get('verifiable', [])]
            reason_codes = positive_codes[:3]  # Keep top 3
        else:
            negative_codes = [code for code in reason_codes 
                            if code in self.reason_codes.get('not_verifiable', [])]
            reason_codes = negative_codes[:3]  # Keep top 3
        
        return status, reason_codes, round(qc_confidence, 3)