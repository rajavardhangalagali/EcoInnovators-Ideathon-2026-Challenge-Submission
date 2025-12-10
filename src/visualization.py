"""
Explainability and visualization module for solar panel detection.
Generates overlay images with bounding boxes, confidence scores, and legends.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json


class VisualizationGenerator:
    """Generates explanatory visualizations for solar panel detections."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize visualization generator.
        
        Args:
            config: Configuration dictionary for visualization settings
        """
        self.config = config or {}
        
        # Visualization settings
        self.box_color = (0, 255, 0)  # Green for solar panels
        self.box_thickness = 2
        self.text_color = (255, 255, 255)  # White text
        self.text_bg_color = (0, 0, 0, 180)  # Semi-transparent black
        self.font_scale = 0.6
        self.font_thickness = 1
        
        # Legend settings
        self.legend_bg_color = (255, 255, 255, 220)  # Semi-transparent white
        self.legend_text_color = (0, 0, 0)
        
    def draw_bounding_boxes(self, image: np.ndarray, 
                           boxes: List[List[float]], 
                           confidences: List[float],
                           class_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Draw bounding boxes on image.
        
        Args:
            image: Input image as numpy array
            boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
            confidences: Confidence scores for each box
            class_names: Optional class names for each box
            
        Returns:
            Image with bounding boxes drawn
        """
        overlay_image = image.copy()
        
        if not boxes:
            return overlay_image
        
        for i, (box, confidence) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            cv2.rectangle(overlay_image, (x1, y1), (x2, y2), 
                         self.box_color, self.box_thickness)
            
            # Prepare label text
            class_name = class_names[i] if class_names and i < len(class_names) else "Solar Panel"
            label = f"{class_name}: {confidence:.2f}"
            
            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness
            )
            
            # Draw text background
            text_bg_x1 = x1
            text_bg_y1 = y1 - text_height - baseline - 5
            text_bg_x2 = x1 + text_width + 10
            text_bg_y2 = y1
            
            # Create overlay for semi-transparent background
            overlay = overlay_image.copy()
            cv2.rectangle(overlay, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2),
                         self.text_bg_color[:3], -1)
            cv2.addWeighted(overlay_image, 0.8, overlay, 0.2, 0, overlay_image)
            
            # Draw text
            cv2.putText(overlay_image, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 
                       self.text_color, self.font_thickness)
        
        return overlay_image
    
    def create_legend(self, detection_results: Dict, 
                     quantification_results: Optional[Dict] = None,
                     qc_results: Optional[Dict] = None) -> Dict[str, str]:
        """
        Create legend information for the visualization.
        Format matches requirements: class, confidence, area.
        
        Args:
            detection_results: Results from detection
            quantification_results: Results from quantification
            qc_results: Results from quality control
            
        Returns:
            Dictionary with legend items
        """
        legend_items = {}
        
        # Detection info - match required format
        has_solar = detection_results.get('has_solar', False)
        max_confidence = detection_results.get('max_confidence', 0.0)
        
        # Required format: "class = Solar Panel"
        if has_solar:
            legend_items['class'] = "class = Solar Panel"
            legend_items['confidence'] = f"confidence = {max_confidence:.2f}"
        else:
            legend_items['class'] = "class = No Solar Panel"
            legend_items['confidence'] = f"confidence = {max_confidence:.2f}"
        
        # Quantification info - match required format
        if quantification_results:
            area = quantification_results.get('total_area_sqm', 0.0)
            if has_solar and area > 0:
                legend_items['area'] = f"area = {area:.1f} m²"
        
        # Additional info (optional but helpful)
        panel_count = detection_results.get('panel_count', 0)
        if has_solar:
            legend_items['panel_count'] = f"panels = {panel_count}"
        
        # QC info
        if qc_results:
            qc_status = qc_results.get('status', 'UNKNOWN')
            legend_items['qc_status'] = f"QC = {qc_status}"
        
        return legend_items
    
    def add_legend_to_image(self, image: np.ndarray, 
                           legend_items: Dict[str, str],
                           position: str = "top-right") -> np.ndarray:
        """
        Add legend to image.
        
        Args:
            image: Input image
            legend_items: Dictionary of legend items
            position: Legend position ("top-right", "top-left", "bottom-right", "bottom-left")
            
        Returns:
            Image with legend added
        """
        if not legend_items:
            return image
        
        # Create a copy
        result_image = image.copy()
        
        # Calculate legend size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        line_height = 25
        margin = 10
        
        max_text_width = 0
        for item in legend_items.values():
            (text_width, _), _ = cv2.getTextSize(item, font, font_scale, font_thickness)
            max_text_width = max(max_text_width, text_width)
        
        legend_width = max_text_width + 2 * margin
        legend_height = len(legend_items) * line_height + 2 * margin
        
        # Determine position
        h, w = image.shape[:2]
        
        if position == "top-right":
            x1, y1 = w - legend_width - 10, 10
        elif position == "top-left":
            x1, y1 = 10, 10
        elif position == "bottom-right":
            x1, y1 = w - legend_width - 10, h - legend_height - 10
        else:  # bottom-left
            x1, y1 = 10, h - legend_height - 10
        
        x2, y2 = x1 + legend_width, y1 + legend_height
        
        # Draw legend background
        overlay = result_image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.addWeighted(result_image, 0.8, overlay, 0.2, 0, result_image)
        
        # Draw border
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 0), 1)
        
        # Draw legend text
        text_y = y1 + margin + 15
        for item in legend_items.values():
            cv2.putText(result_image, item, (x1 + margin, text_y),
                       font, font_scale, (0, 0, 0), font_thickness)
            text_y += line_height
        
        return result_image
    
    def create_overlay_image(self, image: Union[np.ndarray, str],
                           detection_results: Dict,
                           quantification_results: Optional[Dict] = None,
                           qc_results: Optional[Dict] = None,
                           output_path: Optional[str] = None) -> np.ndarray:
        """
        Create complete overlay image with detections, quantification, and QC info.
        
        Args:
            image: Input image (numpy array or path)
            detection_results: Detection results
            quantification_results: Optional quantification results
            qc_results: Optional QC results
            output_path: Optional path to save the overlay image
            
        Returns:
            Overlay image as numpy array
        """
        # Load image if path provided
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image from {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()
        
        # Draw bounding boxes
        boxes = detection_results.get('boxes', [])
        confidences = detection_results.get('confidences', [])
        class_names = detection_results.get('class_names', [])
        
        overlay_img = self.draw_bounding_boxes(img, boxes, confidences, class_names)
        
        # Create and add legend
        legend_items = self.create_legend(detection_results, quantification_results, qc_results)
        overlay_img = self.add_legend_to_image(overlay_img, legend_items)
        
        # Add title
        title = f"Solar Panel Detection - Sample ID: {detection_results.get('sample_id', 'Unknown')}"
        self._add_title(overlay_img, title)
        
        # Save if output path provided
        if output_path:
            self.save_overlay_image(overlay_img, output_path)
        
        return overlay_img
    
    def _add_title(self, image: np.ndarray, title: str) -> None:
        """Add title to the top of the image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(
            title, font, font_scale, font_thickness
        )
        
        # Calculate position (centered at top)
        h, w = image.shape[:2]
        x = (w - text_width) // 2
        y = text_height + 20
        
        # Draw title background
        overlay = image.copy()
        cv2.rectangle(overlay, (x - 10, 5), (x + text_width + 10, y + 10),
                     (0, 0, 0), -1)
        cv2.addWeighted(image, 0.8, overlay, 0.2, 0, image)
        
        # Draw title text
        cv2.putText(image, title, (x, y), font, font_scale, 
                   (255, 255, 255), font_thickness)
    
    def save_overlay_image(self, image: np.ndarray, output_path: str) -> None:
        """
        Save overlay image to file.
        
        Args:
            image: Overlay image as numpy array
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(str(output_path), bgr_image)
        
        if not success:
            raise RuntimeError(f"Failed to save overlay image to {output_path}")
    
    def create_comparison_image(self, original_image: np.ndarray,
                              overlay_image: np.ndarray,
                              output_path: Optional[str] = None) -> np.ndarray:
        """
        Create side-by-side comparison of original and overlay images.
        
        Args:
            original_image: Original image
            overlay_image: Overlay image with annotations
            output_path: Optional output path
            
        Returns:
            Comparison image
        """
        # Ensure both images have same height
        h1, w1 = original_image.shape[:2]
        h2, w2 = overlay_image.shape[:2]
        
        target_height = min(h1, h2)
        
        # Resize if needed
        if h1 != target_height:
            aspect_ratio = w1 / h1
            new_width = int(target_height * aspect_ratio)
            original_image = cv2.resize(original_image, (new_width, target_height))
        
        if h2 != target_height:
            aspect_ratio = w2 / h2
            new_width = int(target_height * aspect_ratio)
            overlay_image = cv2.resize(overlay_image, (new_width, target_height))
        
        # Create comparison
        comparison = np.hstack([original_image, overlay_image])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(comparison, "Detection Results", 
                   (original_image.shape[1] + 10, 30), font, 0.8, (255, 255, 255), 2)
        
        if output_path:
            self.save_overlay_image(comparison, output_path)
        
        return comparison


class ReportGenerator:
    """Generates detailed analysis reports."""
    
    def __init__(self):
        """Initialize report generator."""
        pass
    
    def generate_analysis_report(self, sample_id: str,
                               detection_results: Dict,
                               quantification_results: Dict,
                               qc_results: Dict,
                               image_metadata: Optional[Dict] = None) -> Dict:
        """
        Generate comprehensive analysis report.
        
        Args:
            sample_id: Sample identifier
            detection_results: Detection results
            quantification_results: Quantification results
            qc_results: QC results
            image_metadata: Optional image metadata
            
        Returns:
            Comprehensive report dictionary
        """
        report = {
            "analysis_summary": {
                "sample_id": sample_id,
                "has_solar": detection_results.get('has_solar', False),
                "qc_status": qc_results.get('status'),
                "confidence": qc_results.get('confidence_score', 0.0)
            },
            "detection_details": {
                "panel_count": detection_results.get('panel_count', 0),
                "max_confidence": detection_results.get('max_confidence', 0.0),
                "mean_confidence": np.mean(detection_results.get('confidences', [0.0])),
                "detection_boxes": detection_results.get('boxes', [])
            },
            "quantification_details": quantification_results,
            "quality_assessment": {
                "qc_status": qc_results.get('status'),
                "qc_confidence": qc_results.get('confidence_score', 0.0),
                "reason_codes": qc_results.get('reason_codes', []),
                "image_quality_metrics": qc_results.get('image_quality_metrics', {}),
                "detection_quality_metrics": qc_results.get('detection_quality_metrics', {})
            },
            "metadata": image_metadata or {}
        }
        
        return report
    
    def save_report(self, report: Dict, output_path: str) -> None:
        """Save report to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)