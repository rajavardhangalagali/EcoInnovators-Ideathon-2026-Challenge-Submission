"""
Solar panel detection using YOLOv8 model.
Handles model loading, inference, and confidence scoring.
"""

import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
import cv2
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import warnings


class SolarPanelDetector:
    """YOLOv8-based solar panel detector."""
    
    def __init__(self, model_path: str, device: str = "auto", 
                 confidence_threshold: float = 0.5, iou_threshold: float = 0.45):
        """
        Initialize the solar panel detector.
        
        Args:
            model_path: Path to trained YOLOv8 model (.pt file)
            device: Device to run inference on ('auto', 'cpu', 'cuda:0', etc.)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load model
        self.model = self._load_model()
        
    def _load_model(self) -> YOLO:
        """Load the YOLOv8 model."""
        try:
            if Path(self.model_path).exists():
                # Load custom trained model
                model = YOLO(self.model_path)
                print(f"Loaded trained model from {self.model_path}")
            else:
                # Use pre-trained YOLOv8 model (will need to be retrained)
                model = YOLO('yolov8n.pt')  # Start with nano model
                warnings.warn(f"Model file {self.model_path} not found. Using pre-trained YOLOv8n. "
                            "This model needs to be trained on solar panel data.")
            
            # Move model to device
            model.to(self.device)
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def detect(self, image: Union[str, Path, np.ndarray, Image.Image]) -> Dict:
        """
        Detect solar panels in an image.
        
        Args:
            image: Image to analyze (path, numpy array, or PIL Image)
            
        Returns:
            Detection results dictionary with boxes, confidences, and metadata
        """
        # Run inference
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Extract results from first (and only) image
        result = results[0]
        
        # Parse detections
        detections = {
            'boxes': [],
            'confidences': [],
            'class_names': [],
            'has_solar': False,
            'max_confidence': 0.0,
            'panel_count': 0,
            'image_shape': None
        }
        
        if result.boxes is not None and len(result.boxes) > 0:
            # Get image shape
            detections['image_shape'] = result.orig_shape
            
            # Extract box coordinates (xyxy format)
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            # Get class names
            class_names = [result.names[class_id] for class_id in class_ids]
            
            detections['boxes'] = boxes.tolist()
            detections['confidences'] = confidences.tolist()
            detections['class_names'] = class_names
            detections['has_solar'] = len(boxes) > 0
            detections['max_confidence'] = float(np.max(confidences)) if len(confidences) > 0 else 0.0
            detections['panel_count'] = len(boxes)
            
        return detections
    
    def batch_detect(self, image_paths: List[Union[str, Path]]) -> List[Dict]:
        """
        Run detection on multiple images.
        
        Args:
            image_paths: List of paths to images
            
        Returns:
            List of detection result dictionaries
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.detect(image_path)
                result['image_path'] = str(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                # Add empty result for failed image
                results.append({
                    'image_path': str(image_path),
                    'error': str(e),
                    'has_solar': False,
                    'max_confidence': 0.0,
                    'panel_count': 0,
                    'boxes': [],
                    'confidences': [],
                    'class_names': []
                })
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'model_path': self.model_path,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'model_type': 'YOLOv8',
            'class_names': getattr(self.model, 'names', {})
        }


class ModelTrainer:
    """Helper class for training YOLOv8 models on solar panel data."""
    
    def __init__(self, base_model: str = 'yolov8n.pt'):
        """
        Initialize model trainer.
        
        Args:
            base_model: Base YOLOv8 model to start from
        """
        self.base_model = base_model
    
    def prepare_training_config(self, data_yaml_path: str, 
                              epochs: int = 100, 
                              imgsz: int = 640,
                              batch: int = 16) -> Dict:
        """
        Prepare training configuration with robust augmentation for diverse conditions.
        
        Args:
            data_yaml_path: Path to dataset YAML configuration
            epochs: Number of training epochs
            imgsz: Training image size
            batch: Batch size
            
        Returns:
            Training configuration dictionary with augmentation settings
        """
        return {
            'data': data_yaml_path,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'project': 'models',
            'name': 'yolov8_solar',
            'save': True,
            'save_period': 10,
            'val': True,
            'plots': True,
            # Robust augmentation for diverse roof types and imaging conditions
            'hsv_h': 0.025,  # Hue variation: handles blue/teal/black panel colors
            'hsv_s': 0.7,    # Saturation: adapts to different color intensities
            'hsv_v': 0.5,    # Value/brightness: handles various lighting conditions
            'degrees': 15.0,  # Rotation: handles oblique angles and tilted roofs
            'translate': 0.2,  # Translation: simulates different camera positions
            'scale': 0.6,     # Scale variation: handles different altitudes/zoom levels
            'shear': 5.0,     # Shear: simulates oblique satellite viewing angles
            'perspective': 0.0005,  # Perspective distortion for aerial views
            'flipud': 0.5,    # Vertical flip: handles different orientations
            'fliplr': 0.5,    # Horizontal flip: handles different orientations
            'mosaic': 1.0,    # Mosaic augmentation: improves multi-scale detection
            'mixup': 0.15,    # Mixup: improves generalization across image types
            'copy_paste': 0.1,  # Copy-paste: augments panel variety
            # Training optimization for better generalization
            'patience': 50,   # Early stopping patience: prevents overfitting
            'dropout': 0.0,   # Dropout disabled (YOLOv8 handles this internally)
            'lr0': 0.01,      # Initial learning rate
            'lrf': 0.01,      # Final learning rate (1% of initial)
            'momentum': 0.937,  # SGD momentum
            'weight_decay': 0.0005,  # L2 regularization
            'warmup_epochs': 3.0,  # Warmup epochs for stable start
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,       # Box loss weight
            'cls': 0.5,       # Classification loss weight
            'dfl': 1.5,       # Distribution focal loss weight
            'label_smoothing': 0.0,  # Label smoothing (disabled for small dataset)
            'close_mosaic': 10  # Disable mosaic in last N epochs for fine-tuning
        }
    
    def train(self, data_yaml_path: str, **kwargs) -> str:
        """
        Train YOLOv8 model on solar panel data.
        
        Args:
            data_yaml_path: Path to dataset YAML configuration
            **kwargs: Additional training parameters
            
        Returns:
            Path to trained model
        """
        # Load base model
        model = YOLO(self.base_model)
        
        # Prepare config
        config = self.prepare_training_config(data_yaml_path, **kwargs)
        
        # Train model
        results = model.train(**config)
        
        # Return path to best model
        return str(results.save_dir / 'weights' / 'best.pt')
    
    def create_dataset_yaml(self, train_images_dir: str, 
                           val_images_dir: str,
                           output_path: str = None) -> str:
        """
        Create dataset YAML file for YOLOv8 training.
        
        Args:
            train_images_dir: Directory containing training images
            val_images_dir: Directory containing validation images
            output_path: Output path for YAML file (default: auto-detect from dataset)
            
        Returns:
            Path to created YAML file
        """
        # Auto-detect output path if not provided
        if output_path is None:
            dataset_root = Path(train_images_dir).parent.parent
            output_path = dataset_root / "data.yaml"
        
        yaml_content = f"""# Solar Panel Dataset Configuration
path: {Path(train_images_dir).parent.parent.absolute()}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: valid/images  # val images (relative to 'path')

# Classes
names:
  0: solar_panel

# Number of classes
nc: 1
"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"Created dataset YAML at {output_path}")
        return str(output_path)