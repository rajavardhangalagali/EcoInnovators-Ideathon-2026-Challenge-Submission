"""
Image verification module for production-ready feature.
Compares uploaded image with satellite image to verify location match.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from skimage.metrics import structural_similarity as ssim
from pathlib import Path


class ImageVerifier:
    """Verifies if uploaded image matches satellite image at given coordinates."""
    
    def __init__(self, similarity_threshold: float = 0.7,
                 method: str = "ssim"):
        """
        Initialize image verifier.
        
        Args:
            similarity_threshold: Minimum similarity score (0-1) to consider match
            method: Comparison method ("ssim", "feature_matching", "histogram")
        """
        self.similarity_threshold = similarity_threshold
        self.method = method
    
    def verify_image_match(self, uploaded_image: np.ndarray,
                         satellite_image: np.ndarray) -> Dict:
        """
        Verify if uploaded image matches satellite image.
        
        Args:
            uploaded_image: User-uploaded image as numpy array
            satellite_image: Satellite image fetched at coordinates
            
        Returns:
            Dictionary with verification results
        """
        # Resize images to same size for comparison
        target_size = (640, 640)
        uploaded_resized = cv2.resize(uploaded_image, target_size)
        satellite_resized = cv2.resize(satellite_image, target_size)
        
        # Convert to grayscale if needed
        if len(uploaded_resized.shape) == 3:
            uploaded_gray = cv2.cvtColor(uploaded_resized, cv2.COLOR_RGB2GRAY)
        else:
            uploaded_gray = uploaded_resized
        
        if len(satellite_resized.shape) == 3:
            satellite_gray = cv2.cvtColor(satellite_resized, cv2.COLOR_RGB2GRAY)
        else:
            satellite_gray = satellite_resized
        
        # Calculate similarity based on method
        if self.method == "ssim":
            similarity_score = self._calculate_ssim(uploaded_gray, satellite_gray)
        elif self.method == "feature_matching":
            similarity_score = self._calculate_feature_match(uploaded_gray, satellite_gray)
        elif self.method == "histogram":
            similarity_score = self._calculate_histogram_similarity(uploaded_gray, satellite_gray)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Determine match status
        is_match = similarity_score >= self.similarity_threshold
        
        result = {
            "status": "VERIFIED" if is_match else "MISMATCH",
            "similarity_score": float(similarity_score),
            "threshold": self.similarity_threshold,
            "method": self.method,
            "is_match": is_match
        }
        
        if not is_match:
            result["error"] = "Uploaded image does not match satellite image at given coordinates"
            result["recommendation"] = "Please verify coordinates or upload correct image"
        
        return result
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index (SSIM).
        Good for images from same angle/viewpoint.
        
        Args:
            img1: First image (grayscale)
            img2: Second image (grayscale)
            
        Returns:
            SSIM score (0-1, higher = more similar)
        """
        # Ensure images are same size
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Calculate SSIM
        score = ssim(img1, img2, data_range=255)
        return float(score)
    
    def _calculate_feature_match(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate similarity using feature matching (SIFT/ORB).
        Good for images from different angles.
        
        Args:
            img1: First image (grayscale)
            img2: Second image (grayscale)
            
        Returns:
            Feature match score (0-1, higher = more similar)
        """
        # Initialize ORB detector (faster than SIFT, no license issues)
        orb = cv2.ORB_create()
        
        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None:
            return 0.0
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # Calculate similarity based on number of good matches
        num_matches = len(matches)
        max_possible_matches = min(len(kp1), len(kp2))
        
        if max_possible_matches == 0:
            return 0.0
        
        similarity = num_matches / max_possible_matches
        
        # Normalize to 0-1 range (typically 0-0.5, so scale up)
        return float(min(similarity * 2.0, 1.0))
    
    def _calculate_histogram_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate similarity using histogram comparison.
        Simple but limited - good for same viewpoint.
        
        Args:
            img1: First image (grayscale)
            img2: Second image (grayscale)
            
        Returns:
            Histogram similarity score (0-1, higher = more similar)
        """
        # Calculate histograms
        hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        # Compare using correlation
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return float(similarity)
    
    def verify_from_paths(self, uploaded_image_path: str,
                         satellite_image_path: str) -> Dict:
        """
        Verify images from file paths.
        
        Args:
            uploaded_image_path: Path to uploaded image
            satellite_image_path: Path to satellite image
            
        Returns:
            Verification result dictionary
        """
        uploaded_img = cv2.imread(uploaded_image_path)
        satellite_img = cv2.imread(satellite_image_path)
        
        if uploaded_img is None:
            raise ValueError(f"Could not load uploaded image: {uploaded_image_path}")
        if satellite_img is None:
            raise ValueError(f"Could not load satellite image: {satellite_image_path}")
        
        uploaded_img = cv2.cvtColor(uploaded_img, cv2.COLOR_BGR2RGB)
        satellite_img = cv2.cvtColor(satellite_img, cv2.COLOR_BGR2RGB)
        
        return self.verify_image_match(uploaded_img, satellite_img)


def create_image_verifier(similarity_threshold: float = 0.7,
                         method: str = "ssim") -> ImageVerifier:
    """
    Factory function to create image verifier.
    
    Args:
        similarity_threshold: Minimum similarity for match (0-1)
        method: Comparison method ("ssim", "feature_matching", "histogram")
        
    Returns:
        ImageVerifier instance
    """
    return ImageVerifier(similarity_threshold=similarity_threshold, method=method)

