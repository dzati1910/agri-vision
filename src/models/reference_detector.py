"""
Reference Morphological Pest Detector.

Implements BaseDetector protocol using classical computer vision and morphological
feature extraction tailored to Planococcus citri (waxy white cuticle, oval aspect ratio,
lateral filament frequency).

Provides a deterministic baseline detector that runs without external weights,
ideal for unit testing, reproducible benchmarks, and edge validation.
"""

from __future__ import annotations
from typing import List, Sequence, Tuple
import numpy as np
import cv2

from src.core.interfaces import BaseDetector
from src.domain.entities import BoundingBox, Detection, PestClass


class ReferenceMorphologyDetector(BaseDetector):
    """
    Classical Computer Vision & Morphological feature detector for Planococcus citri.
    Detects waxy-white high-contrast insect bodies on green grapevine parenchyma.
    """

    def __init__(
        self,
        min_area: float = 120.0,
        max_area: float = 12000.0,
        whiteness_threshold: int = 190,
        name: str = "ReferenceMorphologyDetector"
    ) -> None:
        self._name = name
        self.min_area = min_area
        self.max_area = max_area
        self.whiteness_threshold = whiteness_threshold

    @property
    def model_name(self) -> str:
        return self._name

    def predict(self, image: np.ndarray, conf_threshold: float = 0.25) -> List[Detection]:
        """
        Detects pests within image tile or full image.
        
        Steps:
        1. Convert BGR to Grayscale / HSV.
        2. High-pass filter & threshold for chalky white waxy epicuticle.
        3. Morphological opening/closing to isolate distinct individuals.
        4. Contour extraction, bounding box computation, and confidence scoring.
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]

        # Convert to grayscale
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # In BGR, white waxy powder has high values in all 3 channels
            # and low saturation in HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            # Chalky white mask: high brightness, low-to-medium saturation
            mask = (val >= self.whiteness_threshold) & (sat <= 140)
            binary = (mask * 255).astype(np.uint8)
        else:
            gray = image
            _, binary = cv2.threshold(gray, self.whiteness_threshold, 255, cv2.THRESH_BINARY)

        # Morphological filtering to separate adjacent clusters
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Find connected contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[Detection] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = max(bw, bh) / max(1.0, min(bw, bh))

            # Planococcus citri has an oval aspect ratio between 1.1 and 2.5
            if aspect_ratio > 3.5:
                continue

            # Calculate confidence based on whiteness purity and contrast with background
            crop_val = val[y:y + bh, x:x + bw] if image.ndim == 3 else gray[y:y + bh, x:x + bw]
            mean_intensity = float(np.mean(crop_val)) if crop_val.size > 0 else 0.0
            confidence = min(1.0, max(0.3, mean_intensity / 255.0))

            if confidence < conf_threshold:
                continue

            # Determine class based on morphological size
            if area < 1100:
                cid = 0
                cname = "planococcus_citri_nymph"
            elif area < 3200:
                cid = 1
                cname = "planococcus_citri_adult"
            else:
                cid = 2
                cname = "planococcus_citri_ovisac"

            bbox = BoundingBox(
                xmin=float(x),
                ymin=float(y),
                xmax=float(x + bw),
                ymax=float(y + bh)
            )

            detections.append(
                Detection(
                    bbox=bbox,
                    class_id=cid,
                    class_name=cname,
                    confidence=confidence
                )
            )

        return detections

    def predict_batch(self, images: Sequence[np.ndarray], conf_threshold: float = 0.25) -> List[List[Detection]]:
        """Batched prediction across multiple image tiles."""
        return [self.predict(img, conf_threshold=conf_threshold) for img in images]
