"""
Domain Entities for Agricultural Computer Vision Portfolio.
Implements robust, immutable value objects following Clean Architecture and SOLID principles.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, List, Dict, Any, Optional
import numpy as np


class PestClass(str, Enum):
    """Target pest categories according to agricultural entomology."""
    NYMPH = "planococcus_citri_nymph"
    ADULT = "planococcus_citri_adult"
    OVISAC = "planococcus_citri_ovisac"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """
    Immutable Value Object representing an axis-aligned bounding box.
    Coordinates are stored in absolute pixel space: (xmin, ymin, xmax, ymax).
    
    Analogy to C:
    Equivalent to a contiguous struct with 4 float64 fields:
    typedef struct { double xmin, ymin, xmax, ymax; } BoundingBox;
    """
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __post_init__(self) -> None:
        if self.xmax < self.xmin:
            raise ValueError(f"Invalid bounding box: xmax ({self.xmax}) < xmin ({self.xmin})")
        if self.ymax < self.ymin:
            raise ValueError(f"Invalid bounding box: ymax ({self.ymax}) < ymin ({self.ymin})")

    @property
    def width(self) -> float:
        """Width in pixels."""
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        """Height in pixels."""
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        """Surface area in pixel squared."""
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> Tuple[float, float]:
        """(cx, cy) centroid coordinates."""
        return (self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0

    @property
    def aspect_ratio(self) -> float:
        """Width / Height ratio."""
        h = self.height
        return self.width / h if h > 0 else 0.0

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Returns (xmin, ymin, xmax, ymax)."""
        return self.xmin, self.ymin, self.xmax, self.ymax

    def to_xywh(self) -> Tuple[float, float, float, float]:
        """Returns (xmin, ymin, width, height) in absolute pixels."""
        return self.xmin, self.ymin, self.width, self.height

    def to_yolo(self, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        Normalizes coordinates to YOLO format: (norm_cx, norm_cy, norm_w, norm_h).
        Values are clamped between 0.0 and 1.0.
        """
        if img_width <= 0 or img_height <= 0:
            raise ValueError("Image dimensions must be positive integers.")
        cx, cy = self.center
        norm_cx = max(0.0, min(1.0, cx / img_width))
        norm_cy = max(0.0, min(1.0, cy / img_height))
        norm_w = max(0.0, min(1.0, self.width / img_width))
        norm_h = max(0.0, min(1.0, self.height / img_height))
        return norm_cx, norm_cy, norm_w, norm_h

    @classmethod
    def from_yolo(cls, norm_cx: float, norm_cy: float, norm_w: float, norm_h: float,
                  img_width: int, img_height: int) -> BoundingBox:
        """Constructs BoundingBox from normalized YOLO coordinates."""
        if img_width <= 0 or img_height <= 0:
            raise ValueError(f"Image dimensions must be positive integers, got: {img_width}x{img_height}")
        if norm_w < 0.0 or norm_h < 0.0:
            raise ValueError(f"Normalized width and height must be non-negative, got w={norm_w}, h={norm_h}")
        cx = norm_cx * img_width
        cy = norm_cy * img_height
        w = norm_w * img_width
        h = norm_h * img_height
        xmin = cx - w / 2.0
        ymin = cy - h / 2.0
        xmax = cx + w / 2.0
        ymax = cy + h / 2.0
        return cls(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> BoundingBox:
        """Constructs BoundingBox from top-left (x, y) and dimensions (w, h)."""
        return cls(xmin=x, ymin=y, xmax=x + w, ymax=y + h)

    def clip(self, max_width: int, max_height: int) -> BoundingBox:
        """Clips coordinates to fit within [0, max_width] and [0, max_height]."""
        if max_width <= 0 or max_height <= 0:
            raise ValueError(f"Clipping bounds must be positive, got max_width={max_width}, max_height={max_height}")
        c_xmin = max(0.0, min(float(max_width), self.xmin))
        c_ymin = max(0.0, min(float(max_height), self.ymin))
        c_xmax = max(0.0, min(float(max_width), self.xmax))
        c_ymax = max(0.0, min(float(max_height), self.ymax))
        return BoundingBox(xmin=c_xmin, ymin=c_ymin, xmax=max(c_xmin, c_xmax), ymax=max(c_ymin, c_ymax))

    def translate(self, dx: float, dy: float) -> BoundingBox:
        """Translates the bounding box by (dx, dy). Used in SAHI coordinate re-projection."""
        return BoundingBox(
            xmin=self.xmin + dx,
            ymin=self.ymin + dy,
            xmax=self.xmax + dx,
            ymax=self.ymax + dy
        )

    def iou(self, other: BoundingBox) -> float:
        """
        Calculates Intersection-over-Union (Jaccard Index) with another box.
        Formula: IoU = Area(A ∩ B) / Area(A ∪ B)
        """
        ixmin = max(self.xmin, other.xmin)
        iymin = max(self.ymin, other.ymin)
        ixmax = min(self.xmax, other.xmax)
        iymax = min(self.ymax, other.ymax)

        inter_w = max(0.0, ixmax - ixmin)
        inter_h = max(0.0, iymax - iymin)
        inter_area = inter_w * inter_h

        union_area = self.area + other.area - inter_area
        if union_area <= 0.0:
            return 0.0
        return inter_area / union_area


@dataclass(frozen=True, slots=True)
class Detection:
    """
    Immutable representation of an object detection candidate.
    """
    bbox: BoundingBox
    class_id: int
    class_name: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence score must be in [0, 1], got {self.confidence}")

    def with_offset(self, dx: float, dy: float) -> Detection:
        """Returns a new Detection with translated bounding box coordinates."""
        return Detection(
            bbox=self.bbox.translate(dx, dy),
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=self.confidence
        )


@dataclass(frozen=True, slots=True)
class TileWindow:
    """
    Defines a spatial slice/crop within a high-resolution image.
    Follows NumPy matrix slicing convention: img[ymin:ymax, xmin:xmax].
    """
    tile_id: int
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin


@dataclass
class LeafSample:
    """
    Represents an agronomic visual sample with multimodal sensor readings.
    Connects Computer Vision detections with physiological plant data.
    """
    sample_id: str
    plant_id: str
    treatment: str
    spad_reading: float
    image_shape: Tuple[int, int, int]
    ground_truth: List[Detection] = field(default_factory=list)
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def pest_count(self) -> int:
        return len(self.ground_truth)

    @property
    def is_infested(self) -> bool:
        return self.pest_count > 0
