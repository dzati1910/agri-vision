"""
Slicing Aided Hyper Inference (SAHI) / Tiling Engine.

Solves the 'Small Object Detection' bottleneck when inspecting 50MP+ smartphone
imagery for millimeter-scale pests (Planococcus citri) without resolution degradation.

Memory & Mathematical Model:
- Downsampling an 8160x6120 image to 640x640 causes a 12.75x reduction in each axis,
  reducing a 30x30 px nymph into ~2x2 px (lost in interpolation noise).
- SAHI generates a grid of (tile_w, tile_h) crops with overlap step (stride = tile - overlap),
  preserving 100% optical fidelity.
- Crops are memory views into the original NumPy array (no malloc overhead for crops).
"""

from __future__ import annotations
from typing import List, Tuple, Sequence, Optional
import numpy as np

from src.core.interfaces import BaseSlicer
from src.domain.entities import TileWindow, Detection, BoundingBox


class SAHISlicer(BaseSlicer):
    """
    Production-grade image slicer supporting custom tile sizes, overlap ratios,
    edge-padding / boundary-clamping, and zero-copy crop generation.
    """

    def __init__(
        self,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.20,
        overlap_width_ratio: float = 0.20,
        include_full_image: bool = True
    ) -> None:
        """
        Args:
            slice_height: Height of each window in pixels.
            slice_width: Width of each window in pixels.
            overlap_height_ratio: Overlap fraction along height (0.0 to 0.8).
            overlap_width_ratio: Overlap fraction along width (0.0 to 0.8).
            include_full_image: Whether to also evaluate a scaled full-image crop
                                for context and large macro structures.
        """
        if slice_height <= 0 or slice_width <= 0:
            raise ValueError("Slice dimensions must be positive.")
        if not (0.0 <= overlap_height_ratio < 0.9):
            raise ValueError(f"overlap_height_ratio must be in [0, 0.9), got {overlap_height_ratio}")
        if not (0.0 <= overlap_width_ratio < 0.9):
            raise ValueError(f"overlap_width_ratio must be in [0, 0.9), got {overlap_width_ratio}")

        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.include_full_image = include_full_image

    def generate_windows(self, image_shape: Tuple[int, int, ...]) -> List[TileWindow]:
        """
        Calculates window grid covering the entire image canvas.
        Guarantees complete coverage, clamping edges to prevent out-of-bounds coordinates.
        
        Args:
            image_shape: Tuple (height, width, channels) or (height, width).
            
        Returns:
            List of TileWindow instances with global coordinates.
        """
        img_h, img_w = image_shape[:2]
        if img_h <= 0 or img_w <= 0:
            raise ValueError(f"Image dimensions must be positive integers, got shape: {image_shape}")

        step_y = max(1, int(self.slice_height * (1.0 - self.overlap_height_ratio)))
        step_x = max(1, int(self.slice_width * (1.0 - self.overlap_width_ratio)))

        # Determine start coordinates along Y axis
        y_starts: List[int] = []
        cur_y = 0
        while cur_y + self.slice_height < img_h:
            y_starts.append(cur_y)
            cur_y += step_y
        # Ensure the bottom-most boundary is covered
        y_starts.append(max(0, img_h - self.slice_height))
        y_starts = sorted(list(set(y_starts)))

        # Determine start coordinates along X axis
        x_starts: List[int] = []
        cur_x = 0
        while cur_x + self.slice_width < img_w:
            x_starts.append(cur_x)
            cur_x += step_x
        # Ensure the right-most boundary is covered
        x_starts.append(max(0, img_w - self.slice_width))
        x_starts = sorted(list(set(x_starts)))

        windows: List[TileWindow] = []
        tile_id = 0

        for ymin in y_starts:
            ymax = min(img_h, ymin + self.slice_height)
            for xmin in x_starts:
                xmax = min(img_w, xmin + self.slice_width)
                windows.append(
                    TileWindow(
                        tile_id=tile_id,
                        xmin=xmin,
                        ymin=ymin,
                        xmax=xmax,
                        ymax=ymax
                    )
                )
                tile_id += 1

        return windows

    def crop_tile(self, image: np.ndarray, window: TileWindow) -> np.ndarray:
        """
        Extracts tile from image.
        In NumPy, this returns a view (contiguous pointer offset) whenever possible.
        If the tile at boundaries is smaller than (slice_width, slice_height),
        it is padded with zeros (reflection or constant border) to match standard model dimensions.
        """
        h, w = image.shape[:2]
        crop = image[window.ymin:window.ymax, window.xmin:window.xmax]

        crop_h, crop_w = crop.shape[:2]
        if crop_h == self.slice_height and crop_w == self.slice_width:
            return crop

        # Pad to required slice dimensions if needed
        pad_bottom = self.slice_height - crop_h
        pad_right = self.slice_width - crop_w
        if pad_bottom > 0 or pad_right > 0:
            if image.ndim == 3:
                padded = np.pad(crop, ((0, max(0, pad_bottom)), (0, max(0, pad_right)), (0, 0)), mode="constant")
            else:
                padded = np.pad(crop, ((0, max(0, pad_bottom)), (0, max(0, pad_right))), mode="constant")
            return padded

        return crop

    @staticmethod
    def map_tile_detections_to_global(
        tile_detections: Sequence[Detection],
        window: TileWindow,
        img_width: int,
        img_height: int
    ) -> List[Detection]:
        """
        Translates detection coordinates from local tile space to global image space.
        
        Args:
            tile_detections: Detections predicted inside the tile.
            window: Tile spatial location (xmin, ymin, xmax, ymax).
            img_width: Global image width.
            img_height: Global image height.
            
        Returns:
            List of Detections translated and clipped to original canvas.
        """
        if img_width <= 0 or img_height <= 0:
            raise ValueError(f"Image dimensions must be positive integers, got w={img_width}, h={img_height}")

        global_detections: List[Detection] = []
        for det in tile_detections:
            translated_box = det.bbox.translate(dx=window.xmin, dy=window.ymin)
            clipped_box = translated_box.clip(max_width=img_width, max_height=img_height)

            # Ignore invalid degenerate boxes after clipping
            if clipped_box.area > 0:
                global_detections.append(
                    Detection(
                        bbox=clipped_box,
                        class_id=det.class_id,
                        class_name=det.class_name,
                        confidence=det.confidence
                    )
                )

        return global_detections
