"""
High-Resolution Hyper Inference Pipeline (SAHI Architecture).

Integrates spatial slicing, batched model evaluation, global coordinate re-projection,
and multi-tile NMS fusion to solve the small-object vanishing problem.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional
import numpy as np
import cv2

from src.core.interfaces import BaseDetector, BaseSlicer
from src.slicing.sahi_slicer import SAHISlicer
from src.core.nms import apply_nms, weighted_boxes_fusion
from src.domain.entities import BoundingBox, Detection


@dataclass(frozen=True)
class PipelineResult:
    """Summary output of a hyper-inference execution."""
    detections: List[Detection]
    num_tiles_processed: int
    elapsed_seconds: float
    fps: float
    raw_tile_detection_count: int


class HyperInferencePipeline:
    """
    SAHI (Slicing Aided Hyper Inference) Pipeline.
    Orchestrates high-resolution object detection for microscopic pests.
    """

    def __init__(
        self,
        detector: BaseDetector,
        slicer: Optional[BaseSlicer] = None,
        global_nms_iou: float = 0.50,
        use_box_fusion: bool = True,
        batch_size: int = 4
    ) -> None:
        self.detector = detector
        self.slicer = slicer or SAHISlicer(slice_height=640, slice_width=640, overlap_height_ratio=0.20, overlap_width_ratio=0.20)
        self.global_nms_iou = global_nms_iou
        self.use_box_fusion = use_box_fusion
        self.batch_size = max(1, batch_size)

    def run(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        evaluate_full_image: Optional[bool] = None
    ) -> PipelineResult:
        """
        Executes complete SAHI pipeline on a high-resolution image.
        
        Args:
            image: High-resolution image (H, W, 3) BGR/RGB numpy array.
            conf_threshold: Minimum confidence score.
            evaluate_full_image: If True, also includes a full-image pass. If None, respects slicer.include_full_image.
            
        Returns:
            PipelineResult containing consolidated detections and execution profiling.
        """
        if image is None or getattr(image, "size", 0) == 0:
            return PipelineResult(
                detections=[],
                num_tiles_processed=0,
                elapsed_seconds=0.0,
                fps=0.0,
                raw_tile_detection_count=0
            )

        start_time = time.perf_counter()
        img_h, img_w = image.shape[:2]

        windows = self.slicer.generate_windows(image.shape)
        raw_detections: List[Detection] = []

        # 1. Process sliced tiles in batches for hardware throughput
        for i in range(0, len(windows), self.batch_size):
            batch_windows = windows[i:i + self.batch_size]
            batch_crops = [self.slicer.crop_tile(image, win) for win in batch_windows]

            batch_results = self.detector.predict_batch(batch_crops, conf_threshold=conf_threshold)

            for win, tile_dets in zip(batch_windows, batch_results):
                global_dets = SAHISlicer.map_tile_detections_to_global(
                    tile_dets, win, img_width=img_w, img_height=img_h
                )
                raw_detections.extend(global_dets)

        # 2. Optional full-image context pass
        should_eval_full = self.slicer.include_full_image if evaluate_full_image is None else evaluate_full_image
        if should_eval_full:
            full_img_dets = self.detector.predict(image, conf_threshold=conf_threshold)
            raw_detections.extend(full_img_dets)

        # 3. Post-process with Global NMS or Weighted Box Fusion across tile boundaries
        if self.use_box_fusion:
            fused_detections = weighted_boxes_fusion(raw_detections, iou_threshold=self.global_nms_iou)
        else:
            fused_detections = apply_nms(raw_detections, iou_threshold=self.global_nms_iou)

        elapsed = time.perf_counter() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        return PipelineResult(
            detections=fused_detections,
            num_tiles_processed=len(windows),
            elapsed_seconds=elapsed,
            fps=fps,
            raw_tile_detection_count=len(raw_detections)
        )


class StandardResizePipeline:
    """
    Standard naive detection pipeline that resizes the whole high-res image
    directly into standard detector input size (e.g. 640x640).
    Used as an empirical baseline to demonstrate the Small Object Detection problem.
    """

    def __init__(self, detector: BaseDetector, target_size: Tuple[int, int] = (640, 640)) -> None:
        """
        Args:
            detector: Detection model instance.
            target_size: Tuple (target_width, target_height) for downsampling.
        """
        if target_size[0] <= 0 or target_size[1] <= 0:
            raise ValueError(f"target_size dimensions must be positive, got: {target_size}")
        self.detector = detector
        self.target_size = target_size

    def run(self, image: np.ndarray, conf_threshold: float = 0.25) -> PipelineResult:
        if image is None or getattr(image, "size", 0) == 0:
            return PipelineResult(
                detections=[],
                num_tiles_processed=0,
                elapsed_seconds=0.0,
                fps=0.0,
                raw_tile_detection_count=0
            )

        start_time = time.perf_counter()
        orig_h, orig_w = image.shape[:2]

        # Downsample high-res image to detector's input dimensions (width, height)
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

        # Predict on resized image
        downscaled_dets = self.detector.predict(resized, conf_threshold=conf_threshold)

        # Scale detected boxes back to original high-res canvas coordinates
        scale_x = orig_w / float(self.target_size[0])
        scale_y = orig_h / float(self.target_size[1])

        scaled_detections: List[Detection] = []
        for det in downscaled_dets:
            b = det.bbox
            scaled_box = BoundingBox(
                xmin=b.xmin * scale_x,
                ymin=b.ymin * scale_y,
                xmax=b.xmax * scale_x,
                ymax=b.ymax * scale_y
            ).clip(max_width=orig_w, max_height=orig_h)

            scaled_detections.append(
                Detection(
                    bbox=scaled_box,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence
                )
            )

        elapsed = time.perf_counter() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        return PipelineResult(
            detections=scaled_detections,
            num_tiles_processed=1,
            elapsed_seconds=elapsed,
            fps=fps,
            raw_tile_detection_count=len(scaled_detections)
        )
