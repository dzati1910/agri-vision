"""
Production ONNX Runtime Detector for Edge Deployment.

Implements zero-copy tensor preparation, letterboxing, pure NumPy inference decoding,
and vectorized NMS post-processing without PyTorch dependencies at runtime.

Connection to C Architecture:
- Direct mapping to C API: OrtSession, OrtMemoryInfo, and OrtValue tensors.
- Continuous memory buffers (np.ascontiguousarray) avoid memory reallocations.
"""

from __future__ import annotations
from typing import List, Sequence, Tuple, Optional, Dict
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort

from src.core.interfaces import BaseDetector
from src.domain.entities import BoundingBox, Detection
from src.core.nms import apply_nms


class ONNXDetector(BaseDetector):
    """
    Ultra-lightweight edge detector executing ONNX graphs via ONNX Runtime.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        input_size: Tuple[int, int] = (640, 640),
        class_mapping: Optional[Dict[int, str]] = None,
        providers: Optional[List[str]] = None
    ) -> None:
        self.onnx_path = str(onnx_path)
        self.input_size = input_size
        self._class_mapping = class_mapping or {
            0: "planococcus_citri_nymph",
            1: "planococcus_citri_adult",
            2: "planococcus_citri_ovisac"
        }

        # Initialize ONNX Runtime Session
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

    @property
    def model_name(self) -> str:
        return f"ONNX({Path(self.onnx_path).name})"

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Letterbox resize and normalize to float32 [0.0, 1.0] NCHW.
        
        Returns:
            Tuple of (tensor_nchw, scale_ratio, (pad_left, pad_top))
        """
        target_w, target_h = self.input_size
        h, w = image.shape[:2]

        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded canvas
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2

        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        # BGR -> RGB, HWC -> CHW, [0, 255] -> [0.0, 1.0]
        # Memory layout: C-contiguous memory buffer for OrtValue
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        transposed = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
        tensor = np.expand_dims(transposed, axis=0)

        return tensor, scale, (pad_left, pad_top)

    def predict(self, image: np.ndarray, conf_threshold: float = 0.25) -> List[Detection]:
        """
        Runs inference through ONNX session and decodes output boxes.
        """
        if image is None or image.size == 0:
            return []

        orig_h, orig_w = image.shape[:2]
        tensor, scale, (pad_left, pad_top) = self._preprocess(image)

        # Run ONNX inference
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        raw_output = outputs[0]  # Shape: (1, 4 + num_classes, num_anchors) or (1, num_anchors, 4 + num_classes)

        # Normalize shape to (num_anchors, 4 + num_classes)
        if raw_output.shape[1] < raw_output.shape[2]:
            # (1, 7, 8400) -> (8400, 7)
            preds = raw_output[0].T
        else:
            preds = raw_output[0]

        # In YOLO format: cols 0..3 are cx, cy, w, h
        cx = preds[:, 0]
        cy = preds[:, 1]
        w = preds[:, 2]
        h = preds[:, 3]

        # Class scores: cols 4..
        class_scores = preds[:, 4:]
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # Confidence filter mask
        valid_mask = max_scores >= conf_threshold
        if not np.any(valid_mask):
            return []

        cx = cx[valid_mask]
        cy = cy[valid_mask]
        w = w[valid_mask]
        h = h[valid_mask]
        scores = max_scores[valid_mask]
        cids = class_ids[valid_mask]

        # Transform from letterbox back to original image space
        x1_raw = (cx - w / 2.0 - pad_left) / scale
        y1_raw = (cy - h / 2.0 - pad_top) / scale
        x2_raw = (cx + w / 2.0 - pad_left) / scale
        y2_raw = (cy + h / 2.0 - pad_top) / scale

        # Guarantee min/max ordering and clip to original image boundaries
        x1 = np.clip(np.minimum(x1_raw, x2_raw), 0, orig_w)
        y1 = np.clip(np.minimum(y1_raw, y2_raw), 0, orig_h)
        x2 = np.clip(np.maximum(x1_raw, x2_raw), 0, orig_w)
        y2 = np.clip(np.maximum(y1_raw, y2_raw), 0, orig_h)

        candidates: List[Detection] = []
        for i in range(len(scores)):
            if x2[i] <= x1[i] or y2[i] <= y1[i]:
                continue
            cid = int(cids[i])
            cname = self._class_mapping.get(cid, f"class_{cid}")
            bbox = BoundingBox(
                xmin=float(x1[i]),
                ymin=float(y1[i]),
                xmax=float(x2[i]),
                ymax=float(y2[i])
            )
            candidates.append(
                Detection(
                    bbox=bbox,
                    class_id=cid,
                    class_name=cname,
                    confidence=float(scores[i])
                )
            )

        # Post-process with NMS
        return apply_nms(candidates, iou_threshold=0.45)

    def predict_batch(self, images: Sequence[np.ndarray], conf_threshold: float = 0.25) -> List[List[Detection]]:
        return [self.predict(img, conf_threshold=conf_threshold) for img in images]
