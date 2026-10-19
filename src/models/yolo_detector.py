"""
Ultralytics YOLO Adapter (YOLOv8 / YOLOv11).

Encapsulates deep learning inference, adapting raw tensor outputs to
clean domain Detection entities.
"""

from __future__ import annotations
from typing import List, Sequence, Optional, Dict
from pathlib import Path
import numpy as np

from src.core.interfaces import BaseDetector
from src.domain.entities import BoundingBox, Detection


class YOLODetector(BaseDetector):
    """
    Adapter for Ultralytics YOLO models (YOLOv8, YOLOv9, YOLOv11).
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        class_mapping: Optional[Dict[int, str]] = None
    ) -> None:
        self.model_path = str(model_path)
        self.device = device
        self._class_mapping = class_mapping or {
            0: "planococcus_citri_nymph",
            1: "planococcus_citri_adult",
            2: "planococcus_citri_ovisac"
        }

        # Lazy load model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
        return self._model

    @property
    def model_name(self) -> str:
        return f"YOLO({Path(self.model_path).stem})"

    def predict(self, image: np.ndarray, conf_threshold: float = 0.25) -> List[Detection]:
        """
        Runs YOLO inference on a single image.
        """
        if image is None or getattr(image, "size", 0) == 0:
            return []

        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )

        detections: List[Detection] = []
        if not results:
            return detections

        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            box = xyxy[i]
            cid = int(classes[i])
            cname = self._class_mapping.get(cid, f"class_{cid}")
            bbox = BoundingBox(
                xmin=float(box[0]),
                ymin=float(box[1]),
                xmax=float(box[2]),
                ymax=float(box[3])
            )
            detections.append(
                Detection(
                    bbox=bbox,
                    class_id=cid,
                    class_name=cname,
                    confidence=float(confs[i])
                )
            )

        return detections

    def predict_batch(self, images: Sequence[np.ndarray], conf_threshold: float = 0.25) -> List[List[Detection]]:
        """
        Runs batched YOLO inference for optimal GPU / AVX throughput.
        """
        if len(images) == 0:
            return []

        # If any images are empty or None, fallback to individual predict for safety
        valid_images = [img for img in images if img is not None and getattr(img, "size", 0) > 0]
        if len(valid_images) != len(images):
            return [self.predict(img, conf_threshold=conf_threshold) for img in images]

        results = self.model.predict(
            source=list(images),
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )

        all_detections: List[List[Detection]] = []
        for r in results:
            det_list: List[Detection] = []
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)

                for i in range(len(xyxy)):
                    cid = int(classes[i])
                    cname = self._class_mapping.get(cid, f"class_{cid}")
                    bbox = BoundingBox(
                        xmin=float(xyxy[i, 0]),
                        ymin=float(xyxy[i, 1]),
                        xmax=float(xyxy[i, 2]),
                        ymax=float(xyxy[i, 3])
                    )
                    det_list.append(
                        Detection(
                            bbox=bbox,
                            class_id=cid,
                            class_name=cname,
                            confidence=float(confs[i])
                        )
                    )
            all_detections.append(det_list)

        return all_detections
