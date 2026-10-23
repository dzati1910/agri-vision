"""
YOLO Annotation Format Parser and Exporter.

Handles normalized (cx, cy, w, h) coordinate conversion, file serialization,
and dataset YAML configuration generation for Ultralytics YOLOv8/v11.
"""

from __future__ import annotations
import os
from typing import List, Dict, Optional
from pathlib import Path

from src.domain.entities import BoundingBox, Detection


class YOLOParser:
    """Parser and serializer for Ultralytics YOLO .txt annotation format."""

    @staticmethod
    def save_annotation(
        detections: List[Detection],
        output_txt_path: str | Path,
        img_width: int,
        img_height: int
    ) -> None:
        """
        Saves detections to YOLO formatted text file:
        <class_id> <norm_cx> <norm_cy> <norm_w> <norm_h>
        """
        p = Path(output_txt_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = []
        for det in detections:
            cx, cy, w, h = det.bbox.to_yolo(img_width, img_height)
            lines.append(f"{det.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        with open(p, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @staticmethod
    def load_annotation(
        txt_path: str | Path,
        img_width: int,
        img_height: int,
        class_mapping: Optional[Dict[int, str]] = None
    ) -> List[Detection]:
        """
        Loads YOLO annotation txt file and returns list of Detection objects.
        """
        p = Path(txt_path)
        if not p.exists():
            raise FileNotFoundError(f"YOLO annotation file not found: {txt_path}")
        if not p.is_file():
            raise FileNotFoundError(f"YOLO annotation path is not a file: {txt_path}")

        if img_width <= 0 or img_height <= 0:
            raise ValueError(f"Image dimensions must be positive integers, got: {img_width}x{img_height}")

        detections: List[Detection] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                try:
                    cid = int(parts[0])
                    cx = float(parts[1])
                    cy = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    conf = float(parts[5]) if len(parts) >= 6 else 1.0
                except ValueError:
                    continue

                cname = class_mapping.get(cid, f"class_{cid}") if class_mapping else f"class_{cid}"
                try:
                    bbox = BoundingBox.from_yolo(cx, cy, w, h, img_width, img_height)
                except ValueError:
                    continue

                detections.append(
                    Detection(
                        bbox=bbox,
                        class_id=cid,
                        class_name=cname,
                        confidence=conf
                    )
                )

        return detections

    @staticmethod
    def create_dataset_yaml(
        yaml_path: str | Path,
        dataset_root: str | Path,
        train_rel_path: str = "images/train",
        val_rel_path: str = "images/val",
        class_names: Optional[Dict[int, str]] = None
    ) -> None:
        """
        Creates Ultralytics dataset.yaml configuration file.
        """
        if class_names is None:
            class_names = {
                0: "planococcus_citri_nymph",
                1: "planococcus_citri_adult",
                2: "planococcus_citri_ovisac"
            }

        p = Path(yaml_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        content = [
            f"path: {os.path.abspath(dataset_root)}",
            f"train: {train_rel_path}",
            f"val: {val_rel_path}",
            "",
            "names:"
        ]
        for cid, name in sorted(class_names.items()):
            content.append(f"  {cid}: {name}")

        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(content) + "\n")
