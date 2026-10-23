"""
COCO Annotation Format Parser and Exporter.

Standard MS COCO JSON serialization including agronomic metadata extensions
(Plant ID, SPAD-502 index, Treatment group).
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.domain.entities import BoundingBox, Detection, LeafSample


class COCOParser:
    """Handles serialization and parsing of MS COCO standard datasets."""

    @staticmethod
    def export_samples_to_coco(
        samples: List[LeafSample],
        output_json_path: str | Path,
        categories: Optional[Dict[int, str]] = None
    ) -> None:
        """
        Exports LeafSample instances to COCO format JSON.
        """
        if categories is None:
            categories = {
                0: "planococcus_citri_nymph",
                1: "planococcus_citri_adult",
                2: "planococcus_citri_ovisac"
            }

        coco_data: Dict[str, Any] = {
            "info": {
                "description": "IFSertãoPE Vine Pest Dataset - Planococcus citri & SPAD-502",
                "version": "1.0",
                "year": 2026,
                "contributor": "Computer Vision & Agricultural Entomology Lab"
            },
            "licenses": [{"id": 1, "name": "CC-BY-4.0"}],
            "categories": [
                {"id": cid, "name": name, "supercategory": "agricultural_pest"}
                for cid, name in categories.items()
            ],
            "images": [],
            "annotations": []
        }

        ann_id = 1
        for img_id, sample in enumerate(samples, start=1):
            h, w = sample.image_shape[:2]
            coco_data["images"].append({
                "id": img_id,
                "file_name": Path(sample.image_path).name if sample.image_path else f"{sample.sample_id}.jpg",
                "width": w,
                "height": h,
                "plant_id": sample.plant_id,
                "treatment": sample.treatment,
                "spad_reading": sample.spad_reading
            })

            for det in sample.ground_truth:
                xmin, ymin, bw, bh = det.bbox.to_xywh()
                coco_data["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": det.class_id,
                    "bbox": [round(xmin, 2), round(ymin, 2), round(bw, 2), round(bh, 2)],
                    "area": round(det.bbox.area, 2),
                    "segmentation": [],
                    "iscrowd": 0,
                    "confidence": det.confidence
                })
                ann_id += 1

        p = Path(output_json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=2)

    @staticmethod
    def load_coco_annotations(json_path: str | Path) -> Dict[str, List[Detection]]:
        """
        Reads COCO JSON and returns dictionary of {image_file_name: List[Detection]}.
        """
        p = Path(json_path)
        if not p.is_file():
            return {}

        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

        if not isinstance(data, dict):
            return {}

        category_map = {c["id"]: c["name"] for c in data.get("categories", [])}
        img_id_to_name = {img["id"]: img["file_name"] for img in data.get("images", [])}

        results: Dict[str, List[Detection]] = {img["file_name"]: [] for img in data.get("images", [])}

        for ann in data.get("annotations", []):
            img_name = img_id_to_name.get(ann["image_id"])
            if not img_name:
                continue

            x, y, w, h = ann["bbox"]
            cid = ann["category_id"]
            cname = category_map.get(cid, f"class_{cid}")
            conf = ann.get("confidence", 1.0)

            bbox = BoundingBox.from_xywh(x, y, w, h)
            results[img_name].append(
                Detection(
                    bbox=bbox,
                    class_id=cid,
                    class_name=cname,
                    confidence=conf
                )
            )

        return results
