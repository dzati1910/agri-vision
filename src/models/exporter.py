"""
Model Exporter and ONNX Graph Generator.

Supports exporting YOLOv8/v11 weights to ONNX format and creating minimal
reproducible ONNX test graphs for edge validation.
"""

from __future__ import annotations
from typing import Tuple, Optional
from pathlib import Path
import torch
import torch.nn as nn
import onnx


class MiniPestYOLOHead(nn.Module):
    """
    Minimal PyTorch detection head producing YOLO-compatible output tensors:
    Output shape: (Batch, 4 + num_classes, num_anchors).
    """
    def __init__(self, num_classes: int = 3, num_anchors: int = 8400) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        # Lightweight conv layers simulating backbone + head
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((20, 20))
        )
        # Linear projection to generate candidate anchor boxes
        self.head = nn.Linear(16 * 20 * 20, (4 + num_classes) * 84)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        feats = self.backbone(x)
        flat = feats.view(b, -1)
        out = self.head(flat)
        out = out.view(b, 4 + self.num_classes, 84)
        # Repeat to simulate standard 8400 grid points
        repeats = self.num_anchors // 84
        out = out.repeat(1, 1, repeats)
        cx_cy = torch.sigmoid(out[:, :2, :]) * 640.0
        w_h = torch.abs(out[:, 2:4, :]) + 15.0
        boxes = torch.cat([cx_cy, w_h], dim=1)
        scores = torch.sigmoid(out[:, 4:, :])
        return torch.cat([boxes, scores], dim=1)


class ModelExporter:
    """Handles PyTorch to ONNX serialization, validation, and metadata embedding."""

    @staticmethod
    def export_yolo_to_onnx(
        model_pt_path: str | Path,
        output_onnx_path: str | Path,
        imgsz: Tuple[int, int] = (640, 640),
        opset: int = 17
    ) -> str:
        """
        Exports Ultralytics model to ONNX.
        """
        from ultralytics import YOLO
        model = YOLO(str(model_pt_path))
        exported_path = model.export(
            format="onnx",
            imgsz=list(imgsz),
            opset=opset,
            dynamic=False
        )
        return str(exported_path)

    @staticmethod
    def generate_minimal_onnx_fixture(
        output_path: str | Path,
        num_classes: int = 3,
        input_size: Tuple[int, int] = (640, 640)
    ) -> str:
        """
        Generates a valid, self-contained ONNX detection model for automated testing,
        CI/CD, and offline edge verification.
        """
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        model = MiniPestYOLOHead(num_classes=num_classes)
        model.eval()

        dummy_input = torch.randn(1, 3, input_size[1], input_size[0], dtype=torch.float32)

        torch.onnx.export(
            model,
            dummy_input,
            str(p),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["output0"]
        )

        # Validate ONNX graph
        onnx_model = onnx.load(str(p))
        onnx.checker.check_model(onnx_model)

        return str(p)
