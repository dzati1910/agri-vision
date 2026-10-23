"""
Model Exporter and ONNX Graph Generator.

Supports exporting YOLOv8/v11 weights to ONNX format and creating minimal
reproducible ONNX test graphs for edge validation.
"""

from __future__ import annotations
from typing import Tuple, Optional, Any
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
        # 1x1 Conv projection to generate candidate anchor boxes (YOLO architecture)
        self.head = nn.Conv2d(16, 4 + num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        out = self.head(feats).flatten(2)
        repeats = self.num_anchors // out.shape[2]
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
        opset: int = 17,
        dynamic: bool = True
    ) -> str:
        """
        Exports Ultralytics model to ONNX with dynamic batch support.
        """
        import shutil
        from ultralytics import YOLO

        model = YOLO(str(model_pt_path))
        exported_path = model.export(
            format="onnx",
            imgsz=list(imgsz),
            opset=opset,
            dynamic=dynamic
        )

        out_path = Path(output_onnx_path).resolve()
        exp_path = Path(exported_path).resolve()
        if out_path != exp_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(exported_path, str(out_path))
            return str(out_path)

        return str(exported_path)

    @staticmethod
    def quantize_onnx_dynamic(
        input_onnx_path: str | Path,
        output_onnx_path: str | Path,
        weight_type: Optional[Any] = None
    ) -> str:
        """
        Applies dynamic INT8 quantization to an ONNX model, reducing model size
        and accelerating inference on CPU targets using onnxruntime.quantization.
        """
        from onnxruntime.quantization import quantize_dynamic, QuantType

        in_p = Path(input_onnx_path)
        if not in_p.is_file():
            raise FileNotFoundError(f"Input ONNX model not found: {input_onnx_path}")

        out_p = Path(output_onnx_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        w_type = weight_type if weight_type is not None else QuantType.QUInt8
        quantize_dynamic(
            model_input=str(in_p),
            model_output=str(out_p),
            weight_type=w_type
        )
        return str(out_p)

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

        batch_dim = torch.export.Dim("batch_size", min=1, max=256)
        dynamic_shapes = {"x": {0: batch_dim}}

        torch.onnx.export(
            model,
            dummy_input,
            str(p),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["output0"],
            dynamic_shapes=dynamic_shapes,
        )

        # Validate ONNX graph
        onnx_model = onnx.load(str(p))
        onnx.checker.check_model(onnx_model)

        return str(p)
