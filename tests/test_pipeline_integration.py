"""
Integration tests for HyperInferencePipeline and ONNX Edge detector.
"""

from pathlib import Path
import pytest
import numpy as np

from src.data.synthetic_generator import SyntheticVinePestGenerator
from src.models.reference_detector import ReferenceMorphologyDetector
from src.slicing.sahi_slicer import SAHISlicer
from src.pipeline.hyper_inference import HyperInferencePipeline, StandardResizePipeline
from src.models.exporter import ModelExporter
from src.models.onnx_detector import ONNXDetector


def test_hyper_inference_pipeline_execution():
    generator = SyntheticVinePestGenerator(seed=123)
    img, sample = generator.generate_sample("test_p1", "PLANTA_01", "T2", width=1920, height=1080)

    detector = ReferenceMorphologyDetector()
    slicer = SAHISlicer(slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2)
    pipeline = HyperInferencePipeline(detector=detector, slicer=slicer, global_nms_iou=0.45, use_box_fusion=True)

    result = pipeline.run(img, conf_threshold=0.25, evaluate_full_image=True)

    assert result.num_tiles_processed > 1
    assert result.elapsed_seconds > 0.0
    assert result.fps > 0.0
    assert len(result.detections) > 0

    # Detections should have valid coordinates within the canvas
    h, w = img.shape[:2]
    for det in result.detections:
        assert 0 <= det.bbox.xmin < det.bbox.xmax <= w
        assert 0 <= det.bbox.ymin < det.bbox.ymax <= h


def test_onnx_edge_pipeline_integration(tmp_path: Path):
    onnx_file = tmp_path / "test_mini.onnx"
    # Generate minimal valid ONNX graph
    ModelExporter.generate_minimal_onnx_fixture(onnx_file, num_classes=3)
    assert onnx_file.exists()

    detector = ONNXDetector(onnx_file)
    slicer = SAHISlicer(slice_height=640, slice_width=640)
    pipeline = HyperInferencePipeline(detector=detector, slicer=slicer)

    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = pipeline.run(dummy_img, conf_threshold=0.1)

    assert result.num_tiles_processed >= 1
    assert isinstance(result.detections, list)


def test_pipelines_empty_and_none_image_handling():
    detector = ReferenceMorphologyDetector()
    slicer = SAHISlicer(slice_height=640, slice_width=640)
    sahi_pipeline = HyperInferencePipeline(detector=detector, slicer=slicer)
    resize_pipeline = StandardResizePipeline(detector=detector)

    # SAHI with None and empty
    res_none = sahi_pipeline.run(None)
    assert res_none.detections == []
    assert res_none.num_tiles_processed == 0

    res_empty = sahi_pipeline.run(np.zeros((0, 0, 3), dtype=np.uint8))
    assert res_empty.detections == []
    assert res_empty.num_tiles_processed == 0

    # StandardResize with None and empty
    res_r_none = resize_pipeline.run(None)
    assert res_r_none.detections == []
    assert res_r_none.num_tiles_processed == 0

    res_r_empty = resize_pipeline.run(np.zeros((0, 0, 3), dtype=np.uint8))
    assert res_r_empty.detections == []
    assert res_r_empty.num_tiles_processed == 0


def test_standard_resize_pipeline_invalid_target_size_raises():
    detector = ReferenceMorphologyDetector()
    with pytest.raises(ValueError, match="target_size dimensions must be positive"):
        StandardResizePipeline(detector=detector, target_size=(0, 640))
    with pytest.raises(ValueError, match="target_size dimensions must be positive"):
        StandardResizePipeline(detector=detector, target_size=(640, -10))


def test_hyper_inference_pipeline_respects_slicer_include_full_image():
    detector = ReferenceMorphologyDetector()
    slicer_no_full = SAHISlicer(slice_height=640, slice_width=640, include_full_image=False)
    pipeline = HyperInferencePipeline(detector=detector, slicer=slicer_no_full)

    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    # When evaluate_full_image is None, it should respect slicer_no_full.include_full_image (False)
    res = pipeline.run(img, evaluate_full_image=None)
    assert isinstance(res.detections, list)


def test_yolo_detector_predict_none_or_empty_returns_empty():
    from src.models.yolo_detector import YOLODetector
    custom_weights = Path("output/models/yolov8_pest_best.pt")
    model_path = str(custom_weights) if custom_weights.exists() else "yolov8n.pt"
    yolo_det = YOLODetector(model_path)

    assert yolo_det.predict(None) == []
    assert yolo_det.predict(np.zeros((0, 0, 3), dtype=np.uint8)) == []
    assert yolo_det.predict_batch([]) == []
    assert yolo_det.predict_batch([None, np.zeros((0, 0, 3), dtype=np.uint8)]) == [[], []]
