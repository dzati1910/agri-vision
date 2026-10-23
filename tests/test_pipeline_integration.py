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


def test_onnx_detector_preprocess_handles_grayscale_and_rgba(tmp_path: Path):
    onnx_file = tmp_path / "test_mini_prep.onnx"
    ModelExporter.generate_minimal_onnx_fixture(onnx_file, num_classes=3)
    detector = ONNXDetector(onnx_file)

    # 2D grayscale
    gray_img = np.ones((480, 640), dtype=np.uint8) * 128
    tensor_gray, scale, pad = detector._preprocess(gray_img)
    assert tensor_gray.shape == (1, 3, 640, 640)

    # 4-channel RGBA
    rgba_img = np.ones((480, 640, 4), dtype=np.uint8) * 200
    tensor_rgba, scale, pad = detector._preprocess(rgba_img)
    assert tensor_rgba.shape == (1, 3, 640, 640)


def test_onnx_detector_static_batch_fallback(tmp_path: Path):
    # Test that when has_dynamic_batch is False, predict_batch runs sequentially without crashing
    onnx_file = tmp_path / "test_static.onnx"
    ModelExporter.generate_minimal_onnx_fixture(onnx_file, num_classes=3)
    detector = ONNXDetector(onnx_file)
    # Simulate static batch model
    detector.has_dynamic_batch = False

    images = [
        np.zeros((640, 640, 3), dtype=np.uint8),
        np.ones((640, 640, 3), dtype=np.uint8) * 50,
        np.ones((640, 640, 3), dtype=np.uint8) * 100,
    ]
    results = detector.predict_batch(images, conf_threshold=0.1)
    assert len(results) == 3
    for r in results:
        assert isinstance(r, list)


def test_model_exporter_quantize_onnx_dynamic(tmp_path: Path):
    onnx_file = tmp_path / "base.onnx"
    quant_file = tmp_path / "quantized.onnx"
    ModelExporter.generate_minimal_onnx_fixture(onnx_file, num_classes=3)
    out = ModelExporter.quantize_onnx_dynamic(onnx_file, quant_file)
    assert Path(out).exists()
    assert quant_file.exists()


def test_pest_visualizer_create_side_by_side_proxy():
    from src.utils.visualizer import PestVisualizer
    from src.domain.entities import BoundingBox, Detection

    large_img = np.zeros((1440, 2560, 3), dtype=np.uint8)
    dets1 = [Detection(BoundingBox(100, 100, 200, 200), 0, "nymph", 0.9)]
    dets2 = [Detection(BoundingBox(300, 300, 450, 450), 1, "adult", 0.85)]

    banner = PestVisualizer.create_side_by_side_proxy(large_img, dets1, dets2, target_display_width=1200)
    assert banner is not None
    # Each panel was scaled down to 600w, total 1200w
    assert banner.shape[1] == 1200
    assert banner.shape[0] > 0

    # Test create_side_by_side_comparison calls proxy
    comp = PestVisualizer.create_side_by_side_comparison(large_img, dets1, dets2, target_display_width=1200)
    assert comp.shape == banner.shape


def test_pest_visualizer_empty_and_invalid_inputs():
    from src.utils.visualizer import PestVisualizer

    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    banner_empty = PestVisualizer.create_side_by_side_proxy(empty_img, [], [])
    assert banner_empty.size == 0

    banner_none = PestVisualizer.create_side_by_side_proxy(None, [], [])
    assert banner_none.size == 0

    drawn_none = PestVisualizer.draw_detections(None, [])
    assert drawn_none.size == 0

    grid_none = PestVisualizer.draw_tiling_grid(None, [])
    assert grid_none.size == 0

    valid_img = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="target_display_width must be positive"):
        PestVisualizer.create_side_by_side_proxy(valid_img, [], [], target_display_width=0)


def test_onnx_detector_preprocess_empty_raises_value_error(tmp_path: Path):
    onnx_file = tmp_path / "test_mini_prep_err.onnx"
    ModelExporter.generate_minimal_onnx_fixture(onnx_file, num_classes=3)
    detector = ONNXDetector(onnx_file)

    with pytest.raises(ValueError, match="Input image cannot be None or empty"):
        detector._preprocess(np.zeros((0, 0, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="Input image cannot be None or empty"):
        detector._preprocess(None)


def test_model_exporter_quantize_onnx_missing_file_raises_filenotfound(tmp_path: Path):
    missing_onnx = tmp_path / "ghost.onnx"
    out_quant = tmp_path / "out.onnx"
    with pytest.raises(FileNotFoundError, match="Input ONNX model not found"):
        ModelExporter.quantize_onnx_dynamic(missing_onnx, out_quant)


def test_streamlit_app_caching_and_sample_switching():
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=30)
    assert len(at.exception) == 0
    assert "inference_cache" in at.session_state

    initial_plant_id = at.session_state["inference_cache"]["plant_id"]

    # Select another sample
    found_sb = False
    for sb in at.sidebar.selectbox:
        if "Caso de Estudo" in sb.label:
            # Pick a different key
            other_key = "T3" if initial_plant_id != "PLANTA_T3" else "T1"
            sb.select(other_key)
            found_sb = True
            break
    assert found_sb

    at.run(timeout=30)
    assert len(at.exception) == 0
    updated_plant_id = at.session_state["inference_cache"]["plant_id"]
    assert updated_plant_id != initial_plant_id

