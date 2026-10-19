"""
Unit tests for domain entities and evaluation metrics.
"""

import pytest
import numpy as np

from src.domain.entities import BoundingBox, Detection, PestClass
from src.domain.metrics import (
    calculate_iou_matrix_numpy,
    compute_ap,
    evaluate_detections
)


def test_bounding_box_properties():
    box = BoundingBox(xmin=10.0, ymin=20.0, xmax=50.0, ymax=80.0)
    assert box.width == 40.0
    assert box.height == 60.0
    assert box.area == 2400.0
    assert box.center == (30.0, 50.0)
    assert pytest.approx(box.aspect_ratio) == 40.0 / 60.0


def test_bounding_box_invalid_raises():
    with pytest.raises(ValueError, match="xmax"):
        BoundingBox(xmin=50.0, ymin=10.0, xmax=20.0, ymax=60.0)
    with pytest.raises(ValueError, match="ymax"):
        BoundingBox(xmin=10.0, ymin=60.0, xmax=50.0, ymax=20.0)


def test_bounding_box_yolo_roundtrip():
    img_w, img_h = 1920, 1080
    orig_box = BoundingBox(xmin=100.0, ymin=200.0, xmax=300.0, ymax=400.0)
    cx, cy, w, h = orig_box.to_yolo(img_w, img_h)

    reconstructed = BoundingBox.from_yolo(cx, cy, w, h, img_w, img_h)
    assert pytest.approx(reconstructed.xmin, abs=1e-3) == orig_box.xmin
    assert pytest.approx(reconstructed.ymin, abs=1e-3) == orig_box.ymin
    assert pytest.approx(reconstructed.xmax, abs=1e-3) == orig_box.xmax
    assert pytest.approx(reconstructed.ymax, abs=1e-3) == orig_box.ymax


def test_iou_calculation():
    box_a = BoundingBox(xmin=0.0, ymin=0.0, xmax=10.0, ymax=10.0)
    box_b = BoundingBox(xmin=5.0, ymin=0.0, xmax=15.0, ymax=10.0)
    # Intersection: [5, 0] to [10, 10] -> 5 * 10 = 50
    # Area A = 100, Area B = 100, Union = 150
    # IoU = 50 / 150 = 1/3
    assert pytest.approx(box_a.iou(box_b), abs=1e-4) == 1.0 / 3.0


def test_vectorized_iou_matrix():
    boxes_a = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [20.0, 20.0, 30.0, 30.0]
    ], dtype=np.float32)
    boxes_b = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [5.0, 0.0, 15.0, 10.0],
        [100.0, 100.0, 110.0, 110.0]
    ], dtype=np.float32)

    mat = calculate_iou_matrix_numpy(boxes_a, boxes_b)
    assert mat.shape == (2, 3)
    assert pytest.approx(mat[0, 0]) == 1.0
    assert pytest.approx(mat[0, 1], abs=1e-4) == 1.0 / 3.0
    assert mat[0, 2] == 0.0
    assert mat[1, 0] == 0.0


def test_evaluate_detections_metrics():
    # 1 image, 1 GT, 1 perfect prediction
    gt = [Detection(bbox=BoundingBox(10, 10, 50, 50), class_id=0, class_name="nymph")]
    pred = [Detection(bbox=BoundingBox(10, 10, 50, 50), class_id=0, class_name="nymph", confidence=0.95)]

    report = evaluate_detections([[pred[0]]], [[gt[0]]])
    assert report.overall_precision == 1.0
    assert report.overall_recall == 1.0
    assert report.overall_f1 == 1.0
    assert report.mean_ap_50 == 1.0


def test_evaluate_detections_false_positives_without_ground_truth():
    # 1 image with false alarm prediction and NO ground truth
    pred = [Detection(bbox=BoundingBox(10, 10, 50, 50), class_id=0, class_name="nymph", confidence=0.85)]
    report = evaluate_detections([[pred[0]]], [[]])

    assert 0 in report.per_class
    c_metric = report.per_class[0]
    assert c_metric.true_positives == 0
    assert c_metric.false_positives == 1
    assert c_metric.false_negatives == 0
    assert c_metric.precision == 0.0
    assert c_metric.recall == 0.0


def test_bounding_box_from_yolo_invalid_dimensions_raises():
    with pytest.raises(ValueError, match="Image dimensions must be positive"):
        BoundingBox.from_yolo(0.5, 0.5, 0.1, 0.1, img_width=0, img_height=100)
    with pytest.raises(ValueError, match="Image dimensions must be positive"):
        BoundingBox.from_yolo(0.5, 0.5, 0.1, 0.1, img_width=100, img_height=-10)
    with pytest.raises(ValueError, match="Normalized width and height must be non-negative"):
        BoundingBox.from_yolo(0.5, 0.5, -0.1, 0.1, img_width=100, img_height=100)


def test_bounding_box_clip_invalid_dimensions_raises():
    box = BoundingBox(10, 10, 50, 50)
    with pytest.raises(ValueError, match="Clipping bounds must be positive"):
        box.clip(max_width=0, max_height=100)
    with pytest.raises(ValueError, match="Clipping bounds must be positive"):
        box.clip(max_width=100, max_height=-5)


def test_evaluate_detections_mismatched_lengths_raises():
    pred = [[Detection(bbox=BoundingBox(10, 10, 50, 50), class_id=0, class_name="nymph")]]
    with pytest.raises(ValueError, match="Image list count mismatch"):
        evaluate_detections(pred, [[], []])


def test_runtime_type_hints_all_modules():
    import importlib
    import inspect
    import pkgutil
    import typing
    import src

    for module_info in pkgutil.walk_packages(src.__path__, src.__name__ + '.'):
        mod = importlib.import_module(module_info.name)
        for name, obj in inspect.getmembers(mod):
            if inspect.isfunction(obj) and getattr(obj, '__module__', '') == mod.__name__:
                hints = typing.get_type_hints(obj)
                assert isinstance(hints, dict)
            elif inspect.isclass(obj) and getattr(obj, '__module__', '') == mod.__name__:
                for m_name, m_obj in inspect.getmembers(obj, predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x)):
                    hints = typing.get_type_hints(m_obj)
                    assert isinstance(hints, dict)
