"""
Unit tests for vectorized NMS, Soft-NMS, and Weighted Boxes Fusion.
"""

import pytest
import numpy as np

from src.domain.entities import BoundingBox, Detection
from src.core.nms import (
    nms_numpy,
    soft_nms_numpy,
    weighted_boxes_fusion,
    border_aware_box_fusion,
    apply_nms
)


def test_nms_suppression():
    # Two heavily overlapping boxes, one with higher score
    boxes = np.array([
        [10.0, 10.0, 50.0, 50.0],
        [12.0, 12.0, 50.0, 50.0],  # IoU > 0.8
        [100.0, 100.0, 150.0, 150.0]  # Disjoint box
    ], dtype=np.float32)
    scores = np.array([0.9, 0.7, 0.85], dtype=np.float32)

    keep = nms_numpy(boxes, scores, iou_threshold=0.5)
    # Box 1 should be suppressed by Box 0; Box 2 must be kept
    assert set(keep) == {0, 2}
    assert keep[0] == 0  # Sorted by score


def test_soft_nms_decay():
    boxes = np.array([
        [10.0, 10.0, 50.0, 50.0],
        [12.0, 12.0, 50.0, 50.0]
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)

    keep_idx, updated_scores = soft_nms_numpy(boxes, scores, iou_threshold=0.5, sigma=0.5)
    # Soft NMS should keep both, but the second box's score must be penalized
    assert len(keep_idx) == 2
    assert updated_scores[1] < 0.8


def test_weighted_boxes_fusion():
    b1 = Detection(BoundingBox(10.0, 10.0, 50.0, 50.0), class_id=0, class_name="nymph", confidence=0.8)
    b2 = Detection(BoundingBox(12.0, 12.0, 52.0, 52.0), class_id=0, class_name="nymph", confidence=0.8)

    fused = weighted_boxes_fusion([b1, b2], iou_threshold=0.5)
    assert len(fused) == 1
    # Coordinates should be exactly the midpoint since confidences are equal
    assert pytest.approx(fused[0].bbox.xmin) == 11.0
    assert pytest.approx(fused[0].bbox.ymin) == 11.0
    assert pytest.approx(fused[0].bbox.xmax) == 51.0
    assert pytest.approx(fused[0].bbox.ymax) == 51.0


def test_apply_nms_class_aware():
    # Overlapping boxes of DIFFERENT classes should NOT suppress each other
    b1 = Detection(BoundingBox(10.0, 10.0, 50.0, 50.0), class_id=0, class_name="nymph", confidence=0.9)
    b2 = Detection(BoundingBox(11.0, 11.0, 51.0, 51.0), class_id=1, class_name="adult", confidence=0.85)

    kept = apply_nms([b1, b2], iou_threshold=0.5, class_agnostic=False)
    assert len(kept) == 2


def test_soft_nms_unsorted_scores_preserves_original_indices():
    # Box 0 has lower score, Box 1 has higher score and distinct coordinates
    boxes = np.array([
        [0.0, 0.0, 10.0, 10.0],       # Box 0 (weaker)
        [100.0, 100.0, 110.0, 110.0]  # Box 1 (stronger)
    ], dtype=np.float32)
    scores = np.array([0.4, 0.95], dtype=np.float32)

    # Filter with score_threshold=0.5: only Box 1 should survive
    keep_idx, keep_scores = soft_nms_numpy(boxes, scores, score_threshold=0.5)
    assert len(keep_idx) == 1
    assert keep_idx[0] == 1  # Must map to original index 1!
    assert keep_scores[0] == pytest.approx(0.95)
    assert np.allclose(boxes[keep_idx[0]], [100.0, 100.0, 110.0, 110.0])


def test_soft_nms_linear_decay():
    boxes = np.array([
        [10.0, 10.0, 50.0, 50.0],
        [12.0, 12.0, 50.0, 50.0]
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)

    keep_idx, updated_scores = soft_nms_numpy(boxes, scores, iou_threshold=0.3, method="linear")
    assert len(keep_idx) == 2
    assert updated_scores[1] < 0.8


def test_soft_nms_invalid_method_raises():
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    with pytest.raises(ValueError, match="Unknown Soft-NMS method"):
        soft_nms_numpy(boxes, scores, method="unsupported_method")


def test_soft_nms_invalid_sigma_raises():
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    with pytest.raises(ValueError, match="sigma must be positive"):
        soft_nms_numpy(boxes, scores, sigma=0.0)


def test_soft_nms_invalid_boxes_shape_raises():
    boxes_1d = np.array([0, 0, 10, 10], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    with pytest.raises(ValueError, match="boxes must have shape"):
        soft_nms_numpy(boxes_1d, scores)


def test_weighted_boxes_fusion_invalid_iou_raises():
    b1 = Detection(BoundingBox(0, 0, 10, 10), class_id=0, class_name="nymph", confidence=0.8)
    with pytest.raises(ValueError, match="iou_threshold must be in"):
        weighted_boxes_fusion([b1], iou_threshold=-0.5)
    with pytest.raises(ValueError, match="iou_threshold must be in"):
        weighted_boxes_fusion([b1], iou_threshold=1.5)


def test_nms_empty_or_single():
    # Empty
    assert len(nms_numpy(np.zeros((0, 4), dtype=np.float32), np.array([], dtype=np.float32))) == 0
    # Single
    single = nms_numpy(np.array([[0, 0, 10, 10]], dtype=np.float32), np.array([0.9], dtype=np.float32))
    assert list(single) == [0]
    # apply_nms empty and single
    assert apply_nms([]) == []
    b = Detection(BoundingBox(0, 0, 10, 10), class_id=0, class_name="nymph", confidence=0.8)
    assert len(apply_nms([b])) == 1


def test_border_aware_box_fusion_empty_and_single():
    assert border_aware_box_fusion([]) == []
    b1 = Detection(BoundingBox(10, 10, 50, 50), class_id=0, class_name="nymph", confidence=0.9)
    assert len(border_aware_box_fusion([b1])) == 1


def test_border_aware_box_fusion_invalid_params_raises():
    b1 = Detection(BoundingBox(10, 10, 50, 50), class_id=0, class_name="nymph", confidence=0.9)
    with pytest.raises(ValueError, match="iou_threshold must be in"):
        border_aware_box_fusion([b1], iou_threshold=-0.1)
    with pytest.raises(ValueError, match="containment_threshold must be in"):
        border_aware_box_fusion([b1], containment_threshold=1.5)


def test_border_aware_box_fusion_edge_truncation_preserves_outer_extent():
    truncated_det = Detection(BoundingBox(620, 100, 640, 130), class_id=0, class_name="nymph", confidence=0.85)
    full_det = Detection(BoundingBox(620, 100, 655, 130), class_id=0, class_name="nymph", confidence=0.90)

    fused = border_aware_box_fusion([truncated_det, full_det], iou_threshold=0.55, containment_threshold=0.70)
    assert len(fused) == 1
    assert fused[0].bbox.xmax == 655.0
    assert fused[0].bbox.xmin == 620.0
    assert fused[0].confidence == pytest.approx(0.90)


def test_border_aware_box_fusion_dense_colony_preserves_distinct_pests():
    p1 = Detection(BoundingBox(100, 100, 125, 125), class_id=0, class_name="nymph", confidence=0.92)
    p2 = Detection(BoundingBox(120, 100, 145, 125), class_id=0, class_name="nymph", confidence=0.88)

    fused = border_aware_box_fusion([p1, p2], iou_threshold=0.55, containment_threshold=0.70)
    assert len(fused) == 2


def test_border_aware_box_fusion_class_aware():
    p1 = Detection(BoundingBox(100, 100, 150, 150), class_id=0, class_name="nymph", confidence=0.92)
    p2 = Detection(BoundingBox(105, 105, 145, 145), class_id=1, class_name="adult", confidence=0.88)

    fused = border_aware_box_fusion([p1, p2])
    assert len(fused) == 2


def test_border_aware_box_fusion_area_ratio_threshold_validation():
    b1 = Detection(BoundingBox(10, 10, 50, 50), class_id=0, class_name="nymph", confidence=0.9)
    with pytest.raises(ValueError, match="area_ratio_threshold must be in"):
        border_aware_box_fusion([b1], area_ratio_threshold=-0.1)
    with pytest.raises(ValueError, match="area_ratio_threshold must be in"):
        border_aware_box_fusion([b1], area_ratio_threshold=1.5)


def test_border_aware_box_fusion_multi_fragment_connected_components():
    # Multi-tile seam: 2 truncated fragments and 1 full box of the same pest
    d1 = Detection(BoundingBox(100, 100, 120, 140), class_id=0, class_name="nymph", confidence=0.85)
    d2 = Detection(BoundingBox(110, 100, 130, 140), class_id=0, class_name="nymph", confidence=0.90)
    d3 = Detection(BoundingBox(100, 100, 135, 140), class_id=0, class_name="nymph", confidence=0.75)

    fused = border_aware_box_fusion([d1, d2, d3])
    assert len(fused) == 1
    assert fused[0].bbox.xmin == 100.0
    assert fused[0].bbox.xmax == 135.0
    assert fused[0].bbox.ymin == 100.0
    assert fused[0].bbox.ymax == 140.0
    assert fused[0].confidence == pytest.approx(0.90)

