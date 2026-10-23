"""
Metrics and Evaluation Engine for Object Detection and Agronomic Validation.
Implements vectorized IoU calculations, precision-recall curve interpolation,
and standard COCO/PASCAL VOC evaluation metrics (AP@50, AP@50:95).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np

from src.domain.entities import BoundingBox, Detection


@dataclass(frozen=True)
class ClassMetrics:
    """Metrics calculated for an individual class."""
    class_id: int
    class_name: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    average_precision_50: float
    average_precision_75: float
    average_precision_coco: float


@dataclass(frozen=True)
class EvaluationReport:
    """Comprehensive evaluation summary across all dataset samples."""
    per_class: Dict[int, ClassMetrics]
    mean_ap_50: float
    mean_ap_75: float
    mean_ap_coco: float
    overall_precision: float
    overall_recall: float
    overall_f1: float
    total_ground_truths: int
    total_predictions: int


def calculate_iou_matrix_numpy(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Vectorized pairwise IoU calculation between two sets of bounding boxes.
    Equivalent to efficient nested matrix operations in C SIMD.
    
    Args:
        boxes_a: Array of shape (N, 4) with [xmin, ymin, xmax, ymax]
        boxes_b: Array of shape (M, 4) with [xmin, ymin, xmax, ymax]
        
    Returns:
        Matrix of shape (N, M) where [i, j] is IoU between box_a[i] and box_b[j].
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    # Compute intersection coordinates via broadcasting
    # boxes_a[:, None, :] shape: (N, 1, 4), boxes_b[None, :, :] shape: (1, M, 4)
    ixmin = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    iymin = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    ixmax = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    iymax = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = np.maximum(0.0, ixmax - ixmin)
    inter_h = np.maximum(0.0, iymax - iymin)
    inter_area = inter_w * inter_h

    # Compute individual areas
    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])

    union_area = area_a[:, None] + area_b[None, :] - inter_area
    iou_matrix = np.where(union_area > 0, inter_area / union_area, 0.0)
    return iou_matrix.astype(np.float32)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Computes Average Precision (AP) using the continuous area under the
    envelope precision-recall curve (VOC / COCO standard).
    """
    # Prepend and append sentinel boundary values
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope (ensure monotonically decreasing from right to left)
    # Vectorized: accumulate maximum from right in a single C-level pass
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]

    # Find points where recall changes
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    # Sum area of rectangles: ΔRecall * Precision
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return float(ap)


def evaluate_detections(
    predictions_per_image: List[List[Detection]],
    ground_truths_per_image: List[List[Detection]],
    iou_thresholds: Optional[List[float]] = None
) -> EvaluationReport:
    """
    Evaluates detector predictions against ground truth across multiple images.
    
    Args:
        predictions_per_image: List of detections for each image.
        ground_truths_per_image: List of ground truth annotations for each image.
        iou_thresholds: List of IoU thresholds. Defaults to [0.5, 0.55, ..., 0.95].
    """
    if len(predictions_per_image) != len(ground_truths_per_image):
        raise ValueError(
            f"Image list count mismatch: got {len(predictions_per_image)} prediction sets "
            f"and {len(ground_truths_per_image)} ground truth sets."
        )

    if iou_thresholds is None:
        iou_thresholds = list(np.arange(0.50, 0.96, 0.05))

    # Collect all unique class IDs
    all_classes: set[int] = set()
    class_names: Dict[int, str] = {}
    for gts in ground_truths_per_image:
        for g in gts:
            all_classes.add(g.class_id)
            class_names[g.class_id] = g.class_name
    for preds in predictions_per_image:
        for p in preds:
            all_classes.add(p.class_id)
            if p.class_id not in class_names:
                class_names[p.class_id] = p.class_name

    per_class_results: Dict[int, ClassMetrics] = {}
    total_gt = sum(len(gts) for gts in ground_truths_per_image)
    total_pred = sum(len(preds) for preds in predictions_per_image)

    for cid in sorted(all_classes):
        cname = class_names.get(cid, f"class_{cid}")

        # Evaluate across all iou thresholds
        ap_list: List[float] = []
        tp_50 = 0
        fp_50 = 0
        fn_50 = 0
        precision_50 = 0.0
        recall_50 = 0.0
        f1_50 = 0.0

        for thresh in iou_thresholds:
            # Flatten predictions and ground truth matches for this class
            scores: List[float] = []
            is_tp_list: List[int] = []
            num_gt_class = 0

            for preds, gts in zip(predictions_per_image, ground_truths_per_image):
                c_preds = [p for p in preds if p.class_id == cid]
                c_gts = [g for g in gts if g.class_id == cid]
                num_gt_class += len(c_gts)

                if len(c_preds) == 0:
                    continue

                # Sort predictions by confidence descending
                c_preds = sorted(c_preds, key=lambda x: x.confidence, reverse=True)

                if len(c_gts) == 0:
                    for p in c_preds:
                        scores.append(p.confidence)
                        is_tp_list.append(0)
                    continue

                pred_boxes = np.array([p.bbox.to_xyxy() for p in c_preds], dtype=np.float32)
                gt_boxes = np.array([g.bbox.to_xyxy() for g in c_gts], dtype=np.float32)

                iou_mat = calculate_iou_matrix_numpy(pred_boxes, gt_boxes)
                matched_gt = set()

                for p_idx, p in enumerate(c_preds):
                    scores.append(p.confidence)
                    # Mask already-matched GTs and find best remaining match via vectorized argmax
                    row = iou_mat[p_idx].copy()
                    if matched_gt:
                        row[list(matched_gt)] = 0.0
                    best_gt_idx = int(np.argmax(row))
                    best_iou = float(row[best_gt_idx])

                    if best_iou >= thresh and best_iou > 0.0:
                        is_tp_list.append(1)
                        matched_gt.add(best_gt_idx)
                    else:
                        is_tp_list.append(0)

            if num_gt_class == 0:
                ap_list.append(0.0)
                if abs(thresh - 0.50) < 1e-4:
                    tp_50 = 0
                    fp_50 = len(scores)
                    fn_50 = 0
                    precision_50 = 0.0
                    recall_50 = 0.0
                    f1_50 = 0.0
                continue

            # Sort cumulative TP and FP by descending confidence
            sort_order = np.argsort(-np.array(scores)) if len(scores) > 0 else np.array([], dtype=int)
            tp_sorted = np.array(is_tp_list, dtype=np.float32)[sort_order] if len(sort_order) > 0 else np.array([])
            fp_sorted = 1.0 - tp_sorted if len(tp_sorted) > 0 else np.array([])

            acc_tp = np.cumsum(tp_sorted)
            acc_fp = np.cumsum(fp_sorted)

            recalls = acc_tp / max(1, num_gt_class)
            precisions = acc_tp / np.maximum(acc_tp + acc_fp, np.finfo(np.float32).eps)

            current_ap = compute_ap(recalls, precisions)
            ap_list.append(current_ap)

            # Record stats at 0.50 threshold
            if abs(thresh - 0.50) < 1e-4:
                tp_50 = int(acc_tp[-1]) if len(acc_tp) > 0 else 0
                fp_50 = int(acc_fp[-1]) if len(acc_fp) > 0 else 0
                fn_50 = num_gt_class - tp_50
                precision_50 = float(tp_50 / max(1, tp_50 + fp_50)) if (tp_50 + fp_50) > 0 else 0.0
                recall_50 = float(tp_50 / max(1, num_gt_class)) if num_gt_class > 0 else 0.0
                denom = precision_50 + recall_50
                f1_50 = float(2 * precision_50 * recall_50 / denom) if denom > 0 else 0.0

        ap50 = ap_list[0] if len(ap_list) > 0 else 0.0
        ap75 = ap_list[5] if len(ap_list) > 5 else 0.0
        ap_coco = float(np.mean(ap_list)) if len(ap_list) > 0 else 0.0

        per_class_results[cid] = ClassMetrics(
            class_id=cid,
            class_name=cname,
            true_positives=tp_50,
            false_positives=fp_50,
            false_negatives=fn_50,
            precision=precision_50,
            recall=recall_50,
            f1_score=f1_50,
            average_precision_50=ap50,
            average_precision_75=ap75,
            average_precision_coco=ap_coco
        )

    if per_class_results:
        m_ap50 = float(np.mean([m.average_precision_50 for m in per_class_results.values()]))
        m_ap75 = float(np.mean([m.average_precision_75 for m in per_class_results.values()]))
        m_ap_coco = float(np.mean([m.average_precision_coco for m in per_class_results.values()]))
        avg_p = float(np.mean([m.precision for m in per_class_results.values()]))
        avg_r = float(np.mean([m.recall for m in per_class_results.values()]))
        denom = avg_p + avg_r
        overall_f1 = (2 * avg_p * avg_r / denom) if denom > 0 else 0.0
    else:
        m_ap50 = m_ap75 = m_ap_coco = avg_p = avg_r = overall_f1 = 0.0

    return EvaluationReport(
        per_class=per_class_results,
        mean_ap_50=m_ap50,
        mean_ap_75=m_ap75,
        mean_ap_coco=m_ap_coco,
        overall_precision=avg_p,
        overall_recall=avg_r,
        overall_f1=overall_f1,
        total_ground_truths=total_gt,
        total_predictions=total_pred
    )
