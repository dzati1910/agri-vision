"""
Vectorized Non-Maximum Suppression (NMS) and Box Fusion Engine.

Memory Architecture Notes for C Programmers:
- NumPy arrays are row-major contiguous C-arrays (C_CONTIGUOUS).
- Slicing creates views into existing RAM buffers without malloc/memcpy.
- Vectorized broadcasting replaces O(N^2) pointer loops with optimized SIMD instructions (AVX2/AVX-512).
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

from src.domain.entities import BoundingBox, Detection
from src.domain.metrics import calculate_iou_matrix_numpy


def nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5
) -> np.ndarray:
    """
    Standard greedy Non-Maximum Suppression using vectorized NumPy operations.
    
    Args:
        boxes: Array of shape (N, 4) with [xmin, ymin, xmax, ymax].
        scores: Array of shape (N,) with confidence scores.
        iou_threshold: Jaccard overlap threshold for suppression.
        
    Returns:
        Array of selected indices sorted by decreasing confidence.
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    # In C terms: float* x1 = boxes[:, 0], etc.
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    # argsort returns indices that would sort the array descending
    order = np.argsort(scores)[::-1]

    keep: List[int] = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        # Compute intersection of top candidate with remaining candidates
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)

        # Keep only elements with IoU less than or equal to threshold
        remaining_indices = np.where(iou <= iou_threshold)[0]
        # Offset by 1 because order[0] was the current pivot
        order = order[remaining_indices + 1]

    return np.array(keep, dtype=np.int64)


def soft_nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
    sigma: float = 0.5,
    score_threshold: float = 0.05,
    method: str = "gaussian"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Soft-NMS algorithm (Bodla et al., 2017).
    Instead of completely eliminating overlapping candidates, dampens confidence scores.
    Especially effective in high-density insect clusters (e.g., Planococcus citri colonies).
    
    Args:
        boxes: Array of shape (N, 4).
        scores: Array of shape (N,).
        iou_threshold: Overlap threshold for linear decay.
        sigma: Variance parameter for gaussian decay.
        score_threshold: Pruning threshold for decayed scores.
        method: "gaussian" or "linear".
        
    Returns:
        Tuple of (selected_indices, updated_scores).
    """
    if method not in ("gaussian", "linear"):
        raise ValueError(f"Unknown Soft-NMS method: '{method}'. Supported methods are 'gaussian' and 'linear'.")
    if sigma <= 0.0:
        raise ValueError(f"Gaussian sigma must be positive, got: {sigma}")

    N = boxes.shape[0]
    if N == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes must have shape (N, 4), got: {boxes.shape}")

    b = boxes.copy()
    s = scores.copy()
    orig_indices = np.arange(N, dtype=np.int64)
    areas = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])

    keep_indices: List[int] = []
    keep_scores: List[float] = []

    for i in range(N):
        # Find index of max score in remaining items [i:N]
        max_pos = i + int(np.argmax(s[i:]))
        # Swap current item with max item (in-place swap like in C)
        b[[i, max_pos]] = b[[max_pos, i]]
        s[[i, max_pos]] = s[[max_pos, i]]
        areas[[i, max_pos]] = areas[[max_pos, i]]
        orig_indices[[i, max_pos]] = orig_indices[[max_pos, i]]

        pos = i
        if s[pos] < score_threshold:
            # Since s[pos] is the maximum among remaining items, all subsequent scores are below threshold
            break

        keep_indices.append(int(orig_indices[pos]))
        keep_scores.append(float(s[pos]))

        # Compute IoU of pivot box with all subsequent boxes
        xx1 = np.maximum(b[pos, 0], b[pos + 1:, 0])
        yy1 = np.maximum(b[pos, 1], b[pos + 1:, 1])
        xx2 = np.minimum(b[pos, 2], b[pos + 1:, 2])
        yy2 = np.minimum(b[pos, 3], b[pos + 1:, 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[pos] + areas[pos + 1:] - inter
        iou = np.where(union > 0, inter / union, 0.0)

        # Decay scores
        if method == "gaussian":
            decay = np.exp(-(iou ** 2) / sigma)
        else:  # linear
            decay = np.where(iou > iou_threshold, 1.0 - iou, 1.0)

        s[pos + 1:] *= decay

    return np.array(keep_indices, dtype=np.int64), np.array(keep_scores, dtype=np.float32)


def weighted_boxes_fusion(
    detections: List[Detection],
    iou_threshold: float = 0.55
) -> List[Detection]:
    """
    Weighted Boxes Fusion (WBF) across overlapping tiles in SAHI.
    Averaging bounding box coordinates weighted by their confidence
    yields sub-pixel localization accuracy when combining adjacent tile cuts.
    """
    if not (0.0 <= iou_threshold <= 1.0):
        raise ValueError(f"iou_threshold must be in [0.0, 1.0], got: {iou_threshold}")

    if len(detections) <= 1:
        return list(detections)

    # Group detections by class
    classes = set(d.class_id for d in detections)
    final_detections: List[Detection] = []

    for cid in classes:
        c_dets = [d for d in detections if d.class_id == cid]
        # Sort descending by confidence
        c_dets.sort(key=lambda d: d.confidence, reverse=True)

        clusters: List[List[Detection]] = []

        for d in c_dets:
            matched = False
            for cluster in clusters:
                # Compare with cluster representative (first element)
                if d.bbox.iou(cluster[0].bbox) >= iou_threshold:
                    cluster.append(d)
                    matched = True
                    break
            if not matched:
                clusters.append([d])

        # Fuse each cluster
        for cluster in clusters:
            if len(cluster) == 1:
                final_detections.append(cluster[0])
                continue

            weights = np.array([item.confidence for item in cluster], dtype=np.float64)
            w_sum = max(float(np.sum(weights)), 1e-8)

            xmins = np.array([item.bbox.xmin for item in cluster], dtype=np.float64)
            ymins = np.array([item.bbox.ymin for item in cluster], dtype=np.float64)
            xmaxs = np.array([item.bbox.xmax for item in cluster], dtype=np.float64)
            ymaxs = np.array([item.bbox.ymax for item in cluster], dtype=np.float64)

            # Weighted average coordinates
            fused_xmin = float(np.sum(xmins * weights) / w_sum)
            fused_ymin = float(np.sum(ymins * weights) / w_sum)
            fused_xmax = float(np.sum(xmaxs * weights) / w_sum)
            fused_ymax = float(np.sum(ymaxs * weights) / w_sum)

            # Confidence is max confidence or weighted confidence with bonus for consensus
            fused_conf = float(max(d.confidence for d in cluster))

            fused_box = BoundingBox(
                xmin=fused_xmin,
                ymin=fused_ymin,
                xmax=max(fused_xmin, fused_xmax),
                ymax=max(fused_ymin, fused_ymax)
            )

            final_detections.append(
                Detection(
                    bbox=fused_box,
                    class_id=cluster[0].class_id,
                    class_name=cluster[0].class_name,
                    confidence=fused_conf
                )
            )

    return sorted(final_detections, key=lambda d: d.confidence, reverse=True)


def border_aware_box_fusion(
    detections: List[Detection],
    iou_threshold: float = 0.55,
    containment_threshold: float = 0.70,
    area_ratio_threshold: float = 0.80
) -> List[Detection]:
    """
    Border-Aware Box Fusion (BABF) across overlapping tiles in SAHI.
    
    Solves two critical failure modes in high-resolution agricultural pest detection:
    1. Edge-Truncated Duplications & Distortions:
       When a pest is sliced by a tile boundary, a partial/truncated detection
       is produced. Standard WBF either drops the match (IoU < threshold) leading to
       duplicate partials, or naively averages coordinates, which pulls the boundary
       inward and shrinks the insect box.
       BABF detects containment (intersection / min_area >= containment_threshold)
       and guards the outer physical boundaries (min xmin/ymin, max xmax/ymax).
       
    2. Dense Colony Preservation:
       In dense insect clusters (e.g., Planococcus citri colonies), adjacent distinct
       nymphs/adults with low overlap (IoU < iou_threshold and containment < threshold)
       are strictly preserved as distinct entities without collapsing into a single box.
       
    Args:
        detections: List of candidate Detection entities.
        iou_threshold: Intersection-over-Union threshold for similar-sized boxes (0.0 to 1.0).
        containment_threshold: Intersection-over-Area (IoA) threshold on the smaller box
                               to identify edge-truncated duplicates (0.0 to 1.0).
        area_ratio_threshold: Ratio between min and max area below which the smaller detection
                              is treated as edge-truncated rather than an unclipped duplicate.
                              
    Returns:
        Consolidated List of Detection entities, sorted descending by confidence.
    """
    if not (0.0 <= iou_threshold <= 1.0):
        raise ValueError(f"iou_threshold must be in [0.0, 1.0], got: {iou_threshold}")
    if not (0.0 <= containment_threshold <= 1.0):
        raise ValueError(f"containment_threshold must be in [0.0, 1.0], got: {containment_threshold}")
    if not (0.0 <= area_ratio_threshold <= 1.0):
        raise ValueError(f"area_ratio_threshold must be in [0.0, 1.0], got: {area_ratio_threshold}")

    if len(detections) <= 1:
        return list(detections)

    def _boxes_match(d1: Detection, d2: Detection) -> bool:
        iou = d1.bbox.iou(d2.bbox)
        if iou >= iou_threshold:
            return True
        inter = d1.bbox.intersection(d2.bbox)
        min_area = min(d1.bbox.area, d2.bbox.area)
        if min_area > 0:
            ioa = inter / min_area
            if ioa >= containment_threshold:
                larger_bbox = d2.bbox if d2.bbox.area >= d1.bbox.area else d1.bbox
                smaller_bbox = d1.bbox if d2.bbox.area >= d1.bbox.area else d2.bbox
                cx, cy = smaller_bbox.center
                if (larger_bbox.xmin <= cx <= larger_bbox.xmax and
                    larger_bbox.ymin <= cy <= larger_bbox.ymax):
                    return True
        return False

    classes = set(d.class_id for d in detections)
    final_detections: List[Detection] = []

    for cid in classes:
        c_dets = [d for d in detections if d.class_id == cid]
        c_dets.sort(key=lambda d: d.confidence, reverse=True)

        clusters: List[List[Detection]] = []

        for d in c_dets:
            matched_indices = [
                idx for idx, cl in enumerate(clusters)
                if any(_boxes_match(d, rep) for rep in cl)
            ]

            if not matched_indices:
                clusters.append([d])
            elif len(matched_indices) == 1:
                clusters[matched_indices[0]].append(d)
            else:
                primary = clusters[matched_indices[0]]
                primary.append(d)
                for idx in sorted(matched_indices[1:], reverse=True):
                    primary.extend(clusters[idx])
                    del clusters[idx]

        # Fuse each cluster
        fused_class_dets: List[Detection] = []
        for cluster in clusters:
            if len(cluster) == 1:
                fused_class_dets.append(cluster[0])
                continue

            max_area = max(item.bbox.area for item in cluster)
            has_truncated_box = any(
                (item.bbox.area / max_area < area_ratio_threshold)
                for item in cluster
                if max_area > 0
            )

            weights = np.array([item.confidence for item in cluster], dtype=np.float64)
            w_sum = max(float(np.sum(weights)), 1e-8)

            if has_truncated_box:
                # Border-guard: protect outer boundary from being shrunken by truncated slice
                fused_xmin = float(min(item.bbox.xmin for item in cluster))
                fused_ymin = float(min(item.bbox.ymin for item in cluster))
                fused_xmax = float(max(item.bbox.xmax for item in cluster))
                fused_ymax = float(max(item.bbox.ymax for item in cluster))
            else:
                # Sub-pixel weighted average for unclipped, similar-sized boxes
                xmins = np.array([item.bbox.xmin for item in cluster], dtype=np.float64)
                ymins = np.array([item.bbox.ymin for item in cluster], dtype=np.float64)
                xmaxs = np.array([item.bbox.xmax for item in cluster], dtype=np.float64)
                ymaxs = np.array([item.bbox.ymax for item in cluster], dtype=np.float64)

                fused_xmin = float(np.sum(xmins * weights) / w_sum)
                fused_ymin = float(np.sum(ymins * weights) / w_sum)
                fused_xmax = float(np.sum(xmaxs * weights) / w_sum)
                fused_ymax = float(np.sum(ymaxs * weights) / w_sum)

            fused_conf = float(max(d.confidence for d in cluster))

            fused_box = BoundingBox(
                xmin=fused_xmin,
                ymin=fused_ymin,
                xmax=max(fused_xmin, fused_xmax),
                ymax=max(fused_ymin, fused_ymax)
            )

            fused_class_dets.append(
                Detection(
                    bbox=fused_box,
                    class_id=cluster[0].class_id,
                    class_name=cluster[0].class_name,
                    confidence=fused_conf
                )
            )

        # Post-fusion check: ensure no residual contained boxes remain across clusters
        fused_class_dets.sort(key=lambda d: d.confidence, reverse=True)
        consolidated: List[Detection] = []
        for fd in fused_class_dets:
            merged = False
            for c_idx, cd in enumerate(consolidated):
                if _boxes_match(fd, cd):
                    f_xmin = min(fd.bbox.xmin, cd.bbox.xmin)
                    f_ymin = min(fd.bbox.ymin, cd.bbox.ymin)
                    f_xmax = max(fd.bbox.xmax, cd.bbox.xmax)
                    f_ymax = max(fd.bbox.ymax, cd.bbox.ymax)
                    f_conf = max(fd.confidence, cd.confidence)
                    consolidated[c_idx] = Detection(
                        bbox=BoundingBox(f_xmin, f_ymin, f_xmax, f_ymax),
                        class_id=cd.class_id,
                        class_name=cd.class_name,
                        confidence=f_conf
                    )
                    merged = True
                    break
            if not merged:
                consolidated.append(fd)

        final_detections.extend(consolidated)

    return sorted(final_detections, key=lambda d: d.confidence, reverse=True)


def apply_nms(
    detections: List[Detection],
    iou_threshold: float = 0.5,
    class_agnostic: bool = False
) -> List[Detection]:
    """
    High-level NMS utility accepting domain Detection entities.
    """
    if len(detections) <= 1:
        return list(detections)

    if class_agnostic:
        boxes = np.array([d.bbox.to_xyxy() for d in detections], dtype=np.float32)
        scores = np.array([d.confidence for d in detections], dtype=np.float32)
        keep = nms_numpy(boxes, scores, iou_threshold)
        return [detections[i] for i in keep]

    # Class-aware NMS
    unique_classes = set(d.class_id for d in detections)
    kept_detections: List[Detection] = []

    for cid in unique_classes:
        c_indices = [i for i, d in enumerate(detections) if d.class_id == cid]
        c_dets = [detections[i] for i in c_indices]

        boxes = np.array([d.bbox.to_xyxy() for d in c_dets], dtype=np.float32)
        scores = np.array([d.confidence for d in c_dets], dtype=np.float32)
        keep = nms_numpy(boxes, scores, iou_threshold)

        for k in keep:
            kept_detections.append(c_dets[k])

    return sorted(kept_detections, key=lambda d: d.confidence, reverse=True)
