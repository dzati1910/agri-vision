"""
Unit tests for SAHI Slicer, window generation, zero-copy memory views, and coordinate translation.
"""

import pytest
import numpy as np

from src.slicing.sahi_slicer import SAHISlicer
from src.domain.entities import BoundingBox, Detection
from src.utils.memory_profiler import MemoryProfiler


def test_sahi_slicer_window_generation():
    # Image 1280x720, tiles 640x640 with 20% overlap
    slicer = SAHISlicer(slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2)
    windows = slicer.generate_windows(image_shape=(720, 1280, 3))

    assert len(windows) > 0
    # Verify that all windows are within image boundaries
    for win in windows:
        assert 0 <= win.xmin < win.xmax <= 1280
        assert 0 <= win.ymin < win.ymax <= 720

    # Ensure coverage of the right and bottom edges
    max_x = max(w.xmax for w in windows)
    max_y = max(w.ymax for w in windows)
    assert max_x == 1280
    assert max_y == 720


def test_sahi_zero_copy_crop():
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    slicer = SAHISlicer(slice_height=640, slice_width=640)
    windows = slicer.generate_windows(image.shape)

    # First tile does not need padding, so it must be a zero-copy strided view
    crop = slicer.crop_tile(image, windows[0])
    assert MemoryProfiler.verify_zero_copy_slice(image, crop)
    assert crop.shape == (640, 640, 3)


def test_sahi_coordinate_reprojection():
    # Tile at (100, 200)
    from src.domain.entities import TileWindow
    win = TileWindow(tile_id=0, xmin=100, ymin=200, xmax=740, ymax=840)

    # Local detection at (50, 60, 90, 110)
    local_det = Detection(
        bbox=BoundingBox(50, 60, 90, 110),
        class_id=0,
        class_name="nymph",
        confidence=0.92
    )

    global_dets = SAHISlicer.map_tile_detections_to_global(
        tile_detections=[local_det],
        window=win,
        img_width=2000,
        img_height=2000
    )

    assert len(global_dets) == 1
    g = global_dets[0]
    assert g.bbox.xmin == 150  # 50 + 100
    assert g.bbox.ymin == 260  # 60 + 200
    assert g.bbox.xmax == 190  # 90 + 100
    assert g.bbox.ymax == 310  # 110 + 200
    assert g.confidence == 0.92


def test_sahi_invalid_image_shape_raises():
    slicer = SAHISlicer(slice_height=640, slice_width=640)
    with pytest.raises(ValueError, match="Image dimensions must be positive"):
        slicer.generate_windows((0, 0, 3))
    with pytest.raises(ValueError, match="Image dimensions must be positive"):
        slicer.generate_windows((-10, 100, 3))


def test_sahi_small_image_padding():
    # Image smaller than tile size (e.g. 200x300)
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    slicer = SAHISlicer(slice_height=640, slice_width=640)
    windows = slicer.generate_windows(img.shape)
    assert len(windows) == 1
    assert windows[0].xmax == 300
    assert windows[0].ymax == 200

    crop = slicer.crop_tile(img, windows[0])
    assert crop.shape == (640, 640, 3)


def test_sahi_map_tile_detections_invalid_dimensions_raises():
    from src.domain.entities import TileWindow
    win = TileWindow(tile_id=0, xmin=0, ymin=0, xmax=640, ymax=640)
    det = Detection(bbox=BoundingBox(10, 10, 50, 50), class_id=0, class_name="nymph")
    with pytest.raises(ValueError, match="Image dimensions must be positive"):
        SAHISlicer.map_tile_detections_to_global([det], win, img_width=0, img_height=100)
    with pytest.raises(ValueError, match="Image dimensions must be positive"):
        SAHISlicer.map_tile_detections_to_global([det], win, img_width=100, img_height=-10)


def test_sahi_extreme_aspect_ratios():
    slicer = SAHISlicer(slice_height=640, slice_width=640)

    # Wide strip: 200 height x 8000 width
    windows_wide = slicer.generate_windows((200, 8000, 3))
    assert len(windows_wide) > 0
    assert max(w.xmax for w in windows_wide) == 8000
    assert max(w.ymax for w in windows_wide) == 200

    # Tall strip: 8000 height x 200 width
    windows_tall = slicer.generate_windows((8000, 200, 3))
    assert len(windows_tall) > 0
    assert max(w.xmax for w in windows_tall) == 200
    assert max(w.ymax for w in windows_tall) == 8000
