"""
Unit tests for YOLO and COCO dataset parsers and exporters.
"""

from pathlib import Path
import pytest

from src.domain.entities import BoundingBox, Detection, LeafSample
from src.data.yolo_parser import YOLOParser
from src.data.coco_parser import COCOParser


def test_yolo_parser_save_and_load(tmp_path: Path):
    txt_file = tmp_path / "label_001.txt"
    dets = [
        Detection(BoundingBox(100, 200, 300, 400), class_id=0, class_name="nymph", confidence=0.88),
        Detection(BoundingBox(500, 600, 700, 800), class_id=1, class_name="adult", confidence=0.95)
    ]
    img_w, img_h = 1000, 1000

    YOLOParser.save_annotation(dets, txt_file, img_w, img_h)
    assert txt_file.exists()

    loaded = YOLOParser.load_annotation(txt_file, img_w, img_h, class_mapping={0: "nymph", 1: "adult"})
    assert len(loaded) == 2
    assert loaded[0].class_id == 0
    assert loaded[0].class_name == "nymph"
    assert pytest.approx(loaded[0].bbox.xmin, abs=1.0) == 100.0
    assert pytest.approx(loaded[0].bbox.ymin, abs=1.0) == 200.0
    assert pytest.approx(loaded[0].bbox.xmax, abs=1.0) == 300.0
    assert pytest.approx(loaded[0].bbox.ymax, abs=1.0) == 400.0


def test_coco_parser_export_and_load(tmp_path: Path):
    json_file = tmp_path / "coco_test.json"
    sample = LeafSample(
        sample_id="test_leaf_01",
        plant_id="PLANTA_05",
        treatment="T2",
        spad_reading=36.5,
        image_shape=(1000, 1000, 3),
        ground_truth=[
            Detection(BoundingBox(50, 50, 100, 120), class_id=0, class_name="nymph", confidence=1.0)
        ],
        image_path=str(tmp_path / "test_leaf_01.jpg")
    )

    COCOParser.export_samples_to_coco([sample], json_file)
    assert json_file.exists()

    loaded = COCOParser.load_coco_annotations(json_file)
    assert "test_leaf_01.jpg" in loaded
    dets = loaded["test_leaf_01.jpg"]
    assert len(dets) == 1
    assert dets[0].bbox.xmin == 50.0
    assert dets[0].bbox.ymin == 50.0
    assert dets[0].bbox.width == 50.0
    assert dets[0].bbox.height == 70.0


def test_yolo_parser_handles_corrupt_lines(tmp_path: Path):
    corrupt_file = tmp_path / "corrupt_label.txt"
    # Content has header line, malformed line with strings, out-of-bounds line, and 1 valid line
    corrupt_file.write_text(
        "class_id norm_cx norm_cy norm_w norm_h\n"
        "corrupted line with random words\n"
        "0 0.5 0.5 -0.2 0.2\n"  # negative width -> ValueError in from_yolo, safely skipped
        "0 0.5 0.5 0.2 0.2 0.95\n",  # valid line
        encoding="utf-8"
    )

    loaded = YOLOParser.load_annotation(corrupt_file, img_width=1000, img_height=1000)
    assert len(loaded) == 1
    assert loaded[0].class_id == 0
    assert loaded[0].confidence == 0.95


def test_yolo_parser_invalid_dimensions_raises(tmp_path: Path):
    txt_file = tmp_path / "dummy.txt"
    txt_file.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Image dimensions must be positive"):
        YOLOParser.load_annotation(txt_file, img_width=0, img_height=1000)


def test_coco_parser_load_missing_file_returns_empty(tmp_path: Path):
    missing_file = tmp_path / "does_not_exist.json"
    result = COCOParser.load_coco_annotations(missing_file)
    assert result == {}


def test_yolo_parser_missing_file_raises_filenotfound(tmp_path: Path):
    missing_file = tmp_path / "non_existent_label.txt"
    with pytest.raises(FileNotFoundError, match="YOLO annotation file not found"):
        YOLOParser.load_annotation(missing_file, img_width=1000, img_height=1000)


def test_yolo_parser_empty_file_returns_empty_list(tmp_path: Path):
    empty_file = tmp_path / "healthy_leaf.txt"
    empty_file.write_text("", encoding="utf-8")
    result = YOLOParser.load_annotation(empty_file, img_width=1000, img_height=1000)
    assert result == []


def test_coco_parser_corrupted_json_returns_empty(tmp_path: Path):
    corrupt_json = tmp_path / "broken.json"
    corrupt_json.write_text("{ this is broken json :: ;;; }", encoding="utf-8")
    result = COCOParser.load_coco_annotations(corrupt_json)
    assert result == {}


def test_coco_parser_non_dict_json_returns_empty(tmp_path: Path):
    list_json = tmp_path / "list_root.json"
    list_json.write_text("[]", encoding="utf-8")
    result = COCOParser.load_coco_annotations(list_json)
    assert result == {}

