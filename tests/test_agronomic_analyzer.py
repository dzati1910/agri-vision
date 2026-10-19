"""
Unit tests for Agronomic Pest Analyzer and SPAD-502 correlation.
"""

import pytest

from src.domain.entities import LeafSample
from src.pipeline.agronomic_analyzer import AgronomicPestAnalyzer


def test_severity_classification_levels():
    # Healthy
    diag_0 = AgronomicPestAnalyzer.evaluate_sample_severity(0, 48.0)
    assert "SÃO" in diag_0.level
    assert not diag_0.action_required

    # Light
    diag_1 = AgronomicPestAnalyzer.evaluate_sample_severity(5, 43.0)
    assert "LEVE" in diag_1.level
    assert not diag_1.action_required

    # Moderate
    diag_2 = AgronomicPestAnalyzer.evaluate_sample_severity(20, 35.0)
    assert "MODERADA" in diag_2.level
    assert diag_2.action_required

    # Critical
    diag_3 = AgronomicPestAnalyzer.evaluate_sample_severity(45, 25.0)
    assert "CRÍTICA" in diag_3.level
    assert diag_3.action_required


def test_agronomic_correlation_analysis():
    # Create synthetic samples where higher pest count correlates with lower SPAD
    samples = [
        LeafSample(sample_id="S1", plant_id="P1", treatment="T0", spad_reading=48.0, image_shape=(100, 100, 3)),
        LeafSample(sample_id="S2", plant_id="P1", treatment="T1", spad_reading=42.0, image_shape=(100, 100, 3)),
        LeafSample(sample_id="S3", plant_id="P2", treatment="T2", spad_reading=34.0, image_shape=(100, 100, 3)),
        LeafSample(sample_id="S4", plant_id="P2", treatment="T3", spad_reading=26.0, image_shape=(100, 100, 3))
    ]
    # Synthetic detection counts: 0, 8, 20, 50
    preds = [[], [1] * 8, [1] * 20, [1] * 50]

    report = AgronomicPestAnalyzer.analyze_dataset_correlation(samples, predictions=preds)

    assert report.sample_count == 4
    # Pearson r should be strongly negative (more pests = lower chlorophyll)
    assert report.pearson_r < -0.90
    assert report.slope < 0.0
    assert report.r_squared > 0.85
    # Mathematically valid p-value in [0, 1] and statistically significant
    assert 0.0 <= report.p_value <= 1.0
    assert report.p_value < 0.05


def test_agronomic_correlation_mismatched_predictions_raises():
    samples = [
        LeafSample(sample_id="S1", plant_id="P1", treatment="T0", spad_reading=48.0, image_shape=(100, 100, 3)),
        LeafSample(sample_id="S2", plant_id="P1", treatment="T1", spad_reading=42.0, image_shape=(100, 100, 3)),
    ]
    # 3 prediction lists for 2 samples
    mismatched_preds = [[], [], []]
    with pytest.raises(ValueError, match="Sample count mismatch"):
        AgronomicPestAnalyzer.analyze_dataset_correlation(samples, predictions=mismatched_preds)


def test_agronomic_correlation_zero_variance():
    # Constant SPAD reading (zero variance in Y)
    samples = [
        LeafSample(sample_id="S1", plant_id="P1", treatment="T0", spad_reading=45.0, image_shape=(100, 100, 3)),
        LeafSample(sample_id="S2", plant_id="P1", treatment="T1", spad_reading=45.0, image_shape=(100, 100, 3)),
    ]
    report = AgronomicPestAnalyzer.analyze_dataset_correlation(samples)
    assert report.pearson_r == 0.0
    assert report.p_value == 1.0
    assert report.r_squared == 0.0
