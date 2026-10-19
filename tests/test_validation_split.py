"""
Unit tests for Anti-Leakage Dataset Splitter (Anti-Pseudoreplication).
"""

import pytest

from src.domain.entities import LeafSample
from src.data.validation_split import PlantIDGroupSplitter


def test_plant_id_group_kfold_zero_leakage():
    # 6 plants, 3 leaves per plant = 18 samples
    samples = []
    for p_idx in range(1, 7):
        pid = f"PLANTA_{p_idx:02d}"
        for leaf in range(1, 4):
            samples.append(
                LeafSample(
                    sample_id=f"{pid}_L{leaf}",
                    plant_id=pid,
                    treatment="T1",
                    spad_reading=40.0,
                    image_shape=(1000, 1000, 3)
                )
            )

    splitter = PlantIDGroupSplitter(n_splits=3)
    folds = splitter.split(samples, n_splits=3)

    assert len(folds) == 3

    for fold_idx, (train_set, val_set) in enumerate(folds):
        train_plants = set(s.plant_id for s in train_set)
        val_plants = set(s.plant_id for s in val_set)

        # Scientific anti-pseudoreplication invariant:
        assert train_plants.isdisjoint(val_plants), f"Data leakage in fold {fold_idx}!"

        audit = PlantIDGroupSplitter.audit_data_leakage(train_set, val_set)
        assert audit["leaked_plant_count"] == 0
        assert audit["is_scientifically_sound"] is True


def test_naive_split_detects_leakage():
    # 4 plants, 5 leaves per plant = 20 samples
    samples = []
    for p_idx in range(1, 5):
        pid = f"PLANTA_{p_idx:02d}"
        for leaf in range(1, 6):
            samples.append(
                LeafSample(
                    sample_id=f"{pid}_L{leaf}",
                    plant_id=pid,
                    treatment="T2",
                    spad_reading=35.0,
                    image_shape=(1000, 1000, 3)
                )
            )

    comparison = PlantIDGroupSplitter.compare_split_strategies(samples, val_ratio=0.3, seed=42)
    naive = comparison["naive_random_split"]
    group = comparison["strict_group_split"]

    # Naive split leaks plants across train and val
    assert naive["leaked_plant_count"] > 0
    assert naive["is_scientifically_sound"] is False

    # Group split eliminates leakage completely
    assert group["leaked_plant_count"] == 0
    assert group["is_scientifically_sound"] is True


def test_split_insufficient_groups_raises():
    # Only 2 plants, but requesting 5 folds
    samples = [
        LeafSample(sample_id="L1", plant_id="P1", treatment="T0", spad_reading=45.0, image_shape=(100, 100, 3)),
        LeafSample(sample_id="L2", plant_id="P2", treatment="T0", spad_reading=45.0, image_shape=(100, 100, 3))
    ]
    splitter = PlantIDGroupSplitter(n_splits=5)
    with pytest.raises(ValueError, match="At least 5 unique plant IDs"):
        splitter.split(samples, n_splits=5)
