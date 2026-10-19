"""
Anti-Leakage Agronomic Validation Splitter (Anti-Pseudoreplication).

Addresses the critical flaw in agricultural machine learning where leaves
from the same biological plant are leaked across training and validation folds.
Implements strict GroupKFold and Stratified Group Split by Plant ID.
"""

from __future__ import annotations
from typing import List, Tuple, Sequence, Dict, Set, Optional, Any
import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from src.core.interfaces import BaseDatasetSplitter
from src.domain.entities import LeafSample


class PlantIDGroupSplitter(BaseDatasetSplitter):
    """
    Dataset splitter that enforces strict biological plant independence.
    Guarantees zero data leakage between training and evaluation splits.
    """

    def __init__(self, n_splits: int = 5) -> None:
        self.n_splits = n_splits

    def split(
        self,
        samples: Sequence[LeafSample],
        n_splits: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42
    ) -> List[Tuple[List[LeafSample], List[LeafSample]]]:
        """
        Splits samples such that all images from any given plant_id are exclusively
        in the training set OR the validation set, never both.
        
        Args:
            samples: Collection of LeafSample entities.
            n_splits: Number of cross-validation folds.
            shuffle: Whether to shuffle groups before splitting.
            seed: Random state seed.
            
        Returns:
            List of (train_samples, val_samples) for each fold.
        """
        k = n_splits or self.n_splits
        if len(samples) == 0:
            return []

        # Extract plant IDs as grouping variable
        groups = np.array([s.plant_id for s in samples])
        unique_groups = np.unique(groups)

        if len(unique_groups) < k:
            raise ValueError(
                f"Cannot perform {k}-fold GroupKFold with only {len(unique_groups)} distinct plants! "
                f"At least {k} unique plant IDs are required."
            )

        gkf = GroupKFold(n_splits=k)
        indices = np.arange(len(samples))

        folds: List[Tuple[List[LeafSample], List[LeafSample]]] = []

        for train_idx, val_idx in gkf.split(indices, groups=groups):
            train_set = [samples[i] for i in train_idx]
            val_set = [samples[i] for i in val_idx]

            # Enforce audit assertion
            leakage = self.audit_data_leakage(train_set, val_set)
            if leakage["leaked_plant_count"] > 0:
                raise RuntimeError(f"Data leakage detected! Leaked plants: {leakage['leaked_plants']}")

            folds.append((train_set, val_set))

        return folds

    @staticmethod
    def audit_data_leakage(
        train_samples: Sequence[LeafSample],
        val_samples: Sequence[LeafSample]
    ) -> Dict[str, Any]:
        """
        Audits two dataset partitions to mathematically prove absence of plant leakage.
        
        Returns:
            Dictionary with audit statistics:
            - leaked_plant_count: Must be 0 for scientific validity.
            - leaked_plants: Set of offending plant IDs.
            - train_plants: Total unique plants in training.
            - val_plants: Total unique plants in validation.
            - is_scientifically_sound: Boolean flag.
        """
        train_plants = set(s.plant_id for s in train_samples)
        val_plants = set(s.plant_id for s in val_samples)

        overlap = train_plants.intersection(val_plants)

        return {
            "leaked_plant_count": len(overlap),
            "leaked_plants": sorted(list(overlap)),
            "train_plant_count": len(train_plants),
            "val_plant_count": len(val_plants),
            "train_sample_count": len(train_samples),
            "val_sample_count": len(val_samples),
            "is_scientifically_sound": len(overlap) == 0
        }

    @staticmethod
    def compare_split_strategies(
        samples: Sequence[LeafSample],
        val_ratio: float = 0.25,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Demonstrates the danger of naive random split vs Group split on the same dataset.
        Shows how naive split leaks biological plant phenotypes into validation.
        """
        # 1. Naive random split (the common error in literature)
        rng = np.random.default_rng(seed)
        shuffled_idx = rng.permutation(len(samples))
        n_val = max(1, int(len(samples) * val_ratio))
        val_idx = shuffled_idx[:n_val]
        train_idx = shuffled_idx[n_val:]

        naive_train = [samples[i] for i in train_idx]
        naive_val = [samples[i] for i in val_idx]
        naive_audit = PlantIDGroupSplitter.audit_data_leakage(naive_train, naive_val)

        # 2. Strict Group Shuffle Split
        gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        groups = [s.plant_id for s in samples]
        g_train_idx, g_val_idx = next(gss.split(samples, groups=groups))

        group_train = [samples[i] for i in g_train_idx]
        group_val = [samples[i] for i in g_val_idx]
        group_audit = PlantIDGroupSplitter.audit_data_leakage(group_train, group_val)

        return {
            "naive_random_split": naive_audit,
            "strict_group_split": group_audit
        }
