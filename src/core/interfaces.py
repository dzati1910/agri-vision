"""
Core interfaces and abstract protocols defining the contracts for detectors,
slicers, dataset splitters, and model exporters (Clean Architecture / Dependency Inversion).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Sequence, Optional, Protocol, runtime_checkable
import numpy as np

from src.domain.entities import Detection, TileWindow, LeafSample


@runtime_checkable
class BaseDetector(Protocol):
    """
    Protocol defining the detection engine contract.
    Decouples high-level pipelines (SAHI) from low-level models (YOLO, ONNX, Torch).
    """
    @property
    def model_name(self) -> str:
        """Returns model identifier."""
        ...

    def predict(self, image: np.ndarray, conf_threshold: float = 0.25) -> List[Detection]:
        """
        Runs object detection on a single image (H, W, C) uint8 RGB/BGR.
        """
        ...

    def predict_batch(self, images: Sequence[np.ndarray], conf_threshold: float = 0.25) -> List[List[Detection]]:
        """
        Runs vectorized/batched inference on a sequence of image tiles.
        """
        ...


class BaseSlicer(ABC):
    """
    Abstract contract for spatial image slicing (SAHI-style tiling).
    """
    @abstractmethod
    def generate_windows(self, image_shape: Tuple[int, int, ...]) -> List[TileWindow]:
        """Generates tiling grid with specified overlap and boundary handling."""
        pass

    @abstractmethod
    def crop_tile(self, image: np.ndarray, window: TileWindow) -> np.ndarray:
        """Extracts the image slice corresponding to window."""
        pass


class BaseDatasetSplitter(ABC):
    """
    Abstract contract for dataset splitting ensuring scientific integrity
    against pseudoreplication and data leakage.
    """
    @abstractmethod
    def split(
        self,
        samples: Sequence[LeafSample],
        n_splits: int = 5,
        shuffle: bool = True,
        seed: int = 42
    ) -> List[Tuple[List[LeafSample], List[LeafSample]]]:
        """
        Splits dataset into list of (train_samples, validation_samples) folds.
        """
        pass
