"""
Memory Profiler and Buffer Inspector for Low-Level Analysis.

Bridges Python 3.13 NumPy memory model with C pointer and memory layout concepts:
- Strided slicing (char* offset arithmetic without malloc).
- C_CONTIGUOUS vs F_CONTIGUOUS cache line locality.
- Buffer sharing (inspecting array.base and __array_interface__).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np


@dataclass(frozen=True)
class BufferLayout:
    """Detailed memory descriptor equivalent to a C struct array header."""
    pointer_address: int
    total_bytes: int
    shape: Tuple[int, ...]
    strides_bytes: Tuple[int, ...]
    is_c_contiguous: bool
    is_fortran_contiguous: bool
    owns_memory: bool
    data_type: str


class MemoryProfiler:
    """Inspects low-level memory buffers of image arrays and slices."""

    @staticmethod
    def inspect_buffer(arr: np.ndarray) -> BufferLayout:
        """
        Extracts raw memory attributes from a NumPy array.
        In C: struct PyArrayObject -> data, dimensions, strides, flags.
        """
        interface = arr.__array_interface__
        ptr = interface["data"][0]

        return BufferLayout(
            pointer_address=ptr,
            total_bytes=arr.nbytes,
            shape=arr.shape,
            strides_bytes=arr.strides,
            is_c_contiguous=arr.flags["C_CONTIGUOUS"],
            is_fortran_contiguous=arr.flags["F_CONTIGUOUS"],
            owns_memory=arr.flags["OWNDATA"],
            data_type=str(arr.dtype)
        )

    @staticmethod
    def verify_zero_copy_slice(original: np.ndarray, sliced: np.ndarray) -> bool:
        """
        Proves whether `sliced` is a zero-copy pointer view into `original`'s RAM buffer.
        If True, creating this slice involved 0 bytes of heap allocation (malloc).
        """
        # Sliced array's base pointer references the original array or its parent
        return sliced.base is original or (original.base is not None and sliced.base is original.base)

    @staticmethod
    def compute_memory_savings(
        image_shape: Tuple[int, int, int],
        num_tiles: int,
        tile_shape: Tuple[int, int, int]
    ) -> Dict[str, Any]:
        """
        Calculates RAM saved by using strided views vs copying tiles into RAM.
        """
        bytes_per_pixel = 1  # uint8 = 1 byte
        h, w, c = image_shape
        th, tw, tc = tile_shape

        orig_mb = (h * w * c * bytes_per_pixel) / (1024 * 1024)
        naive_copied_tiles_mb = (num_tiles * th * tw * tc * bytes_per_pixel) / (1024 * 1024)
        view_metadata_bytes = num_tiles * 128  # ~128 bytes per PyArrayObject header
        view_metadata_mb = view_metadata_bytes / (1024 * 1024)

        savings_ratio = (naive_copied_tiles_mb / max(view_metadata_mb, 1e-6))

        return {
            "original_image_mb": round(orig_mb, 2),
            "naive_malloc_tiles_mb": round(naive_copied_tiles_mb, 2),
            "zero_copy_views_mb": round(view_metadata_mb, 4),
            "ram_multiplication_avoided": round(naive_copied_tiles_mb / orig_mb, 2),
            "theoretical_savings_factor": round(savings_ratio, 1)
        }
