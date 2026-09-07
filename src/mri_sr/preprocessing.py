"""Volume normalization and aspect-preserving HR preparation."""

from typing import Tuple

import cv2
import numpy as np


def normalize_volume_to_float01(
    volume: np.ndarray,
    low_percentile: float = 1,
    high_percentile: float = 99,
) -> np.ndarray:
    """Normalize one complete volume using finite voxels only."""
    values = np.asarray(volume, dtype=np.float32)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("The volume contains no finite intensity values")

    low = float(np.percentile(finite_values, low_percentile))
    high = float(np.percentile(finite_values, high_percentile))
    if high <= low:
        normalized = np.zeros_like(values, dtype=np.float32)
    else:
        normalized = (values - low) / (high - low)
        normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized.astype(np.float32)


def resize_with_padding(image: np.ndarray, target_size: int) -> np.ndarray:
    """Resize a slice without distortion and pad it to a square."""
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D slice, got shape {image.shape}")
    if target_size < 1:
        raise ValueError("Target size must be positive")

    height, width = image.shape
    scale = min(target_size / height, target_size / width)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    output = np.zeros((target_size, target_size), dtype=np.float32)
    top = (target_size - resized_height) // 2
    left = (target_size - resized_width) // 2
    output[top : top + resized_height, left : left + resized_width] = resized
    return output


def prepare_hr_reference(normalized_slice: np.ndarray, target_size: int) -> np.ndarray:
    """Prepare one already volume-normalized slice as the HR reference."""
    return np.clip(resize_with_padding(normalized_slice, target_size), 0.0, 1.0)


def create_brain_mask(reference: np.ndarray, threshold: float) -> np.ndarray:
    """Create a simple foreground mask that excludes padded background."""
    if reference.ndim != 2:
        raise ValueError("The reference image must be two-dimensional")
    mask = np.isfinite(reference) & (reference > threshold)
    if not np.any(mask):
        raise ValueError("The brain mask is empty; lower the mask threshold")
    return mask