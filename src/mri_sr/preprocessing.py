"""Preprocessing helpers for robust MRI slice normalization."""

import cv2
import numpy as np
from typing import Optional


def robust_normalize_to_float01(
    img: np.ndarray,
    low_percentile: float = 1,
    high_percentile: float = 99,
) -> np.ndarray:
    """Clip to robust percentiles and normalize image values to [0, 1]."""
    img = img.astype(np.float32)

    lo = float(np.percentile(img, low_percentile))
    hi = float(np.percentile(img, high_percentile))

    if hi <= lo:
        hi = lo + 1e-6

    img = (img - lo) / (hi - lo)
    img = np.clip(img, 0.0, 1.0)
    return img.astype(np.float32)


def float01_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert [0, 1] float image to uint8 for PNG export."""
    return (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def prepare_hr_reference(img: np.ndarray, force_square_size: Optional[int] = None) -> np.ndarray:
    """Build the HR reference image used by metrics and degradation."""
    img = robust_normalize_to_float01(img)

    # Optional resize keeps all methods on a fixed input resolution.
    if force_square_size is not None:
        img = cv2.resize(
            img,
            (force_square_size, force_square_size),
            interpolation=cv2.INTER_CUBIC,
        )

    return np.clip(img, 0.0, 1.0).astype(np.float32)