"""Classical interpolation methods."""

from typing import Tuple

import cv2
import numpy as np

from .config import METHODS


INTERPOLATION_CODES = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


def upscale_image(lr_image: np.ndarray, method: str, target_shape: Tuple[int, int]) -> np.ndarray:
    """Upscale the same LR image with one selected classical method."""
    if method not in METHODS:
        raise ValueError(f"Unknown interpolation method: {method}")
    if lr_image.ndim != 2:
        raise ValueError("The LR image must be two-dimensional")
    height, width = target_shape
    result = cv2.resize(lr_image, (width, height), interpolation=INTERPOLATION_CODES[method])
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def upscale_all_methods(lr_image: np.ndarray, target_shape: Tuple[int, int]) -> dict[str, np.ndarray]:
    """Reconstruct all methods from the identical LR array."""
    return {method: upscale_image(lr_image, method, target_shape) for method in METHODS}