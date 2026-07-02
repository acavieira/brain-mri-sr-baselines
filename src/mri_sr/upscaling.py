from typing import Tuple

import cv2
import numpy as np


INTERPOLATION_METHODS = ["nearest", "bilinear", "bicubic", "lanczos"]


_INTERPOLATION_MAP = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


def upscale_image(method: str, lr_img: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if method not in _INTERPOLATION_MAP:
        raise ValueError(f"Unknown method: {method}")

    h, w = target_shape
    sr_img = cv2.resize(lr_img, (w, h), interpolation=_INTERPOLATION_MAP[method])
    return np.clip(sr_img, 0.0, 1.0).astype(np.float32)