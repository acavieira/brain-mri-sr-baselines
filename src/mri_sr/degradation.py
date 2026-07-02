import cv2
import numpy as np


def degrade_image(
    hr_img: np.ndarray,
    scale: int,
    blur_sigma: float,
    noise_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if scale < 2:
        raise ValueError("Scale must be >= 2")

    degraded = hr_img.astype(np.float32)

    if blur_sigma > 0:
        degraded = cv2.GaussianBlur(degraded, (0, 0), blur_sigma)

    h, w = degraded.shape
    lr_w = max(8, int(round(w / scale)))
    lr_h = max(8, int(round(h / scale)))

    lr_img = cv2.resize(degraded, (lr_w, lr_h), interpolation=cv2.INTER_AREA)

    if noise_sigma > 0:
        noise = rng.normal(0.0, noise_sigma, size=lr_img.shape).astype(np.float32)
        lr_img = lr_img + noise

    return np.clip(lr_img, 0.0, 1.0).astype(np.float32)