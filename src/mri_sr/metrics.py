import math
import warnings
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    from image_similarity_measures.quality_metrics import issm as issm_metric
except Exception:
    issm_metric = None

_issm_disabled = False


def mse(ref: np.ndarray, pred: np.ndarray) -> float:
    diff = ref.astype(np.float32) - pred.astype(np.float32)
    return float(np.mean(diff ** 2))


def mae(ref: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(ref.astype(np.float32) - pred.astype(np.float32))))


def rmse(ref: np.ndarray, pred: np.ndarray) -> float:
    return float(math.sqrt(mse(ref, pred)))


def nrmse(ref: np.ndarray, pred: np.ndarray) -> float:
    value_range = float(ref.max() - ref.min())
    if value_range <= 1e-8:
        return 0.0
    return float(rmse(ref, pred) / value_range)


def psnr(ref: np.ndarray, pred: np.ndarray, max_val: float = 1.0) -> float:
    value = mse(ref, pred)
    if value <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(max_val / math.sqrt(value)))


def pearson(ref: np.ndarray, pred: np.ndarray) -> float:
    x = ref.astype(np.float32).ravel()
    y = pred.astype(np.float32).ravel()

    if np.std(x) <= 1e-8 or np.std(y) <= 1e-8:
        return 0.0

    return float(np.corrcoef(x, y)[0, 1])


def ssim_and_map(ref: np.ndarray, pred: np.ndarray) -> Tuple[float, np.ndarray]:
    x = ref.astype(np.float64)
    y = pred.astype(np.float64)

    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)

    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)

    sigma_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x ** 2
    sigma_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y ** 2
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x2 + sigma_y2 + c2)

    ssim_map = numerator / (denominator + 1e-12)
    ssim_map = np.clip(ssim_map, -1.0, 1.0).astype(np.float32)

    return float(np.mean(ssim_map)), ssim_map


def gradient_mse(ref: np.ndarray, pred: np.ndarray) -> float:
    ref = ref.astype(np.float32)
    pred = pred.astype(np.float32)

    ref_x = cv2.Sobel(ref, cv2.CV_32F, 1, 0, ksize=3)
    ref_y = cv2.Sobel(ref, cv2.CV_32F, 0, 1, ksize=3)
    pred_x = cv2.Sobel(pred, cv2.CV_32F, 1, 0, ksize=3)
    pred_y = cv2.Sobel(pred, cv2.CV_32F, 0, 1, ksize=3)

    ref_grad = np.sqrt(ref_x ** 2 + ref_y ** 2)
    pred_grad = np.sqrt(pred_x ** 2 + pred_y ** 2)

    return float(np.mean((ref_grad - pred_grad) ** 2))


def hfen(ref: np.ndarray, pred: np.ndarray) -> float:
    ref = ref.astype(np.float32)
    pred = pred.astype(np.float32)

    ref_hf = cv2.GaussianBlur(ref, (0, 0), 1.5) - cv2.GaussianBlur(ref, (0, 0), 3.0)
    pred_hf = cv2.GaussianBlur(pred, (0, 0), 1.5) - cv2.GaussianBlur(pred, (0, 0), 3.0)

    return float(np.linalg.norm(ref_hf - pred_hf) / (np.linalg.norm(ref_hf) + 1e-8))


def issm_optional(ref: np.ndarray, pred: np.ndarray) -> Optional[float]:
    global _issm_disabled

    if issm_metric is None or _issm_disabled:
        return None

    try:
        ref_3d = ref.astype(np.float32)[..., np.newaxis]
        pred_3d = pred.astype(np.float32)[..., np.newaxis]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            value = float(issm_metric(org_img=ref_3d, pred_img=pred_3d))

        if not np.isfinite(value):
            _issm_disabled = True
            return None

        return value
    except Exception:
        _issm_disabled = True
        return None


def compute_metrics(ref: np.ndarray, pred: np.ndarray) -> Dict[str, object]:
    ssim_value, ssim_map = ssim_and_map(ref, pred)
    mae_value = mae(ref, pred)

    return {
        "ssim": ssim_value,
        "psnr": psnr(ref, pred),
        "mse": mse(ref, pred),
        "mae": mae_value,
        "rmse": rmse(ref, pred),
        "nrmse": nrmse(ref, pred),
        "pearson": pearson(ref, pred),
        "gradient_mse": gradient_mse(ref, pred),
        "hfen": hfen(ref, pred),
        "diff_percent": mae_value * 100.0,
        "issm": issm_optional(ref, pred),
        "ssim_map": ssim_map,
    }
