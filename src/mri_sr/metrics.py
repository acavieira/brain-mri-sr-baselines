"""Standard image quality metrics for normalized slices."""

import math
from typing import Dict, Optional

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def mse(ref: np.ndarray, pred: np.ndarray) -> float:
    """Mean squared error."""
    diff = ref.astype(np.float32) - pred.astype(np.float32)
    return float(np.mean(diff ** 2))


def mae(ref: np.ndarray, pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(ref.astype(np.float32) - pred.astype(np.float32))))


def rmse(ref: np.ndarray, pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(math.sqrt(mse(ref, pred)))


def psnr(ref: np.ndarray, pred: np.ndarray) -> float:
    """Return standard PSNR for normalized images with data range 1.0."""
    value = mse(ref, pred)
    if value == 0.0:
        return float("inf")
    return float(peak_signal_noise_ratio(ref, pred, data_range=1.0))


def masked_psnr(ref: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    """Return PSNR using only pixels selected by a boolean mask."""
    if ref.shape != pred.shape or ref.shape != mask.shape:
        raise ValueError("Reference, prediction, and mask must have equal shapes")
    if not np.any(mask):
        raise ValueError("The PSNR mask is empty")
    return psnr(ref[mask], pred[mask])


def isnr(ref: np.ndarray, pred: np.ndarray, baseline: np.ndarray) -> float:
    """Return improvement in dB over a baseline reconstruction."""
    if ref.shape != pred.shape or ref.shape != baseline.shape:
        raise ValueError("Reference, reconstruction, and baseline must have equal shapes")
    baseline_error = mse(ref, baseline)
    reconstruction_error = mse(ref, pred)
    if reconstruction_error == 0.0:
        return float("inf") if baseline_error > 0.0 else 0.0
    if baseline_error == 0.0:
        return float("-inf")
    return float(10.0 * np.log10(baseline_error / reconstruction_error))


def compute_metrics(
    ref: np.ndarray,
    pred: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    brain_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute metrics after validating equal image shapes."""
    if ref.shape != pred.shape:
        raise ValueError(f"HR and reconstruction shapes differ: {ref.shape} != {pred.shape}")
    reference = ref.astype(np.float32)
    reconstruction = pred.astype(np.float32)
    metrics = {
        "psnr_full": psnr(reference, reconstruction),
        "mse": mse(reference, reconstruction),
        "mae": mae(reference, reconstruction),
        "rmse": rmse(reference, reconstruction),
        "ssim": float(structural_similarity(reference, reconstruction, data_range=1.0)),
    }
    if brain_mask is not None:
        metrics["psnr_brain"] = masked_psnr(reference, reconstruction, brain_mask)
    if baseline is not None:
        metrics["isnr"] = isnr(reference, reconstruction, baseline.astype(np.float32))
    return metrics
