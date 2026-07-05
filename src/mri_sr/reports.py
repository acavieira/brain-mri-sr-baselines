"""Helpers to write visual reports and metrics CSV files."""

import csv
import os
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .io import ensure_dir
from .preprocessing import float01_to_uint8


def save_image(path: str, img: np.ndarray) -> None:
    """Save a normalized float image as an 8-bit PNG."""
    cv2.imwrite(path, float01_to_uint8(img))


def save_visual_report(
    method: str,
    axis: str,
    slice_index: int,
    hr_img: np.ndarray,
    lr_img: np.ndarray,
    sr_img: np.ndarray,
    ssim_map: np.ndarray,
    metrics: Dict[str, object],
    output_dir: str,
    error_vmax: float,
) -> None:
    """Create a 2x3 visual panel with reference, prediction, and error maps."""
    ensure_dir(output_dir)

    # Resize LR only for display; metrics are computed on original arrays.
    h, w = hr_img.shape
    lr_display = cv2.resize(lr_img, (w, h), interpolation=cv2.INTER_NEAREST)

    abs_error = np.abs(hr_img - sr_img)
    signed_error = hr_img - sr_img

    # Top row: reference/degraded/reconstruction.
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=160)

    axes[0, 0].imshow(hr_img, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("HR reference")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(lr_display, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("LR degraded")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(sr_img, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title(f"Upscaled: {method}")
    axes[0, 2].axis("off")

    # Bottom row: absolute error, signed error, and local SSIM map.
    im1 = axes[1, 0].imshow(abs_error, cmap="turbo", vmin=0, vmax=error_vmax)
    axes[1, 0].set_title("Absolute error")
    axes[1, 0].axis("off")
    fig.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im2 = axes[1, 1].imshow(signed_error, cmap="seismic", vmin=-error_vmax, vmax=error_vmax)
    axes[1, 1].set_title("Signed error")
    axes[1, 1].axis("off")
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im3 = axes[1, 2].imshow(ssim_map, cmap="viridis", vmin=0, vmax=1)
    axes[1, 2].set_title("Local SSIM")
    axes[1, 2].axis("off")
    fig.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{axis} slice {slice_index} | {method} | "
        f"SSIM={metrics['ssim']:.4f} | PSNR={metrics['psnr']:.2f} | "
        f"MSE={metrics['mse']:.6f} | HFEN={metrics['hfen']:.4f}",
        fontsize=11,
    )

    plt.tight_layout()
    path = os.path.join(output_dir, f"report_{axis}_{slice_index:04d}_{method}.png")
    fig.savefig(path)
    plt.close(fig)


def save_metrics_csv(rows: List[Dict[str, object]], path: str) -> None:
    """Write metric rows to a CSV file."""
    if not rows:
        return

    ensure_dir(os.path.dirname(path))

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)