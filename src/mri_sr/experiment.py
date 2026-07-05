"""Main baseline experiment loop and aggregate metric helpers."""

import os
import time
from typing import Dict, List

import numpy as np

from .config import ExperimentConfig
from .degradation import degrade_image
from .io import choose_slice_indices, ensure_dir, extract_slice, load_nifti_volume
from .metrics import compute_metrics
from .preprocessing import prepare_hr_reference
from .reports import save_metrics_csv, save_visual_report
from .upscaling import INTERPOLATION_METHODS, upscale_image


def _collect_numeric_values(rows: List[Dict[str, object]], metric: str) -> np.ndarray:
    """Collect finite numeric values for one metric key."""
    values = []

    for row in rows:
        value = row.get(metric)
        if value in ("", None):
            continue

        try:
            numeric_value = float(value)
        except Exception:
            continue

        if np.isfinite(numeric_value):
            values.append(numeric_value)

    return np.array(values, dtype=np.float32)


def aggregate_metrics(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Aggregate per-slice rows into per-method mean/std summary rows."""
    methods = sorted(set(row["method"] for row in rows))
    metric_names = [
        "ssim",
        "psnr",
        "mse",
        "mae",
        "rmse",
        "nrmse",
        "pearson",
        "gradient_mse",
        "hfen",
        "diff_percent",
        "isnr",
        "issm",
        "degradation_time_ms",
        "upscaling_time_ms",
        "metrics_time_ms",
        "total_method_time_ms",
    ]

    aggregate_rows = []

    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        aggregate = {
            "method": method,
            "num_slices": len(method_rows),
        }

        # Compute mean and std for each available metric.
        for metric in metric_names:
            values = _collect_numeric_values(method_rows, metric)
            if values.size == 0:
                aggregate[f"{metric}_mean"] = ""
                aggregate[f"{metric}_std"] = ""
                continue

            aggregate[f"{metric}_mean"] = float(values.mean())
            aggregate[f"{metric}_std"] = float(values.std())

        aggregate_rows.append(aggregate)

    # Primary ranking order used throughout the baseline pipeline.
    aggregate_rows.sort(
        key=lambda row: (
            -row["ssim_mean"],
            -row["psnr_mean"],
            row["mse_mean"],
        )
    )

    return aggregate_rows


def run_experiment(config: ExperimentConfig) -> None:
    """Run the full baseline workflow for one anatomical axis."""
    rng = np.random.default_rng(config.random_seed)

    ensure_dir(config.output_dir)
    figures_dir = os.path.join(config.output_dir, "figures")
    ensure_dir(figures_dir)

    volume = load_nifti_volume(config.input_path)
    slice_indices = choose_slice_indices(
        volume,
        axis=config.slice_axis,
        center_index=config.slice_index,
        num_slices=config.num_slices,
    )

    central_slice = slice_indices[len(slice_indices) // 2]

    print(f"Input volume: {config.input_path}")
    print(f"Volume shape: {volume.shape}")
    print(f"Axis: {config.slice_axis}")
    print(f"Slices: {slice_indices}")
    print(f"Scale: x{config.scale}")
    print(f"ISNR baseline: {config.isnr_baseline_method}")

    if config.isnr_baseline_method not in INTERPOLATION_METHODS:
        raise ValueError(
            f"Invalid ISNR baseline method: {config.isnr_baseline_method}. "
            f"Expected one of {INTERPOLATION_METHODS}"
        )

    rows: List[Dict[str, object]] = []

    for slice_index in slice_indices:
        print(f"\nProcessing slice {slice_index}...")

        raw_slice = extract_slice(volume, config.slice_axis, slice_index)
        hr_img = prepare_hr_reference(raw_slice, config.force_square_size)

        degradation_start = time.perf_counter()
        lr_img = degrade_image(
            hr_img,
            scale=config.scale,
            blur_sigma=config.blur_sigma,
            noise_sigma=config.noise_sigma,
            rng=rng,
        )
        degradation_time_ms = (time.perf_counter() - degradation_start) * 1000.0

        baseline_sr = upscale_image(config.isnr_baseline_method, lr_img, hr_img.shape)
        hr_height, hr_width = hr_img.shape
        lr_height, lr_width = lr_img.shape

        # Evaluate every classical interpolation method on the same LR input.
        for method in INTERPOLATION_METHODS:
            method_start = time.perf_counter()

            upscale_start = time.perf_counter()
            sr_img = upscale_image(method, lr_img, hr_img.shape)
            upscaling_time_ms = (time.perf_counter() - upscale_start) * 1000.0

            metrics_start = time.perf_counter()
            metrics = compute_metrics(hr_img, sr_img, baseline_pred=baseline_sr)
            metrics_time_ms = (time.perf_counter() - metrics_start) * 1000.0
            total_method_time_ms = (time.perf_counter() - method_start) * 1000.0

            # Keep one row per method and per slice for later aggregation.
            row = {
                "slice_axis": config.slice_axis,
                "slice_index": slice_index,
                "method": method,
                "ssim": metrics["ssim"],
                "psnr": metrics["psnr"],
                "mse": metrics["mse"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "nrmse": metrics["nrmse"],
                "pearson": metrics["pearson"],
                "gradient_mse": metrics["gradient_mse"],
                "hfen": metrics["hfen"],
                "diff_percent": metrics["diff_percent"],
                "isnr": "" if metrics["isnr"] is None else metrics["isnr"],
                "issm": "" if metrics["issm"] is None else metrics["issm"],
                "scale": config.scale,
                "isnr_baseline_method": config.isnr_baseline_method,
                "hr_height": hr_height,
                "hr_width": hr_width,
                "lr_height": lr_height,
                "lr_width": lr_width,
                "degradation_time_ms": degradation_time_ms,
                "upscaling_time_ms": upscaling_time_ms,
                "metrics_time_ms": metrics_time_ms,
                "total_method_time_ms": total_method_time_ms,
            }
            rows.append(row)

            isnr_text = "n/a" if metrics["isnr"] is None else f"{metrics['isnr']:.3f}"
            print(
                f"  {method:8s} | "
                f"SSIM={metrics['ssim']:.4f} | "
                f"PSNR={metrics['psnr']:.2f} | "
                f"MSE={metrics['mse']:.6f} | "
                f"HFEN={metrics['hfen']:.4f} | "
                f"ISNR={isnr_text} | "
                f"time={total_method_time_ms:.1f} ms"
            )

            # Save only the central slice by default to keep outputs compact.
            if config.save_all_figures or slice_index == central_slice:
                save_visual_report(
                    method=method,
                    axis=config.slice_axis,
                    slice_index=slice_index,
                    hr_img=hr_img,
                    lr_img=lr_img,
                    sr_img=sr_img,
                    ssim_map=metrics["ssim_map"],
                    metrics=metrics,
                    output_dir=figures_dir,
                    error_vmax=config.error_vmax,
                )

    metrics_path = os.path.join(config.output_dir, "metrics_by_slice.csv")
    aggregate_path = os.path.join(config.output_dir, "metrics_aggregate.csv")

    # Export per-slice and aggregate CSVs.
    save_metrics_csv(rows, metrics_path)
    aggregate_rows = aggregate_metrics(rows)
    save_metrics_csv(aggregate_rows, aggregate_path)

    print("\nDone.")
    print(f"Metrics by slice: {metrics_path}")
    print(f"Aggregate metrics: {aggregate_path}")

    print("\nRanking:")
    for row in aggregate_rows:
        isnr_mean = row.get("isnr_mean")
        isnr_text = "n/a" if isnr_mean in ("", None) else f"{float(isnr_mean):.3f}"

        print(
            f"- {row['method']:8s} | "
            f"SSIM={row['ssim_mean']:.4f} ± {row['ssim_std']:.4f} | "
            f"PSNR={row['psnr_mean']:.2f} ± {row['psnr_std']:.2f} | "
            f"MSE={row['mse_mean']:.6f} ± {row['mse_std']:.6f} | "
            f"ISNR={isnr_text}"
        )