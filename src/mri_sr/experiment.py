import os
from typing import Dict, List

import numpy as np

from .config import ExperimentConfig
from .degradation import degrade_image
from .io import choose_slice_indices, ensure_dir, extract_slice, load_nifti_volume
from .metrics import compute_metrics
from .preprocessing import prepare_hr_reference
from .reports import save_metrics_csv, save_visual_report
from .upscaling import INTERPOLATION_METHODS, upscale_image


def aggregate_metrics(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
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
    ]

    aggregate_rows = []

    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        aggregate = {
            "method": method,
            "num_slices": len(method_rows),
        }

        for metric in metric_names:
            values = np.array([float(row[metric]) for row in method_rows], dtype=np.float32)
            aggregate[f"{metric}_mean"] = float(values.mean())
            aggregate[f"{metric}_std"] = float(values.std())

        aggregate_rows.append(aggregate)

    aggregate_rows.sort(
        key=lambda row: (
            -row["ssim_mean"],
            -row["psnr_mean"],
            row["mse_mean"],
        )
    )

    return aggregate_rows


def run_experiment(config: ExperimentConfig) -> None:
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

    rows: List[Dict[str, object]] = []

    for slice_index in slice_indices:
        print(f"\nProcessing slice {slice_index}...")

        raw_slice = extract_slice(volume, config.slice_axis, slice_index)
        hr_img = prepare_hr_reference(raw_slice, config.force_square_size)
        lr_img = degrade_image(
            hr_img,
            scale=config.scale,
            blur_sigma=config.blur_sigma,
            noise_sigma=config.noise_sigma,
            rng=rng,
        )
        save_report = config.save_all_figures or slice_index == central_slice

        for method in INTERPOLATION_METHODS:
            sr_img = upscale_image(method, lr_img, hr_img.shape)
            metrics = compute_metrics(
                hr_img,
                sr_img,
                include_ssim_map=save_report,
                compute_issm=config.compute_issm,
            )

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
                "issm": "" if metrics["issm"] is None else metrics["issm"],
            }
            rows.append(row)

            print(
                f"  {method:8s} | "
                f"SSIM={metrics['ssim']:.4f} | "
                f"PSNR={metrics['psnr']:.2f} | "
                f"MSE={metrics['mse']:.6f} | "
                f"HFEN={metrics['hfen']:.4f}"
            )

            if save_report:
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

    save_metrics_csv(rows, metrics_path)
    aggregate_rows = aggregate_metrics(rows)
    save_metrics_csv(aggregate_rows, aggregate_path)

    print("\nDone.")
    print(f"Metrics by slice: {metrics_path}")
    print(f"Aggregate metrics: {aggregate_path}")

    print("\nRanking:")
    for row in aggregate_rows:
        print(
            f"- {row['method']:8s} | "
            f"SSIM={row['ssim_mean']:.4f} ± {row['ssim_std']:.4f} | "
            f"PSNR={row['psnr_mean']:.2f} ± {row['psnr_std']:.2f} | "
            f"MSE={row['mse_mean']:.6f} ± {row['mse_std']:.6f}"
        )