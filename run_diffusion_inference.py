"""CLI entrypoint to evaluate a trained diffusion checkpoint on MRI slices."""

import argparse
import os
import time
from typing import Dict, List

import numpy as np

from src.mri_sr.degradation import degrade_image
from src.mri_sr.diffusion.inference import DiffusionSuperResolver
from src.mri_sr.experiment import aggregate_metrics
from src.mri_sr.io import choose_slice_indices, ensure_dir, extract_slice, load_nifti_volume
from src.mri_sr.metrics import compute_metrics
from src.mri_sr.preprocessing import prepare_hr_reference
from src.mri_sr.reports import save_metrics_csv, save_visual_report
from src.mri_sr.upscaling import upscale_image


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for diffusion inference and evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate trained diffusion U-Net model on synthetic 9.4T -> 1.5T setup")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-path", default="data/sub-0_ses-1_T1w.nii")
    parser.add_argument("--output-dir", default="results/diffusion/evaluation/diffusion_eval")
    parser.add_argument("--slice-axis", default="sagittal", choices=["sagittal", "coronal", "axial"])
    parser.add_argument("--slice-index", type=int, default=-1)
    parser.add_argument("--num-slices", type=int, default=9)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--blur-sigma", type=float, default=0.65)
    parser.add_argument("--noise-sigma", type=float, default=0.02)
    parser.add_argument("--force-square-size", type=int, default=256)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--isnr-baseline-method", default="bilinear")
    parser.add_argument("--save-all-figures", action="store_true")
    parser.add_argument("--random-seed", type=int, default=23)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    """Run diffusion inference, compute metrics, and save CSV/figure outputs."""
    args = parse_args()

    # Prepare output structure before processing slices.
    ensure_dir(args.output_dir)
    figures_dir = os.path.join(args.output_dir, "figures")
    ensure_dir(figures_dir)

    resolver = DiffusionSuperResolver.from_checkpoint(args.checkpoint, device=args.device)

    volume = load_nifti_volume(args.input_path)
    slice_indices = choose_slice_indices(
        volume,
        axis=args.slice_axis,
        center_index=args.slice_index,
        num_slices=args.num_slices,
    )
    central_slice = slice_indices[len(slice_indices) // 2]

    # Use one deterministic RNG for the synthetic degradation path.
    rng = np.random.default_rng(args.random_seed)
    rows: List[Dict[str, object]] = []

    for slice_index in slice_indices:
        raw_slice = extract_slice(volume, args.slice_axis, slice_index)
        hr_img = prepare_hr_reference(raw_slice, force_square_size=args.force_square_size)

        degradation_start = time.perf_counter()
        lr_img = degrade_image(
            hr_img,
            scale=args.scale,
            blur_sigma=args.blur_sigma,
            noise_sigma=args.noise_sigma,
            rng=rng,
        )
        degradation_time_ms = (time.perf_counter() - degradation_start) * 1000.0

        baseline_sr = upscale_image(args.isnr_baseline_method, lr_img, hr_img.shape)

        # Measure pure diffusion sampling time separately.
        infer_start = time.perf_counter()
        sr_img = resolver.super_resolve(
            lr_img=lr_img,
            output_shape=hr_img.shape,
            num_steps=args.sampling_steps,
        )
        upscaling_time_ms = (time.perf_counter() - infer_start) * 1000.0

        metrics_start = time.perf_counter()
        metrics = compute_metrics(hr_img, sr_img, baseline_pred=baseline_sr)
        metrics_time_ms = (time.perf_counter() - metrics_start) * 1000.0

        row = {
            "slice_axis": args.slice_axis,
            "slice_index": slice_index,
            "method": "diffusion_unet",
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
            "scale": args.scale,
            "isnr_baseline_method": args.isnr_baseline_method,
            "hr_height": hr_img.shape[0],
            "hr_width": hr_img.shape[1],
            "lr_height": lr_img.shape[0],
            "lr_width": lr_img.shape[1],
            "degradation_time_ms": degradation_time_ms,
            "upscaling_time_ms": upscaling_time_ms,
            "metrics_time_ms": metrics_time_ms,
            "total_method_time_ms": degradation_time_ms + upscaling_time_ms + metrics_time_ms,
        }
        rows.append(row)

        print(
            f"Slice {slice_index:04d} | SSIM={metrics['ssim']:.4f} | "
            f"PSNR={metrics['psnr']:.2f} | HFEN={metrics['hfen']:.4f} | "
            f"time={row['total_method_time_ms']:.1f} ms"
        )

        if args.save_all_figures or slice_index == central_slice:
            save_visual_report(
                method="diffusion_unet",
                axis=args.slice_axis,
                slice_index=slice_index,
                hr_img=hr_img,
                lr_img=lr_img,
                sr_img=sr_img,
                ssim_map=metrics["ssim_map"],
                metrics=metrics,
                output_dir=figures_dir,
                error_vmax=0.35,
            )

    metrics_by_slice_path = os.path.join(args.output_dir, "metrics_by_slice.csv")
    metrics_aggregate_path = os.path.join(args.output_dir, "metrics_aggregate.csv")

    # Keep the same CSV format used by the baseline pipeline.
    save_metrics_csv(rows, metrics_by_slice_path)
    save_metrics_csv(aggregate_metrics(rows), metrics_aggregate_path)

    print("\nDone.")
    print(f"Metrics by slice: {metrics_by_slice_path}")
    print(f"Aggregate metrics: {metrics_aggregate_path}")


if __name__ == "__main__":
    main()
