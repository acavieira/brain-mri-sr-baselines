"""Explicit execution flow for the MRI SR comparison."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import ExperimentConfig, METHODS, ORIENTATIONS
from .degradation import degrade_image
from .interpolation import upscale_image
from .metrics import compute_metrics
from .nifti_io import ORIENTATION_AXES, extract_anatomical_slice, find_nifti_files, load_canonical_volume, select_slice_indices
from .preprocessing import create_brain_mask, normalize_volume_to_float01, prepare_hr_reference
from .reporting import generate_reports, save_example


def timestamped_output_dir(output_root: str) -> Path:
    """Create a new output directory without reusing previous results."""
    timestamp = time.strftime("run_%Y%m%d_%H%M%S")
    output_dir = Path(output_root) / timestamp
    suffix = 1
    while output_dir.exists():
        output_dir = Path(output_root) / f"{timestamp}_{suffix}"
        suffix += 1
    output_dir.mkdir(parents=True)
    return output_dir


def run_experiment(config: ExperimentConfig, output_dir: Optional[Path] = None) -> Tuple[Path, List[Dict[str, object]]]:
    """Run HR -> degradation -> LR -> interpolation -> metrics for all inputs."""
    rng = np.random.default_rng(config.random_seed)
    input_paths = find_nifti_files(config.input_dir)
    if config.isnr_baseline_method not in METHODS:
        raise ValueError(f"Unknown ISNR baseline method: {config.isnr_baseline_method}")
    if output_dir is None:
        output_dir = timestamped_output_dir(config.output_root)

    rows: List[Dict[str, object]] = []
    for input_path in input_paths:
        print(f"Loading canonical volume: {input_path.name}")
        volume = load_canonical_volume(input_path)
        normalized_volume = normalize_volume_to_float01(
            volume,
            low_percentile=config.low_percentile,
            high_percentile=config.high_percentile,
        )
        print(f"  canonical shape={volume.shape}; orientations={', '.join(ORIENTATIONS)}")

        for orientation in ORIENTATIONS:
            axis = ORIENTATION_AXES[orientation]
            indices = select_slice_indices(volume.shape[axis], config.slices_per_volume)
            print(f"  {orientation}: {len(indices)} slices ({indices[0]}..{indices[-1]})")

            for slice_position, slice_index in enumerate(indices):
                normalized_slice = extract_anatomical_slice(normalized_volume, orientation, slice_index)
                hr_image = prepare_hr_reference(normalized_slice, config.target_size)
                brain_mask = create_brain_mask(hr_image, config.brain_mask_threshold)
                degradation_start = time.perf_counter()
                lr_image = degrade_image(
                    hr_image,
                    scale=config.scale,
                    blur_sigma=config.blur_sigma,
                    noise_sigma=config.noise_sigma,
                    rng=rng,
                )
                degradation_time_ms = (time.perf_counter() - degradation_start) * 1000.0

                reconstructions = {}
                interpolation_times = {}
                for method in METHODS:
                    interpolation_start = time.perf_counter()
                    reconstructions[method] = upscale_image(lr_image, method, hr_image.shape)
                    interpolation_times[method] = (time.perf_counter() - interpolation_start) * 1000.0

                for method in METHODS:
                    reconstruction = reconstructions[method]
                    metrics_start = time.perf_counter()
                    metrics = compute_metrics(
                        hr_image,
                        reconstruction,
                        baseline=reconstructions[config.isnr_baseline_method],
                        brain_mask=brain_mask,
                    )
                    metrics_time_ms = (time.perf_counter() - metrics_start) * 1000.0
                    rows.append(
                        {
                            "volume": input_path.name,
                            "orientation": orientation,
                            "slice_index": slice_index,
                            "method": method,
                            **metrics,
                            "degradation_time_ms": degradation_time_ms,
                            "interpolation_time_ms": interpolation_times[method],
                            "metrics_time_ms": metrics_time_ms,
                            "processing_time_ms": interpolation_times[method],
                            "hr_height": hr_image.shape[0],
                            "hr_width": hr_image.shape[1],
                            "lr_height": lr_image.shape[0],
                            "lr_width": lr_image.shape[1],
                        }
                    )

                should_save_example = config.save_all_examples or slice_position == len(indices) // 2
                if should_save_example:
                    example_name = f"report_{orientation}_{slice_index:04d}_example.png"
                    save_example(
                        path=output_dir / "figures" / "examples" / example_name,
                        hr_image=hr_image,
                        lr_image=lr_image,
                        reconstructions=reconstructions,
                        title=f"{input_path.name} | {orientation} | slice {slice_index}",
                        error_vmax=0.35,
                    )

    ranking = generate_reports(rows, output_dir, {**config.to_dict(), "input_files": [path.name for path in input_paths]})
    print("\nRanking by mean PSNR:")
    for row in ranking:
        print(f"  {row['rank']}. {row['method']}: {row['psnr_full_mean']:.4f} dB")
    return output_dir, rows
