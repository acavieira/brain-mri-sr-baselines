"""CSV tables and dissertation-ready figures for one experiment run."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.metrics import structural_similarity

from .config import METHODS, ORIENTATIONS
from .metrics import mse, psnr

METRIC_NAMES = ("psnr_full", "psnr_brain", "mse", "mae", "rmse", "ssim", "isnr")
BY_SLICE_COLUMNS = (
    "volume",
    "orientation",
    "slice_index",
    "method",
    "psnr_full",
    "psnr_brain",
    "mse",
    "mae",
    "rmse",
    "ssim",
    "isnr",
    "degradation_time_ms",
    "interpolation_time_ms",
    "metrics_time_ms",
    "processing_time_ms",
    "hr_height",
    "hr_width",
    "lr_height",
    "lr_width",
)


def write_csv(rows: List[Mapping[str, object]], path: Path, columns: Optional[Iterable[str]] = None) -> None:
    """Write rows with stable columns and full numeric precision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(columns) if columns is not None else list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finite_values(rows: List[Mapping[str, object]], key: str) -> List[float]:
    values = []
    for row in rows:
        value = float(row[key])
        if math.isfinite(value):
            values.append(value)
    return values


def _summary(values: List[float]) -> Dict[str, object]:
    if not values:
        return {"mean": "", "std": "", "median": "", "min": "", "max": ""}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def aggregate_metrics(rows: List[Mapping[str, object]], group_keys: tuple[str, ...]) -> List[Dict[str, object]]:
    """Aggregate complete per-slice rows by the requested keys."""
    grouped: Dict[tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    output: List[Dict[str, object]] = []
    for key_values, group in grouped.items():
        aggregate: Dict[str, object] = dict(zip(group_keys, key_values))
        aggregate["num_slices"] = len(group)
        for metric in METRIC_NAMES:
            stats = _summary(_finite_values(group, metric))
            for statistic, value in stats.items():
                aggregate[f"{metric}_{statistic}"] = value
        output.append(aggregate)
    return output


def aggregate_runtime(rows: List[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Aggregate processing time by orientation and method."""
    grouped: Dict[tuple[object, object], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(row["orientation"], row["method"])].append(row)
    output = []
    for (orientation, method), group in grouped.items():
        row = {"orientation": orientation, "method": method, "num_slices": len(group)}
        for time_name in ("degradation_time_ms", "interpolation_time_ms", "metrics_time_ms", "processing_time_ms"):
            stats = _summary(_finite_values(group, time_name))
            row.update({f"{time_name}_{key}": value for key, value in stats.items()})
        output.append(row)
    return output


def rank_methods(summary_rows: List[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Rank methods by mean PSNR, then mean SSIM."""
    ranked = sorted(
        (dict(row) for row in summary_rows),
        key=lambda row: (-float(row["psnr_full_mean"]), -float(row["ssim_mean"]), METHODS.index(str(row["method"]))),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def save_example(
    path: Path,
    hr_image: np.ndarray,
    lr_image: np.ndarray,
    reconstructions: Mapping[str, np.ndarray],
    title: str,
    error_vmax: float,
) -> None:
    """Save one reference-style report for each interpolation method."""
    path.parent.mkdir(parents=True, exist_ok=True)
    display_lr = np.asarray(lr_image)
    display_lr = cv2.resize(display_lr, (hr_image.shape[1], hr_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    for method in METHODS:
        reconstruction = reconstructions[method]
        local_ssim, ssim_map = structural_similarity(
            hr_image,
            reconstruction,
            data_range=1.0,
            full=True,
        )
        absolute_error = np.abs(hr_image - reconstruction)
        signed_error = hr_image - reconstruction
        report_path = path.with_name(path.stem.replace("_example", "") + f"_{method}.png")
        figure, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=160)

        panels = (
            (axes[0, 0], hr_image, "HR reference", "gray", 0.0, 1.0),
            (axes[0, 1], display_lr, "LR degraded", "gray", 0.0, 1.0),
            (axes[0, 2], reconstruction, f"Upscaled: {method}", "gray", 0.0, 1.0),
        )
        for axis, image, panel_title, color_map, minimum, maximum in panels:
            axis.imshow(image, cmap=color_map, vmin=minimum, vmax=maximum)
            axis.set_title(panel_title)
            axis.axis("off")

        error_axis = axes[1, 0]
        error_axis.imshow(absolute_error, cmap="turbo", vmin=0.0, vmax=error_vmax)
        error_axis.set_title("Absolute error")
        error_axis.axis("off")

        signed_axis = axes[1, 1]
        signed_axis.imshow(signed_error, cmap="seismic", vmin=-error_vmax, vmax=error_vmax)
        signed_axis.set_title("Signed error")
        signed_axis.axis("off")

        ssim_axis = axes[1, 2]
        ssim_axis.imshow(ssim_map, cmap="viridis", vmin=0.0, vmax=1.0)
        ssim_axis.set_title("Local SSIM")
        ssim_axis.axis("off")

        figure.suptitle(
            f"{title} | {method} | SSIM={local_ssim:.4f} | "
            f"PSNR={_psnr_for_report(hr_image, reconstruction):.2f} | "
            f"MSE={_mse_for_report(hr_image, reconstruction):.6f}",
            fontsize=11,
        )
        figure.tight_layout()
        figure.savefig(report_path)
        plt.close(figure)


def _mse_for_report(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    """Compute MSE for the visual report title."""
    return mse(reference, reconstruction)


def _psnr_for_report(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    """Compute standard PSNR for the visual report title."""
    return psnr(reference, reconstruction)


def _save_psnr_by_method(summary_rows: List[Mapping[str, object]], figures_dir: Path) -> None:
    ordered = sorted(summary_rows, key=lambda row: METHODS.index(str(row["method"])))
    figure, axis = plt.subplots(figsize=(7, 4), dpi=180)
    axis.bar([row["method"] for row in ordered], [float(row["psnr_full_mean"]) for row in ordered], color="#356859")
    axis.set_title("Mean PSNR by method")
    axis.set_xlabel("Method")
    axis.set_ylabel("PSNR (dB)")
    figure.tight_layout()
    figure.savefig(figures_dir / "psnr_by_method.png")
    plt.close(figure)


def _save_psnr_by_axis(axis_rows: List[Mapping[str, object]], figures_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4), dpi=180)
    positions = np.arange(len(ORIENTATIONS))
    width = 0.18
    for method_index, method in enumerate(METHODS):
        values = [float(next(row for row in axis_rows if row["orientation"] == orientation and row["method"] == method)["psnr_full_mean"]) for orientation in ORIENTATIONS]
        axis.bar(positions + (method_index - 1.5) * width, values, width, label=method)
    axis.set_xticks(positions, ORIENTATIONS)
    axis.set_title("Mean PSNR by orientation")
    axis.set_xlabel("Orientation")
    axis.set_ylabel("PSNR (dB)")
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(figures_dir / "psnr_by_axis.png")
    plt.close(figure)


def _save_psnr_distribution(rows: List[Mapping[str, object]], figures_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4), dpi=180)
    axis.boxplot([[float(row["psnr_full"]) for row in rows if row["method"] == method] for method in METHODS], tick_labels=METHODS)
    axis.set_title("PSNR distribution")
    axis.set_xlabel("Method")
    axis.set_ylabel("PSNR (dB)")
    figure.tight_layout()
    figure.savefig(figures_dir / "psnr_distribution_by_method.png")
    plt.close(figure)


def generate_reports(rows: List[Mapping[str, object]], output_dir: Path, parameters: Mapping[str, object]) -> List[Dict[str, object]]:
    """Write tables and figures for a completed run."""
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    by_axis = aggregate_metrics(rows, ("orientation", "method"))
    summary = aggregate_metrics(rows, ("method",))
    ranking = rank_methods(summary)
    runtime = aggregate_runtime(rows)

    write_csv(rows, tables_dir / "metrics_by_slice.csv", BY_SLICE_COLUMNS)
    write_csv(by_axis, tables_dir / "metrics_by_axis.csv")
    write_csv(summary, tables_dir / "metrics_summary.csv")
    write_csv(runtime, tables_dir / "runtime_summary.csv")
    write_csv(ranking, tables_dir / "ranking.csv")
    (output_dir / "run_parameters.json").write_text(json.dumps(parameters, indent=2, sort_keys=True), encoding="utf-8")

    _save_psnr_by_method(summary, figures_dir)
    _save_psnr_by_axis(by_axis, figures_dir)
    _save_psnr_distribution(rows, figures_dir)
    summary_text = ["MRI classical baseline experiment", "", f"Slices evaluated: {len({(row['volume'], row['orientation'], row['slice_index']) for row in rows})}", "", "Ranking by mean full-image PSNR (tie-break: mean SSIM):"]
    summary_text.extend(f"{row['rank']}. {row['method']}: PSNR={row['psnr_full_mean']:.6f} dB, SSIM={row['ssim_mean']:.6f}" for row in ranking)
    (output_dir / "execution_summary.txt").write_text("\n".join(summary_text) + "\n", encoding="utf-8")
    return ranking
