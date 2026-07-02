"""
Runs the MRI SR baseline experiment for all anatomical orientations:
sagittal, coronal, and axial.

It creates one result folder per orientation and then generates a final
comparison CSV and Markdown summary.
"""

import csv
import os
from pathlib import Path
from typing import Dict, List

from src.mri_sr.config import ExperimentConfig
from src.mri_sr.experiment import run_experiment


AXES = ["sagittal", "coronal", "axial"]


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: object, default: float = float("nan")) -> float:
    try:
        if value == "" or value is None:
            return default
        return float(value)
    except Exception:
        return default


def add_axis_to_aggregate_rows(rows: List[Dict[str, str]], axis: str) -> List[Dict[str, object]]:
    updated = []
    for row in rows:
        new_row: Dict[str, object] = {"slice_axis": axis}
        new_row.update(row)
        updated.append(new_row)
    return updated


def rank_all_results(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            -to_float(row.get("ssim_mean")),
            -to_float(row.get("psnr_mean")),
            to_float(row.get("mse_mean")),
            to_float(row.get("hfen_mean")),
        ),
    )


def create_final_summary(rows: List[Dict[str, object]], output_path: Path) -> None:
    ranked = rank_all_results(rows)

    lines = []
    lines.append("# Final Results Summary")
    lines.append("")
    lines.append("This document summarizes the aggregate results obtained across the three anatomical orientations.")
    lines.append("")
    lines.append("Ranking criterion used in this summary:")
    lines.append("")
    lines.append("```text")
    lines.append("1. Higher mean SSIM")
    lines.append("2. Higher mean PSNR")
    lines.append("3. Lower mean MSE")
    lines.append("4. Lower mean HFEN")
    lines.append("```")
    lines.append("")

    if ranked:
        best = ranked[0]
        lines.append("## Best overall result")
        lines.append("")
        lines.append(f"- Orientation: `{best['slice_axis']}`")
        lines.append(f"- Method: `{best['method']}`")
        lines.append(f"- SSIM mean: `{to_float(best.get('ssim_mean')):.4f}`")
        lines.append(f"- PSNR mean: `{to_float(best.get('psnr_mean')):.2f}`")
        lines.append(f"- MSE mean: `{to_float(best.get('mse_mean')):.6f}`")
        lines.append(f"- HFEN mean: `{to_float(best.get('hfen_mean')):.4f}`")
        lines.append("")

    lines.append("## Global ranking")
    lines.append("")
    lines.append("| Rank | Orientation | Method | SSIM mean | PSNR mean | MSE mean | HFEN mean |")
    lines.append("|---:|---|---|---:|---:|---:|---:|")

    for idx, row in enumerate(ranked, start=1):
        lines.append(
            f"| {idx} | `{row['slice_axis']}` | `{row['method']}` | "
            f"{to_float(row.get('ssim_mean')):.4f} | "
            f"{to_float(row.get('psnr_mean')):.2f} | "
            f"{to_float(row.get('mse_mean')):.6f} | "
            f"{to_float(row.get('hfen_mean')):.4f} |"
        )

    lines.append("")
    lines.append("## Best method per anatomical orientation")
    lines.append("")

    for axis in AXES:
        axis_rows = [row for row in ranked if row["slice_axis"] == axis]
        if not axis_rows:
            continue

        best_axis = axis_rows[0]
        lines.append(f"### {axis}")
        lines.append("")
        lines.append(f"- Best method: `{best_axis['method']}`")
        lines.append(f"- SSIM mean: `{to_float(best_axis.get('ssim_mean')):.4f}`")
        lines.append(f"- PSNR mean: `{to_float(best_axis.get('psnr_mean')):.2f}`")
        lines.append(f"- MSE mean: `{to_float(best_axis.get('mse_mean')):.6f}`")
        lines.append(f"- HFEN mean: `{to_float(best_axis.get('hfen_mean')):.4f}`")
        lines.append("")

    lines.append("## Interpretation notes")
    lines.append("")
    lines.append("- Higher SSIM indicates better structural similarity between HR and SR.")
    lines.append("- Higher PSNR indicates lower global reconstruction error.")
    lines.append("- Lower MSE/RMSE/MAE indicates lower pixel-level error.")
    lines.append("- Lower HFEN suggests better preservation of high-frequency detail.")
    lines.append("- The best classical method should not be interpreted as recovering lost anatomical information. It only produced the closest interpolation-based reconstruction under this controlled degradation setting.")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    input_path = "data/sub-0_ses-1_T1w.nii"
    base_results_dir = Path("results")
    final_dir = base_results_dir / "final_comparison"
    final_dir.mkdir(parents=True, exist_ok=True)

    all_aggregate_rows: List[Dict[str, object]] = []

    for axis in AXES:
        output_dir = base_results_dir / f"scale4_{axis}"

        config = ExperimentConfig(
            input_path=input_path,
            output_dir=str(output_dir),
            slice_axis=axis,
            slice_index=-1,
            num_slices=9,
            scale=4,
            # blur_sigma=1.0,
            # noise_sigma=0.0,
            blur_sigma=0.55,
            noise_sigma=0.01,
            force_square_size=None,
            save_all_figures=True,
        )

        print("\n" + "=" * 80)
        print(f"Running experiment for axis: {axis}")
        print("=" * 80)

        run_experiment(config)

        aggregate_path = output_dir / "metrics_aggregate.csv"
        if aggregate_path.exists():
            axis_rows = read_csv_rows(aggregate_path)
            all_aggregate_rows.extend(add_axis_to_aggregate_rows(axis_rows, axis))
        else:
            print(f"Warning: aggregate file not found: {aggregate_path}")

    all_aggregate_rows = rank_all_results(all_aggregate_rows)

    final_csv_path = final_dir / "all_axes_aggregate.csv"
    final_md_path = final_dir / "final_results_summary.md"

    write_csv_rows(all_aggregate_rows, final_csv_path)
    create_final_summary(all_aggregate_rows, final_md_path)

    print("\nDone.")
    print(f"Final aggregate CSV: {final_csv_path}")
    print(f"Final Markdown summary: {final_md_path}")


if __name__ == "__main__":
    main()
