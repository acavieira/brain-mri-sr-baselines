"""
Runs the MRI SR baseline experiment for all anatomical orientations:
sagittal, coronal, and axial.

It creates one result folder per orientation and then generates a final
comparison CSV and Markdown summary.
"""

import csv
import argparse
from pathlib import Path
from typing import Dict, List

from src.mri_sr.config import ExperimentConfig
from src.mri_sr.experiment import run_experiment
from src.mri_sr.thesis_tables import (
    build_composite_ranking,
    build_image_size_table,
    build_method_ranking,
    build_processing_time_table,
    build_psnr_reference_template,
    rank_all_results,
    to_float,
    write_csv_rows,
    write_thesis_markdown_bundle,
)
from src.mri_sr.upscaling import INTERPOLATION_METHODS


AXES = ["sagittal", "coronal", "axial"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the full baseline run across all axes."""
    parser = argparse.ArgumentParser(description="Run MRI SR experiments for all anatomical orientations")
    parser.add_argument("--input-path", default="data/sub-0_ses-1_T1w.nii")
    parser.add_argument("--base-results-dir", default="results/baseline_methods")
    parser.add_argument(
        "--analysis-dir",
        default="results/analysis/final_comparison",
        help="Folder where aggregate comparison, tables, and analysis outputs are stored.",
    )
    parser.add_argument("--slice-index", type=int, default=-1)
    parser.add_argument("--num-slices", type=int, default=30)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--blur-sigma", type=float, default=0.45)
    parser.add_argument("--noise-sigma", type=float, default=0.02)
    parser.add_argument(
        "--isnr-baseline-method",
        default="bilinear",
        choices=INTERPOLATION_METHODS,
        help="Metodo de referencia para calculo de ISNR.",
    )
    parser.add_argument("--force-square-size", type=int, default=None)
    parser.add_argument("--save-all-figures", action="store_true")
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read a CSV file and return rows as dictionaries."""
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def add_axis_to_rows(rows: List[Dict[str, str]], axis: str) -> List[Dict[str, object]]:
    """Guarantee a `slice_axis` value for each row."""
    updated = []
    for row in rows:
        new_row: Dict[str, object] = dict(row)
        new_row["slice_axis"] = new_row.get("slice_axis") or axis
        updated.append(new_row)
    return updated


def _format_optional(value: object, digits: int = 4) -> str:
    """Format numeric values and return n/a when not available."""
    numeric_value = to_float(value)
    if numeric_value != numeric_value:
        return "n/a"
    return f"{numeric_value:.{digits}f}"


def create_final_summary(rows: List[Dict[str, object]], output_path: Path) -> None:
    """Create a human-readable Markdown summary for final aggregate metrics."""
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
    lines.append("5. Higher mean ISNR (tie-break)")
    lines.append("```")
    lines.append("")

    if ranked:
        best = ranked[0]
        lines.append("## Best overall result")
        lines.append("")
        lines.append(f"- Orientation: `{best['slice_axis']}`")
        lines.append(f"- Method: `{best['method']}`")
        lines.append(f"- SSIM mean: `{_format_optional(best.get('ssim_mean'), 4)}`")
        lines.append(f"- PSNR mean: `{_format_optional(best.get('psnr_mean'), 2)}`")
        lines.append(f"- MSE mean: `{_format_optional(best.get('mse_mean'), 6)}`")
        lines.append(f"- HFEN mean: `{_format_optional(best.get('hfen_mean'), 4)}`")
        lines.append(f"- ISNR mean: `{_format_optional(best.get('isnr_mean'), 3)}`")
        lines.append("")

    lines.append("## Global ranking")
    lines.append("")
    lines.append("| Rank | Orientation | Method | SSIM mean | PSNR mean | MSE mean | HFEN mean | ISNR mean |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|")

    for idx, row in enumerate(ranked, start=1):
        lines.append(
            f"| {idx} | `{row['slice_axis']}` | `{row['method']}` | "
            f"{_format_optional(row.get('ssim_mean'), 4)} | "
            f"{_format_optional(row.get('psnr_mean'), 2)} | "
            f"{_format_optional(row.get('mse_mean'), 6)} | "
            f"{_format_optional(row.get('hfen_mean'), 4)} | "
            f"{_format_optional(row.get('isnr_mean'), 3)} |"
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
        lines.append(f"- SSIM mean: `{_format_optional(best_axis.get('ssim_mean'), 4)}`")
        lines.append(f"- PSNR mean: `{_format_optional(best_axis.get('psnr_mean'), 2)}`")
        lines.append(f"- MSE mean: `{_format_optional(best_axis.get('mse_mean'), 6)}`")
        lines.append(f"- HFEN mean: `{_format_optional(best_axis.get('hfen_mean'), 4)}`")
        lines.append(f"- ISNR mean: `{_format_optional(best_axis.get('isnr_mean'), 3)}`")
        lines.append("")

    lines.append("## Interpretation notes")
    lines.append("")
    lines.append("- Higher SSIM indicates better structural similarity between HR and SR.")
    lines.append("- Higher PSNR indicates lower global reconstruction error.")
    lines.append("- Lower MSE/RMSE/MAE indicates lower pixel-level error.")
    lines.append("- Lower HFEN suggests better preservation of high-frequency detail.")
    lines.append("- ISNR measures the gain over a baseline interpolation (configured in the CLI).")
    lines.append("- The best classical method should not be interpreted as recovering lost anatomical information. It only produced the closest interpolation-based reconstruction under this controlled degradation setting.")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run baseline experiments for all axes and export final comparison files."""
    args = parse_args()

    input_path = args.input_path
    base_results_dir = Path(args.base_results_dir)
    final_dir = Path(args.analysis_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    all_aggregate_rows: List[Dict[str, object]] = []
    all_by_slice_rows: List[Dict[str, object]] = []

    # Execute one baseline run per anatomical orientation.
    for axis in AXES:
        output_dir = base_results_dir / f"scale{args.scale}_{axis}"

        config = ExperimentConfig(
            input_path=input_path,
            output_dir=str(output_dir),
            slice_axis=axis,
            slice_index=args.slice_index,
            num_slices=args.num_slices,
            scale=args.scale,
            blur_sigma=args.blur_sigma,
            noise_sigma=args.noise_sigma,
            isnr_baseline_method=args.isnr_baseline_method,
            force_square_size=args.force_square_size,
            save_all_figures=args.save_all_figures,
        )

        print("\n" + "=" * 80)
        print(f"Running experiment for axis: {axis}")
        print("=" * 80)

        run_experiment(config)

        aggregate_path = output_dir / "metrics_aggregate.csv"
        by_slice_path = output_dir / "metrics_by_slice.csv"

        # Merge per-axis outputs into final comparison tables.
        if aggregate_path.exists():
            axis_rows = read_csv_rows(aggregate_path)
            all_aggregate_rows.extend(add_axis_to_rows(axis_rows, axis))
        else:
            print(f"Warning: aggregate file not found: {aggregate_path}")

        if by_slice_path.exists():
            by_slice_rows = read_csv_rows(by_slice_path)
            all_by_slice_rows.extend(add_axis_to_rows(by_slice_rows, axis))
        else:
            print(f"Warning: by-slice file not found: {by_slice_path}")

    # Keep the final aggregate file already sorted by quality ranking.
    all_aggregate_rows = rank_all_results(all_aggregate_rows)

    final_csv_path = final_dir / "all_axes_aggregate.csv"
    final_by_slice_csv_path = final_dir / "all_axes_by_slice.csv"
    final_md_path = final_dir / "final_results_summary.md"

    # Save final aggregate/by-slice CSVs and summary markdown.
    write_csv_rows(all_aggregate_rows, final_csv_path)
    write_csv_rows(all_by_slice_rows, final_by_slice_csv_path)
    create_final_summary(all_aggregate_rows, final_md_path)

    # Build thesis-friendly tables directly from final merged data.
    tables_dir = final_dir / "tables"
    ranking_rows = build_composite_ranking(all_aggregate_rows)
    method_ranking_rows = build_method_ranking(all_aggregate_rows)
    image_rows = build_image_size_table(all_by_slice_rows)
    timing_rows = build_processing_time_table(all_by_slice_rows)
    psnr_reference_rows = build_psnr_reference_template(all_aggregate_rows)

    write_csv_rows(ranking_rows, tables_dir / "ranking_global_composto.csv")
    write_csv_rows(method_ranking_rows, tables_dir / "ranking_global_metodos.csv")
    write_csv_rows(image_rows, tables_dir / "tamanho_imagens.csv")
    write_csv_rows(timing_rows, tables_dir / "tempos_processamento.csv")
    write_csv_rows(psnr_reference_rows, tables_dir / "psnr_vs_literatura_template.csv")
    write_thesis_markdown_bundle(
        tables_dir / "tabelas_relatorio.md",
        ranking_rows=ranking_rows,
        image_rows=image_rows,
        timing_rows=timing_rows,
        method_ranking_rows=method_ranking_rows,
    )

    print("\nDone.")
    print(f"Final aggregate CSV: {final_csv_path}")
    print(f"Final by-slice CSV: {final_by_slice_csv_path}")
    print(f"Final Markdown summary: {final_md_path}")
    print(f"Thesis tables: {tables_dir}")


if __name__ == "__main__":
    main()
