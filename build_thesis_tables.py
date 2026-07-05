"""Build thesis-oriented CSV/Markdown tables from experiment outputs."""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from src.mri_sr.thesis_tables import (
    build_composite_ranking,
    build_image_size_table,
    build_method_ranking,
    build_processing_time_table,
    build_psnr_reference_template,
    write_csv_rows,
    write_thesis_markdown_bundle,
)


AXES = ["sagittal", "coronal", "axial"]


def parse_args() -> argparse.Namespace:
    """Parse CLI options for table generation."""
    parser = argparse.ArgumentParser(description="Build thesis-ready result tables")
    parser.add_argument("--base-results-dir", default="results/baseline_methods")
    parser.add_argument(
        "--analysis-dir",
        default="results/analysis/final_comparison",
        help="Folder where aggregate comparison files are stored and tables are written.",
    )
    parser.add_argument("--scale", type=int, default=2)
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read CSV rows as dictionaries using UTF-8 encoding."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def collect_by_slice_rows(base_results_dir: Path, scale: int) -> List[Dict[str, object]]:
    """Collect per-slice metrics from each anatomical axis folder."""
    rows: List[Dict[str, object]] = []
    for axis in AXES:
        path = base_results_dir / f"scale{scale}_{axis}" / "metrics_by_slice.csv"
        if not path.exists():
            continue

        for row in read_csv_rows(path):
            out_row: Dict[str, object] = dict(row)
            out_row["slice_axis"] = out_row.get("slice_axis") or axis
            rows.append(out_row)
    return rows


def main() -> None:
    """Create all thesis tables and write them under final_comparison/tables."""
    args = parse_args()

    base_results_dir = Path(args.base_results_dir)
    final_dir = Path(args.analysis_dir)
    tables_dir = final_dir / "tables"

    aggregate_path = final_dir / "all_axes_aggregate.csv"
    legacy_aggregate_path = base_results_dir.parent / "final_comparison" / "all_axes_aggregate.csv"
    if not aggregate_path.exists() and legacy_aggregate_path.exists():
        aggregate_path = legacy_aggregate_path
    if not aggregate_path.exists():
        raise FileNotFoundError(
            f"Missing aggregate file: {aggregate_path}. "
            "Run run_all_experiments.py first."
        )

    by_slice_path = final_dir / "all_axes_by_slice.csv"
    legacy_by_slice_path = base_results_dir.parent / "final_comparison" / "all_axes_by_slice.csv"
    source_by_slice_path = by_slice_path
    if not source_by_slice_path.exists() and legacy_by_slice_path.exists():
        source_by_slice_path = legacy_by_slice_path

    # Read aggregate results and reuse the unified per-slice file when available.
    aggregate_rows = [dict(row) for row in read_csv_rows(aggregate_path)]
    by_slice_rows = (
        [dict(row) for row in read_csv_rows(source_by_slice_path)]
        if source_by_slice_path.exists()
        else collect_by_slice_rows(base_results_dir, scale=args.scale)
    )

    # Persist the fallback by-slice merge so future runs can reuse it directly.
    if by_slice_rows and not by_slice_path.exists():
        write_csv_rows(by_slice_rows, by_slice_path)

    # Keep aggregate CSV inside analysis-dir for a unified folder layout.
    if aggregate_rows and aggregate_path != (final_dir / "all_axes_aggregate.csv"):
        write_csv_rows(aggregate_rows, final_dir / "all_axes_aggregate.csv")

    # Build all thesis-facing tables from aggregate and by-slice data.
    ranking_rows = build_composite_ranking(aggregate_rows)
    method_ranking_rows = build_method_ranking(aggregate_rows)
    image_rows = build_image_size_table(by_slice_rows)
    timing_rows = build_processing_time_table(by_slice_rows)
    psnr_reference_rows = build_psnr_reference_template(aggregate_rows)

    write_csv_rows(ranking_rows, tables_dir / "ranking_global_composto.csv")
    write_csv_rows(method_ranking_rows, tables_dir / "ranking_global_metodos.csv")
    write_csv_rows(image_rows, tables_dir / "tamanho_imagens.csv")
    write_csv_rows(timing_rows, tables_dir / "tempos_processamento.csv")
    write_csv_rows(psnr_reference_rows, tables_dir / "psnr_vs_literatura_template.csv")

    # Also generate a ready-to-paste markdown bundle for the report.
    write_thesis_markdown_bundle(
        tables_dir / "tabelas_relatorio.md",
        ranking_rows=ranking_rows,
        image_rows=image_rows,
        timing_rows=timing_rows,
        method_ranking_rows=method_ranking_rows,
    )

    print("Done.")
    print(f"Tables folder: {tables_dir}")
    print(f"Ranking CSV: {tables_dir / 'ranking_global_composto.csv'}")
    print(f"Method ranking CSV: {tables_dir / 'ranking_global_metodos.csv'}")
    print(f"Image size CSV: {tables_dir / 'tamanho_imagens.csv'}")
    print(f"Processing time CSV: {tables_dir / 'tempos_processamento.csv'}")
    print(f"PSNR literature template CSV: {tables_dir / 'psnr_vs_literatura_template.csv'}")
    print(f"Markdown bundle: {tables_dir / 'tabelas_relatorio.md'}")


if __name__ == "__main__":
    main()
