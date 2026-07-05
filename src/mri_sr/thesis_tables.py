"""Utilities to generate thesis-friendly ranking and summary tables."""

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def to_float(value: object, default: float = float("nan")) -> float:
    """Convert values to float and keep a fallback for invalid inputs."""
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def write_csv_rows(rows: List[Dict[str, object]], path: Path) -> None:
    """Write rows to CSV, creating parent directories when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rank_all_results(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Apply the baseline ranking order used in report summaries."""
    return sorted(
        rows,
        key=lambda row: (
            -to_float(row.get("ssim_mean")),
            -to_float(row.get("psnr_mean")),
            to_float(row.get("mse_mean")),
            to_float(row.get("hfen_mean")),
        ),
    )


def _normalize_series(values: List[float], higher_is_better: bool) -> List[float]:
    """Min-max normalize a list while handling missing/non-finite values."""
    finite_values = [v for v in values if math.isfinite(v)]
    if not finite_values:
        return [0.5 for _ in values]

    min_value = min(finite_values)
    max_value = max(finite_values)
    if abs(max_value - min_value) <= 1e-12:
        return [0.5 for _ in values]

    normalized = []
    for value in values:
        if not math.isfinite(value):
            normalized.append(0.5)
            continue

        ratio = (value - min_value) / (max_value - min_value)
        normalized.append(ratio if higher_is_better else (1.0 - ratio))
    return normalized


def build_composite_ranking(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Build a weighted composite score and rank for each row."""
    if not rows:
        return []

    # Weights can be tuned later, but all scores are normalized first.
    weights = {
        "ssim_mean": (0.30, True),
        "psnr_mean": (0.25, True),
        "mse_mean": (0.20, False),
        "hfen_mean": (0.15, False),
        "isnr_mean": (0.05, True),
        "total_method_time_ms_mean": (0.05, False),
    }

    normalized_by_metric: Dict[str, List[float]] = {}
    active_metrics: List[Tuple[str, float]] = []

    # Keep only metrics that actually exist in current data.
    for metric, (weight, higher_is_better) in weights.items():
        raw_values = [to_float(row.get(metric)) for row in rows]
        if not any(math.isfinite(v) for v in raw_values):
            continue
        normalized_by_metric[metric] = _normalize_series(raw_values, higher_is_better)
        active_metrics.append((metric, weight))

    if not active_metrics:
        active_metrics = [("ssim_mean", 1.0)]
        normalized_by_metric["ssim_mean"] = _normalize_series(
            [to_float(row.get("ssim_mean")) for row in rows],
            higher_is_better=True,
        )

    total_weight = sum(weight for _, weight in active_metrics)
    normalized_weight = {metric: weight / total_weight for metric, weight in active_metrics}

    scored_rows: List[Dict[str, object]] = []
    for index, row in enumerate(rows):
        score = 0.0
        for metric, weight in normalized_weight.items():
            score += normalized_by_metric[metric][index] * weight

        out_row = dict(row)
        out_row["overall_score"] = round(score, 6)
        scored_rows.append(out_row)

    scored_rows.sort(
        key=lambda row: (
            -to_float(row.get("overall_score")),
            -to_float(row.get("ssim_mean")),
            -to_float(row.get("psnr_mean")),
            to_float(row.get("mse_mean")),
        )
    )

    for rank, row in enumerate(scored_rows, start=1):
        row["overall_rank"] = rank

    return scored_rows


def _mean(values: Iterable[float]) -> float:
    """Return arithmetic mean or NaN when the input is empty."""
    values = list(values)
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _std(values: Iterable[float]) -> float:
    """Return population standard deviation or NaN when empty."""
    values = list(values)
    if not values:
        return float("nan")
    mean_value = _mean(values)
    return math.sqrt(sum((v - mean_value) ** 2 for v in values) / len(values))


def _weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    """Return weighted mean while ignoring invalid values and weights."""
    weighted_pairs = []
    for value, weight in zip(values, weights):
        if not (math.isfinite(value) and math.isfinite(weight) and weight > 0):
            continue
        weighted_pairs.append((value, weight))

    if not weighted_pairs:
        return float("nan")

    total_weight = sum(weight for _, weight in weighted_pairs)
    if total_weight <= 1e-12:
        return float("nan")

    weighted_sum = sum(value * weight for value, weight in weighted_pairs)
    return weighted_sum / total_weight


def build_method_ranking(aggregate_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Build a method-only ranking by combining all anatomical axes.

    Metrics are averaged with `num_slices` as weight so each axis contributes
    proportionally to the number of slices used in that run.
    """
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in aggregate_rows:
        method = str(row.get("method", "unknown"))
        grouped[method].append(row)

    summary_rows: List[Dict[str, object]] = []
    tracked_metrics = [
        "ssim_mean",
        "psnr_mean",
        "mse_mean",
        "hfen_mean",
        "isnr_mean",
        "total_method_time_ms_mean",
    ]

    for method, rows in sorted(grouped.items()):
        weights: List[float] = []
        for row in rows:
            num_slices = to_float(row.get("num_slices"), default=1.0)
            if not (math.isfinite(num_slices) and num_slices > 0):
                num_slices = 1.0
            weights.append(num_slices)

        summary: Dict[str, object] = {
            "method": method,
            "num_axes": len(rows),
            "num_slices_total": int(round(sum(weights))),
        }

        for metric in tracked_metrics:
            values = [to_float(row.get(metric)) for row in rows]
            weighted_value = _weighted_mean(values, weights)
            summary[metric] = round(weighted_value, 6) if math.isfinite(weighted_value) else ""

        summary_rows.append(summary)

    return build_composite_ranking(summary_rows)


def build_processing_time_table(by_slice_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Aggregate processing times per axis and method."""
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in by_slice_rows:
        axis = str(row.get("slice_axis", "unknown"))
        method = str(row.get("method", "unknown"))
        grouped[(axis, method)].append(row)

    output_rows: List[Dict[str, object]] = []
    for (axis, method), rows in sorted(grouped.items()):
        degradation = [to_float(r.get("degradation_time_ms")) for r in rows]
        upscaling = [to_float(r.get("upscaling_time_ms")) for r in rows]
        metrics = [to_float(r.get("metrics_time_ms")) for r in rows]
        total = [to_float(r.get("total_method_time_ms")) for r in rows]

        finite_total = [v for v in total if math.isfinite(v)]

        output_rows.append(
            {
                "slice_axis": axis,
                "method": method,
                "num_rows": len(rows),
                "degradation_time_ms_mean": round(_mean([v for v in degradation if math.isfinite(v)]), 4),
                "degradation_time_ms_std": round(_std([v for v in degradation if math.isfinite(v)]), 4),
                "upscaling_time_ms_mean": round(_mean([v for v in upscaling if math.isfinite(v)]), 4),
                "upscaling_time_ms_std": round(_std([v for v in upscaling if math.isfinite(v)]), 4),
                "metrics_time_ms_mean": round(_mean([v for v in metrics if math.isfinite(v)]), 4),
                "metrics_time_ms_std": round(_std([v for v in metrics if math.isfinite(v)]), 4),
                "total_method_time_ms_mean": round(_mean(finite_total), 4),
                "total_method_time_ms_std": round(_std(finite_total), 4),
            }
        )

    # Fastest methods appear first within each axis.
    output_rows.sort(
        key=lambda row: (
            row["slice_axis"],
            to_float(row.get("total_method_time_ms_mean")),
            row["method"],
        )
    )
    return output_rows


def build_image_size_table(by_slice_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Build one row per axis with average HR/LR image dimensions."""
    grouped_by_axis_slice: Dict[str, Dict[int, Dict[str, object]]] = defaultdict(dict)
    for row in by_slice_rows:
        axis = str(row.get("slice_axis", "unknown"))
        slice_index = int(to_float(row.get("slice_index"), default=-1))
        if slice_index < 0:
            continue
        if slice_index not in grouped_by_axis_slice[axis]:
            grouped_by_axis_slice[axis][slice_index] = row

    output_rows: List[Dict[str, object]] = []
    for axis in sorted(grouped_by_axis_slice.keys()):
        rows = list(grouped_by_axis_slice[axis].values())
        if not rows:
            continue

        hr_heights = [int(to_float(row.get("hr_height"), default=0)) for row in rows]
        hr_widths = [int(to_float(row.get("hr_width"), default=0)) for row in rows]
        lr_heights = [int(to_float(row.get("lr_height"), default=0)) for row in rows]
        lr_widths = [int(to_float(row.get("lr_width"), default=0)) for row in rows]
        scales = [to_float(row.get("scale"), default=float("nan")) for row in rows]

        valid_hr_heights = [v for v in hr_heights if v > 0]
        valid_hr_widths = [v for v in hr_widths if v > 0]
        valid_lr_heights = [v for v in lr_heights if v > 0]
        valid_lr_widths = [v for v in lr_widths if v > 0]
        valid_scales = [v for v in scales if math.isfinite(v)]

        has_size_data = bool(valid_hr_heights and valid_hr_widths and valid_lr_heights and valid_lr_widths)
        if has_size_data:
            hr_height: object = int(round(_mean(valid_hr_heights)))
            hr_width: object = int(round(_mean(valid_hr_widths)))
            lr_height: object = int(round(_mean(valid_lr_heights)))
            lr_width: object = int(round(_mean(valid_lr_widths)))
            hr_pixels: object = int(hr_height * hr_width)
            lr_pixels: object = int(lr_height * lr_width)
            pixel_ratio: object = round(float(hr_pixels / lr_pixels), 4) if lr_pixels > 0 else ""
        else:
            hr_height = ""
            hr_width = ""
            lr_height = ""
            lr_width = ""
            hr_pixels = ""
            lr_pixels = ""
            pixel_ratio = ""

        output_rows.append(
            {
                "slice_axis": axis,
                "num_slices": len(rows),
                "scale_mean": round(_mean(valid_scales), 4) if valid_scales else "",
                "hr_height": hr_height,
                "hr_width": hr_width,
                "hr_pixels": hr_pixels,
                "lr_height": lr_height,
                "lr_width": lr_width,
                "lr_pixels": lr_pixels,
                "pixel_ratio_hr_over_lr": pixel_ratio,
            }
        )

    return output_rows


def build_psnr_reference_template(aggregate_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Create an editable CSV template to compare PSNR with literature."""
    template_rows: List[Dict[str, object]] = []
    for row in rank_all_results(aggregate_rows):
        ours_psnr = to_float(row.get("psnr_mean"))
        template_rows.append(
            {
                "slice_axis": row.get("slice_axis", ""),
                "method": row.get("method", ""),
                "our_psnr_mean_db": round(ours_psnr, 4) if math.isfinite(ours_psnr) else "",
                "paper_id": "",
                "paper_dataset": "",
                "paper_psnr_db": "",
                "delta_db_ours_minus_paper": "",
                "notes": "",
            }
        )
    return template_rows


def write_thesis_markdown_bundle(
    output_path: Path,
    ranking_rows: List[Dict[str, object]],
    image_rows: List[Dict[str, object]],
    timing_rows: List[Dict[str, object]],
    method_ranking_rows: Optional[List[Dict[str, object]]] = None,
) -> None:
    """Write a markdown file containing all major thesis tables."""
    def fmt(value: object, digits: int = 4) -> str:
        numeric = to_float(value)
        if not math.isfinite(numeric):
            return "-"
        return f"{numeric:.{digits}f}"

    lines: List[str] = []
    # Keep section titles in Portuguese to match report language.
    lines.append("# Tabelas para relatorio")
    lines.append("")
    lines.append("## Ranking global composto")
    lines.append("")
    lines.append("| Rank | Eixo | Metodo | Score | SSIM | PSNR (dB) | MSE | HFEN | ISNR (dB) |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for row in ranking_rows:
        lines.append(
            f"| {row.get('overall_rank', '')} | `{row.get('slice_axis', '')}` | `{row.get('method', '')}` | "
            f"{fmt(row.get('overall_score'))} | "
            f"{fmt(row.get('ssim_mean'))} | "
            f"{fmt(row.get('psnr_mean'), 2)} | "
            f"{fmt(row.get('mse_mean'), 6)} | "
            f"{fmt(row.get('hfen_mean'))} | "
            f"{fmt(row.get('isnr_mean'), 3)} |"
        )

    if method_ranking_rows:
        lines.append("")
        lines.append("## Ranking global por metodo")
        lines.append("")
        lines.append("| Rank | Metodo | Score | Eixos | Fatias | SSIM | PSNR (dB) | MSE | HFEN | ISNR (dB) |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in method_ranking_rows:
            lines.append(
                f"| {row.get('overall_rank', '')} | `{row.get('method', '')}` | "
                f"{fmt(row.get('overall_score'))} | "
                f"{row.get('num_axes', '')} | "
                f"{row.get('num_slices_total', '')} | "
                f"{fmt(row.get('ssim_mean'))} | "
                f"{fmt(row.get('psnr_mean'), 2)} | "
                f"{fmt(row.get('mse_mean'), 6)} | "
                f"{fmt(row.get('hfen_mean'))} | "
                f"{fmt(row.get('isnr_mean'), 3)} |"
            )

    lines.append("")
    lines.append("## Tamanho medio das imagens")
    lines.append("")
    lines.append("| Eixo | Fatias | Escala | HR (HxW) | LR (HxW) | HR pixels | LR pixels | Razao HR/LR |")
    lines.append("|---|---:|---:|---|---|---:|---:|---:|")
    for row in image_rows:
        hr_h = row.get("hr_height", "")
        hr_w = row.get("hr_width", "")
        lr_h = row.get("lr_height", "")
        lr_w = row.get("lr_width", "")

        hr_shape = f"{hr_h}x{hr_w}" if hr_h not in ("", None) and hr_w not in ("", None) else "-"
        lr_shape = f"{lr_h}x{lr_w}" if lr_h not in ("", None) and lr_w not in ("", None) else "-"
        hr_pixels = row.get("hr_pixels", "")
        lr_pixels = row.get("lr_pixels", "")

        lines.append(
            f"| `{row.get('slice_axis', '')}` | {row.get('num_slices', '')} | {fmt(row.get('scale_mean'), 2)} | "
            f"{hr_shape} | "
            f"{lr_shape} | "
            f"{hr_pixels if hr_pixels not in ('', None) else '-'} | "
            f"{lr_pixels if lr_pixels not in ('', None) else '-'} | "
            f"{fmt(row.get('pixel_ratio_hr_over_lr'), 2)} |"
        )

    lines.append("")
    lines.append("## Tempos de processamento medios")
    lines.append("")
    lines.append("| Eixo | Metodo | Degradacao (ms) | Upscaling (ms) | Metricas (ms) | Total por metodo (ms) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in timing_rows:
        lines.append(
            f"| `{row.get('slice_axis', '')}` | `{row.get('method', '')}` | "
            f"{fmt(row.get('degradation_time_ms_mean'), 2)} | "
            f"{fmt(row.get('upscaling_time_ms_mean'), 2)} | "
            f"{fmt(row.get('metrics_time_ms_mean'), 2)} | "
            f"{fmt(row.get('total_method_time_ms_mean'), 2)} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
