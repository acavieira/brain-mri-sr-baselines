"""
Generate baseline pipeline diagrams for method comparison.

Scope:
    This diagram is intentionally baseline-only (classical interpolation methods).
    Diffusion/U-Net belongs to a later phase and is not shown here.
"""

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def parse_args() -> argparse.Namespace:
    """Parse output paths for the English and Portuguese diagrams."""
    parser = argparse.ArgumentParser(description="Create baseline MRI SR pipeline diagrams (EN/PT)")
    parser.add_argument("--output-en", default="results/analysis/final_comparison/diagrams/pipeline_baseline_en.png")
    parser.add_argument("--output-pt", default="results/analysis/final_comparison/diagrams/pipeline_baseline_pt.png")
    return parser.parse_args()


def _draw_pipeline(steps: Iterable[str], title: str, output_path: Path) -> None:
    """Draw a horizontal flow diagram where each item in `steps` is one stage."""
    step_list: List[str] = list(steps)
    if not step_list:
        raise ValueError("At least one pipeline step is required.")

    fig_width = max(14.0, 2.8 * len(step_list))
    fig, ax = plt.subplots(figsize=(fig_width, 3.6), dpi=180)
    ax.axis("off")

    # Compute a simple responsive layout so labels stay readable.
    left_margin = 0.02
    right_margin = 0.02
    usable_width = 1.0 - left_margin - right_margin
    box_width = min(0.125, usable_width / len(step_list) * 0.78)
    gap = (usable_width - box_width * len(step_list)) / max(len(step_list) - 1, 1)

    y = 0.45
    box_height = 0.22
    face_color = "#E8F0FE"
    edge_color = "#1A73E8"

    centers: List[float] = []
    for index, text in enumerate(step_list):
        x = left_margin + index * (box_width + gap)
        centers.append(x + box_width / 2.0)

        box = FancyBboxPatch(
            (x, y),
            box_width,
            box_height,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.4,
            edgecolor=edge_color,
            facecolor=face_color,
        )
        ax.add_patch(box)
        ax.text(
            x + box_width / 2.0,
            y + box_height / 2.0,
            text,
            ha="center",
            va="center",
            fontsize=9.5,
            color="#0B1F33",
            wrap=True,
        )

    # Link consecutive boxes with right-pointing arrows.
    for idx in range(len(centers) - 1):
        start = (centers[idx] + box_width / 2.0 - 0.006, y + box_height / 2.0)
        end = (centers[idx + 1] - box_width / 2.0 + 0.006, y + box_height / 2.0)
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.2,
            color="#4A5568",
        )
        ax.add_patch(arrow)

    ax.text(0.5, 0.88, title, ha="center", va="center", fontsize=13, fontweight="bold")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate both language variants of the baseline pipeline diagram."""
    args = parse_args()

    english_steps = [
        "3D NIfTI volume",
        "2D slice extraction",
        "Robust normalization",
        "Synthetic blur\n(Gaussian)",
        "Downsampling\n(HR -> LR)",
        "Optional LR noise",
        "Upscaling / SR\n(classical methods)",
        "Metrics, timing\nand final tables",
    ]

    portuguese_steps = [
        "Volume NIfTI 3D",
        "Extracao de fatias 2D",
        "Normalizacao robusta",
        "Synthetic blur\n(Gaussiano)",
        "Downsampling\n(HR -> LR)",
        "Ruido opcional na LR",
        "Upscaling / SR\n(metodos classicos)",
        "Metricas, tempos\ne tabelas finais",
    ]

    _draw_pipeline(
        steps=english_steps,
        title="MRI Super-Resolution Baseline Pipeline (method comparison)",
        output_path=Path(args.output_en),
    )
    _draw_pipeline(
        steps=portuguese_steps,
        title="Pipeline baseline de Super-Resolucao MRI (comparacao de metodos)",
        output_path=Path(args.output_pt),
    )

    print("Done.")
    print(f"EN diagram: {args.output_en}")
    print(f"PT diagram: {args.output_pt}")


if __name__ == "__main__":
    main()
