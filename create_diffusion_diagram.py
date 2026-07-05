"""Generate a diffusion model diagram inside the diffusion results folder."""

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for diffusion diagram generation."""
    parser = argparse.ArgumentParser(description="Create diffusion model pipeline diagram")
    parser.add_argument(
        "--output",
        default="results/diffusion/diagrams/diffusion_model_pipeline.png",
        help="Output PNG path for the diffusion diagram.",
    )
    return parser.parse_args()


def _draw_box(ax, x: float, y: float, w: float, h: float, text: str, fc: str = "#EAF4FF") -> None:
    """Draw one rounded box with centered text."""
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#1A73E8",
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=9.5, wrap=True)


def _arrow(ax, start: Tuple[float, float], end: Tuple[float, float]) -> None:
    """Draw a simple directional arrow between two points."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.2,
            color="#4A5568",
        )
    )


def main() -> None:
    """Create a two-lane diagram for diffusion training and inference."""
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 7), dpi=180)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.96, "Conditional Diffusion U-Net for MRI Super-Resolution", ha="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.92, "Training lane (top) and inference lane (bottom)", ha="center", fontsize=10, color="#374151")

    train_steps: List[str] = [
        "HR slice\nx0",
        "Sample t and noise",
        "Forward noise\nq(xt | x0)",
        "Condition LR->HR\ninterpolation",
        "U-Net predicts noise\ne_theta(xt, cond, t)",
        "MSE loss\nvs true noise",
        "Update weights",
    ]

    infer_steps: List[str] = [
        "Condition LR->HR\ninterpolation",
        "Start from random noise\nxT",
        "Reverse step\np(x(t-1) | xt, cond)",
        "Repeat for t=T..1",
        "Final SR image\nx0 hat",
    ]

    train_y = 0.62
    infer_y = 0.24
    box_h = 0.14

    train_w = 0.12
    train_gap = 0.02
    train_x0 = 0.03

    infer_w = 0.15
    infer_gap = 0.03
    infer_x0 = 0.07

    train_centers: List[Tuple[float, float]] = []
    for idx, text in enumerate(train_steps):
        x = train_x0 + idx * (train_w + train_gap)
        _draw_box(ax, x, train_y, train_w, box_h, text, fc="#E8F0FE")
        train_centers.append((x + train_w / 2.0, train_y + box_h / 2.0))

    infer_centers: List[Tuple[float, float]] = []
    for idx, text in enumerate(infer_steps):
        x = infer_x0 + idx * (infer_w + infer_gap)
        _draw_box(ax, x, infer_y, infer_w, box_h, text, fc="#EAFBF1")
        infer_centers.append((x + infer_w / 2.0, infer_y + box_h / 2.0))

    for idx in range(len(train_centers) - 1):
        _arrow(ax, (train_centers[idx][0] + train_w / 2.0 - 0.01, train_centers[idx][1]), (train_centers[idx + 1][0] - train_w / 2.0 + 0.01, train_centers[idx + 1][1]))

    for idx in range(len(infer_centers) - 1):
        _arrow(ax, (infer_centers[idx][0] + infer_w / 2.0 - 0.01, infer_centers[idx][1]), (infer_centers[idx + 1][0] - infer_w / 2.0 + 0.01, infer_centers[idx + 1][1]))

    # Link training and inference lanes through the trained U-Net checkpoint.
    ax.text(0.5, 0.50, "Saved checkpoint (trained U-Net)", ha="center", fontsize=10, color="#1F2937")
    _arrow(ax, (0.50, train_y), (0.50, 0.52))
    _arrow(ax, (0.50, 0.48), (0.50, infer_y + box_h))

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print("Done.")
    print(f"Diffusion diagram: {output_path}")


if __name__ == "__main__":
    main()
