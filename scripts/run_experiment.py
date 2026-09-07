"""Command-line entry point for the classical MRI SR experiment."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mri_sr.config import ExperimentConfig
from src.mri_sr.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    """Parse the few options that are useful at execution time."""
    parser = argparse.ArgumentParser(description="Run classical MRI super-resolution baselines")
    parser.add_argument("--input-dir", default="data", help="Directory containing .nii or .nii.gz volumes")
    parser.add_argument("--output-root", default="results", help="Root directory for timestamped runs")
    parser.add_argument("--save-all-examples", action="store_true", help="Save an example for every slice")
    return parser.parse_args()


def main() -> None:
    """Build the configuration and execute the complete experiment."""
    args = parse_args()
    config = ExperimentConfig(
        input_dir=args.input_dir,
        output_root=args.output_root,
        save_all_examples=args.save_all_examples,
    )
    output_dir, rows = run_experiment(config)
    print(f"\nSaved {len(rows)} metric rows to {output_dir}")


if __name__ == "__main__":
    main()
