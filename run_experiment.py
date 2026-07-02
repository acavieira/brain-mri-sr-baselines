import argparse

from src.mri_sr.config import ExperimentConfig
from src.mri_sr.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single MRI SR experiment")
    parser.add_argument("--input-path", default="data/sub-0_ses-1_T1w.nii")
    parser.add_argument("--output-dir", default="results/scale2_sagittal")
    parser.add_argument("--slice-axis", default="sagittal", choices=["sagittal", "coronal", "axial"])
    parser.add_argument("--slice-index", type=int, default=-1)
    parser.add_argument("--num-slices", type=int, default=30)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--blur-sigma", type=float, default=0.65)
    parser.add_argument("--noise-sigma", type=float, default=0.02)
    parser.add_argument("--force-square-size", type=int, default=None)
    parser.add_argument("--save-all-figures", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = ExperimentConfig(
        input_path=args.input_path,
        output_dir=args.output_dir,
        slice_axis=args.slice_axis,
        slice_index=args.slice_index,
        num_slices=args.num_slices,
        scale=args.scale,
        blur_sigma=args.blur_sigma,
        noise_sigma=args.noise_sigma,
        force_square_size=args.force_square_size,
        save_all_figures=args.save_all_figures,
    )

    run_experiment(config)