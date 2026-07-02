from src.mri_sr.config import ExperimentConfig
from src.mri_sr.experiment import run_experiment


if __name__ == "__main__":
    config = ExperimentConfig(
        input_path="data/sub-0_ses-1_T1w.nii",
        output_dir="results/scale4_sagittal",
        slice_axis="sagittal",
        slice_index=-1,
        num_slices=9,
        scale=4,
    #blur_sigma: float = 1.0
    #noise_sigma: float = 0.0        
        blur_sigma=0.55,
        noise_sigma=0.01,
        force_square_size=None,
        save_all_figures=True,
    )

    run_experiment(config)