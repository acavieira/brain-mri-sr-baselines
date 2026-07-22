from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    input_path: str
    output_dir: str

    slice_axis: str = "sagittal"
    slice_index: int = -1
    num_slices: int = 9

    scale: int = 4
    blur_sigma: float = 0.55
    noise_sigma: float = 0.01

    force_square_size: Optional[int] = None
    save_all_figures: bool = False
    compute_issm: bool = False
    error_vmax: float = 0.35
    random_seed: int = 23