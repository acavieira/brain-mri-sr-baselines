"""Configuration object for diffusion model training."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DiffusionConfig:
    """All settings required to train and checkpoint diffusion U-Net."""

    train_glob: str
    output_dir: str

    image_size: int = 256
    scale: int = 2
    blur_sigma: float = 0.65
    noise_sigma: float = 0.02
    slices_per_volume: int = 30
    axes: Tuple[str, ...] = ("sagittal", "coronal", "axial")

    batch_size: int = 4
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    num_workers: int = 0

    num_diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    save_every: int = 5
    random_seed: int = 23
    device: str = "auto"
