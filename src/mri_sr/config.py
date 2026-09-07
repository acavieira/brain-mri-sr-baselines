"""Single source of truth for the classical baseline experiment."""

from dataclasses import asdict, dataclass
from typing import Any, Dict


ORIENTATIONS = ("axial", "coronal", "sagittal")
METHODS = ("nearest", "bilinear", "bicubic", "lanczos")
ISNR_BASELINE_METHOD = "bilinear"
TARGET_SIZE = 256
SLICES_PER_VOLUME = 30
SCALE = 2
BLUR_SIGMA = 0.65
NOISE_SIGMA = 0.02
RANDOM_SEED = 23
LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0
BRAIN_MASK_THRESHOLD = 0.05


@dataclass(frozen=True)
class ExperimentConfig:
    """Experimental parameters saved alongside every run."""

    input_dir: str = "data"
    output_root: str = "results"
    slices_per_volume: int = SLICES_PER_VOLUME
    target_size: int = TARGET_SIZE
    scale: int = SCALE
    blur_sigma: float = BLUR_SIGMA
    noise_sigma: float = NOISE_SIGMA
    random_seed: int = RANDOM_SEED
    low_percentile: float = LOW_PERCENTILE
    high_percentile: float = HIGH_PERCENTILE
    brain_mask_threshold: float = BRAIN_MASK_THRESHOLD
    isnr_baseline_method: str = ISNR_BASELINE_METHOD
    save_all_examples: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Return serializable parameters for run provenance."""
        return asdict(self)