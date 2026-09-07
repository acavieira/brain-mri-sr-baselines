"""NIfTI loading and anatomical slice selection."""

from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np


ORIENTATION_AXES: Dict[str, int] = {"sagittal": 0, "coronal": 1, "axial": 2}


def find_nifti_files(input_dir: str) -> List[Path]:
    """Return sorted NIfTI files from a directory."""
    directory = Path(input_dir)
    paths = sorted(list(directory.glob("*.nii")) + list(directory.glob("*.nii.gz")))
    if not paths:
        raise FileNotFoundError(f"No .nii or .nii.gz files found in {directory}")
    return paths


def load_canonical_volume(path: Path) -> np.ndarray:
    """Load a 3D volume reoriented to nibabel's canonical RAS+ orientation."""
    if path.suffix not in {".nii", ".gz"} and not path.name.endswith(".nii.gz"):
        raise ValueError(f"Input must be a NIfTI file: {path}")
    image = nib.as_closest_canonical(nib.load(str(path)))
    volume = image.get_fdata(dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got shape {volume.shape}")
    return volume


def select_slice_indices(axis_length: int, count: int) -> List[int]:
    """Select exactly `count` centered, consecutive indices."""
    if count < 1:
        raise ValueError("The slice count must be positive")
    if axis_length < count:
        raise ValueError(f"Axis length {axis_length} is smaller than {count} slices")
    start = max(0, (axis_length - count) // 2)
    return list(range(start, start + count))


def extract_anatomical_slice(volume: np.ndarray, orientation: str, index: int) -> np.ndarray:
    """Extract one slice after canonical orientation using an explicit axis map."""
    if orientation not in ORIENTATION_AXES:
        raise ValueError(f"Unknown orientation: {orientation}")
    axis = ORIENTATION_AXES[orientation]
    if not 0 <= index < volume.shape[axis]:
        raise IndexError(f"Slice {index} is outside axis {orientation} with length {volume.shape[axis]}")

    if orientation == "sagittal":
        slice_image = volume[index, :, :]
    elif orientation == "coronal":
        slice_image = volume[:, index, :]
    else:
        slice_image = volume[:, :, index]

    # Keep saved anatomical views upright and consistent across orientations.
    oriented_slice = np.flipud(np.transpose(slice_image))
    return np.asarray(oriented_slice, dtype=np.float32)