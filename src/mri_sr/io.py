"""I/O helpers for loading NIfTI data and extracting 2D slices."""

import os
from typing import List

import nibabel as nib
import numpy as np


def ensure_dir(path: str) -> None:
    """Create a directory when it does not exist."""
    os.makedirs(path, exist_ok=True)


def load_nifti_volume(path: str) -> np.ndarray:
    """Load one 3D NIfTI volume as float32."""
    if not (path.lower().endswith(".nii") or path.lower().endswith(".nii.gz")):
        raise ValueError("Input must be a NIfTI file: .nii or .nii.gz")

    nii = nib.load(path)
    data = nii.get_fdata(dtype=np.float32)

    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI volume, got shape {data.shape}")

    return data


def get_axis_length(data: np.ndarray, axis: str) -> int:
    """Return axis size using anatomical axis names."""
    if axis == "sagittal":
        return data.shape[0]
    if axis == "coronal":
        return data.shape[1]
    if axis == "axial":
        return data.shape[2]
    raise ValueError(f"Unknown axis: {axis}")


def choose_slice_indices(data: np.ndarray, axis: str, center_index: int, num_slices: int) -> List[int]:
    """Return valid indices around a center location for one axis."""
    axis_len = get_axis_length(data, axis)

    if center_index < 0:
        center_index = axis_len // 2

    center_index = int(np.clip(center_index, 0, axis_len - 1))
    num_slices = max(1, int(num_slices))

    half = num_slices // 2
    start = center_index - half
    end = center_index + half + 1

    indices = list(range(start, end))
    return [idx for idx in indices if 0 <= idx < axis_len]


def extract_slice(data: np.ndarray, axis: str, slice_index: int) -> np.ndarray:
    """Extract one 2D anatomical slice and apply display orientation fix."""
    if axis == "sagittal":
        img = data[slice_index, :, :]
    elif axis == "coronal":
        img = data[:, slice_index, :]
    elif axis == "axial":
        img = data[:, :, slice_index]
    else:
        raise ValueError(f"Unknown axis: {axis}")

    # Keep a consistent orientation in saved figures.
    img = np.transpose(img)
    img = np.flipud(img)
    return img.astype(np.float32)