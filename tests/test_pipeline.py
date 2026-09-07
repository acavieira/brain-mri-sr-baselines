"""Focused tests for the classical baseline pipeline."""

import csv

import numpy as np
from skimage.metrics import peak_signal_noise_ratio

from src.mri_sr.config import METHODS, ORIENTATIONS
from src.mri_sr.degradation import degrade_image
from src.mri_sr.interpolation import upscale_all_methods
from src.mri_sr.metrics import compute_metrics, isnr, psnr
from src.mri_sr.nifti_io import extract_anatomical_slice, select_slice_indices
from src.mri_sr.preprocessing import normalize_volume_to_float01, prepare_hr_reference, resize_with_padding
from src.mri_sr.reporting import generate_reports


def test_selects_exactly_30_slices():
    indices = select_slice_indices(97, 30)
    assert len(indices) == 30
    assert len(set(indices)) == 30


def test_canonical_axis_mapping_is_explicit():
    volume = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    assert extract_anatomical_slice(volume, "sagittal", 1).shape == (6, 5)
    assert extract_anatomical_slice(volume, "coronal", 1).shape == (6, 4)
    assert extract_anatomical_slice(volume, "axial", 1).shape == (5, 4)


def test_normalization_ignores_nan_and_stays_in_unit_interval():
    volume = np.array([[[np.nan, 1.0], [2.0, 3.0]], [[4.0, 5.0], [np.inf, -np.inf]]], dtype=np.float32)
    normalized = normalize_volume_to_float01(volume)
    assert np.isfinite(normalized).all()
    assert 0.0 <= normalized.min() <= normalized.max() <= 1.0


def test_constant_volume_normalizes_to_zero():
    normalized = normalize_volume_to_float01(np.full((3, 3, 3), 7.0, dtype=np.float32))
    assert np.all(normalized == 0.0)


def test_resize_preserves_aspect_ratio_with_padding():
    image = np.ones((20, 40), dtype=np.float32)
    resized = resize_with_padding(image, 256)
    assert resized.shape == (256, 256)
    assert np.count_nonzero(resized[0]) == 0
    assert np.count_nonzero(resized[64]) == 256
    assert np.count_nonzero(resized[:, 0]) == 128


def test_same_lr_drives_all_methods_and_shapes():
    hr = np.linspace(0.0, 1.0, 256 * 256, dtype=np.float32).reshape(256, 256)
    lr = degrade_image(hr, scale=2, blur_sigma=0.5, noise_sigma=0.0, rng=np.random.default_rng(23))
    reconstructions = upscale_all_methods(lr, hr.shape)
    assert lr.shape == (128, 128)
    assert list(reconstructions) == list(METHODS)
    assert all(image.shape == hr.shape for image in reconstructions.values())


def test_psnr_matches_skimage_and_perfect_image_is_infinite():
    reference = np.zeros((16, 16), dtype=np.float32)
    prediction = np.full((16, 16), 0.1, dtype=np.float32)
    assert psnr(reference, prediction) == peak_signal_noise_ratio(reference, prediction, data_range=1.0)
    assert np.isinf(psnr(reference, reference))


def test_isnr_uses_fixed_baseline_error():
    reference = np.zeros((8, 8), dtype=np.float32)
    baseline = np.full((8, 8), 0.2, dtype=np.float32)
    reconstruction = np.full((8, 8), 0.1, dtype=np.float32)
    assert isnr(reference, baseline, baseline) == 0.0
    assert np.isclose(isnr(reference, reconstruction, baseline), 10.0 * np.log10(4.0))
    assert np.isinf(isnr(reference, reference, baseline))


def test_report_tables_are_created(tmp_path):
    rows = []
    for orientation in ORIENTATIONS:
        for method in METHODS:
            rows.append(
                {
                    "volume": "synthetic.nii",
                    "orientation": orientation,
                    "slice_index": 10,
                    "method": method,
                    "psnr": 20.0 + METHODS.index(method),
                    "mse": 0.01,
                    "mae": 0.05,
                    "rmse": 0.1,
                    "ssim": 0.8,
                        "isnr": 0.0,
                    "processing_time_ms": 1.0,
                    "hr_height": 256,
                    "hr_width": 256,
                    "lr_height": 128,
                    "lr_width": 128,
                }
            )
    generate_reports(rows, tmp_path, {"random_seed": 23})
    expected = [
        "metrics_by_slice.csv",
        "metrics_by_axis.csv",
        "metrics_summary.csv",
        "runtime_summary.csv",
        "ranking.csv",
    ]
    for filename in expected:
        assert (tmp_path / "tables" / filename).exists()
    with (tmp_path / "tables" / "metrics_by_slice.csv").open(newline="", encoding="utf-8") as handle:
        assert "volume" in next(csv.reader(handle))
    assert (tmp_path / "execution_summary.txt").exists()
