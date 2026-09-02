"""Dataset utilities for diffusion training on MRI slices."""

from .imports import *

def collect_nifti_paths(pattern: str) -> List[str]:
    """Collect NIfTI file paths matching a glob pattern."""
    paths = sorted(glob.glob(pattern))
    return [path for path in paths if path.endswith(".nii") or path.endswith(".nii.gz")]


class NiftiSlicePairDataset(Dataset):
    """Yield HR and conditioning tensors for diffusion noise prediction."""

    def __init__(
        self,
        nifti_paths: List[str],
        image_size: int,
        scale: int,
        blur_sigma: float,
        noise_sigma: float,
        slices_per_volume: int,
        axes: Tuple[str, ...],
        random_seed: int = 23,
    ) -> None:
        if not nifti_paths:
            raise ValueError("No NIfTI paths provided for diffusion dataset.")

        self.nifti_paths = sorted(nifti_paths)
        self.image_size = int(image_size)
        self.scale = int(scale)
        self.blur_sigma = float(blur_sigma)
        self.noise_sigma = float(noise_sigma)
        self.slices_per_volume = int(slices_per_volume)
        self.axes = tuple(axes)
        self.random_seed = int(random_seed)

        # Volumes are cached to avoid repeated disk reads per slice.
        self._volume_cache: Dict[str, np.ndarray] = {}
        self._items: List[Tuple[str, str, int]] = self._build_index()

    def _build_index(self) -> List[Tuple[str, str, int]]:
        """Build a flat index of (volume path, axis, slice index)."""
        items: List[Tuple[str, str, int]] = []

        for path in self.nifti_paths:
            volume = load_nifti_volume(path)
            for axis in self.axes:
                slice_indices = choose_slice_indices(
                    volume,
                    axis=axis,
                    center_index=-1,
                    num_slices=self.slices_per_volume,
                )
                for slice_index in slice_indices:
                    items.append((path, axis, slice_index))

        if not items:
            raise ValueError("No slices were indexed for diffusion training.")
        return items

    def _load_volume(self, path: str) -> np.ndarray:
        """Load and cache one volume."""
        if path not in self._volume_cache:
            self._volume_cache[path] = load_nifti_volume(path)
        return self._volume_cache[path]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Dict[str, object]:
        """Return one training sample with HR target and LR conditioning image."""
        path, axis, slice_index = self._items[index]
        volume = self._load_volume(path)

        raw_slice = extract_slice(volume, axis, slice_index)
        hr_img = prepare_hr_reference(raw_slice, force_square_size=self.image_size)

        # Use deterministic per-index noise so training is reproducible.
        rng = np.random.default_rng(self.random_seed + index)
        lr_img = degrade_image(
            hr_img,
            scale=self.scale,
            blur_sigma=self.blur_sigma,
            noise_sigma=self.noise_sigma,
            rng=rng,
        )

        # Condition is LR resized to HR size so model input shapes always match.
        target_h, target_w = hr_img.shape
        cond_img = cv2.resize(lr_img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        hr_tensor = torch.from_numpy(hr_img).unsqueeze(0).float()
        cond_tensor = torch.from_numpy(cond_img).unsqueeze(0).float()

        return {
            "hr": hr_tensor,
            "cond": cond_tensor,
            "path": str(Path(path)),
            "slice_axis": axis,
            "slice_index": int(slice_index),
        }
