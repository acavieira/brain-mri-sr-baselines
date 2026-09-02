"""Inference utilities for conditional diffusion super-resolution."""

from .imports import *

def _resolve_device(device_arg: str) -> torch.device:
    """Resolve user device preference to an available torch device."""
    if device_arg and device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DiffusionSuperResolver:
    """Load a trained model and run reverse diffusion sampling."""

    def __init__(
        self,
        model: ConditionalUNet2D,
        scheduler: DiffusionScheduler,
        device: torch.device,
        image_size: int,
        beta_start: float,
        beta_end: float,
    ) -> None:
        self.model = model
        self.scheduler = scheduler
        self.device = device
        self.image_size = int(image_size)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "auto") -> "DiffusionSuperResolver":
        """Instantiate the resolver from a training checkpoint file."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})
        model_kwargs = checkpoint.get("model_kwargs", {})

        # Restore architecture and weights exactly as saved during training.
        model = ConditionalUNet2D(**model_kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])

        resolved_device = _resolve_device(device)
        model = model.to(resolved_device)
        model.eval()

        num_steps = int(config.get("num_diffusion_steps", 1000))
        beta_start = float(config.get("beta_start", 1e-4))
        beta_end = float(config.get("beta_end", 2e-2))
        scheduler = DiffusionScheduler(
            num_steps=num_steps,
            beta_start=beta_start,
            beta_end=beta_end,
        ).to(resolved_device)

        image_size = int(config.get("image_size", 256))
        return cls(
            model=model,
            scheduler=scheduler,
            device=resolved_device,
            image_size=image_size,
            beta_start=beta_start,
            beta_end=beta_end,
        )

    @torch.no_grad()
    def super_resolve(
        self,
        lr_img: np.ndarray,
        output_shape: Optional[Tuple[int, int]] = None,
        num_steps: Optional[int] = None,
    ) -> np.ndarray:
        """Generate SR output from LR input using reverse diffusion."""
        if lr_img.ndim != 2:
            raise ValueError(f"Expected 2D grayscale image. Got shape: {lr_img.shape}")

        if output_shape is None:
            output_shape = (self.image_size, self.image_size)

        target_h, target_w = output_shape
        cond_img = cv2.resize(
            np.clip(lr_img.astype(np.float32), 0.0, 1.0),
            (target_w, target_h),
            interpolation=cv2.INTER_CUBIC,
        )

        cond_tensor = torch.from_numpy(cond_img).unsqueeze(0).unsqueeze(0).float().to(self.device)

        scheduler = self.scheduler
        # Allow a custom number of sampling steps at inference time.
        if num_steps is not None and int(num_steps) != scheduler.num_steps:
            scheduler = DiffusionScheduler(
                num_steps=int(num_steps),
                beta_start=self.beta_start,
                beta_end=self.beta_end,
            ).to(self.device)

        sr_tensor = scheduler.sample(self.model, cond_tensor, clip_output=True)
        sr_img = sr_tensor.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
        return np.clip(sr_img, 0.0, 1.0)
