"""DDPM scheduler with forward noise and reverse sampling steps."""

from dataclasses import dataclass

import torch


def _extract(values: torch.Tensor, timesteps: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Gather timestep-specific coefficients and reshape for broadcasting."""
    out = values.gather(0, timesteps)
    return out.view(timesteps.shape[0], *([1] * (len(target_shape) - 1)))


@dataclass
class DiffusionScheduler:
    """Linear-beta diffusion scheduler used in training and inference."""

    num_steps: int
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    def __post_init__(self) -> None:
        """Precompute all scalar schedules used by q and p transitions."""
        if self.num_steps <= 0:
            raise ValueError("num_steps must be > 0")

        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_steps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype=torch.float32), self.alphas_cumprod[:-1]],
            dim=0,
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod + 1e-12)
        )

    def to(self, device: torch.device) -> "DiffusionScheduler":
        """Move all cached tensors to the selected torch device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps uniformly for one training batch."""
        return torch.randint(0, self.num_steps, (batch_size,), device=device, dtype=torch.long)

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: add noise to x_start at timestep t."""
        sqrt_alphas_cumprod_t = _extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = _extract(
            self.sqrt_one_minus_alphas_cumprod,
            timesteps,
            x_start.shape,
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(
        self,
        model,
        x: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Reverse diffusion step: sample x_{t-1} from x_t."""
        betas_t = _extract(self.betas, timesteps, x.shape)
        sqrt_one_minus_alphas_cumprod_t = _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x.shape)
        sqrt_recip_alphas_t = _extract(self.sqrt_recip_alphas, timesteps, x.shape)

        model_output = model(x, condition, timesteps)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * model_output / (sqrt_one_minus_alphas_cumprod_t + 1e-12))

        posterior_variance_t = _extract(self.posterior_variance, timesteps, x.shape)
        noise = torch.randn_like(x)

        nonzero_mask = (timesteps != 0).float().view(-1, *([1] * (x.ndim - 1)))
        return model_mean + nonzero_mask * torch.sqrt(torch.clamp(posterior_variance_t, min=1e-20)) * noise

    @torch.no_grad()
    def sample(
        self,
        model,
        condition: torch.Tensor,
        clip_output: bool = True,
    ) -> torch.Tensor:
        """Run full reverse diffusion chain starting from Gaussian noise."""
        device = condition.device
        x = torch.randn_like(condition)

        for step in reversed(range(self.num_steps)):
            timesteps = torch.full((condition.shape[0],), step, device=device, dtype=torch.long)
            x = self.p_sample(model, x, condition, timesteps)

        if clip_output:
            x = torch.clamp(x, 0.0, 1.0)
        return x
