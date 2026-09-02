"""Conditional U-Net building blocks for diffusion noise prediction."""

from .imports import *

def _group_count(channels: int) -> int:
    """Pick a valid GroupNorm group count for the given channel size."""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    """Classic sinusoidal embedding for diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Encode integer timesteps into continuous features."""
        device = timesteps.device
        half_dim = self.dim // 2
        if half_dim == 0:
            return timesteps.float().unsqueeze(-1)

        exponent = -math.log(10000) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=device) * exponent)
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResidualBlock(nn.Module):
    """Residual convolution block conditioned by time embeddings."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()

        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.activation = nn.SiLU()

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Apply two conv layers and add residual skip connection."""
        h = self.conv1(self.activation(self.norm1(x)))
        h = h + self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.activation(self.norm2(h)))
        return h + self.skip(x)


class ConditionalUNet2D(nn.Module):
    """2D U-Net that predicts diffusion noise from noisy HR and condition image."""

    def __init__(
        self,
        in_channels: int = 1,
        cond_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: Sequence[int] = (1, 2, 4),
        time_dim: int = 256,
    ) -> None:
        super().__init__()

        # Time embedding MLP used by all residual blocks.
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        self.input_conv = nn.Conv2d(in_channels + cond_channels, base_channels, kernel_size=3, padding=1)

        down_channels = [base_channels * mult for mult in channel_multipliers]
        self.down_blocks = nn.ModuleList()
        current_channels = base_channels

        for out_channels in down_channels:
            self.down_blocks.append(
                nn.ModuleDict(
                    {
                        "block1": ResidualBlock(current_channels, out_channels, time_dim),
                        "block2": ResidualBlock(out_channels, out_channels, time_dim),
                        "downsample": nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    }
                )
            )
            current_channels = out_channels

        # Bottleneck blocks operate at the lowest spatial resolution.
        self.mid_block1 = ResidualBlock(current_channels, current_channels, time_dim)
        self.mid_block2 = ResidualBlock(current_channels, current_channels, time_dim)

        self.up_blocks = nn.ModuleList()
        for out_channels in reversed(down_channels):
            self.up_blocks.append(
                nn.ModuleDict(
                    {
                        "upsample": nn.ConvTranspose2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1),
                        "block1": ResidualBlock(out_channels * 2, out_channels, time_dim),
                        "block2": ResidualBlock(out_channels, out_channels, time_dim),
                    }
                )
            )
            current_channels = out_channels

        self.output_norm = nn.GroupNorm(_group_count(current_channels), current_channels)
        self.output_conv = nn.Conv2d(current_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x_noisy: torch.Tensor, condition: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass for one diffusion denoising step."""
        time_emb = self.time_embedding(timesteps)

        x = torch.cat([x_noisy, condition], dim=1)
        x = self.input_conv(x)

        # Save encoder outputs for decoder skip connections.
        skip_connections = []
        for block in self.down_blocks:
            x = block["block1"](x, time_emb)
            x = block["block2"](x, time_emb)
            skip_connections.append(x)
            x = block["downsample"](x)

        x = self.mid_block1(x, time_emb)
        x = self.mid_block2(x, time_emb)

        for block in self.up_blocks:
            x = block["upsample"](x)
            skip = skip_connections.pop()

            # Guard against odd-size rounding during down/up sampling.
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            x = torch.cat([x, skip], dim=1)
            x = block["block1"](x, time_emb)
            x = block["block2"](x, time_emb)

        x = self.output_conv(F.silu(self.output_norm(x)))
        return x
