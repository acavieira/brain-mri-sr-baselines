"""Training loop for the conditional diffusion U-Net model."""

import csv
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..io import ensure_dir
from .config import DiffusionConfig
from .dataset import NiftiSlicePairDataset, collect_nifti_paths
from .model import ConditionalUNet2D
from .scheduler import DiffusionScheduler


DEFAULT_MODEL_KWARGS: Dict[str, object] = {
    "in_channels": 1,
    "cond_channels": 1,
    "base_channels": 64,
    "channel_multipliers": (1, 2, 4),
    "time_dim": 256,
}


def resolve_device(device_arg: str) -> torch.device:
    """Resolve user preference to an available training device."""
    if device_arg and device_arg != "auto":
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_checkpoint(
    path: Path,
    model: ConditionalUNet2D,
    optimizer: AdamW,
    epoch: int,
    best_loss: float,
    config: DiffusionConfig,
    model_kwargs: Dict[str, object],
) -> None:
    """Serialize model, optimizer, and metadata into one checkpoint file."""
    payload = {
        "epoch": epoch,
        "best_loss": best_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "model_kwargs": model_kwargs,
    }
    torch.save(payload, path)


def _append_history_row(path: Path, row: Dict[str, object]) -> None:
    """Append one epoch summary row to training_history.csv."""
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _build_dataloader(config: DiffusionConfig) -> Tuple[DataLoader, int]:
    """Create dataset and dataloader for diffusion training."""
    nifti_paths = collect_nifti_paths(config.train_glob)
    if not nifti_paths:
        raise FileNotFoundError(f"No NIfTI files matched train_glob: {config.train_glob}")

    dataset = NiftiSlicePairDataset(
        nifti_paths=nifti_paths,
        image_size=config.image_size,
        scale=config.scale,
        blur_sigma=config.blur_sigma,
        noise_sigma=config.noise_sigma,
        slices_per_volume=config.slices_per_volume,
        axes=config.axes,
        random_seed=config.random_seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return dataloader, len(dataset)


def train_diffusion(config: DiffusionConfig) -> None:
    """Train diffusion model and save periodic checkpoints."""
    ensure_dir(config.output_dir)
    output_dir = Path(config.output_dir)

    # Persist exact training configuration for reproducible experiments.
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    seed_everything(config.random_seed)
    device = resolve_device(config.device)
    print(f"Using device: {device}")

    dataloader, dataset_size = _build_dataloader(config)
    print(f"Dataset samples: {dataset_size}")
    print(f"Batches per epoch: {len(dataloader)}")

    model_kwargs = dict(DEFAULT_MODEL_KWARGS)
    model = ConditionalUNet2D(**model_kwargs).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = DiffusionScheduler(
        num_steps=config.num_diffusion_steps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
    ).to(device)

    best_loss = float("inf")
    history_path = output_dir / "training_history.csv"

    # Standard DDPM objective: predict noise from noisy target and condition.
    total_start = time.perf_counter()
    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()

        running_loss = 0.0
        sample_count = 0

        for batch in dataloader:
            hr = batch["hr"].to(device)
            cond = batch["cond"].to(device)
            timesteps = scheduler.sample_timesteps(hr.shape[0], device=device)
            noise = torch.randn_like(hr)
            noisy_hr = scheduler.q_sample(hr, timesteps, noise)

            predicted_noise = model(noisy_hr, cond, timesteps)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = hr.size(0)
            running_loss += float(loss.item()) * batch_size
            sample_count += batch_size

        epoch_loss = running_loss / max(sample_count, 1)
        epoch_seconds = time.perf_counter() - epoch_start

        row = {
            "epoch": epoch,
            "loss_mse_noise": round(epoch_loss, 8),
            "epoch_seconds": round(epoch_seconds, 3),
            "samples": sample_count,
            "device": str(device),
        }
        _append_history_row(history_path, row)

        print(
            f"[Epoch {epoch:03d}/{config.num_epochs}] "
            f"loss={epoch_loss:.6f} | time={epoch_seconds:.2f}s"
        )

        # Save the best-performing checkpoint based on epoch loss.
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            _save_checkpoint(
                path=output_dir / "best_model.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_loss=best_loss,
                config=config,
                model_kwargs=model_kwargs,
            )

        # Save periodic checkpoints for later comparisons and recovery.
        if epoch % config.save_every == 0 or epoch == config.num_epochs:
            _save_checkpoint(
                path=output_dir / f"checkpoint_epoch_{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_loss=best_loss,
                config=config,
                model_kwargs=model_kwargs,
            )

    total_seconds = time.perf_counter() - total_start
    print("\nTraining complete.")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Total training time: {total_seconds / 60.0:.2f} min")
    print(f"Artifacts: {output_dir}")
