"""CLI entrypoint to train the diffusion U-Net model from scratch."""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments used for diffusion training."""
    parser = argparse.ArgumentParser(description="Train conditional U-Net diffusion SR model from scratch")
    parser.add_argument("--train-glob", default="data/train_94t/*.nii*")
    parser.add_argument("--output-dir", default="results/diffusion/training/diffusion_unet")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--blur-sigma", type=float, default=0.65)
    parser.add_argument("--noise-sigma", type=float, default=0.02)
    parser.add_argument("--slices-per-volume", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-diffusion-steps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=23)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    """Build a DiffusionConfig from CLI arguments and start training."""
    args = parse_args()

    try:
        # Keep heavy imports local to provide a clear installation error message.
        from src.mri_sr.diffusion.config import DiffusionConfig
        from src.mri_sr.diffusion.train import train_diffusion
    except Exception as exc:
        raise RuntimeError(
            "Diffusion training requires PyTorch. Install it first, for example:\n"
            "pip install torch torchvision"
        ) from exc

    config = DiffusionConfig(
        train_glob=args.train_glob,
        output_dir=args.output_dir,
        image_size=args.image_size,
        scale=args.scale,
        blur_sigma=args.blur_sigma,
        noise_sigma=args.noise_sigma,
        slices_per_volume=args.slices_per_volume,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        num_diffusion_steps=args.num_diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        save_every=args.save_every,
        random_seed=args.random_seed,
        device=args.device,
    )

    # The full training pipeline is implemented in src/mri_sr/diffusion/train.py.
    train_diffusion(config)


if __name__ == "__main__":
    main()
