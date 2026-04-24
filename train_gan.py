from pathlib import Path

import torch
import torch.nn as nn

from src.config import DEFAULT_CONFIG
from src.data.datasets import create_data_bundle
from src.evaluation.visualization import plot_gan_losses
from src.models.gan import Discriminator, Generator
from src.training.train_gan import train_gan
from src.utils.device import print_device_summary
from src.utils.paths import get_project_paths
from src.utils.reproducibility import seed_everything


def main():
    config = DEFAULT_CONFIG
    paths = get_project_paths(Path(__file__).resolve())

    device = print_device_summary()
    seed_everything(config.seed)

    data_bundle = create_data_bundle(config, paths)
    print("Dataset size:", len(data_bundle.dataset))
    print("Train subset size:", len(data_bundle.train_subset))
    print("Test subset size:", len(data_bundle.test_subset))

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    adversarial_loss = nn.BCELoss()
    reconstruction_loss = nn.L1Loss()

    g_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=config.gan_learning_rate,
        betas=config.gan_betas,
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.gan_learning_rate,
        betas=config.gan_betas,
    )

    g_losses, d_losses = train_gan(
        generator,
        discriminator,
        data_bundle.train_loader,
        adversarial_loss,
        reconstruction_loss,
        g_optimizer,
        d_optimizer,
        device,
        epochs=config.gan_epochs,
        lambda_l1=config.lambda_l1,
    )

    plot_gan_losses(g_losses, d_losses)

    checkpoint_path = paths.checkpoints_dir / "gan_best.pth"
    torch.save(generator.state_dict(), checkpoint_path)
    print(f"Saved GAN checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
