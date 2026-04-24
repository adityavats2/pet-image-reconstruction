"""Run the full edge-to-pet translation experiment using the modular src package."""

from pathlib import Path
import sys

import torch
import torch.nn as nn

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import DEFAULT_CONFIG
from src.data.datasets import create_data_bundle
from src.data.preprocessing import get_edge_map
from src.evaluation.metrics import evaluate_l1
from src.evaluation.visualization import (
    plot_gan_losses,
    plot_loss_curve,
    show_dataset_samples,
    show_edge_target_pair,
    show_model_comparison,
    show_original_and_edge,
    show_prediction_triplets,
    show_two_column_batch,
)
from src.models.cnn import SimpleCNN
from src.models.gan import Discriminator, Generator
from src.training.train_cnn import train_cnn
from src.training.train_gan import train_gan
from src.utils.device import print_device_summary
from src.utils.paths import get_project_paths
from src.utils.reproducibility import seed_everything


def main():
    config = DEFAULT_CONFIG
    paths = get_project_paths(Path(__file__).resolve())

    device = print_device_summary()
    seed_everything(config.seed)

    print("Project folder:", paths.project_dir)
    print("Data folder:", paths.data_dir)
    print("Results folder:", paths.results_dir)
    print("Checkpoints folder:", paths.checkpoints_dir)

    data_bundle = create_data_bundle(config, paths)
    print("Dataset size:", len(data_bundle.dataset))

    base_loader = torch.utils.data.DataLoader(
        data_bundle.dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    images = next(iter(base_loader))[0]
    show_dataset_samples(images)

    sample_image = images[0]
    edge_image = get_edge_map(sample_image)
    show_original_and_edge(sample_image, edge_image)

    print("Edge dataset size:", len(data_bundle.edge_dataset))
    edge_sample, image_sample = data_bundle.edge_dataset[0]
    show_edge_target_pair(edge_sample, image_sample)

    print("Train subset size:", len(data_bundle.train_subset))
    print("Test subset size:", len(data_bundle.test_subset))

    edge_batch, image_batch = next(iter(data_bundle.train_loader))
    show_two_column_batch(edge_batch, image_batch, "Train Edge", "Train Target")

    cnn_model = SimpleCNN().to(device)
    cnn_criterion = nn.L1Loss()
    cnn_optimizer = torch.optim.Adam(
        cnn_model.parameters(),
        lr=config.cnn_learning_rate,
    )

    print(cnn_model)
    edge_batch_device = edge_batch.to(device)
    with torch.no_grad():
        cnn_shape_output = cnn_model(edge_batch_device)

    print("Input shape :", edge_batch_device.shape)
    print("Output shape:", cnn_shape_output.shape)

    cnn_losses = train_cnn(
        cnn_model,
        data_bundle.train_loader,
        cnn_criterion,
        cnn_optimizer,
        device,
        epochs=config.cnn_epochs,
    )
    plot_loss_curve(cnn_losses, "CNN Training Loss", "L1 Loss")

    cnn_model.eval()
    edge_batch, target_batch = next(iter(data_bundle.test_loader))
    edge_batch_device = edge_batch.to(device)
    with torch.no_grad():
        cnn_output_batch = cnn_model(edge_batch_device).cpu()

    show_prediction_triplets(
        edge_batch.cpu(),
        cnn_output_batch,
        target_batch.cpu(),
        "CNN Output",
    )

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    edge_batch, image_batch = next(iter(data_bundle.train_loader))
    edge_batch_device = edge_batch.to(device)
    image_batch_device = image_batch.to(device)

    with torch.no_grad():
        fake_images = generator(edge_batch_device)
        discriminator_output = discriminator(edge_batch_device, fake_images)

    print("Edge batch shape:       ", edge_batch_device.shape)
    print("Fake image shape:       ", fake_images.shape)
    print("Discriminator out shape:", discriminator_output.shape)

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

    real_labels = torch.ones((config.batch_size, 1, 1, 1), device=device)
    fake_labels = torch.zeros((config.batch_size, 1, 1, 1), device=device)
    print("Real labels shape:", real_labels.shape)
    print("Fake labels shape:", fake_labels.shape)

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

    generator.eval()
    edge_batch, target_batch = next(iter(data_bundle.test_loader))
    edge_batch_device = edge_batch.to(device)
    with torch.no_grad():
        gan_output_batch = generator(edge_batch_device).cpu()

    show_prediction_triplets(
        edge_batch.cpu(),
        gan_output_batch,
        target_batch.cpu(),
        "GAN Output",
    )

    cnn_model.eval()
    generator.eval()
    edge_batch, target_batch = next(iter(data_bundle.test_loader))
    edge_batch_device = edge_batch.to(device)
    with torch.no_grad():
        cnn_output_batch = cnn_model(edge_batch_device).cpu()
        gan_output_batch = generator(edge_batch_device).cpu()

    show_model_comparison(
        edge_batch.cpu(),
        cnn_output_batch,
        gan_output_batch,
        target_batch.cpu(),
        "Edge",
    )

    print("Shifted test subset size:", len(data_bundle.shifted_test_subset))
    shifted_edges, shifted_targets = next(iter(data_bundle.shifted_test_loader))
    show_two_column_batch(shifted_edges, shifted_targets, "Shifted Edge", "Target")

    edge_batch, target_batch = next(iter(data_bundle.shifted_test_loader))
    edge_batch_device = edge_batch.to(device)
    with torch.no_grad():
        cnn_output_batch = cnn_model(edge_batch_device).cpu()
        gan_output_batch = generator(edge_batch_device).cpu()

    show_model_comparison(
        edge_batch.cpu(),
        cnn_output_batch,
        gan_output_batch,
        target_batch.cpu(),
        "Shifted Edge",
    )

    cnn_l1 = evaluate_l1(cnn_model, data_bundle.test_loader, device)
    gan_l1 = evaluate_l1(generator, data_bundle.test_loader, device)

    print("CNN L1 Loss:", cnn_l1)
    print("GAN L1 Loss:", gan_l1)

    cnn_checkpoint_path = paths.checkpoints_dir / "cnn_best.pth"
    gan_checkpoint_path = paths.checkpoints_dir / "gan_best.pth"
    torch.save(cnn_model.state_dict(), cnn_checkpoint_path)
    torch.save(generator.state_dict(), gan_checkpoint_path)
    print(f"Weights saved successfully to {paths.checkpoints_dir}")


if __name__ == "__main__":
    main()
