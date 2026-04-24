from pathlib import Path

import torch

from src.config import DEFAULT_CONFIG
from src.data.datasets import create_data_bundle
from src.evaluation.metrics import evaluate_l1
from src.evaluation.visualization import (
    show_model_comparison,
    show_prediction_triplets,
)
from src.models.cnn import SimpleCNN
from src.models.gan import Generator
from src.utils.device import print_device_summary
from src.utils.paths import get_project_paths


def main():
    config = DEFAULT_CONFIG
    paths = get_project_paths(Path(__file__).resolve())
    device = print_device_summary()

    data_bundle = create_data_bundle(config, paths, download=False)

    cnn_model = SimpleCNN().to(device)
    generator = Generator().to(device)

    cnn_checkpoint = paths.checkpoints_dir / "cnn_best.pth"
    gan_checkpoint = paths.checkpoints_dir / "gan_best.pth"

    cnn_model.load_state_dict(torch.load(cnn_checkpoint, map_location=device))
    generator.load_state_dict(torch.load(gan_checkpoint, map_location=device))

    cnn_l1 = evaluate_l1(cnn_model, data_bundle.test_loader, device)
    gan_l1 = evaluate_l1(generator, data_bundle.test_loader, device)

    print("CNN L1 Loss:", cnn_l1)
    print("GAN L1 Loss:", gan_l1)

    edge_batch, target_batch = next(iter(data_bundle.test_loader))
    edge_batch_device = edge_batch.to(device)

    with torch.no_grad():
        cnn_output_batch = cnn_model(edge_batch_device).cpu()
        gan_output_batch = generator(edge_batch_device).cpu()

    show_prediction_triplets(edge_batch.cpu(), cnn_output_batch, target_batch.cpu(), "CNN Output")
    show_prediction_triplets(edge_batch.cpu(), gan_output_batch, target_batch.cpu(), "GAN Output")
    show_model_comparison(
        edge_batch.cpu(),
        cnn_output_batch,
        gan_output_batch,
        target_batch.cpu(),
        "Edge",
    )


if __name__ == "__main__":
    main()
