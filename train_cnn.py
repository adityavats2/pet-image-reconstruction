from pathlib import Path

import torch
import torch.nn as nn

from src.config import DEFAULT_CONFIG
from src.data.datasets import create_data_bundle
from src.evaluation.visualization import plot_loss_curve
from src.models.cnn import SimpleCNN
from src.training.train_cnn import train_cnn
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

    model = SimpleCNN().to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.cnn_learning_rate)

    losses = train_cnn(
        model,
        data_bundle.train_loader,
        criterion,
        optimizer,
        device,
        epochs=config.cnn_epochs,
    )

    plot_loss_curve(losses, "CNN Training Loss", "L1 Loss")

    checkpoint_path = paths.checkpoints_dir / "cnn_best.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved CNN checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
