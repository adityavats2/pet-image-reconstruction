from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    image_size: tuple[int, int] = (128, 128)
    batch_size: int = 16
    train_size: int = 3000
    test_size: int = 500
    cnn_epochs: int = 10
    gan_epochs: int = 8
    cnn_learning_rate: float = 0.001
    gan_learning_rate: float = 0.0002
    gan_betas: tuple[float, float] = (0.5, 0.999)
    lambda_l1: int = 100
    dataset_split: str = "trainval"
    target_types: str = "category"


DEFAULT_CONFIG = ExperimentConfig()
