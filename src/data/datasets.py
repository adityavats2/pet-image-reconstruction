from dataclasses import dataclass

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from src.config import ExperimentConfig
from src.data.preprocessing import get_blurred_edge_map, get_edge_map
from src.utils.paths import ProjectPaths


class EdgeToImageDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, _ = self.base_dataset[idx]
        edge = get_edge_map(image)
        return edge, image


class BlurredEdgeToImageDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, _ = self.base_dataset[idx]
        edge = get_blurred_edge_map(image)
        return edge, image


@dataclass(frozen=True)
class DataBundle:
    dataset: OxfordIIITPet
    edge_dataset: EdgeToImageDataset
    shifted_dataset: BlurredEdgeToImageDataset
    train_subset: Subset
    test_subset: Subset
    shifted_test_subset: Subset
    train_loader: DataLoader
    test_loader: DataLoader
    shifted_test_loader: DataLoader


def build_transform(config: ExperimentConfig):
    return transforms.Compose(
        [
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
        ]
    )


def load_base_dataset(config: ExperimentConfig, paths: ProjectPaths, download: bool = True):
    return OxfordIIITPet(
        root=str(paths.data_dir),
        split=config.dataset_split,
        target_types=config.target_types,
        download=download,
        transform=build_transform(config),
    )


def create_data_bundle(
    config: ExperimentConfig,
    paths: ProjectPaths,
    download: bool = True,
) -> DataBundle:
    dataset = load_base_dataset(config, paths, download=download)
    edge_dataset = EdgeToImageDataset(dataset)
    shifted_dataset = BlurredEdgeToImageDataset(dataset)

    train_indices = list(range(config.train_size))
    test_indices = list(range(config.train_size, config.train_size + config.test_size))

    train_subset = Subset(edge_dataset, train_indices)
    test_subset = Subset(edge_dataset, test_indices)
    shifted_test_subset = Subset(shifted_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=config.batch_size, shuffle=False)
    shifted_test_loader = DataLoader(
        shifted_test_subset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    return DataBundle(
        dataset=dataset,
        edge_dataset=edge_dataset,
        shifted_dataset=shifted_dataset,
        train_subset=train_subset,
        test_subset=test_subset,
        shifted_test_subset=shifted_test_subset,
        train_loader=train_loader,
        test_loader=test_loader,
        shifted_test_loader=shifted_test_loader,
    )
