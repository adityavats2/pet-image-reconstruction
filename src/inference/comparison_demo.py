from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

from src.config import ExperimentConfig
from src.data.datasets import build_transform
from src.data.preprocessing import get_blurred_edge_map, get_edge_map
from src.models.cnn import SimpleCNN
from src.models.gan import Generator


@dataclass(frozen=True)
class ComparisonArtifacts:
    original: Path
    edge: Path
    cnn_output: Path
    gan_output: Path
    comparison: Path


@dataclass(frozen=True)
class ComparisonDemoResult:
    input_path: Path
    cnn_checkpoint_path: Path
    gan_checkpoint_path: Path
    device: str
    artifacts: ComparisonArtifacts


def _load_checkpoint(model, checkpoint_path: Path, device: torch.device, model_name: str):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{model_name} checkpoint not found at {checkpoint_path}. "
            f"Make sure {checkpoint_path.name} exists or pass a valid override."
        )

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_cnn_checkpoint(checkpoint_path: Path, device: torch.device) -> SimpleCNN:
    model = SimpleCNN().to(device)
    return _load_checkpoint(model, checkpoint_path, device, "CNN")


def load_gan_checkpoint(checkpoint_path: Path, device: torch.device) -> Generator:
    model = Generator().to(device)
    return _load_checkpoint(model, checkpoint_path, device, "GAN")


def preprocess_image(image_path: Path, config: ExperimentConfig) -> torch.Tensor:
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = build_transform(config)(image)

    return tensor


def run_comparison_inference_with_models(
    image_tensor: torch.Tensor,
    cnn_model: SimpleCNN,
    gan_model: Generator,
    device: torch.device,
    edge_mode: str = "standard",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if edge_mode == "standard":
        edge_tensor = get_edge_map(image_tensor)
    elif edge_mode == "blurred":
        edge_tensor = get_blurred_edge_map(image_tensor)
    else:
        raise ValueError(f"Unsupported edge mode: {edge_mode}")

    batched_edge = edge_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        cnn_output = cnn_model(batched_edge).squeeze(0).cpu()
        gan_output = gan_model(batched_edge).squeeze(0).cpu()

    return image_tensor.cpu(), edge_tensor.cpu(), cnn_output, gan_output


def run_comparison_inference(
    image_tensor: torch.Tensor,
    cnn_checkpoint_path: Path,
    gan_checkpoint_path: Path,
    device: torch.device,
    edge_mode: str = "standard",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cnn_model = load_cnn_checkpoint(cnn_checkpoint_path, device)
    gan_model = load_gan_checkpoint(gan_checkpoint_path, device)
    return run_comparison_inference_with_models(
        image_tensor=image_tensor,
        cnn_model=cnn_model,
        gan_model=gan_model,
        device=device,
        edge_mode=edge_mode,
    )


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    return to_pil_image(image_tensor.clamp(0.0, 1.0))


def _add_label(image: Image.Image, label: str) -> Image.Image:
    label_height = 28
    labelled = Image.new("RGB", (image.width, image.height + label_height), color="white")
    labelled.paste(image, (0, label_height))

    draw = ImageDraw.Draw(labelled)
    draw.text((10, 8), label, fill="black")
    return labelled


def build_comparison_image(
    original_tensor: torch.Tensor,
    edge_tensor: torch.Tensor,
    cnn_tensor: torch.Tensor,
    gan_tensor: torch.Tensor,
) -> Image.Image:
    original_image = _add_label(tensor_to_pil(original_tensor), "Original")
    edge_image = _add_label(tensor_to_pil(edge_tensor), "Edge Map")
    cnn_image = _add_label(tensor_to_pil(cnn_tensor), "CNN Output")
    gan_image = _add_label(tensor_to_pil(gan_tensor), "GAN Output")

    panel_width, panel_height = original_image.size
    comparison = Image.new("RGB", (panel_width * 4, panel_height), color="white")
    comparison.paste(original_image, (0, 0))
    comparison.paste(edge_image, (panel_width, 0))
    comparison.paste(cnn_image, (panel_width * 2, 0))
    comparison.paste(gan_image, (panel_width * 3, 0))
    return comparison


def save_comparison_outputs(
    image_tensor: torch.Tensor,
    edge_tensor: torch.Tensor,
    cnn_tensor: torch.Tensor,
    gan_tensor: torch.Tensor,
    input_path: Path,
    output_dir: Path,
) -> ComparisonArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    original_path = output_dir / f"{stem}_original.png"
    edge_path = output_dir / f"{stem}_edge.png"
    cnn_output_path = output_dir / f"{stem}_cnn_output.png"
    gan_output_path = output_dir / f"{stem}_gan_output.png"
    comparison_path = output_dir / f"{stem}_comparison.png"

    tensor_to_pil(image_tensor).save(original_path)
    tensor_to_pil(edge_tensor).save(edge_path)
    tensor_to_pil(cnn_tensor).save(cnn_output_path)
    tensor_to_pil(gan_tensor).save(gan_output_path)
    build_comparison_image(image_tensor, edge_tensor, cnn_tensor, gan_tensor).save(comparison_path)

    return ComparisonArtifacts(
        original=original_path,
        edge=edge_path,
        cnn_output=cnn_output_path,
        gan_output=gan_output_path,
        comparison=comparison_path,
    )


def run_comparison_demo_from_path(
    image_path: Path,
    cnn_checkpoint_path: Path,
    gan_checkpoint_path: Path,
    output_dir: Path,
    config: ExperimentConfig,
    device: torch.device,
    edge_mode: str = "standard",
) -> ComparisonDemoResult:
    image_tensor = preprocess_image(image_path, config)
    original_tensor, edge_tensor, cnn_tensor, gan_tensor = run_comparison_inference(
        image_tensor=image_tensor,
        cnn_checkpoint_path=cnn_checkpoint_path,
        gan_checkpoint_path=gan_checkpoint_path,
        device=device,
        edge_mode=edge_mode,
    )

    artifacts = save_comparison_outputs(
        image_tensor=original_tensor,
        edge_tensor=edge_tensor,
        cnn_tensor=cnn_tensor,
        gan_tensor=gan_tensor,
        input_path=image_path,
        output_dir=output_dir,
    )

    return ComparisonDemoResult(
        input_path=image_path,
        cnn_checkpoint_path=cnn_checkpoint_path,
        gan_checkpoint_path=gan_checkpoint_path,
        device=str(device),
        artifacts=artifacts,
    )
