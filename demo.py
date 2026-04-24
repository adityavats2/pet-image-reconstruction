import argparse
from pathlib import Path
import sys

from src.config import DEFAULT_CONFIG
from src.inference.comparison_demo import run_comparison_demo_from_path
from src.utils.device import print_device_summary
from src.utils.paths import get_project_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pretrained CNN and GAN on a single pet image and save "
            "a side-by-side comparison."
        ),
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        help="Path to the input image.",
    )
    input_group.add_argument(
        "--sample",
        action="store_true",
        help="Run the demo on a built-in Oxford-IIIT Pet sample image.",
    )
    parser.add_argument(
        "--cnn-checkpoint",
        default=None,
        help="Optional override for the CNN checkpoint path.",
    )
    parser.add_argument(
        "--gan-checkpoint",
        default=None,
        help="Optional override for the GAN checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for the directory where demo artifacts are saved.",
    )
    return parser


def get_sample_image_path(paths) -> Path:
    sample_path = paths.data_dir / "oxford-iiit-pet" / "images" / "Abyssinian_1.jpg"
    if not sample_path.exists():
        raise FileNotFoundError(
            "Sample image not found. Download the dataset first or run the demo "
            "with --image path/to/your_pet.jpg instead."
        )
    return sample_path


def format_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    paths = get_project_paths(Path(__file__).resolve())
    config = DEFAULT_CONFIG
    try:
        input_path = (
            get_sample_image_path(paths)
            if args.sample
            else Path(args.image).expanduser().resolve()
        )
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")

        cnn_checkpoint_path = (
            Path(args.cnn_checkpoint).expanduser().resolve()
            if args.cnn_checkpoint
            else paths.checkpoints_dir / "cnn_best.pth"
        )
        gan_checkpoint_path = (
            Path(args.gan_checkpoint).expanduser().resolve()
            if args.gan_checkpoint
            else paths.checkpoints_dir / "gan_best.pth"
        )
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else paths.results_dir / "demo"
        )

        device = print_device_summary()
        print("\nRunning comparison demo...")
        result = run_comparison_demo_from_path(
            image_path=input_path,
            cnn_checkpoint_path=cnn_checkpoint_path,
            gan_checkpoint_path=gan_checkpoint_path,
            output_dir=output_dir,
            config=config,
            device=device,
        )

        print("\nComparison demo complete.")
        print(f"Input image: {format_path(result.input_path, paths.project_dir)}")
        print(f"CNN checkpoint: {format_path(result.cnn_checkpoint_path, paths.project_dir)}")
        print(f"GAN checkpoint: {format_path(result.gan_checkpoint_path, paths.project_dir)}")
        print(f"Device: {result.device}")
        print(f"Output directory: {format_path(output_dir, paths.project_dir)}")
        print("Saved artifacts:")
        print(f"  Original:   {format_path(result.artifacts.original, paths.project_dir)}")
        print(f"  Edge map:   {format_path(result.artifacts.edge, paths.project_dir)}")
        print(f"  CNN output: {format_path(result.artifacts.cnn_output, paths.project_dir)}")
        print(f"  GAN output: {format_path(result.artifacts.gan_output, paths.project_dir)}")
        print(f"  Comparison: {format_path(result.artifacts.comparison, paths.project_dir)}")
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except OSError as exc:
        print(f"Error: Could not read or write one of the files. {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
