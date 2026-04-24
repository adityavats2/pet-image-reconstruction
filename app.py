from pathlib import Path
import tempfile
import time

import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.config import DEFAULT_CONFIG
from src.inference.comparison_demo import (
    ComparisonDemoResult,
    load_cnn_checkpoint,
    load_gan_checkpoint,
    preprocess_image,
    run_comparison_inference_with_models,
    save_comparison_outputs,
)
from src.utils.device import get_device
from src.utils.paths import get_project_paths


PAGE_TITLE = "Pet Image Reconstruction Demo"
PAGE_ICON = "🐾"
ALLOWED_FILE_TYPES = ["png", "jpg", "jpeg"]
TECH_STACK = "Built with: PyTorch | OpenCV | Streamlit | NumPy | Pillow"
EDGE_MODE_OPTIONS = {
    "Standard": "standard",
    "Blurred (domain shift)": "blurred",
}
SAMPLE_IMAGE_OPTIONS = {
    "Abyssinian": "Abyssinian_1.jpg",
    "Bengal": "Bengal_48.jpg",
    "Pug": "pug_1.jpg",
}


def get_named_sample_image_path(paths, sample_name: str) -> Path:
    filename = SAMPLE_IMAGE_OPTIONS[sample_name]
    sample_path = paths.data_dir / "oxford-iiit-pet" / "images" / filename
    if not sample_path.exists():
        raise FileNotFoundError(
            f"Sample image not found: {filename}. Add the Oxford-IIIT Pet dataset or upload your own image."
        )
    return sample_path


def save_uploaded_image(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def display_results(result, edge_mode_label: str) -> None:
    st.subheader("Inference Results")
    st.caption(
        "Key Finding: While the CNN captures general shapes, the GAN "
        "(using adversarial loss) successfully recovers fine-grained fur "
        "textures and eye details that traditional regression models blur."
    )
    columns = st.columns(3)
    images = [
        ("Original Image", result.artifacts.original),
        (f"{edge_mode_label} Edge Map", result.artifacts.edge),
        ("GAN Output", result.artifacts.gan_output),
    ]

    for column, (label, image_path) in zip(columns, images):
        column.subheader(label)
        column.image(str(image_path), use_container_width=True)


def display_comparison_results(result) -> None:
    st.subheader("Model Comparison")
    columns = st.columns(2)
    comparison_images = [
        ("CNN Output", result.artifacts.cnn_output),
        ("Four-Panel Comparison", result.artifacts.comparison),
    ]

    for column, (label, image_path) in zip(columns, comparison_images):
        column.subheader(label)
        column.image(str(image_path), use_container_width=True)


def display_run_metrics(inference_seconds: float, device: str, processed_resolution: tuple[int, int]) -> None:
    st.caption("Technical Summary")
    columns = st.columns(3)
    metric_values = [
        ("Inference Time", f"{inference_seconds:.2f}s"),
        ("Runtime Device", device.upper()),
        ("Processed Resolution", f"{processed_resolution[0]}x{processed_resolution[1]}"),
    ]

    for column, (label, value) in zip(columns, metric_values):
        column.metric(label, value)


def display_saved_artifacts(result, edge_mode_label: str) -> None:
    st.write(f"Edge mode used: `{edge_mode_label}`.")
    with st.expander("Technical Output Details"):
        st.caption("Artifacts are saved locally for reproducibility and debugging.")
        st.code(
            "\n".join(
                [
                    f"Original:   {result.artifacts.original.name}",
                    f"Edge Map:   {result.artifacts.edge.name}",
                    f"CNN Output: {result.artifacts.cnn_output.name}",
                    f"GAN Output: {result.artifacts.gan_output.name}",
                    f"Comparison: {result.artifacts.comparison.name}",
                ]
            )
        )


@st.cache_resource(show_spinner=False)
def load_demo_models():
    paths = get_project_paths(Path(__file__).resolve())
    device = get_device()
    cnn_model = load_cnn_checkpoint(paths.checkpoints_dir / "cnn_best.pth", device)
    gan_model = load_gan_checkpoint(paths.checkpoints_dir / "gan_best.pth", device)
    return cnn_model, gan_model, device


def run_demo(image_path: Path, output_dir: Path, save_stem: str, edge_mode: str):
    cnn_model, gan_model, device = load_demo_models()
    paths = get_project_paths(Path(__file__).resolve())
    image_tensor = preprocess_image(image_path, DEFAULT_CONFIG)
    original_tensor, edge_tensor, cnn_tensor, gan_tensor = run_comparison_inference_with_models(
        image_tensor=image_tensor,
        cnn_model=cnn_model,
        gan_model=gan_model,
        device=device,
        edge_mode=edge_mode,
    )
    artifacts = save_comparison_outputs(
        image_tensor=original_tensor,
        edge_tensor=edge_tensor,
        cnn_tensor=cnn_tensor,
        gan_tensor=gan_tensor,
        input_path=Path(save_stem),
        output_dir=output_dir,
    )
    return ComparisonDemoResult(
        input_path=image_path,
        cnn_checkpoint_path=paths.checkpoints_dir / "cnn_best.pth",
        gan_checkpoint_path=paths.checkpoints_dir / "gan_best.pth",
        device=str(device),
        artifacts=artifacts,
    )


def get_input_image(paths, input_mode: str, uploaded_file, sample_name: str) -> tuple[Path, str, Path | None]:
    if input_mode == "Sample image":
        image_path = get_named_sample_image_path(paths, sample_name)
        return image_path, Path(image_path.name).stem, None

    if uploaded_file is None:
        raise ValueError("Upload an image or switch to the sample image option.")

    try:
        with Image.open(uploaded_file) as uploaded_image:
            uploaded_image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc

    uploaded_file.seek(0)
    temp_path = save_uploaded_image(uploaded_file)
    return temp_path, Path(uploaded_file.name).stem, temp_path


def preview_input_image(image_path: Path) -> None:
    st.subheader("Selected Input")
    st.image(str(image_path), use_container_width=False, width=280)


def display_pipeline_architecture() -> None:
    with st.expander("View Pipeline Architecture"):
        st.markdown(
            "- **Original Image**: A raw pet image is uploaded or selected from the sample set.\n"
            "- **Canny Edge Conditioning**: The image is resized and converted into a shared edge map.\n"
            "- **Latent Representation**: The model encodes the structural edge signal into internal feature space.\n"
            "- **Reconstruction**: CNN and GAN models decode the conditioned representation into pet image outputs.\n"
            "- **Comparison**: The app displays the GAN-focused demo view, with optional CNN baseline comparison."
        )


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

    paths = get_project_paths(Path(__file__).resolve())
    output_dir = paths.results_dir / "demo"

    st.title(PAGE_TITLE)
    st.caption(TECH_STACK)
    st.write(
        "This interactive demo showcases an edge-to-image translation pipeline comparing "
        "standard CNN baselines against Generative Adversarial Networks (GANs). By using "
        "shared Canny edge maps as conditioning, this project demonstrates the GAN's "
        "superior ability to reconstruct high-fidelity textures and realistic pet features."
    )
    st.caption("Full modular implementation and training logs are available on GitHub.")
    display_pipeline_architecture()

    with st.sidebar:
        st.header("Demo Controls")
        input_mode = st.radio(
            "Image source",
            ("Upload image", "Sample image"),
        )
        sample_name = st.selectbox(
            "Choose a sample image",
            options=list(SAMPLE_IMAGE_OPTIONS.keys()),
            disabled=input_mode != "Sample image",
        )
        uploaded_file = st.file_uploader(
            "Choose a pet image",
            type=ALLOWED_FILE_TYPES,
            disabled=input_mode != "Upload image",
        )
        edge_mode_label = st.radio(
            "Edge input mode",
            tuple(EDGE_MODE_OPTIONS.keys()),
        )
        show_comparison = st.checkbox(
            "Show CNN comparison",
            value=False,
            help="Adds the CNN output and the full four-panel comparison below the main GAN-focused results.",
        )
        run_button = st.button("Run Demo", type="primary", use_container_width=True)

    if input_mode == "Sample image":
        try:
            preview_input_image(get_named_sample_image_path(paths, sample_name))
        except FileNotFoundError as exc:
            st.warning(str(exc))
    elif uploaded_file is not None:
        try:
            uploaded_preview = Image.open(uploaded_file)
            st.subheader("Selected Input")
            st.image(uploaded_preview, use_container_width=False, width=280)
            uploaded_file.seek(0)
        except (UnidentifiedImageError, OSError):
            st.warning("Preview unavailable for the uploaded file.")

    if not run_button:
        st.info("Choose an image source and click Run Demo.")
        return

    temp_path: Path | None = None
    try:
        image_path, save_stem, temp_path = get_input_image(
            paths=paths,
            input_mode=input_mode,
            uploaded_file=uploaded_file,
            sample_name=sample_name,
        )
        edge_mode = EDGE_MODE_OPTIONS[edge_mode_label]
        artifact_stem = save_stem if edge_mode == "standard" else f"{save_stem}_{edge_mode}"

        with st.spinner("Running pretrained inference..."):
            start_time = time.perf_counter()
            result = run_demo(
                image_path=image_path,
                output_dir=output_dir,
                save_stem=artifact_stem,
                edge_mode=edge_mode,
            )
            inference_seconds = time.perf_counter() - start_time

        st.success("Demo complete.")
        display_results(result, edge_mode_label=edge_mode_label)
        if show_comparison:
            display_comparison_results(result)
        display_run_metrics(
            inference_seconds=inference_seconds,
            device=result.device,
            processed_resolution=DEFAULT_CONFIG.image_size,
        )
        display_saved_artifacts(result, edge_mode_label=edge_mode_label)
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Unexpected error during inference: {exc}")
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    main()
