import cv2
import numpy as np
import torch


def _tensor_to_uint8_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    return (image * 255).astype(np.uint8)


def get_edge_map(image_tensor: torch.Tensor) -> torch.Tensor:
    img_uint8 = _tensor_to_uint8_image(image_tensor)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200) / 255.0
    stacked_edges = np.stack([edges, edges, edges], axis=0)
    return torch.tensor(stacked_edges, dtype=torch.float32)


def get_blurred_edge_map(image_tensor: torch.Tensor) -> torch.Tensor:
    img_uint8 = _tensor_to_uint8_image(image_tensor)
    blurred = cv2.GaussianBlur(img_uint8, (9, 9), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200) / 255.0
    stacked_edges = np.stack([edges, edges, edges], axis=0)
    return torch.tensor(stacked_edges, dtype=torch.float32)
