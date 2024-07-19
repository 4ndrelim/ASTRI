"""
Display
"""
from typing import List

import cv2
import numpy as np
import torch

def load_image_as_tensor(image_path: str) -> torch.Tensor:
    """
    Loads image in grayscale, and converts into tensor.
    :param image_path: The path to the image
    :return: Image in pytorch tensor
    """
    # pylint: disable=no-member
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.float32) / 255.0 # normalize
    tensor = torch.tensor(image)
    tensor = torch.unsqueeze(tensor, 0) # add channel dim

    return tensor


def load_images_as_batch(image_paths: List[str]) -> torch.Tensor:
    """
    Loads multiple images and convert them into pytorch tensors.
    :param image_paths: List of image paths
    :return: Tensors
    """
    tensors = [load_image_as_tensor(image_path) for image_path in image_paths]
    return torch.stack(tensors) # return as tensors with batch dimension
