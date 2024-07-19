"""
A Residual NN model to simulate Gerchberg-Saxton algorithm.

"""

from typing import Callable, List

import cv2
import numpy as np
import torch
import torch.fft
from torch import nn
import torch.nn.functional as F


# Utility Functions
def load_image_as_tensor(image_path: str) -> torch.Tensor:
    """
    Loads an image from the given path, converts it to grayscale, and then to a PyTorch tensor.
    :param image_path: Path to the image file
    :return: Tensor representing the image
    """
    # Load image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Convert image to a NumPy array and normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    # Convert NumPy array to PyTorch tensor and add a channel dimension
    tensor = torch.tensor(image)
    return tensor


def load_images_as_batch(image_paths: List[str]) -> torch.Tensor:
    tensors = [load_image_as_tensor(image_path) for image_path in image_paths]
    batch_tensor = torch.stack(tensors)
    return batch_tensor


def fourier_transform(input: torch.Tensor) -> torch.Tensor:
    """
    Performs Fourier transform on the given input batch.
    :param input: Input tensor of shape (batch_size, height, width)
    :return: Transformed tensor with shape (batch_size, 2, height, width) where 
             the first channel is the real part and the second channel is the imaginary part.
    """
    # Perform 2D Fourier transform
    dft = torch.fft.fft2(input)

    # Get real and imaginary parts
    real = dft.real
    imag = dft.imag

    # Stack the real and imaginary parts along the channel dimension
    output = torch.stack((real, imag), dim=1)
    
    return output

    

class ForwardConvBlock(nn.Module):
    """
    Forward block as described in the paper. Output has 2*p layers.
    Output of each block will have the same dimension as input, circular padding is performed.
    """
    def __init__(self, in_channels: int, p: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=3, padding='same', padding_mode='circular')
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 30, kernel_size=3, padding='same', padding_mode='circular')
        self.bn2 = nn.BatchNorm2d(30)
        self.conv3 = nn.Conv2d(30, 30, kernel_size=3, padding='same', padding_mode='circular')
        self.bn3 = nn.BatchNorm2d(30)
        self.conv4 = nn.Conv2d(30, 40, kernel_size=3, padding='same', padding_mode='circular')
        self.bn4 = nn.BatchNorm2d(40)
        self.conv5 = nn.Conv2d(40, p*2, kernel_size=3, padding='same', padding_mode='circular')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method.
        :param x: Input array
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        return x

class BackwardConvBlock(nn.Module):
    """
    Backward block as described in the paper.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=3, padding='same', padding_mode='circular')
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 30, kernel_size=3, padding='same', padding_mode='circular')
        self.bn2 = nn.BatchNorm2d(30)
        self.conv3 = nn.Conv2d(30, 30, kernel_size=3, padding='same', padding_mode='circular')
        self.bn3 = nn.BatchNorm2d(30)
        self.conv4 = nn.Conv2d(30, 40, kernel_size=3, padding='same', padding_mode='circular')
        self.bn4 = nn.BatchNorm2d(40)
        self.conv5 = nn.Conv2d(40, 1, kernel_size=3, padding='same', padding_mode='circular')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method.
        :param x: Input array
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        return x


class RGSN(nn.Module):
    """
    Model simulating the unrolled residual Gerchberg-Saxton algorithm.
    """
    def __init__(self, N: int, p: int):
        super().__init__()
        self.N = N
        self.p = p
        # 3 input channels for real, imaginary parts, and goal amplitude
        self.forward_blocks = nn.ModuleList(
            [ForwardConvBlock(3, p) for _ in range(N)]
            )
        # 2 * p input channels for complex field
        self.backward_blocks = nn.ModuleList(
            [BackwardConvBlock(2 * p) for _ in range(N)]
            )

    def forward(self, f0: torch.Tensor, ggoal, A: Callable, A_inv: Callable) -> torch.Tensor:
        """
        Forward method.
        :param f0: Initial hologram tensor
        :param ggoal: Goal amplitude tensor
        :param A: Forward operator function
        :param A_inv: Inverse operator function
        :return: Final hologram tensor
        """
        f = f0
        for n in range(self.N):
            fn = A(f)
            fn = torch.cat((fn, ggoal), dim=1)
            fn = self.forward_blocks[n](fn)
            fn = A_inv(fn)
            fn = self.backward_blocks[n](fn)
            f = f + fn
        return f



# pylint: disable=pointless-string-statement
"""
1. How can we train a model where input is all 1s but expect different outputs?
Input must somehow interact with the model!

2. what is A and A_inv?
FFT

3. input has an imaginary field?? where and how to get


"""