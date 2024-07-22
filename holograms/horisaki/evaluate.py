"""
Evaluate using the trained NN
"""
from typing import Tuple
import os
import math

import numpy as np
import torch
import torch.fft
from torch import nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import joblib
from sklearn.base import BaseEstimator

# USER CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# saved model directory, need scaler and model
MODEL_DIR = os.path.join(BASE_DIR, 'saved_model')
MODEL_PATH = os.path.join(MODEL_DIR, "my_model.pth")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

DATASET_PATH = os.path.join(BASE_DIR, '4digitsonly.npy') # dataset path [[CHANGE THIS!]


# Model
class DownSampleBlock(nn.Module):
    """
    Downsample block as described.
    Essentially halfs the image dimensions.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2,
                               padding=1, padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, padding_mode='zeros')
        self.rconv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2,
                               padding=1, padding_mode='zeros')


    def forward(self, x: torch.Tensor):
        """
        Forward method.
        :param x: Input tensor
        :return: Transformed tensor
        """
        rx = x.clone() # residual
        rx = self.rconv(rx)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)

        return x + rx


class UpSampleBlock(nn.Module):
    """
    UpsampleBlock as described.
    Essentially doubles the image dimension.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.tconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, padding_mode='zeros')
        self.rtconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


    def forward(self, x: torch.Tensor):
        """
        Forward method.
        :param x: Input tensor
        :return: Transformed tensor
        """
        rx = x.clone()
        rx = self.rtconv(rx)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.tconv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)

        return x + rx


class RSubBlock(nn.Module):
    """
    RSubBlock as described in the paper. 
    Note convolutional layers are padded to maintain the same size across input and output.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, padding_mode='zeros')

    def forward(self, x: torch.Tensor):
        """
        Forward method.
        :param x: Input tensor
        :return: Transformed tensor
        """
        rx = x.clone() # residual
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)

        return x + rx


class RBlock(nn.Module):
    """
    RBlock that is comprised of 2 sub-RBlocks.
    It is clear from the paper that the output should be the original NxN with 1 channel,
    but it is not clear how K channels is compressed to 1 channel.

    RSubBlock presumably outputs dimension of (batch_size, K, N, N). K channels because notice there
    is a residual layer that connects the input (which has K channels) to RSubBlock to its output.

    So, a final convolution layer is added to compress to 1 output channel.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()
        self.rsub1 = RSubBlock(in_channels, in_channels)
        self.rsub2 = RSubBlock(in_channels, in_channels)
        self.output = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                                padding=1, padding_mode='zeros')

    def forward(self, x: torch.Tensor):
        """
        Forward method.
        :param x: Input tensor
        :return: Transformed tensor
        """
        x = self.rsub1(x)
        x = self.rsub2(x)
        x = self.output(x)
        return x


class SBlock(nn.Module):
    """
    SBlock as described.
    Note convolutional layers are padded to maintain the same size across input and output.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor):
        """
        Forward method.
        :param x: Input tensor
        :return: Transformed tensor
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        return x


class MultiscaleResNet(nn.Module):
    """
    Proposed ResNet model.
    K channels in the intermediate layers.
    """
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 N: int,
                 K: int
                 ):
        super().__init__()

        num_iter = math.log2(N)
        assert num_iter.is_integer(), f"N must be a power of 2 but it is {N}!"
        self.num_iter = int(num_iter)

        # IMPT: Torch does not recognise regular lists as containers for sub-modules!
        self.downsample_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        self.sblocks = nn.ModuleList()
        # add layers to the model, no. layers depend on image size
        for i in range(self.num_iter):
            if i == 0:
                self.downsample_blocks.append(
                    DownSampleBlock(in_channels=in_ch, out_channels=K)
                    )
                self.sblocks.append(
                    SBlock(in_channels=1, out_channels=1)
                    )
                self.upsample_blocks.append(
                    UpSampleBlock(in_channels=K, out_channels=K)
                )
            else:
                self.downsample_blocks.append(
                    DownSampleBlock(in_channels=K, out_channels=K)
                    )
                self.sblocks.append(
                    SBlock(in_channels=K, out_channels=K)
                    )
                # note in-channels 2K because prev layer output will be concatenated with skip layer
                self.upsample_blocks.append(
                    UpSampleBlock(in_channels=2*K, out_channels=K)
                    )
        self.rblock = RBlock(in_channels=K+1, out_channels=out_ch)

    def forward(self, x: torch.Tensor):
        """
        Forward method.
        :param x: Input tensor
        :return: Transformed tensor
        """
        skips = []
        for i in range(self.num_iter):
            sx = x.clone()  # clone for skip layer
            sx = self.sblocks[i](sx) # get result from skip layer
            skips.append(sx)

            x = self.downsample_blocks[i](x) # downsample

        for i in range(self.num_iter):
            # note 1st upsample block no need concat
            x = self.upsample_blocks[i](x) # upsample
            x = torch.cat([x, skips[-1-i]], dim=1) # concat along channel dim

        x = self.rblock(x) # final block

        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overloaded method to conform to ML terminology.
        """
        with torch.no_grad():
            return self.forward(x)


# Loading dataset and model
def load_model(model: MultiscaleResNet, load_path: str) -> None:
    """
    Load the model from a saved state.
    :param model: Model instance to load the state into
    :param load_path: Path to the saved model state
    """
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(load_path,
                                         map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(load_path))
    model.eval()


def load_data_for_eval(path: str) -> np.ndarray:
    """
    Loads dataset saved as npy format
    Note: Input should be 64x64

    :param path: data file path
    :return: image feature
    """
    assert path is not None, "Make sure dataset path exists!"
    features: np.ndarray = np.load(path)
    # assert format of path
    # make sure there's a batch dimension
    assert features.ndim == 3, "Data should have a batch dimension!"
    assert features.shape[1] == features.shape[2] == 64, \
        "Data does not have the correct dimensions."
    features /= 255.0 # PLEASE REMOVE ONCE FIXED!!!!!!!!!!!!!!!!!!!
    return features


def load_data_with_holo(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    A helper that loads original image and its hologram.
    Note: Input should be 64 x 128
    """
    assert path is not None, "Make sure dataset path exists!"
    data: np.ndarray = np.load(path)
    # assert format of path
    # make sure there's a batch dimension
    assert data.ndim == 3, "Data should have a batch dimension!"
    assert data.shape[1] == 64 and data.shape[2] == 128, \
        "Data does not have the correct dimensions."
    features = data[:, :, :64]
    holograms = data[:, :, 64:]
    return features, holograms


def prepare_data_for_evaluation(imgs: np.ndarray, scaler: BaseEstimator, device: torch.device):
    """
    Loads and scales the image.
    """
    imgs_reshaped = imgs.reshape(imgs.shape[0], -1) # reshape to 1D

    scaled_imgs = scaler.transform(imgs_reshaped)
    scaled_imgs = scaled_imgs.reshape(imgs.shape) # revert to original shape

    scaled_imgs = np.expand_dims(scaled_imgs, axis=1)
    scaled_imgs = torch.from_numpy(scaled_imgs).float().to(device)

    return scaled_imgs


# Utility Functions
def display_hologram_and_transformed(original: np.ndarray, 
                                     hologram: np.ndarray, 
                                     transformed: np.ndarray):
    """
    Utility function to display input, 
    hologram (model's output),
    and transformed (FFT on output) side-by-side.
    """
    assert hologram.ndim == 2 and transformed.ndim == 2, \
        "Ensure images have 2 dimensions before display!"

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(hologram, cmap='gray')
    plt.title('Generated Hologram')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(transformed, cmap='gray')
    plt.title('Applied FFT')
    plt.axis('off')

    plt.show()


def display_image(x: np.ndarray, title: str):
    """
    Utility function to display an (greyscale) image from numpy format
    """
    assert x.ndim == 2, "Ensure image has 2 dimensions before display!"
    plt.subplot(1, 1, 1)
    plt.imshow(x, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()



def apply_fresnel_propagation_torch_test(phase_patterns, wavelength=632.8e-9, pixel_pitch=36e-6, distance=0.13):
    """
    Some algorithm I am taking on faith. 
    Should be applied on GRESYCALE images only.
    Accounts for batch size.
    """
    # Convert to complex exponential
    complex_patterns = np.exp(1j * phase_patterns * 2 * np.pi)  # Phase patterns assumed to be in [0, 1]

    # Get dimensions
    N = phase_patterns.shape[-1]
    # Create spatial frequency coordinates
    fx = np.fft.fftfreq(N, pixel_pitch)
    fy = np.fft.fftfreq(N, pixel_pitch)
    FX, FY = np.meshgrid(fx, fy)

    # Convert NumPy array to PyTorch tensor
    FX = torch.tensor(FX, dtype=torch.float32)
    FY = torch.tensor(FY, dtype=torch.float32)
    complex_patterns_tensor = torch.tensor(complex_patterns, dtype=torch.complex64)
    # Compute 2D Fourier transform
    fft_result = torch.fft.fft2(complex_patterns_tensor)
    fft_result = torch.fft.fftshift(fft_result)

    # Compute the quadratic phase factor for Fresnel propagation
    H = torch.exp(-1j * torch.pi * wavelength * distance * (FX**2 + FY**2))
    H = torch.tensor(H, dtype=torch.complex64)

    # Apply the quadratic phase factor
    fourier_transformed = fft_result * H
    # Compute the magnitude (intensity pattern)
    magnitude_patterns = torch.abs(fourier_transformed) ** 2  # Intensity is the square of the magnitude

    # Normalize to the range 0-1
    magnitude_patterns = (magnitude_patterns - magnitude_patterns.min()) / (magnitude_patterns.max() - magnitude_patterns.min())
    
    # Convert the PyTorch tensor back to a NumPy array as float32
    magnitude_patterns = magnitude_patterns.cpu().numpy().astype(np.float32)
    assert not np.isnan(complex_patterns_tensor).any(), "stoopid"
    return magnitude_patterns


def apply_fresnel_propagation(phase_patterns: torch.tensor) -> torch.tensor:
    # Normalized phase
    phase_patterns = phase_patterns * 2 * torch.pi - torch.pi # between -pi and pi
    # Convert to complex exponential
    complex_patterns = 1/phase_patterns.shape[-1] * torch.exp(1j * phase_patterns)
    # Ensure it is in complex64
    complex_patterns = complex_patterns.to(torch.complex64)
    # Compute 2D Fourier transform
    fft_result = 1/phase_patterns.shape[-1] * torch.fft.fft2(complex_patterns)
    # Fourier shift
    fft_result = torch.fft.fftshift(fft_result, dim=(-2, -1))
    # Compute the magnitude (intensity pattern)
    magnitude_patterns = torch.abs(fft_result)
    # Convert the result back to a NumPy array and return
    magnitude_patterns = magnitude_patterns.numpy().astype(np.float32)

    magnitude_patterns = (magnitude_patterns-magnitude_patterns.min()) / (magnitude_patterns.max() - magnitude_patterns.min())

    return magnitude_patterns


# Main script
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Using Cuda..")
        device_ = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
    else:
        device_ = torch.device("cpu")

    loaded_model = MultiscaleResNet(1, 1, N=64, K=32)
    loaded_model.to(device_)
    load_model(loaded_model, MODEL_PATH)
    loaded_scaler: BaseEstimator = joblib.load(SCALER_PATH)




    images = load_data_for_eval(DATASET_PATH)
    print(np.min(images[0]), np.max(images[0]))
    features = prepare_data_for_evaluation(images, loaded_scaler, device_)

    print("Predictions:")
    predictions = loaded_model.predict(features)
    assert predictions.shape[0] == features.shape[0], "Something went really wrong.."

    # Remove the channel dimension
    predictions = np.squeeze(predictions, axis=1)
    print(predictions[0])
    transformed = apply_fresnel_propagation(predictions[0])
    # transformed[transformed > 0.6] = 0
    display_hologram_and_transformed(images[0],
                                     predictions[0],
                                     transformed)
    
    


    # BELOW IS FOR MY USE!
    # images, provided_holograms = load_data_with_holo(DATASET_PATH)
    # features = prepare_data_for_evaluation(images, loaded_scaler, device_)

    # print("Predictions:")
    # predictions = loaded_model.predict(features)
    # assert predictions.shape[0] == features.shape[0], "Something went really wrong.."

    # # Remove the channel dimension
    # predictions = np.squeeze(predictions, axis=1)
    # print(predictions[0])
    # display_hologram_and_transformed(images[0],
    #                                  predictions[0],
    #                                  apply_fresnel_propagation(predictions[0]))
    # display_image(provided_holograms[0], "Known Hologram")
