"""
Bunch of utils function used across the codebase.
"""

import numpy as np
import torch
import torch.fft

from typing import Union
from config import IMAGE_SIZE, DTYPE_NP, DTYPE_TORCH


########## FOURIER TRANSFORM FUNCTIONS ##########
def apply_gerchberg_saxton(phase_patterns: np.ndarray) -> np.ndarray:
    """
    Generates holograms from targets.
    """
    iterative = 10
    xx, yy = np.meshgrid(np.linspace(-np.pi, np.pi, IMAGE_SIZE), np.linspace(-np.pi, np.pi, IMAGE_SIZE))
    initial_phase = np.cos(xx) + np.cos(yy)
    aphase_estimate = torch.tensor(initial_phase, dtype=DTYPE_TORCH)
    # Convert phase_patterns into a torch tensor
    phase_patterns_tensor = torch.from_numpy(phase_patterns).to(DTYPE_TORCH)
    # phase_patterns_tensor = torch.fft.ifftshift(phase_patterns_tensor, dim=(-2, -1))
    known_abs_spatial = torch.ones(phase_patterns_tensor.shape)
    for _ in range(iterative):
        asignal_spatial = known_abs_spatial * torch.exp(1j * aphase_estimate)
        atemp_1 = 1/(IMAGE_SIZE) * torch.fft.fft2(asignal_spatial, dim=(-2,-1))
        atemp_ang = atemp_1.angle()
        asignal_fourier = phase_patterns_tensor.mul(torch.exp(1j * atemp_ang))
        atemp_2 = torch.fft.ifft2(asignal_fourier)
        aphase_estimate = atemp_2.angle()
    res = aphase_estimate/(2*torch.pi) + 0.5
    return res.numpy().astype(DTYPE_NP)

def apply_fresnel_propagation(phase_patterns: torch.Tensor) -> torch.Tensor:
    """
    Just take this on faith.
    """
    # Normalized phase
    phase_patterns = (phase_patterns - 0.5) * 2*torch.pi # between -pi and pi
    # Convert to complex exponential
    complex_patterns = 1/phase_patterns.shape[-1] * torch.exp(1j * phase_patterns)
    # ensure it is in complex64
    complex_patterns = complex_patterns.to(torch.complex64)
    # Compute 2D Fourier transform
    fft_result = 1/phase_patterns.shape[-1] * torch.fft.fft2(complex_patterns)
    # Fourier shift
    # fft_result = torch.fft.fftshift(fft_result, dim=(-2, -1))
    # Compute the magnitude (intensity pattern)
    magnitude_patterns = torch.abs(fft_result)
    return magnitude_patterns.to(DTYPE_TORCH)

def apply_fresnel_propagation_np(phase_patterns: np.ndarray) -> np.ndarray:
    """
    Same as apply_fresnel_propagation but handles in numpy format.
    """
    phase_patterns = torch.from_numpy(phase_patterns).to(DTYPE_TORCH)
    magnitude_patterns = apply_fresnel_propagation(phase_patterns)

    # Convert the result back to a NumPy array and return
    magnitude_patterns = magnitude_patterns.numpy().astype(DTYPE_NP)
    assert not np.isnan(magnitude_patterns).any(), "Something went wrong.."
    return magnitude_patterns

def apply_fresnel_propagation_np_normalized(phase_patterns: np.ndarray) -> np.ndarray:
    """
    Same as apply_fresnel_propagation_NP but normalizes to [0, 1].
    """
    magnitude_patterns = apply_fresnel_propagation_np(phase_patterns)
    # return normalized magnitude_patterns
    return normalize(magnitude_patterns)


def apply_fresnel_propagation_torch_test(phase_patterns, wavelength=632.8e-9, pixel_pitch=36e-6, distance=0.13):
    """
    Some algorithm I am taking on faith with the power of GPT. But this is unused lol..
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


########## NORMALIZATION STUFF / FOR DISPLAY ##########
def normalize(data: Union[np.ndarray, torch.Tensor]):
    """
    Scales to between 0 and 1 for more stable comparison.
    """
    if isinstance(data, np.ndarray):
        data_min = data.min(axis=(-2,-1), keepdims=True)
        data_max = data.max(axis=(-2,-1), keepdims=True)
        normalized_data = (data - data_min) / (data_max - data_min + 1e-8)
    elif isinstance(data, torch.Tensor):
        data_min = data.amin(dim=(-2, -1), keepdim=True)
        data_max = data.amax(dim=(-2, -1), keepdim=True)
        normalized_data = (data - data_min) / (data_max - data_min + 1e-8)
    else:
        assert False, "Not supported data type"

    return normalized_data

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    hist, bins = np.histogram(image.flatten(), bins=1024, range=[0, 1])
    # Calculate cumulative distribution function (CDF)
    cdf = hist.cumsum()
    print(cdf)
    cdf_normalized = cdf / cdf[-1]  # Normalize to [0, 1]
    # Use linear interpolation of the CDF to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    # Reshape the flat array back to the original image shape
    image_equalized = image_equalized.reshape(image.shape)
    
    return image_equalized

def threshold_image(im, th):
    thresholded_img = np.zeros(im.shape)
    thresholded_img[im > th] = im[im > th]
    return thresholded_img

def compute_otsu_criteria(im, th):
    thresholded_img = threshold_image(im, th)
    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(thresholded_img)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1-weight1
    if weight1 == 1 or weight0 == 0:
        return np.inf
    val_pixels1 = im[thresholded_img == im]
    val_pixels0 = im[thresholded_img == 0]
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
    return weight0*var0 + weight1*var1

def find_best_threshold(im):
    threshold_range = range(np.max(im)+1)
    criterias = [compute_otsu_criteria(im, th) for th in threshold_range]
    best_threshold = threshold_range[np.argmin(criterias)]
    return best_threshold
