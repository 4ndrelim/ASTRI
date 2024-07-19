import torch
import torch.fft
import cv2
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(base_dir, 'illumination.png')


def fourier_transform(input: np.ndarray):
    """
    Performs Fourier transform on the given input.
    :return: Transformed input with real and imaginary parts in the first and second channel resp.
    """
    # Perform DFT using OpenCV
    dft = cv2.dft(np.float32(input), flags=cv2.DFT_COMPLEX_OUTPUT)
    # Apply shift of origin from upper left corner to center of image
    dft_shift = np.fft.fftshift(dft, axes=(0, 1))
    # Extract magnitude and phase images
    mag, phase = cv2.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1])
    # Raise magnitude to some power near 1
    mag = cv2.pow(mag, 1.1)
    # Convert magnitude and phase into cartesian real and imaginary components
    real, imag = cv2.polarToCart(mag, phase)


    return real, imag


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
    # Convert NumPy array to PyTorch tensor
    tensor = torch.tensor(image)
    return tensor


def torch_fourier_transform(input: torch.Tensor) -> torch.Tensor:
    """
    Performs Fourier transform on the given input batch.
    :param input: Input tensor of shape (batch_size, height, width)
    :return: Transformed tensor with shape (batch_size, 2, height, width) where 
             the first channel is the real part and the second channel is the imaginary part.
    """
    # Perform 2D Fourier transform
    dft = torch.fft.fft2(input)
    # Apply shift of origin from upper left corner to center of image
    dft_shift = torch.fft.fftshift(dft)
    
    # Get real and imaginary parts
    real = dft_shift.real
    imag = dft_shift.imag

    # Stack the real and imaginary parts along the channel dimension
    output = torch.stack((real, imag), dim=1)
    
    return output

# Example usage
print("-------------------------------------------")
input_tensor = load_image_as_tensor(image_path)
print("input shape: ", input_tensor.shape)
input_tensor = torch.unsqueeze(input_tensor, 0)
output_tensor = torch_fourier_transform(input_tensor)
print(output_tensor.shape)  # Should be (batch_size, 2, height, width)

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
real_cv2, imag_cv2 = fourier_transform(img)

# Now print both outputs
print("Real part from OpenCV: ", real_cv2)
print("Imaginary part from OpenCV: ", imag_cv2)
print("Real part from PyTorch: ", output_tensor[0, 0, :, :])
print("Imaginary part from PyTorch: ", output_tensor[0, 1, :, :])