import torch
import torch.fft
import cv2
import numpy as np
import os

# Define the base directory and image path
base_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(base_dir, 'stupid_guy.jpg')


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
    Performs Fourier transform on the given input.
    :param input: Input tensor of shape (height, width)
    :return: Transformed tensor with shape (2, height, width) where 
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
    output = torch.stack((real, imag), dim=0)
    
    return output


def torch_inverse_fourier_transform(input: torch.Tensor) -> torch.Tensor:
    """
    Performs inverse Fourier transform on the given input.
    :param input: Transformed tensor with shape (2, height, width)
    :return: Reconstructed tensor with shape (height, width)
    """
    # Extract real and imaginary parts
    real = input[0, :, :]
    imag = input[1, :, :]

    # Combine real and imaginary parts into a complex tensor
    complex_tensor = torch.complex(real, imag)
    # Apply inverse shift of origin
    dft_shift = torch.fft.ifftshift(complex_tensor)
    # Perform 2D inverse Fourier transform
    idft = torch.fft.ifft2(dft_shift)
    # Take the real part of the inverse transform
    output = idft.real
    
    return output


# Example usage
input_tensor = load_image_as_tensor(image_path)
print("Input shape: ", input_tensor.shape)

# Perform Fourier transform
fourier_transformed = torch_fourier_transform(input_tensor)
print("Fourier transformed shape: ", fourier_transformed.shape)

# Perform inverse Fourier transform
reconstructed_tensor = torch_inverse_fourier_transform(fourier_transformed)
print("Reconstructed shape: ", reconstructed_tensor.shape)

# Compare the original and reconstructed images
original_image = input_tensor.numpy()
reconstructed_image = reconstructed_tensor.numpy()

# Display original and reconstructed images using OpenCV
cv2.imshow("Original Image", original_image)
cv2.imshow("Reconstructed Image", reconstructed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# print(original_image)
# print("--------------------------")
# print(reconstructed_image)
