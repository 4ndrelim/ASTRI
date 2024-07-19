"""
Generate some lame data
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.fft


def translate(array, shift_x, shift_y):
    """
    Translate the array by shift_x and shift_y.
    
    Parameters:
    - array: Input 2D array.
    - shift_x: Number of pixels to shift along the x-axis.
    - shift_y: Number of pixels to shift along the y-axis.
    
    Returns:
    - Translated array.
    """
    translated_array = np.roll(array, shift_y, axis=0)  # Shift along y-axis
    translated_array = np.roll(translated_array, shift_x, axis=1)  # Shift along x-axis
    
    return translated_array

def generate_radial(size, frequency, min_val, max_val, noise_std, random_val):
    # Create a grid of coordinates
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    
    # Calculate the distance from the center
    radius = np.sqrt(xx**2 + yy**2)
    
    # generate based on thresholds
    if random_val <= 0.03:
        pattern = np.sin(frequency * np.pi * radius**2)
    elif random_val <= 0.03 + 0.04:
        pattern = np.sin(frequency * np.pi * np.exp(-radius))
    elif random_val <= 0.03 + 0.04 + 0.06:
        pattern = np.sin(frequency * np.pi * np.exp(-radius))
        attenuation = np.exp(-radius)
        pattern *= attenuation
    elif random_val <= 0.03 + 0.04 + 0.06 + 0.07:
        pattern = np.sin(frequency * np.pi * radius)
        attenuation = np.exp(-radius)
        pattern *= attenuation
    else:
        pattern = np.sin(frequency * np.pi * radius)

    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, pattern.shape)
    pattern += noise
    
    # Normalize to the range min_val to max_val
    pattern_min = pattern.min()
    pattern_max = pattern.max()
    pattern_normalized = (pattern - pattern_min) / (pattern_max - pattern_min)  # Normalize to [0, 1]
    pattern_normalized = pattern_normalized * (max_val - min_val) + min_val  # Scale to [min_val, max_val]
    
    # Treat as float by dividing by 255
    pattern_float = pattern_normalized / 255.0
    
    return pattern_float


def generate_radial_patterns(num_images, size):
    dataset = []
    for _ in range(num_images):
        # Random frequency and intensity range
        frequency = np.random.uniform(15, 25)  # Random frequency for variety
        min_val = np.random.randint(0, 105)   # Random minimum value for greyish black
        max_val = np.random.randint(155, 255)  # Random maximum value
        noise_std = np.random.uniform(0, 0.3)
        random_val = np.random.rand()
        
        pattern = generate_radial(size, frequency, min_val, max_val, noise_std, random_val)
        dataset.append(pattern)
    
    dataset = np.array(dataset)
    return dataset

# Parameters
num_images = 1000  # Number of images to generate
image_size = 64  # Size of each image (NxN)

# Generate the dataset
dataset = generate_radial_patterns(num_images, image_size)

# Save dataset as .npy file
# np.save('random_radial_patterns_with_noise.npy', dataset)

# Display a few examples
# for i in range(1):
#     print(dataset[i].shape)
#     plt.imshow(dataset[i], cmap='gray')
#     plt.axis('off')
#     plt.show()


# def load_as_tensor(x: np.ndarray):
#     image = x.astype(np.float32) / 255.0
#     tensor = torch.tensor(image)
#     tensor = torch.unsqueeze(tensor, 0) # add channel dim
#     return tensor


# def load_dataset(dataset):
#     tensors = [load_as_tensor(data) for data in dataset]
#     batch_tensor = torch.stack(tensors)
#     return batch_tensor


# Parameters
num_images = 10  # Number of images to generate
image_size = 64  # Size of each image (NxN)

def generate_random_phase_patterns(num_images, size):
    # Generate random intensity values between 0 and 255
    random_intensities = np.random.randint(0, 256, (num_images, size, size), dtype=np.uint8)
    # Normalize to the range 0 to 1
    random_phases = random_intensities.astype(np.float32) / 255.0
    return random_phases

def apply_fresnel_propagation(phase_patterns):
    # Convert to complex exponential
    complex_patterns = np.exp(1j * phase_patterns * 2 * np.pi)  # Phase patterns assumed to be in [0, 1]
    # Apply 2D Fourier transform and shift the zero frequency component to the center
    fourier_transformed = np.fft.fftshift(np.fft.fft2(complex_patterns))
    # Compute the magnitude (intensity pattern)
    magnitude_patterns = np.abs(fourier_transformed)
    # Normalize to the range 0-1
    magnitude_patterns = (magnitude_patterns - magnitude_patterns.min()) / (magnitude_patterns.max() - magnitude_patterns.min())
    return magnitude_patterns

def apply_fresnel_propagation_torch(phase_patterns, wavelength=632.8e-9, pixel_pitch=36e-6, distance=0.13):
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

def create_dataset(num_images, image_size):
    # Generate random phase patterns
    phase_patterns = generate_random_phase_patterns(num_images, image_size)
    # phase_patterns = generate_radial_patterns(num_images, image_size)
    
    # Apply Fresnel propagation to generate target patterns
    target_patterns = apply_fresnel_propagation_torch(phase_patterns)
    print(phase_patterns.shape, target_patterns.shape)
    return phase_patterns, target_patterns

# Generate the dataset
phase_patterns, target_patterns = create_dataset(num_images, image_size)

# Save dataset as .npy files
# np.save('phase_patterns.npy', phase_patterns)
# np.save('target_patterns.npy', target_patterns)

# Display a few examples
for i in range(10):
    plt.subplot(1, 2, 1)
    plt.imshow(phase_patterns[i], cmap='gray')
    plt.title('Phase Pattern')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(target_patterns[i], cmap='gray')
    plt.title('Target Pattern')
    plt.axis('off')
    
    plt.show()

# Helper function to load and convert dataset for PyTorch
def load_dataset(phase_file, target_file):
    # Load the datasets
    phase_patterns = np.load(phase_file)
    target_patterns = np.load(target_file)
    
    # Convert to float32 and normalize phase patterns to range 0-1
    phase_patterns = phase_patterns.astype(np.float32) / (2 * np.pi)
    target_patterns = target_patterns.astype(np.float32)
    
    # Convert to PyTorch tensors
    phase_tensors = torch.tensor(phase_patterns).unsqueeze(1)  # Add channel dimension
    target_tensors = torch.tensor(target_patterns).unsqueeze(1)  # Add channel dimension
    
    return phase_tensors, target_tensors

# Example usage
# phase_tensors, target_tensors = load_dataset('phase_patterns.npy', 'target_patterns.npy')
# print(phase_tensors.shape)  # Should be (num_images, 1, image_size, image_size)
# print(target_tensors.shape)  # Should be (num_images, 1, image_size, image_size)

