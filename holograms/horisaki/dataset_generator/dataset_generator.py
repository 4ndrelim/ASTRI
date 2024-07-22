import os
from typing import List

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.fft
import keras


# USER CONFIG
NUM_IMAGES = 9 # Number of images to generate
IMAGE_SIZE = 64 # Size of each image (NxN)
RATIOS =  [0.33, 0.33, 0.34] # Ratios of random:radial:digits
FILENAME = "t.npy" # Name of the file to save the dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'dataset')
os.makedirs(DATA_DIR, exist_ok=True)
SAVE_PATH = os.path.join(DATA_DIR, FILENAME)


class DatasetGenerator:
    def __init__(self, num_images: int, image_size: int, ratios: List[int], save_path: str, shuffle: bool=True):
        self.num_images = num_images
        self.image_size = image_size
        self.ratios = ratios
        self.save_path = save_path
        self.shuffle = shuffle

    def generate_radial(self, size: int, frequency: int, noise_std: float, random_val: float):
        # Create a grid of coordinates
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        xx, yy = np.meshgrid(x, y)
        
        # Calculate the distance from the center
        radius = np.sqrt(xx**2 + yy**2)
        
        # generate based on thresholds
        if random_val <= 0.02:
            pattern = np.sin(frequency * np.pi * radius**2)
        elif random_val <= 0.02 + 0.03:
            pattern = np.sin(frequency * np.pi * np.exp(-radius))
        elif random_val <= 0.02 + 0.03 + 0.05:
            pattern = np.sin(frequency * np.pi * np.exp(-radius))
            attenuation = np.exp(-radius)
            pattern *= attenuation
        elif random_val <= 0.02 + 0.03 + 0.05 + 0.06:
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
        
        return pattern_normalized
    

    def generate_radial_patterns(self, num_images: int):
        dataset = []
        for _ in range(num_images):
            # Random frequency and intensity range
            frequency = np.random.uniform(5, 25)  # Random frequency for variety
            noise_std = np.random.uniform(0, 0.3)
            random_val = np.random.rand()
    
            pattern = self.generate_radial(self.image_size, frequency, noise_std, random_val)
            dataset.append(pattern)
        dataset = np.array(dataset)
        return dataset
    

    def generate_random_phase_patterns(self, num_images: int):
        random_intensities = np.random.randint(0, 256, (num_images, self.image_size, self.image_size), dtype=np.uint8)
        return random_intensities.astype(np.float32) / 255.0
    
    def generate_digits(self, num_images: int):
        # load data
        (_, _), (images, _) = keras.datasets.mnist.load_data()
        # reshape to be [samples][width][height][channels]
        images = images.reshape(images.shape[0], 28, 28).astype('float32')
        n = len(images)
        scaled_images = np.zeros((num_images, self.image_size, self.image_size), dtype=np.uint8)
        # Repeat the process num_images times
        for x in range(num_images):
            d1 = np.random.randint(0, n)
            d2 = np.random.randint(0, n)
            while d1 == d2:
                d2= np.random.randint(0, n)
            d3 = np.random.randint(0, n)
            while d3 == d1 or d3 == d2:
                d3 = np.random.randint(0, n)
            d4 = np.random.randint(0, n)
            while d4 == d1 or d4 == d2 or d4 == d3:
                d4 = np.random.randint(0, n)
            img = np.vstack((np.hstack((images[d1], images[d2])), np.hstack((images[d3], images[d4]))))
            # Resize the image
            scaled_img = cv2.resize(img, (self.image_size, self.image_size))
            scaled_images[x] = scaled_img.reshape(self.image_size, self.image_size)

        ret = scaled_images.astype(np.float32) / 255.0 # make sure output between 0 and 1
        assert np.all(ret >= 0) and np.all(ret <= 1), "Scaled images are not within the range [0, 1]"
        return ret
    
    
    def apply_gerchberg_saxton(self, phase_patterns: np.ndarray) -> np.ndarray:
        iterative = 15
        aphase_estimate = torch.rand(self.image_size, self.image_size)
        # Convert phase_patterns into a torch tensor
        phase_patterns_tensor = torch.from_numpy(phase_patterns).float()
        phase_patterns_tensor = torch.fft.ifftshift(phase_patterns_tensor, dim=(-2, -1))
        known_abs_spatial = torch.ones(phase_patterns_tensor.shape)
        for _ in range(iterative):
            asignal_spatial = known_abs_spatial * torch.exp(1j * aphase_estimate)
            atemp_1 = 1/(self.image_size) * torch.fft.fft2(asignal_spatial)
            atemp_ang = atemp_1.angle()
            asignal_fourier = phase_patterns_tensor.mul(torch.exp(1j * atemp_ang))
            atemp_2 = torch.fft.ifft2(asignal_fourier)
            aphase_estimate = atemp_2.angle()
        return aphase_estimate/(2*torch.pi) + 0.5 # 0,1


    def apply_fresnel_propagation(self, phase_patterns: np.ndarray):
        # Convert to tensor equivalent
        phase_patterns = torch.from_numpy(phase_patterns).float()
        # Normalized phase, phase patterns assumed to be in [0, 1]
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
        
        assert not np.isnan(magnitude_patterns).any(), "Something went wrong.."
        return magnitude_patterns
    
    def apply_fresnel_propagation2(self, phase_patterns: np.ndarray):
        # Convert to tensor equivalent
        phase_patterns = torch.from_numpy(phase_patterns).float()
        # Normalized phase, phase patterns assumed to be in [0, 1]
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
        
        assert not np.isnan(magnitude_patterns).any(), "Something went wrong.."
        return magnitude_patterns

    def create_pattern_dataset(self, num_images: int, choice: int):
        if num_images == 0:
            # Return an empty array with appropriate shape if no images
            return np.zeros((0, self.image_size, self.image_size, 2), dtype=np.float32)
        
        # target_patterns on left, phase_patterns on right
        if choice == 0:
            hologram_phase_patterns = self.generate_random_phase_patterns(num_images)
            target_patterns = self.apply_fresnel_propagation(hologram_phase_patterns)
        elif choice == 1:
            hologram_phase_patterns = self.generate_radial_patterns(num_images)
            target_patterns = self.apply_fresnel_propagation(hologram_phase_patterns)
        elif choice == 2:
            target_patterns = self.generate_digits(num_images)
            hologram_phase_patterns = self.apply_gerchberg_saxton(target_patterns)
        else:
            assert False, "Not supported"

        combined_patterns = np.concatenate((target_patterns, hologram_phase_patterns), axis=2)
        return combined_patterns
    

    def create_fulldataset(self):
        num_random_images = int(self.ratios[0] * self.num_images)
        num_radial_images = int(self.ratios[1] * self.num_images)
        num_digits_images = self.num_images - num_random_images - num_radial_images
        to_concat = []
        if num_random_images > 0:
            random_patterns = self.create_pattern_dataset(num_random_images, 0)
            print("shape of random_patterns: ", random_patterns.shape)
            to_concat.append(random_patterns)
        if num_radial_images > 0:
            radian_patterns = self.create_pattern_dataset(num_radial_images, 1)
            print("shape of radian_patterns: ", radian_patterns.shape)
            to_concat.append(radian_patterns)
        if num_digits_images > 0:
            digits_patterns = self.create_pattern_dataset(num_digits_images, 2)
            print("shape of digits_patterns: ", digits_patterns.shape)
            to_concat.append(digits_patterns)

        dataset = np.concatenate(to_concat, axis=0)
        if self.shuffle:
            # Shuffle the dataset while maintaining pairs
            indices = np.arange(dataset.shape[0])
            np.random.shuffle(indices)
            dataset = dataset[indices]
            print("dataset shuffled")
    
        return dataset

    def save_dataset(self):
        dataset = self.create_fulldataset()
        np.save(self.save_path, dataset)
        print(f"Dataset shape: {dataset.shape}")
        return dataset
    

if __name__ == "__main__":
    generator = DatasetGenerator(NUM_IMAGES, IMAGE_SIZE, RATIOS, SAVE_PATH)
    # if shuffle, add False to end of function call
    data = generator.save_dataset()

    # # Display a few examples
    for i in range(min(NUM_IMAGES, 20)):
        plt.subplot(1, 4, 1)
        plt.imshow(data[i][:, :IMAGE_SIZE], cmap='gray')
        plt.title(f'image_{i+1}')
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(data[i][:, IMAGE_SIZE:], cmap='gray')
        plt.title(f'hologram_{i+1}')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(generator.apply_fresnel_propagation(data[i][:, IMAGE_SIZE:]), cmap='gray')
        plt.title(f'reconstructed_image_{i+1}')
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.imshow(generator.apply_fresnel_propagation2(data[i][:, IMAGE_SIZE:]), cmap='gray')
        plt.title(f'reconstructed_image_{i+1}')
        plt.axis('off')

        plt.show()