import os
from typing import List

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.fft
import keras

import random

from utils import apply_gerchberg_saxton, apply_fresnel_propagation_np, normalize
from config import DTYPE_NP, DTYPE_TORCH, IMAGE_SIZE


# USER CONFIG
NUM_IMAGES = 10 # Number of images to generate (40000, 100)
RATIOS =  [0, 0, 1] # Ratios of random:radial:digits
FILENAME = "10digits.npy" # Name of the file to save the dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
os.makedirs(DATA_DIR, exist_ok=True)
SAVE_PATH = os.path.join(DATA_DIR, FILENAME)


class DatasetGenerator:
    """
    A class for generating synthetic dataset.
    """
    def __init__(self, num_images: int, image_size: int, ratios: List[int], save_path: str, shuffle: bool=True):
        self.num_images = num_images
        self.image_size = image_size
        self.ratios = ratios
        self.save_path = save_path
        self.shuffle = shuffle

    def generate_radial(self, size: int, frequency: int, noise_std: float, random_val: float) -> np.ndarray:
        """
        Creates cool concentric circles (of widening gaps) with some noise.
        """
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

        normalized = normalize(pattern)
        return normalized


    def generate_radial_patterns(self, num_images: int):
        """
        Creates cool concentric circles with random configurations.
        """
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


    def generate_random_target_patterns(self, num_images: int):
        """
        Generate random target patterns.
        """
        random_intensities = np.random.randint(0, 256, (num_images, self.image_size, self.image_size), dtype=np.uint16)
        normalized = normalize(random_intensities)
        return normalized
        
    def generate_digits(self, num_images: int):
        """
        Geenrate 4 digits.
        """
        # load data
        (_, _), (images, _) = keras.datasets.mnist.load_data()
        # reshape to be [samples][width][height][channels]
        images = images.reshape(images.shape[0], 28, 28).astype('float32')
        n = len(images)
        scaled_images = np.zeros((num_images, self.image_size, self.image_size), dtype=np.uint8)
        # Repeat the process num_images times
        for i in range(num_images):
            random_indices = random.sample(range(n), 4)
            d1, d2, d3, d4 = random_indices
            img = np.vstack(
                (np.hstack((images[d1], images[d2])), 
                 np.hstack((images[d3], images[d4])))
                 )
            # Resize the image
            scaled_img = cv2.resize(img, (self.image_size, self.image_size))
            scaled_images[i] = scaled_img
      
        # normalize
        scaled_images = scaled_images.astype(DTYPE_NP)
        ret = normalize(scaled_images)
        assert np.all(ret >= 0) and np.all(ret <= 1), "Scaled images are not within the range [0, 1]"
        return ret
        
    def create_pattern_dataset(self, num_images: int, choice: int):
        """
        Creates dataset for a specified type.
        """
        if num_images == 0:
            # Return an empty array with appropriate shape if no images
            return np.zeros((0, self.image_size, self.image_size, 2), dtype=DTYPE_NP)
 
        # target_patterns on left, phase_patterns on right
        if choice == 0:
            target_patterns = self.generate_random_target_patterns(num_images)
            hologram_phase_patterns = apply_gerchberg_saxton(target_patterns)
        elif choice == 1:
            hologram_phase_patterns = self.generate_radial_patterns(num_images)
            target_patterns = apply_fresnel_propagation_np(hologram_phase_patterns)
        elif choice == 2:
            target_patterns = self.generate_digits(num_images)
            hologram_phase_patterns = apply_gerchberg_saxton(target_patterns)
        else:
            assert False, "Not supported"

        combined_patterns = np.concatenate((target_patterns, hologram_phase_patterns), axis=2)
        return combined_patterns


    def create_fulldataset(self):
        """
        Creates full dataset with the size of each type based on some predefined ratios.
        """
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
        """
        Saves the dataset.
        """
        dataset = self.create_fulldataset()
        np.save(self.save_path, dataset)
        print(f"Dataset shape: {dataset.shape}")
        return dataset
    

if __name__ == "__main__":
    generator = DatasetGenerator(NUM_IMAGES, IMAGE_SIZE, RATIOS, SAVE_PATH)
    data = generator.save_dataset()

    # # Display a few examples
    # for i in range(min(NUM_IMAGES, 20)):
    #     target = data[i][:, :IMAGE_SIZE]
    #     hologram = data[i][:, IMAGE_SIZE:]
    #     noise = np.random.normal(0, 0.2, hologram.shape)
    #     hologram1 = hologram + noise
    #     transformed = apply_fresnel_propagation_np(hologram)
    #     transformed1 = apply_fresnel_propagation_np(hologram1)
        
    #     plt.subplot(1, 4, 1)
    #     plt.imshow(target, cmap='gray')
    #     plt.title(f'image_{i+1}')
    #     plt.axis('off')

    #     plt.subplot(1, 4, 2)
    #     plt.imshow(hologram, cmap='gray')
    #     plt.title(f'hologram_{i+1}')
    #     plt.axis('off')

    #     plt.subplot(1, 4, 3)
    #     plt.imshow(transformed, cmap='gray')
    #     plt.title(f'reconst{i+1}')
    #     plt.axis('off')

    #     plt.subplot(1, 4, 4)
    #     plt.imshow(transformed1, cmap='gray')
    #     plt.title(f'reconst{i+1}')
    #     plt.axis('off')

    #     print("Pixel values original: ", np.sum(target**2))
    #     print("Pixel values hologram: ", np.sum(hologram))
    #     print("Pixel values fresnel: ", np.sum(transformed**2))
    #     print("\n")
        
    #     plt.show()
