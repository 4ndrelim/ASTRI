import os
from typing import List

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.fft
import keras
import random

from utils import apply_fresnel_propagation_np, apply_gerchberg_saxton
from config import DTYPE_NP, IMAGE_SIZE


# USER CONFIG
NUM_IMAGES = 250000 # Number of images to generate
RATIOS =  [20000/NUM_IMAGES, 130000/NUM_IMAGES, 100000/NUM_IMAGES] # Ratios of random:icons:digits
FILENAME = "train.npy" # Name of the file to save the dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, '..', 'resources')
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
os.makedirs(DATA_DIR, exist_ok=True)
SAVE_PATH = os.path.join(DATA_DIR, FILENAME)


class DatasetGenerator:
    """
    A class for generating synthetic dataset.

    Saves in NP format because its more portable and uses less space than tensors.
    """
    def __init__(self, num_images: int, image_size: int, ratios: List[int], save_path: str, shuffle: bool=True):
        self.num_images = num_images
        self.image_size = image_size
        self.ratios = ratios
        self.save_path = save_path
        self.shuffle = shuffle

    def generate_icons(self, num_images: int):
        """
        Add variety to dataset. Use real life images from CIFAR.
        """
        iconset = os.path.join(RESOURCE_DIR, "icons_50.npy")
        data = np.load(iconset)

        data = np.transpose(data, (0, 2, 3, 1))
        # Convert to grayscale
        data_gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in data])
        # Resize images
        data_resized = np.array([cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) for img in data_gray])
        # Invert the grayscale images
        processed_data = 255 - data_resized
        # Randomly select num_images from processed_data
        indices = np.random.choice(processed_data.shape[0], num_images, replace=False)
        ret = processed_data[indices]
        return ret

    def generate_random_target_patterns(self, num_images: int):
        """
        Generate random target patterns.
        """
        random_intensities = np.random.randint(0, 256, (num_images, self.image_size, self.image_size), dtype=np.uint16)
        scale = np.sqrt(np.sum(random_intensities**2, axis=(-2,-1)))
        normalized = (1/scale)[:, np.newaxis, np.newaxis] * random_intensities.astype(DTYPE_NP)
        return normalized
    
    def generate_digits(self, num_images: int):
        """
        Geenrate 4 digits.
        """
        # load data
        (_, _), (images, _) = keras.datasets.mnist.load_data()
        images = images.astype(np.uint8)
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
            # Resize the image to desired size
            scaled_img = cv2.resize(img, (self.image_size, self.image_size))
            scaled_images[i] = scaled_img
      
        # normalize
        scaled_images = scaled_images.astype(DTYPE_NP)
        scale = np.sqrt(np.sum(scaled_images**2, axis=(-2,-1)))
        scale = scale[:, np.newaxis, np.newaxis]
        ret =  scaled_images / scale
        # assert np.all(ret >= 0) and np.all(ret <= 1), "Scaled images are not within the range [0, 1]"
        return ret

    def create_pattern_dataset(self, num_images: int, choice: int):
        """
        Creates dataset for a specified type.
        """
        if num_images == 0:
            # Return an empty array with appropriate shape if no images
            return np.zeros((0, self.image_size, self.image_size), dtype=DTYPE_NP)
 
        # target_patterns on left, phase_patterns on right
        if choice == 0:
            target_patterns = self.generate_random_target_patterns(num_images)
        elif choice == 1:
            target_patterns = self.generate_icons(num_images)
        elif choice == 2:
            target_patterns = self.generate_digits(num_images)
        else:
            assert False, "Not supported"

        return target_patterns


    def create_fulldataset(self):
        """
        Creates full dataset with the size of each type based on some predefined ratios.
        """
        num_random_images = int(self.ratios[0] * self.num_images)
        num__images = int(self.ratios[1] * self.num_images)
        num_digits_images = self.num_images - num_random_images - num__images
        to_concat = []
        if num_random_images > 0:
            random_targets = self.create_pattern_dataset(num_random_images, 0)
            print("shape of random_patterns: ", random_targets.shape)
            to_concat.append(random_targets)
        if num__images > 0:
            icon_targets = self.create_pattern_dataset(num__images, 1)
            print("shape of radian_patterns: ", icon_targets.shape)
            to_concat.append(icon_targets)
        if num_digits_images > 0:
            digits_targets = self.create_pattern_dataset(num_digits_images, 2)
            print("shape of digits_patterns: ", digits_targets.shape)
            to_concat.append(digits_targets)

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

    # Display a few examples
    # for i in range(min(NUM_IMAGES, 20)):
    #     target = data[i]
    #     hologram = apply_gerchberg_saxton(target)
    #     transformed = apply_fresnel_propagation_np(hologram)

    #     plt.subplot(1, 3, 1)
    #     plt.imshow(target, cmap='gray')
    #     plt.title(f'image_{i+1}')
    #     plt.axis('off')

    #     plt.subplot(1, 3, 2)
    #     plt.imshow(hologram, cmap='gray')
    #     plt.title(f'hologram_{i+1}')
    #     plt.axis('off')

    #     plt.subplot(1, 3, 3)
    #     plt.imshow(transformed, cmap='gray')
    #     plt.title(f'reconst{i+1}')
    #     plt.axis('off')

    #     print("Sq pixel values of target: ", np.sum(target**2))
    #     print("\n")

    #     plt.show()
