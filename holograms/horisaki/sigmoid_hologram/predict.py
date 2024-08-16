"""
Make predictions using trained model
"""
from typing import Tuple
import os

import numpy as np
import torch
import torch.fft

import matplotlib.pyplot as plt

import joblib
from sklearn.base import BaseEstimator, TransformerMixin

from model import MultiscaleResNet
from config import IMAGE_SIZE, DTYPE_NP, DTYPE_TORCH
from utils import apply_fresnel_propagation, normalize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'saved_model') # scaler and saved_model
MODEL_PATH = os.path.join(MODEL_DIR, "my_model.pth")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

DATASET_PATH = os.path.join(BASE_DIR, '10digits.npy') # dataset path [[CHANGE THIS!]

class IdentityScaler(BaseEstimator, TransformerMixin):
    """
    Dummy class that does not perform feature scaling.
    """
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X
    def fit_transform(self, X, y=None):
        return X
    def inverse_transform(self, X):
        return X
    
# Loading dataset and model
def load_model(model: MultiscaleResNet, load_path: str, device: torch.device) -> None:
    """
    Load the model from a saved state.
    :param model: Model instance to load the state into
    :param load_path: Path to the saved model state
    """
    model.load_state_dict(torch.load(load_path,
                                     map_location=device))
    model.to(device) # load above as was saved before attempting to convert
    model.eval()


def load_data(path: str) -> np.ndarray:
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
    return features


def prepare_data_for_evaluation(imgs: np.ndarray, scaler: BaseEstimator, device: torch.device):
    """
    Loads and scales the image.
    """
    imgs_reshaped = imgs.reshape(imgs.shape[0], -1) # reshape to 1D

    scaled_imgs = scaler.transform(imgs_reshaped)
    scaled_imgs = scaled_imgs.reshape(imgs.shape) # revert to original shape

    scaled_imgs = np.expand_dims(scaled_imgs, axis=1)
    scaled_imgs = torch.from_numpy(scaled_imgs).to(DTYPE_TORCH).to(device)

    return scaled_imgs


# Utility Functions
def display(original: np.ndarray,
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



# Main script
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Using Cuda..")
        device_ = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
    else:
        print("Using CPU..")
        device_ = torch.device("cpu")

    loaded_model = MultiscaleResNet(1, 1, N=IMAGE_SIZE, K=IMAGE_SIZE//2)
    load_model(loaded_model, MODEL_PATH, device_)
    loaded_scaler: BaseEstimator = joblib.load(SCALER_PATH)



    images = load_data(DATASET_PATH)
    features = prepare_data_for_evaluation(images, loaded_scaler, device_)

    print("Predictions:")
    predictions = loaded_model.predict(features)
    assert predictions.shape[0] == features.shape[0], "Something went really wrong.."

    # Remove the channel dimension
    print(predictions.shape)
    predictions = np.squeeze(predictions, axis=1)
    for i in range(predictions.shape[0]):
        print("Hologram pixel sum: ", torch.sum(predictions[i]))
        print(predictions[i])
        transformed = apply_fresnel_propagation(predictions[i]).cpu().numpy().astype(DTYPE_NP)
        transformed[transformed > 0.29] = 0
        display(images[i],
                predictions[i].cpu().numpy(),
                transformed)
