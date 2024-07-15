"""
Evaluate using the trained NN
"""
import os
from typing import List
import numpy as np

import torch
from torch import nn

import joblib
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score

# USER CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# saved model directory, need scaler and model
MODEL_DIR = os.path.join(BASE_DIR, 'saved_model')
MODEL_PATH = os.path.join(MODEL_DIR, "my_model.pth")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

DATASET_PATH = None # dataset path [[CHANGE THIS!]


# Model
class PINN(nn.Module):
    """
    Physics-Informed Neural Network
    https://github.com/omniscientoctopus/Physics-Informed-Neural-Networks/tree/main/
    """

    # pylint: disable=too-many-arguments
    def __init__(self,
                 layers: List[List[int]],
                 f_hat: torch.Tensor,
                 device: torch.device,
                 activation: nn.Module=nn.ReLU(),
                 loss_function: nn.Module=nn.MSELoss(reduction='mean')
                 ):
        super().__init__()  # initialize from parent class
        self.device = device
        self.f_hat = f_hat.to(device)
        self.activation = activation
        self.loss_function = loss_function
        # PINN layers
        self.linears_a = nn.ModuleList(
            [nn.Linear(layers[0][i], layers[0][i+1]) for i in range(len(layers[0])-1)])
        self.linears_b = nn.ModuleList(
            [nn.Linear(layers[1][i], layers[1][i+1]) for i in range(len(layers[1])-1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate input through the layers of the model
        :param x: Input
        :return: Model's output
        """
        a = x[:, :]
        b = x[:, :3]

        # iteratively pass the results through each layer
        for layer in self.linears_a[:-1]:
            a = self.activation(layer(a))
        a = self.linears_a[-1](a) # no need activation function for the output

        # iteratively pass the results through each layer
        for layer in self.linears_b[:-1]:
            b = self.activation(layer(b))
        b = self.linears_b[-1](b) # no need activation function for the output

        return a * b

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overloaded method to conform to ML terminology.
        """
        with torch.no_grad():
            return self.forward(x)


# For loading of model
def load_model(model: PINN, load_path: str) -> None:
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
    Note: Features should only have 11 dimensions

    :param path: data file path
    :return: features
    """
    assert path, "Make sure dataset path exists!"
    features: np.ndarray = np.load(path)
    # assert format of path
    # make sure there's a batch dimension
    assert features.ndim == 2, "Data should have a batch dimension!"
    # check for 11 dimensions in data
    assert features.shape[1] == 11, "Data does not have the correct number of 11 dimensions."
    return features


def prepare_data_for_evaluation(features: np.ndarray, scaler: BaseEstimator, device: torch.device):
    """
    Loads and scales the features.
    """
    scaled_features = scaler.transform(features)
    scaled_features = torch.from_numpy(scaled_features).float().to(device)

    return scaled_features


# Main script
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Using Cuda..")
        device_ = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
    else:
        device_ = torch.device("cpu")

    f_hat_ = torch.zeros(128, 1)
    loaded_model = PINN([[11, 256, 64, 16, 3], [3, 128, 16, 3]], f_hat_, device=device_)
    loaded_model.to(device_)
    load_model(loaded_model, MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)

    features = load_data_for_eval(DATASET_PATH)
    data = prepare_data_for_evaluation(features, loaded_scaler, device_)

    print("Predictions:")
    predictions = loaded_model.predict(data)

    assert predictions.shape[0] == data.shape[0], "Something went really wrong.."

    print(predictions[0]) # just show first 1
