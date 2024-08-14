"""
Training code
"""

from typing import Optional, Tuple
import os

import numpy as np
import torch
import torch.fft
from torch.utils.data import DataLoader

from model import MultiscaleResNet
from config import EPOCHS, LEARNING_RATE, IMAGE_SIZE, LOSS_FN, DTYPE_NP, DTYPE_TORCH, INITIALIZER, GAMMA, MILESTONES
from utils import apply_fresnel_propagation, normalize

import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset') # dataset directory; assumed to exist
MODEL_DIR = os.path.join(BASE_DIR, 'saved_model')
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.npy")
TEST_PATH = os.path.join(DATA_DIR, "test.npy")
MODEL_PATH = os.path.join(MODEL_DIR, "my_model.pth") # can be None -> will not save
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")  # can be None; note special_loss needs scaler

# My own shit
TRACKED_NP = np.zeros((EPOCHS+1, IMAGE_SIZE, 3*IMAGE_SIZE), dtype=DTYPE_NP)
TRACKED_COUNTER = 0


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

# Utility functions
def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads dataset saved as npy format, and splits into features and labels.
    Note: Each data instance is 64 x 128 where the first 64x64 is feature image 
    and the next 64x64 is the label image.

    :param path: data file path
    :param N: Image size
    :return: feature and label numpy array depending on dataset_type
    """
    # assert format of path
    data = np.load(path).astype(DTYPE_NP)
    assert data.shape[2] == data.shape[1] * 2, "Col dimension should be twice that of row"

    features = data[:, :, :64] # take first 64 as features
    labels = data[:, :, 64:] # take last 64 as labels

    return features, labels


def feature_scaling(train: np.ndarray,
                    test: np.ndarray,
                    scaler: BaseEstimator,
                    save_path: Optional[str]=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Feature scale train and test set. Reshape to 1D to scale and the revert to 2D.
    :param train: numpy array of training features
    :param test: numpy array of test features
    :param scaler: Scaler, defaults to MinMaxScaler
    :param save_path: Saves scaler at this path if specified, else do nothing
    :return: scaled numpy array of training and testing features
    """
    # Reshape first to 1D to apply Scaler
    train_reshaped = train.reshape(train.shape[0], -1)
    test_reshaped = test.reshape(test.shape[0], -1)

    train_scaled = scaler.fit_transform(train_reshaped)
    test_scaled = scaler.transform(test_reshaped)

    # Reshape back
    train_scaled = train_scaled.reshape(train.shape)
    test_scaled = test_scaled.reshape(test.shape)

    # pylint: disable=line-too-long
    assert train.shape == train_scaled.shape, "Scaler should not change dimensions of train set."
    assert test.shape == test_scaled.shape, "Scaler should not change dimensions of test set."
    assert np.max(train_scaled) <= 1 + 0.1, "Does not support custom scaler that allows for training feature to exceed 1."
    assert np.min(train_scaled) >= 0, "Does not support custom scaler that allows for training feature below 0."

    # Clip the values to ensure they fall within [0, 1]
    # necessary even with scaling due to precision errors
    # train_scaled = np.clip(train_scaled, 0, 1)
    # test_scaled = np.clip(test_scaled, 0, 1)

    # save
    if save_path is not None:
        joblib.dump(scaler, save_path)

    return train_scaled, test_scaled

# pylint: disable=invalid-name
def prepare_data_for_training(train_path: str, test_path: str, dev: torch.device,
                              scaler: Optional[BaseEstimator]=MinMaxScaler(),
                              scaler_path: Optional[str]=None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Produce DataLoader of train, validation, and test sets, and store tensors at specified device.
    Note: 15% of train dataset to be used as validation dataset.
    :param train_path: Training set path
    :param test_path: Test set path
    :param dev: device where tensors are held
    :return: Train, validation, test DataLoaders
    """
    X_train, y_train = load_data(train_path)
    X_test, y_test = load_data(test_path)
 
    # perform scaling
    X_train_scaled, X_test_scaled = feature_scaling(X_train, X_test, scaler, save_path=scaler_path)

    # Add channel dimension: (num_samples, height, width) -> (num_samples, 1, height, width)
    X_train_scaled = np.expand_dims(X_train_scaled, axis=1)
    X_test_scaled = np.expand_dims(X_test_scaled, axis=1)
    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)

    # Convert to tensors and store at device
    X_train_scaled = torch.from_numpy(X_train_scaled).to(DTYPE_TORCH).to(dev)
    X_test_scaled = torch.from_numpy(X_test_scaled).to(DTYPE_TORCH).to(dev)
    y_train = torch.from_numpy(y_train).to(DTYPE_TORCH).to(dev)
    y_test = torch.from_numpy(y_test).to(DTYPE_TORCH).to(dev)

    # permuate indices for train and validation set later on
    indices = np.random.permutation(X_train_scaled.shape[0])
    train_len = int(X_train_scaled.shape[0] * 0.85) # 15% of dataset to be used for validation
    training_idx, val_idx = indices[:train_len], indices[train_len:]
    X_training_scaled, y_training = X_train_scaled[training_idx, :], y_train[training_idx, :]
    X_val_scaled, y_val = X_train_scaled[val_idx, :], y_train[val_idx, :]

    train_loader = DataLoader(list(zip(X_training_scaled, y_training)),
                              shuffle=True, batch_size=50, drop_last=True)
    val_loader = DataLoader(list(zip(X_val_scaled, y_val)),
                            shuffle=False, batch_size=128, drop_last=False)
    test_loader = DataLoader(list(zip(X_test_scaled, y_test)),
                             shuffle=False, batch_size=128, drop_last=False)

    return train_loader, val_loader, test_loader


# Main training and evaluation loop
# pylint: disable=too-many-locals, too-many-arguments, invalid-name
def train_model(
        model: MultiscaleResNet,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]=None,
        num_epochs: int=100,
        save_path: Optional[str]=None):
    """
    Trains the model.
    :param model: Model
    :param train_loader: Train set
    :param validation_loader: Validation set
    :param optimizer: Controls and apply gradient descent
    :Optional[param] scheduler: Adaptive adjustment of learning rate
    :param num_epochs: Number of epochs to run the model. Default of 100 as per paper
    :param save_path: Where the saved model is stored
    """

    model_state = None
    scaler = joblib.load(SCALER_PATH) # for the special_loss of the model

    print("epoch-----" +
          "Train Loss-----" + 
          "Val MAE Loss"
    )

    best_vloss_found = float('inf')

    for epoch in range(num_epochs):
        model.train() # set to training mode
        train_loss, batch_num = 0.0, 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            # loss = model.loss(X_batch, y_batch)
            # loss = model.fresnel_loss(X_batch, scaler)
            loss = model.special_loss(X_batch, y_batch, scaler)

            # apply update
            loss.backward()
            optimizer.step()
            # accum loss
            train_loss += loss.detach().cpu().item()
            batch_num += 1

        model.eval() # set to eval mode to evaluate on validation set
        with torch.no_grad(): # disable gradient
            val_loss = evaluate_model(model, validation_loader)
            user_msg = f"{epoch+1:03d}-------" + \
                       f"{train_loss / batch_num:.8f}-----" + \
                       f"{val_loss:.8f}"

            if val_loss < best_vloss_found:
                best_vloss_found = val_loss
                if save_path:
                    # inform the user better model is saved
                    user_msg += " Saving.."
                    model_state = model.state_dict()
                    assert model_state is not None, "Something went wrong while saving the model.."
                    torch.save(model_state, save_path)

            print(user_msg)
        # adjust lr after every epoch, if specified
        if scheduler:
            scheduler.step()


def evaluate_model(model: MultiscaleResNet,
                   data_loader: DataLoader) -> float:
    """
    Evaluate model on a dataset using MAE or MSE between original and reconstructed image.
    :param model: model
    :param data_loader: Test set
    :return: loss
    """
    scaler = joblib.load(SCALER_PATH) # for reversing
    model.eval()
    total_loss = 0

    #
    to_remove = 0
    with torch.no_grad():
        for X, y in data_loader:
            unscaled_X_cloned = X.clone().detach().cpu().numpy()
            unscaled_X_cloned = np.squeeze(unscaled_X_cloned, axis=1)
            unscaled_X_flattened = unscaled_X_cloned.reshape(unscaled_X_cloned.shape[0], -1)
            unscaled_X_flattened = scaler.inverse_transform(unscaled_X_flattened)
            unscaled_X = unscaled_X_flattened.reshape((X.shape[0], X.shape[-2], X.shape[-1]))
            unscaled_X = np.expand_dims(unscaled_X, axis=1) # channel dim

            predictions = model.forward(X)
            z = apply_fresnel_propagation(predictions)
            predictions_np = predictions.cpu().numpy()
            z_np = z.cpu().numpy()

            for i in range(unscaled_X.shape[0]):
                total_loss += mean_absolute_error(np.ravel(unscaled_X[i][0]),
                                                 np.ravel(normalize(z_np[i][0])))

            if to_remove < 1:
                # just take the first
                global TRACKED_COUNTER
                img = np.concatenate((unscaled_X[0][0], predictions_np[0][0], z_np[0][0]), axis=1)
                TRACKED_NP[TRACKED_COUNTER] = img
                original, hologram, transformed = unscaled_X[0][0], predictions_np[0][0], z_np[0][0]
                print("Predictions sum: ", np.sum(hologram))
                np.save(os.path.join(BASE_DIR, "train_results.npy"), TRACKED_NP)
                # plt.subplot(1,3,1)
                # plt.imshow(TRACKED_NP[TRACKED_COUNTER][:, :IMAGE_SIZE], cmap='gray')
                # plt.title('Original')
                # plt.axis('off')

                # plt.subplot(1,3,2)
                # plt.imshow(TRACKED_NP[TRACKED_COUNTER][:, IMAGE_SIZE:2*IMAGE_SIZE], cmap='gray')
                # plt.title('Predictions')
                # plt.axis('off')

                # plt.subplot(1,3,3)
                # plt.imshow(TRACKED_NP[TRACKED_COUNTER][:, 2*IMAGE_SIZE:], cmap='gray')
                # plt.title('Reconstructed')
                # plt.axis('off')

                # plt.show()

                # #histogram section
                # plt.subplot(1, 1, 1)
                # plt.hist(TRACKED_NP[TRACKED_COUNTER][:, IMAGE_SIZE:2*IMAGE_SIZE].flatten(), bins=100, edgecolor='black')
                # plt.title('Histogram Hologram')
                # plt.xlabel('Pixel Value')
                # plt.ylabel('Frequency')
                # plt.show()
                
                TRACKED_COUNTER = TRACKED_COUNTER + 1
                to_remove += 1

    num_samples = len(data_loader.dataset)
    average_loss = total_loss / num_samples
    return average_loss


# Main script
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Using cuda..")
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache() # clear cache (ADDED THIS)
    else:
        print("Using CPU..")
        device = torch.device("cpu")

    NO_SCALER = IdentityScaler()
    train_set, val_set, test_set = prepare_data_for_training(TRAIN_PATH, TEST_PATH,
                                                             device, NO_SCALER,
                                                             scaler_path=SCALER_PATH)

    # train_set, val_set, test_set = prepare_data_for_training(TRAIN_PATH, TEST_PATH,
    #                                                          device, scaler_path=SCALER_PATH)

    my_model = MultiscaleResNet(1, 1,
                                N=IMAGE_SIZE, K=IMAGE_SIZE//2,
                                initializer=INITIALIZER,
                                criterion=LOSS_FN)
    my_model = my_model.to(device)

    my_optimizer = torch.optim.AdamW(my_model.parameters(),
                                     lr=LEARNING_RATE)

    my_scheduler = torch.optim.lr_scheduler.MultiStepLR(my_optimizer,
                                                        milestones=MILESTONES, gamma=GAMMA)

    # print(sum(p.numel() for p in my_model.parameters())) see the size

    train_model(
        model = my_model,
        train_loader = train_set,
        validation_loader = val_set,
        optimizer = my_optimizer,
        scheduler= my_scheduler,
        save_path=MODEL_PATH,
        num_epochs=EPOCHS
        )

    print("Model trained. Evaluating on test set..")
    test_loss = evaluate_model(my_model, test_set)
    print(f"Loss on TEST set: {test_loss}")
