"""
A residual NN model for Computer-Generated Holograms (CGH)

# torch.arg(torch.ifft2d(img))
# np.abs(torch.fft2d(hologram*i))
"""

from typing import Callable, Optional, Tuple, Union
import math
import os

import numpy as np
import torch
import torch.fft
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

#pylint: disable=pointless-string-statement
"""
Down blocks need to pad input tensors with zero-padding of 1.

64 -> 32 (64+2-3 /2 + 1)
32 -> 16 (32+2-3 /2 + 1)
16 -> 8  (16+2-3 / 2 + 1)
8  -> 4
4  -> 2 
2  -> 1

Upsample via tconv
1  -> 2
2  -> 4
4  -> 8
8  -> 16
16  -> 32
32 -> 64
"""

# USER CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, 'dataset') # dataset directory
# saved model directory, need scaler and model
MODEL_DIR = os.path.join(BASE_DIR, 'saved_model')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.npy")
TEST_PATH = os.path.join(DATA_DIR, "test.npy")
MODEL_PATH = os.path.join(MODEL_DIR, "my_model.pth") # can be None -> will not save
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

IMAGE_SIZE = 64
LEARNING_RATE = 0.0005
EPOCHS = 20

# My own shit
TRACKED_NP = np.zeros((EPOCHS+1, IMAGE_SIZE, 3*IMAGE_SIZE), dtype=np.float32)
TRACKED_COUNTER = 0

# Utility functions
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
    data = np.load(path)
    assert data.shape[2] == data.shape[1] * 2, "Col dimension should be twice that of row"

    features = data[:, :, :64] # take first 64 as features
    labels = data[:, :, 64:] # take last 64 as labels

    return features, labels


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
    X_train, y_train = X_train[:2000], y_train[:2000]
    # perform scaling
    X_train_scaled, X_test_scaled = feature_scaling(X_train, X_test, scaler, save_path=scaler_path)

    # Add channel dimension: (num_samples, height, width) -> (num_samples, 1, height, width)
    X_train_scaled = np.expand_dims(X_train_scaled, axis=1)
    X_test_scaled = np.expand_dims(X_test_scaled, axis=1)
    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)

    # Convert to tensors and store at device
    X_train_scaled = torch.from_numpy(X_train_scaled).float().to(dev)
    X_test_scaled = torch.from_numpy(X_test_scaled).float().to(dev)
    y_train = torch.from_numpy(y_train).float().to(dev)
    y_test = torch.from_numpy(y_test).float().to(dev)

    # permuate indices for train and validation set later on
    indices = np.random.permutation(X_train_scaled.shape[0])
    train_len = int(X_train_scaled.shape[0] * 0.85) # 15% of dataset to be used for validation
    training_idx, val_idx = indices[:train_len], indices[train_len:]
    X_training_scaled, y_training = X_train_scaled[training_idx, :], y_train[training_idx, :]
    X_val_scaled, y_val = X_train_scaled[val_idx, :], y_train[val_idx, :]

    # set drop_last=True for consistent batch size during training
    train_loader = DataLoader(list(zip(X_training_scaled, y_training)),
                              shuffle=True, batch_size=50, drop_last=True)
    val_loader = DataLoader(list(zip(X_val_scaled, y_val)),
                            shuffle=False, batch_size=128, drop_last=False)
    test_loader = DataLoader(list(zip(X_test_scaled, y_test)),
                             shuffle=False, batch_size=128, drop_last=False)

    return train_loader, val_loader, test_loader


class DownSampleBlock(nn.Module):
    """
    Downsample block as described.
    Essentially halfs the image dimensions.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 initializer: Callable):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2,
                               padding=1, padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, padding_mode='zeros')
        self.rconv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2,
                               padding=1, padding_mode='zeros')

        self._initialize_weights(initializer)

    def _initialize_weights(self, initializer: Callable):
        """
        Initializes the weights for all convolutional layers in this block.
        Bias are initialized to 0.
        :param initializer: Callable that initializes the weights
        """
        # By default, conv layers should be declared with bias=True
        assert self.conv1.bias is not None and \
            self.conv2.bias is not None and \
            self.rconv.bias is not None, "Something went wrong!"

        initializer(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)

        initializer(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        initializer(self.rconv.weight)
        nn.init.zeros_(self.rconv.bias)

    def forward(self, x: torch.Tensor):
        """
        Forward method.
        :param x: Input tensor
        :return: Transformed tensor
        """
        rx = x.clone() # residual
        rx = self.rconv(rx)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)

        return x + rx


class UpSampleBlock(nn.Module):
    """
    UpsampleBlock as described.
    Essentially doubles the image dimension.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 initializer: Callable):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.tconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, padding_mode='zeros')
        self.rtconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self._initialize_weights(initializer)

    def _initialize_weights(self, initializer: Callable):
        """
        Initializes the weights for all convolutional layers in this block.
        Bias are initialized to 0.
        :param initializer: Callable that initializes the weights
        """
        # By default, conv layers should be declared with bias=True
        assert self.tconv1.bias is not None and \
            self.conv2.bias is not None and \
            self.rtconv.bias is not None, "Something went wrong!"

        initializer(self.tconv1.weight)
        nn.init.zeros_(self.tconv1.bias)

        initializer(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        initializer(self.rtconv.weight)
        nn.init.zeros_(self.rtconv.bias)


    def forward(self, x: torch.Tensor):
        """
        Forward method.
        :param x: Input tensor
        :return: Transformed tensor
        """
        rx = x.clone()
        rx = self.rtconv(rx)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.tconv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)

        return x + rx


class RSubBlock(nn.Module):
    """
    RSubBlock as described in the paper. 
    Note convolutional layers are padded to maintain the same size across input and output.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 initializer: Callable):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, padding_mode='zeros')

        self._initialize_weights(initializer)

    def _initialize_weights(self, initializer: Callable):
        """
        Initializes the weights for all convolutional layers in this block.
        Bias are initialized to 0.
        :param initializer: Callable that initializes the weights
        """
        # By default, conv layers should be declared with bias=True
        assert self.conv1.bias is not None and \
            self.conv2.bias is not None, "Something went wrong!"

        initializer(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)

        initializer(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor):
        """
        Forward method.
        :param x: Input tensor
        :return: Transformed tensor
        """
        rx = x.clone() # residual
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)

        return x + rx


class RBlock(nn.Module):
    """
    RBlock that is comprised of 2 sub-RBlocks.
    It is clear from the paper that the output should be the original NxN with 1 channel,
    but it is not clear how K channels is compressed to 1 channel.

    RSubBlock presumably outputs dimension of (batch_size, K, N, N). K channels because notice there
    is a residual layer that connects the input (which has K channels) to RSubBlock to its output.

    So, a final convolution layer is added to compress to 1 output channel.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 initializer: Callable):
        super().__init__()
        self.rsub1 = RSubBlock(in_channels, in_channels, initializer)
        self.rsub2 = RSubBlock(in_channels, in_channels, initializer)
        self.output = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                                padding=1, padding_mode='zeros')

        self._initialize_weights(initializer)

    def _initialize_weights(self, initializer: Callable):
        """
        Initializes the weights for all convolutional layers in this block.
        Bias are initialized to 0.
        :param initializer: Callable that initializes the weights
        """
        # By default, conv layers should be declared with bias=True
        assert self.output.bias is not None, "Something went wrong!"
        # r-subblocks should be initialized during creation
        initializer(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor):
        """
        Forward method.
        :param x: Input tensor
        :return: Transformed tensor
        """
        x = self.rsub1(x)
        x = self.rsub2(x)
        x = self.output(x)
        return x


class SBlock(nn.Module):
    """
    SBlock as described.
    Note convolutional layers are padded to maintain the same size across input and output.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 initializer: Callable):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(out_channels)

        self._initialize_weights(initializer)

    def _initialize_weights(self, initializer: Callable):
        """
        Initializes the weights for all convolutional layers in this block.
        Bias are initialized to 0.
        :param initializer: Callable that initializes the weights
        """
        # By default, conv layers should be declared with bias=True
        assert self.conv1.bias is not None and \
            self.conv2.bias is not None, "Something went wrong!"

        initializer(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)

        initializer(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor):
        """
        Forward method.
        :param x: Input tensor
        :return: Transformed tensor
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        return x


class MultiscaleResNet(nn.Module):
    """
    Proposed ResNet model.
    K channels in the intermediate layers.
    """
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 N: int,
                 K: int,
                 initializer: Callable=lambda tensor: nn.init.kaiming_normal_(tensor, nonlinearity='relu'),
                 criterion: nn.Module=nn.MSELoss(reduction='mean')
                 ):
        super().__init__()

        self.loss_fn = criterion

        num_iter = math.log2(N)
        assert num_iter.is_integer(), f"N must be a power of 2 but it is {N}!"
        self.num_iter = int(num_iter)

        # IMPT: Torch does not recognise regular lists as containers for sub-modules!
        self.downsample_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        self.sblocks = nn.ModuleList()
        # add layers to the model, no. layers depend on image size
        for i in range(self.num_iter):
            if i == 0:
                self.downsample_blocks.append(
                    DownSampleBlock(in_channels=in_ch, out_channels=K, initializer=initializer)
                    )
                self.sblocks.append(
                    SBlock(in_channels=1, out_channels=1, initializer=initializer)
                    )
                self.upsample_blocks.append(
                    UpSampleBlock(in_channels=K, out_channels=K, initializer=initializer)
                )
            else:
                self.downsample_blocks.append(
                    DownSampleBlock(in_channels=K, out_channels=K, initializer=initializer)
                    )
                self.sblocks.append(
                    SBlock(in_channels=K, out_channels=K, initializer=initializer)
                    )
                # note in-channels 2K because prev layer output will be concatenated with skip layer
                self.upsample_blocks.append(
                    UpSampleBlock(in_channels=2*K, out_channels=K, initializer=initializer)
                    )
        self.rblock = RBlock(in_channels=K+1, out_channels=out_ch, initializer=initializer)

        self._initialize_weights(initializer) # Redundant since weights initialized in sub-blocks

    def _initialize_weights(self, initializer: Callable):
        # Initialization handled by sub-blocks
        pass

    def forward(self, x: torch.Tensor):
        """
        Forward method.
        :param x: Input tensor
        :return: Transformed tensor
        """
        skips = []
        for i in range(self.num_iter):
            sx = x.clone()  # clone for skip layer
            sx = self.sblocks[i](sx) # get result from skip layer
            skips.append(sx)

            x = self.downsample_blocks[i](x) # downsample

        for i in range(self.num_iter):
            # note 1st upsample block no need concat
            x = self.upsample_blocks[i](x) # upsample
            x = torch.cat([x, skips[-1-i]], dim=1) # concat along channel dim

        x = self.rblock(x) # final block

        return x

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes loss based on model's criterion.
        :param x: Inputs
        :param y: Labels
        :return: loss value
        """
        return self.loss_fn(self.forward(x), y)
    

    def apply_fresnel_propagation(self, phase_patterns: torch.Tensor) -> torch.Tensor:
        """
        Just take this on faith.
        """
        # Normalized phase
        phase_patterns = phase_patterns * 2 * torch.pi - torch.pi # between -pi and pi
        # Convert to complex exponential
        complex_patterns = 1/phase_patterns.shape[-1] * torch.exp(1j * phase_patterns)
        # ensure it is in complex64
        complex_patterns = complex_patterns.to(torch.complex64)
        # Compute 2D Fourier transform
        fft_result = 1/phase_patterns.shape[-1] * torch.fft.fft2(complex_patterns)
        # Fourier shift
        fft_result = torch.fft.fftshift(fft_result, dim=(-2, -1))
        # Compute the magnitude (intensity pattern)
        magnitude_patterns = torch.abs(fft_result)

        return magnitude_patterns

    def special_loss(self, x: torch.Tensor, y: torch.Tensor, scaler: BaseEstimator) -> torch.Tensor:
        """
        Computes weighted loss based on model's criterion and 
        MSE difference between FFT(model's output) and original
        """
        # # unscale x to get original image
        # # Be careful here tho..
        # # 1. need to clone, detach from computational graph, and shift to cpu and convert numpy format
        # # reason: scikit-learn expect np arrays and np arrays operate on cpu data and
        # cloned_x = x.clone().detach().cpu().numpy()
        # # 2. squeeze to remove channel dim and reshape for scaler to unscale
        # cloned_reshaped_x = np.squeeze(cloned_x, axis=1)
        # cloned_reshaped_x = cloned_reshaped_x.reshape(cloned_reshaped_x.shape[0], -1)
        # unscaled_x = scaler.inverse_transform(cloned_reshaped_x)
        # # 3. shape it back and get tensor
        # unscaled_x = unscaled_x.reshape(x.shape[0], x.shape[-2], x.shape[-1])
        # unscaled_x = np.expand_dims(unscaled_x, axis=1)
        # unscaled_x = torch.from_numpy(unscaled_x).float().to(x.device)

        predictions = self.forward(x)
        loss1 = self.loss_fn(predictions, y)

        # z = self.apply_fresnel_propagation(predictions)
        # normalized_z, normalized_unscaled_x = normalize(z), normalize(unscaled_x)
        # loss2 = nn.SmoothL1Loss(reduction='mean', beta=0.15)(normalized_z, normalized_unscaled_x)

        return loss1


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
            loss = model.special_loss(X_batch, y_batch, scaler)
            # apply update
            loss.backward()
            optimizer.step()
            # accum loss
            train_loss += loss.detach().cpu().item()
            batch_num += 1

        model.eval() # set to eval mode to evaluate on validation set
        with torch.no_grad(): # disable gradient
            # val_loss = evaluate_model(model, validation_loader)
            val_loss = evaluate_model(model, validation_loader)
            user_msg = f"{epoch+1:03d}-------" + \
                       f"{train_loss / batch_num:.8f}-----" + \
                       f"{val_loss:.8f}"

            if val_loss < best_vloss_found:
                best_vloss_found = val_loss
                if save_path:
                    # inform the user better model is saved
                    user_msg += " Tracking.."
                    model_state = model.state_dict()

            print(user_msg)
        # adjust lr after every epoch, if specified
        if scheduler:
            scheduler.step()
    # save model
    if save_path:
        assert model_state is not None, "Something went wrong while saving the model.."
        torch.save(model_state, save_path)


# pylint: disable=invalid-name
def evaluate_model(model: MultiscaleResNet,
                   data_loader: DataLoader) -> float:
    """
    Evaluate model on a dataset using MSE.
    :param model: model
    :param data_loader: Test set
    :return: loss, r2 score
    """
    scaler = joblib.load(SCALER_PATH) # for reversing
    model.eval()
    total_mae = 0.0 # use MAE as a metric

    #
    to_remove = 0
    with torch.no_grad():  # disable gradient computation
        for X, y in data_loader:
            # loss += model.loss(X, y).detach().cpu().item()
            predictions = model.forward(X)
            # reshape
            y_np = y.cpu().numpy().reshape(y.shape[0], -1)
            predictions_np = predictions.cpu().numpy().reshape(predictions.shape[0], -1)
            # accumulate MAE
            for i in range(y_np.shape[0]):
                total_mae += mean_absolute_error(y_np[i], predictions_np[i])
                # total_mae += np.sum(np.abs(y_np[i] - predictions_np[i]))


            unscaled_X_cloned = X.clone().detach().cpu().numpy()
            unscaled_X_cloned = np.squeeze(unscaled_X_cloned, axis=1)
            unscaled_X_flattened = unscaled_X_cloned.reshape(unscaled_X_cloned.shape[0], -1)
            unscaled_X_flattened = scaler.inverse_transform(unscaled_X_flattened)
            unscaled_X = unscaled_X_flattened.reshape((X.shape[0], X.shape[-2], X.shape[-1]))
            unscaled_X = np.expand_dims(unscaled_X, axis=1) # channel dim
            z = model.apply_fresnel_propagation(predictions)
            predictions_np = predictions.cpu().numpy()
            z_np = z.cpu().numpy()

            if to_remove < 1:
                # just take the first
                global TRACKED_COUNTER
                img = np.concatenate((unscaled_X[0][0], predictions_np[0][0], z_np[0][0]), axis=1)
                TRACKED_NP[TRACKED_COUNTER] = img
                # plt.subplot(1,3,1)
                # plt.imshow(TRACKED_NP[TRACKED_COUNTER][:, :IMAGE_SIZE], cmap='gray')
                # plt.title('Original')
                # plt.axis('off')

                # plt.subplot(1,3,2)
                # plt.imshow(TRACKED_NP[TRACKED_COUNTER][:, IMAGE_SIZE:2*IMAGE_SIZE], cmap='gray')
                # plt.title('Predictions')
                # plt.axis('off')

                # plt.subplot(1,3,3)
                # transformed = TRACKED_NP[TRACKED_COUNTER][:, 2*IMAGE_SIZE:]
                # transformed[transformed > 0.5] = 0
                # plt.imshow(transformed, cmap='gray')
                # plt.title('Reconstructed')
                # plt.axis('off')

                # plt.show()
                TRACKED_COUNTER = TRACKED_COUNTER + 1
                to_remove += 1

    num_samples = len(data_loader.dataset)
    average_mae = total_mae / num_samples
    return average_mae


# Main script
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Using cuda..")
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    # NO_SCALER = IdentityScaler()
    # train_set, val_set, test_set = prepare_data_for_training(TRAIN_PATH, TEST_PATH,
    #                                                          device, NO_SCALER,
    #                                                          scaler_path=SCALER_PATH)

    train_set, val_set, test_set = prepare_data_for_training(TRAIN_PATH, TEST_PATH,
                                                             device, scaler_path=SCALER_PATH)

    my_model = MultiscaleResNet(1, 1, N=IMAGE_SIZE, K=IMAGE_SIZE//2, 
                                criterion=nn.SmoothL1Loss(reduction='mean', beta=0.15))
    my_model = my_model.to(device)

    my_optimizer = torch.optim.AdamW(my_model.parameters(),
                                     lr=LEARNING_RATE)

    my_scheduler = torch.optim.lr_scheduler.MultiStepLR(my_optimizer,
                                                        milestones=[10, 20, 25, 30], gamma=0.5)

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
    test_mae1 = evaluate_model(my_model, test_set)
    print(f"MAE on TEST set: {test_mae1}")
