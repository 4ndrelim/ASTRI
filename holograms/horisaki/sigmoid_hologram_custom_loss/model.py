"""
A residual NN model for Computer-Generated Holograms (CGH) based on Horisaki (2018)
"""

from typing import Callable
import math
import os

from config import DTYPE_TORCH
from utils import apply_fresnel_propagation, normalize

import numpy as np
import torch
import torch.fft
from torch import nn
import torch.nn.functional as F

from sklearn.base import BaseEstimator

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
        self.bn1 = nn.BatchNorm2d(in_channels).to(DTYPE_TORCH)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2,
                               padding=1, padding_mode='zeros').to(DTYPE_TORCH)
        self.bn2 = nn.BatchNorm2d(out_channels).to(DTYPE_TORCH)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, padding_mode='zeros').to(DTYPE_TORCH)
        self.rconv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2,
                               padding=1, padding_mode='zeros').to(DTYPE_TORCH)

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
        self.bn1 = nn.BatchNorm2d(in_channels).to(DTYPE_TORCH)
        self.tconv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size=2, stride=2).to(DTYPE_TORCH)
        self.bn2 = nn.BatchNorm2d(out_channels).to(DTYPE_TORCH)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, padding_mode='zeros').to(DTYPE_TORCH)
        self.rtconv = nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size=2, stride=2).to(DTYPE_TORCH)

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
        self.bn1 = nn.BatchNorm2d(in_channels).to(DTYPE_TORCH)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, padding_mode='zeros').to(DTYPE_TORCH)
        self.bn2 = nn.BatchNorm2d(out_channels).to(DTYPE_TORCH)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, padding_mode='zeros').to(DTYPE_TORCH)

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
                                padding=1, padding_mode='zeros').to(DTYPE_TORCH)

        # self._initialize_weights(initializer)
        self._initialize_weights(lambda tensor: nn.init.xavier_normal_(tensor))
        #self._initialize_weights(lambda tensor: nn.init.kaiming_normal_(tensor, mode='fan_in', nonlinearity='linear'))

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
                               padding=1, padding_mode='zeros').to(DTYPE_TORCH)
        self.bn1 = nn.BatchNorm2d(out_channels).to(DTYPE_TORCH)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, padding_mode='zeros').to(DTYPE_TORCH)
        self.bn2 = nn.BatchNorm2d(out_channels).to(DTYPE_TORCH)

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
        m = nn.Sigmoid() # was nn.tanh
        return m(x)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes loss based on model's criterion.
        :param x: Inputs
        :param y: Labels
        :return: loss value
        """
        return self.loss_fn(self.forward(x), y)
    
    def fresnel_loss(self, x: torch.Tensor, scaler: BaseEstimator) -> torch.Tensor:
        """
        Computes loss between FFT(prediction) and original based on model's criterion
        """
        # unscale x to get original image
        # Be careful here tho..
        # 1. need to clone, detach from computational graph, and shift to cpu and convert numpy format
        # reason: scikit-learn expect np arrays and np arrays operate on cpu data and
        cloned_x = x.clone().detach().cpu().numpy()
        # 2. squeeze to remove channel dim and reshape for scaler to unscale
        cloned_reshaped_x = np.squeeze(cloned_x, axis=1)
        cloned_reshaped_x = cloned_reshaped_x.reshape(cloned_reshaped_x.shape[0], -1)
        unscaled_x = scaler.inverse_transform(cloned_reshaped_x)
        # 3. shape it back and get tensor
        unscaled_x = np.expand_dims(unscaled_x, axis=1)
        unscaled_x = unscaled_x.reshape(x.shape)
        unscaled_x = torch.from_numpy(unscaled_x).to(DTYPE_TORCH).to(x.device)

        predictions = self.forward(x)

        z = apply_fresnel_propagation(predictions)
        normalized_z = normalize(z)
        loss2 = self.loss_fn(unscaled_x, normalized_z)

        return loss2
    

    def special_loss(self, x: torch.Tensor, y: torch.Tensor, scaler: BaseEstimator) -> torch.Tensor:
        """
        Computes weighted loss based on model's criterion and 
        loss between FFT(model's output) and original
        """
        # unscale x to get original image
        # Be careful here tho..
        # 1. need to clone, detach from computational graph, and shift to cpu and convert numpy format
        # reason: scikit-learn expect np arrays and np arrays operate on cpu data and
        cloned_x = x.clone().detach().cpu().numpy()
        # 2. squeeze to remove channel dim and reshape for scaler to unscale
        cloned_reshaped_x = np.squeeze(cloned_x, axis=1)
        cloned_reshaped_x = cloned_reshaped_x.reshape(cloned_reshaped_x.shape[0], -1)
        unscaled_x = scaler.inverse_transform(cloned_reshaped_x)
        # 3. shape it back and get tensor
        unscaled_x = np.expand_dims(unscaled_x, axis=1)
        unscaled_x = unscaled_x.reshape(x.shape)
        unscaled_x = torch.from_numpy(unscaled_x).to(DTYPE_TORCH).to(x.device)

        predictions = self.forward(x)
        # loss1 = self.loss_fn(predictions, y)
        mse_loss_fn = nn.MSELoss()
        loss1 = torch.sqrt(mse_loss_fn(predictions, y))

        z = apply_fresnel_propagation(predictions)
        normalized_z = normalize(z)
        loss2 = self.loss_fn(normalized_z, unscaled_x)

        return 0.7*loss1 + 0.3*loss2

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overloaded method to conform to ML terminology.
        """
        with torch.no_grad():
            return self.forward(x)

# def check_model_dtype(model, expected_dtype):
#     for name, param in model.named_parameters():
#         if param.dtype != expected_dtype:
#             print(f"Parameter {name} has dtype {param.dtype}, expected {expected_dtype}.")

#     for name, buffer in model.named_buffers():
#         if buffer.dtype != expected_dtype and name != 'num_batches_tracked':
#             print(f"Buffer {name} has dtype {buffer.dtype}, expected {expected_dtype}.")

# check_model_dtype(MultiscaleResNet(1, 1, 64, 32), DTYPE_TORCH)
