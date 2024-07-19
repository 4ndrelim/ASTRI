"""
A neural network to simulate the Gerchberg-Saxton algorithm
"""
from typing import Callable, Optional, Tuple
import math
import os

import numpy as np
import torch
import torch.fft
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import joblib
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 initializer: Callable=lambda tensor: nn.init.kaiming_normal_(tensor, nonlinearity='relu')):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=3,
                               padding='same', padding_mode='circular')
        self.bn1 = nn.BatchNorm2d(10)

        self.conv2 = nn.Conv2d(10, 30, kernel_size=3,
                               padding='same', padding_mode='circular')
        self.bn2 = nn.BatchNorm2d(30)

        self.conv3 = nn.Conv2d(30, 30, kernel_size=3,
                               padding='same', padding_mode='circular')
        self.bn3 = nn.BatchNorm2d(30)

        self.conv4 = nn.Conv2d(30, 40, kernel_size=3,
                               padding='same', padding_mode='circular')
        self.bn4 = nn.BatchNorm2d(40)

        self.conv5 = nn.Conv2d(40, out_channels, kernel_size=3,
                               padding='same', padding_mode='circular')

        self._initialize_weights(initializer)

    def _initialize_weights(self, initializer: Callable):
        initializer(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)

        initializer(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        initializer(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

        initializer(self.conv4.weight)
        nn.init.zeros_(self.conv4.bias)

        initializer(self.conv5.weight)
        nn.init.zeros_(self.conv5.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(self.bn1(x)))
        x = F.relu(self.conv2(self.bn2(x)))
        x = F.relu(self.conv3(self.bn3(x)))
        x = F.relu(self.conv4(self.bn4(x)))
        x = self.conv5(x)
        return x


class LRGS(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, N: int, p: int):
        super().__init__()
        # in_channel should account for concatenated goal amplitudes
        self.forward_blocks = nn.ModuleList([ConvBlock(in_channel+1, 2*p) for _ in range(N)])   
        self.backward_blocks = nn.ModuleList([ConvBlock(2*p, out_channel) for _ in range(N)]) 
        self.N = N
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Handled by subblocks
        """
        pass

    def A_func(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def A_inverse_func(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    def p_transduce(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor, gg: torch.Tensor) -> torch.Tensor:
        for i in range(self.N):
            rx = x.clone()
            x = self.A_func(x)
            x = torch.cat([x, gg], dim=1) # concat along channel dimension
            x = self.forward_blocks[i](x)
            x = self.A_inverse_func(x)
            x = self.backward_blocks[i](x)
            x = self.p_transduce(x)
            x = x + rx # add residual
        return x

