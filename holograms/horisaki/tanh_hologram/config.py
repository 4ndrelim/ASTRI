"""
For setting file-level configuration
"""
import torch
from torch import nn
import numpy as np

# Ensure the below the same!
DTYPE_TORCH = torch.float64
DTYPE_NP = np.float64

IMAGE_SIZE = 64

EPOCHS = 50
LEARNING_RATE = 0.0016
LOSS_FN = nn.SmoothL1Loss(reduction='mean', beta=0.22)
INITIALIZER = lambda tensor: nn.init.kaiming_normal_(tensor, nonlinearity='relu')
GAMMA = 0.5
MILESTONES = [7, 15, 25, 35]