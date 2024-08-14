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

EPOCHS = 80
LEARNING_RATE = 0.003
TRAIN_BATCH_SIZE = 50
LOSS_FN = nn.MSELoss(reduction='mean')
INITIALIZER = lambda tensor: nn.init.kaiming_normal_(tensor, nonlinearity='relu')
GAMMA = 0.5
MILESTONES = [10, 25, 45, 70]