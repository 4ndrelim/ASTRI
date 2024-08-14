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

EPOCHS = 85 
LEARNING_RATE = 0.002
LOSS_FN = nn.SmoothL1Loss(reduction='mean', beta=0.13) 
# LOSS_FN = nn.MSELoss(reduction='mean')
INITIALIZER = lambda tensor: nn.init.kaiming_normal_(tensor, nonlinearity='relu')
GAMMA = 0.5
MILESTONES = [3, 10, 25, 40, 60]


####################### USELESS ####################### 
class CustomLoss(nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        diff = torch.abs(x - target)
        beta = self.beta
        # Quadratic term for small differences
        quadratic = torch.where(diff < beta, 0.5 * (diff ** 2) / beta, torch.zeros_like(diff))
        # Linear term for large differences
        linear = torch.where(diff >= beta, diff - 0.5 * beta, torch.zeros_like(diff))
        # Combine the loss
        loss = quadratic + linear
        return loss.mean()
