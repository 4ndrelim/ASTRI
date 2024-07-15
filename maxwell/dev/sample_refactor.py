import torch
from torch.utils.data import DataLoader
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import numpy as np
import os
import math
from typing import Tuple, List

# Utility functions

def load_data_forGridSearch(path: str, dataset_type: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(os.path.join(path, f"{dataset_type}.npy"))
    feature = data[:, :-3]
    label = data[:, -3:]

    assert feature.shape[1] == 11, "Imported features do not have the correct dimension"
    assert label.shape[1] == 3, "Imported labels do not have the correct dimension"

    return feature, label

def feature_scaling_forGridSearch(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    minmax_scale = MinMaxScaler().fit(train)
    train_transformed = minmax_scale.transform(train)
    test_transformed = minmax_scale.transform(test)

    assert train.shape == train_transformed.shape, "Feature dimension for the training set is changed due to scaling"
    assert test.shape == test_transformed.shape, "Feature dimension for the testing set is changed due to scaling"
    assert np.max(train_transformed) <= 1 + math.exp(-9), "The max. value of the training feature exceeds 1"
    assert np.min(train_transformed) >= 0, "The min. value of the training feature belows 0"

    return train_transformed, test_transformed

class NeuralNetwork(nn.Module):
    def __init__(self, layers: List[List[int]], f_hat: torch.Tensor, device: torch.device):
        super(NeuralNetwork, self).__init__()
        self.device = device
        self.f_hat = f_hat.to(device)
        self.activation = nn.ReLU()
        self.loss_function = nn.MSELoss(reduction='mean')
        
        self.linears_a = nn.ModuleList([nn.Linear(layers[0][i], layers[0][i+1]) for i in range(len(layers[0])-1)])
        self.linears_b = nn.ModuleList([nn.Linear(layers[1][i], layers[1][i+1]) for i in range(len(layers[1])-1)])
        
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.linears_a:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in self.linears_b:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x[:, :]
        b = x[:, :3]

        for layer in self.linears_a[:-1]:
            a = self.activation(layer(a))
        a = self.linears_a[-1](a)

        for layer in self.linears_b[:-1]:
            b = self.activation(layer(b))
        b = self.linears_b[-1](b)

        return a * b

    def loss_bc(self, x_bc: torch.Tensor, y_bc: torch.Tensor) -> torch.Tensor:
        return self.loss_function(self.forward(x_bc), y_bc)

    def loss_pde(self, x_pde: torch.Tensor) -> torch.Tensor:
        g = x_pde.clone()
        g.requires_grad = True
        f = self.forward(g)

        fx_x_y_z = torch.autograd.grad(f[:, 0].reshape(-1, 1), g, grad_outputs=torch.ones([g.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
        fy_x_y_z = torch.autograd.grad(f[:, 1].reshape(-1, 1), g, grad_outputs=torch.ones([g.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
        fz_x_y_z = torch.autograd.grad(f[:, 2].reshape(-1, 1), g, grad_outputs=torch.ones([g.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]

        divergence = self.loss_function(fx_x_y_z[:, 0] + fy_x_y_z[:, 1] + fz_x_y_z[:, 2], self.f_hat)
        curl_x = self.loss_function(fz_x_y_z[:, 1] - fy_x_y_z[:, 2], self.f_hat)
        curl_y = self.loss_function(fx_x_y_z[:, 2] - fz_x_y_z[:, 0], self.f_hat)
        curl_z = self.loss_function(fy_x_y_z[:, 0] - fx_x_y_z[:, 1], self.f_hat)

        return divergence + curl_x + curl_y + curl_z

    def loss(self, x_bc: torch.Tensor, y_bc: torch.Tensor, x_pde: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_bc = self.loss_bc(x_bc, y_bc)
        loss_pde = self.loss_pde(x_pde)
        return loss_bc + loss_pde * alpha, loss_bc, loss_pde

# Main training and evaluation loop

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, device: torch.device, max_epochs: int = 32, lr: float = 0.001, alpha: float = 0.01):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

    best_R2 = [0, 0, 0, 0, 0, 0]
    for epoch in range(max_epochs):
        model.train()
        total_loss, total_bc, total_pde, count_iter = 0.0, 0.0, 0.0, 0.0
        for X, Y in train_loader:
            X, Y = X.to(device).float(), Y.to(device).float()
            optimizer.zero_grad()
            loss, loss_bc, loss_pde = model.loss(X, Y, X, alpha=alpha)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bc += loss_bc.item()
            total_pde += loss_pde.item()
            count_iter += 1

        model.eval()
        with torch.no_grad():
            val_loss, R2 = evaluate_model(model, val_loader, device)
            if R2 >= best_R2[5]:
                best_R2 = [epoch, total_loss / count_iter, total_bc / count_iter, total_pde / count_iter, val_loss, R2]
                if best_R2[3] * alpha > best_R2[2] * 0.5:
                    alpha *= 0.1
                test_loss, test_R2 = evaluate_model(model, test_loader, device)
                print(f'{epoch:03d}-------{best_R2[1]:.8f}--------{best_R2[2]:.8f}---{best_R2[3]:.8f}-----{best_R2[4]:.8f}---{best_R2[5]:.5f}------{test_loss:.8f}----{test_R2:.5f}')
            else:
                print(f'{epoch:03d}-------{total_loss / count_iter:.8f}--------{total_bc / count_iter:.8f}---{total_pde / count_iter:.8f}-----{val_loss:.8f}---{R2:.5f}')
        scheduler.step()

def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    total_loss, R2, total_count = 0.0, 0.0, 0.0
    for X, Y in data_loader:
        X, Y = X.to(device).float(), Y.to(device).float()
        total_loss += model.loss_bc(X, Y).item()
        Y_pred = model(X)
        R2 += r2_score(y_true=Y.cpu().numpy(), y_pred=Y_pred.cpu().numpy()) * len(Y)
        total_count += len(Y)
    return total_loss / len(data_loader), R2 / total_count

# Data preparation

def prepare_data(path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    X_train, y_train = load_data_forGridSearch(path, "train")
    X_test, y_test = load_data_forGridSearch(path, "test")
    X_train_transformed, X_test_transformed = feature_scaling_forGridSearch(X_train, X_test)

    indices = np.random.permutation(X_train_transformed.shape[0])
    train_len = int(X_train_transformed.shape[0] * 0.7)
    training_idx, val_idx = indices[:train_len], indices[train_len:]
    X_val_transformed, y_val = X_train_transformed[val_idx, :], y_train[val_idx, :]
    X_train_transformed, y_train = X_train_transformed[training_idx, :], y_train[training_idx, :]

    train_loader = DataLoader(list(zip(X_train_transformed, y_train)), shuffle=True, batch_size=128, drop_last=True)
    val_loader = DataLoader(list(zip(X_val_transformed, y_val)), shuffle=False, batch_size=128, drop_last=False)
    test_loader = DataLoader(list(zip(X_test_transformed, y_test)), shuffle=False, batch_size=128, drop_last=False)

    return train_loader, val_loader, test_loader

# Main script

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    path = r"C:\Users\Peng XIE\OneDrive\maxwell\maxwell"
    train_loader, val_loader, test_loader = prepare_data(path)

    F_hat = torch.zeros(128, 1)
    model = NeuralNetwork([[11, 128, 64, 16, 3], [3, 64, 16, 3]], F_hat, device=device)
    model = model.to(device).float()

    train_model(model, train_loader, val_loader, test_loader, device)
