"""
Training a neural network to solve maxwell equations
Based on equations below:
Want to solve magnetic field B of maxwell equation
$$ \nabla \times B = 0 $$
$$ \nabla \cdot B = 0 $$

The magnetic field B is based on current I:
$$B=A\times I$$
where A is independent of I


The input of NN is x_1, x_2, x_3, I_1, I_2, ..., I_8<br>
The output is B_1, B_2, B_3<br>

The deep learning will train this equation with the loss:
$$L = L_{MSE} + \alpha L_{PINNS}$$
$$L_{MSE} = \sum \|B^{exact}_i-B_i\|$$
$$L_{PINNS} = \|\nabla \times B\| + \|\nabla \cdot B\|$$
"""
import os
from typing import Tuple, List, Callable, Optional
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn

import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# USER CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))\

DATA_DIR = os.path.join(BASE_DIR, 'dataset') # dataset directory
# saved model directory, need scaler and model
MODEL_DIR = os.path.join(BASE_DIR, 'saved_model')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.npy")
TEST_PATH = os.path.join(DATA_DIR, "test.npy")
MODEL_PATH = os.path.join(MODEL_DIR, "my_model.pth") # can be None -> will not save
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")  # can be None -> will not save


# Utlity Functions
def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads dataset saved as npy format, and splits into features and labels.
    Note: Data has 14 dimensions - First 11 dimensions are features and next 3 are labels.

    :param path: data file path
    :return: feature and label numpy array depending on dataset_type
    """
    # assert format of path
    data = np.load(path)
    # check for 14 dimensions in data
    assert data.shape[1] == 14, "Data does not have the correct number of 14 dimensions."

    features = data[:, :-3] # take first 11 as features
    labels = data[:, -3:] # take last 3 as labels

    return features, labels


def feature_scaling(train: np.ndarray,
                    test: np.ndarray,
                    scaler=MinMaxScaler(),
                    save_path: Optional[str]=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Feature scale train and test set.
    :param train: numpy array of training features
    :param test: numpy array of test features
    :param scaler: Scaler, defaults to MinMaxScaler
    :param save_path: Saves scaler at this path if specified, else do nothing
    :return: scaled numpy array of training and testing features
    """
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    # pylint: disable=line-too-long
    assert train.shape == train_scaled.shape, "Scaler should not change dimensions of train set."
    assert test.shape == test_scaled.shape, "Scaler should not change dimensions of test set."
    assert np.max(train_scaled) <= 1 + 10e-9, "Does not support custom scaler that allows for training feature to exceed 1."
    assert np.min(train_scaled) >= 0, "Does not support custom scaler that allows for training feature below 0."

    # save
    if save_path is not None:
        joblib.dump(scaler, save_path)
    return train_scaled, test_scaled

# pylint: disable=invalid-name
def prepare_data_for_training(train_path: str, test_path: str, device: torch.device,
                              scaler_path: Optional[str]=None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Produce DataLoader of train, validation, and test sets, and store tensors at specified device.
    Note: 15% of train data used as validation dataset.
    :param train_path: Training set path
    :param test_path: Test set path
    :param device: device where tensors are held
    :param: scaler_path: Saves scaler at this path
    :return: Train, validation, test DataLoaders
    """
    X_train, y_train = load_data(train_path)
    X_test, y_test = load_data(test_path)
    X_train_scaled, X_test_scaled = feature_scaling(X_train, X_test, save_path=scaler_path)

    # Convert to tensors and store at device
    X_train_scaled = torch.from_numpy(X_train_scaled).float().to(device)
    X_test_scaled = torch.from_numpy(X_test_scaled).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)

    # permuate indices for train and validation set later on
    indices = np.random.permutation(X_train_scaled.shape[0])
    train_len = int(X_train_scaled.shape[0] * 0.85) # 15% of data used as validation
    training_idx, val_idx = indices[:train_len], indices[train_len:]
    X_training_scaled, y_training = X_train_scaled[training_idx, :], y_train[training_idx, :]
    X_val_scaled, y_val = X_train_scaled[val_idx, :], y_train[val_idx, :]

    # set drop_last=True for consistent batch size during trng
    train_loader = DataLoader(list(zip(X_training_scaled, y_training)),
                              shuffle=True, batch_size=128, drop_last=True)
    val_loader = DataLoader(list(zip(X_val_scaled, y_val)),
                            shuffle=False, batch_size=128, drop_last=False)
    test_loader = DataLoader(list(zip(X_test_scaled, y_test)),
                             shuffle=False, batch_size=128, drop_last=False)

    return train_loader, val_loader, test_loader


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
                 initalizer: Callable=nn.init.xavier_normal_,
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

        self._initialize_weights(initalizer)


    def _initialize_weights(self, initializer: Callable) -> None:
        """
        Initializes weights between layers of the model.
        :param: initializer: Callable that initializes the weights
        """
        for layer in self.linears_a:
            initializer(layer.weight, gain=1)
            nn.init.zeros_(layer.bias)

        for layer in self.linears_b:
            initializer(layer.weight, gain=1)
            nn.init.zeros_(layer.bias)


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


    def loss_exp(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        """
        Computes experimental loss, by default MSE loss.
        """
        return self.loss_function(self.forward(x), y)


    def loss_bc(self,
                x_bc: torch.Tensor,
                y_bc: torch.Tensor) -> torch.Tensor:
        """
        Computes loss from Boundary Conditions. 
        There's some equation on the Github but leaving it as a dummy function due to missing data
        """
        x_bc = None
        y_bc = None
        return 0


    def loss_pde(self, x_pde: torch.Tensor) -> torch.Tensor:
        """
        Computes PDE loss
        """
        g = x_pde.clone()
        g.requires_grad = True # tracks gradient for training
        f = self.forward(g)

        # create graph for higher order derivative
        fx_x_y_z = torch.autograd.grad(f[:, 0].reshape(-1, 1), g,
                                       grad_outputs=torch.ones([g.shape[0], 1]).to(self.device),
                                       retain_graph=True, create_graph=True)[0]
        fy_x_y_z = torch.autograd.grad(f[:, 1].reshape(-1, 1), g,
                                       grad_outputs=torch.ones([g.shape[0], 1]).to(self.device),
                                       retain_graph=True, create_graph=True)[0]
        fz_x_y_z = torch.autograd.grad(f[:, 2].reshape(-1, 1), g,
                                       grad_outputs=torch.ones([g.shape[0], 1]).to(self.device),
                                       retain_graph=True, create_graph=True)[0]

        divergence = self.loss_function(fx_x_y_z[:,[0]]+fy_x_y_z[:,[1]]+fz_x_y_z[:,[2]], self.f_hat)
        curl_x = self.loss_function(fz_x_y_z[:,[1]]-fy_x_y_z[:,[2]], self.f_hat)
        curl_y = self.loss_function(fx_x_y_z[:,[2]]-fz_x_y_z[:,[0]], self.f_hat)
        curl_z = self.loss_function(fy_x_y_z[:,[0]]-fx_x_y_z[:,[1]], self.f_hat)

        return divergence + curl_x + curl_y + curl_z


    def loss(self,
             x: torch.Tensor,
             y: torch.Tensor,
             x_bc: torch.Tensor,
             y_bc: torch.Tensor,
             x_pde: torch.Tensor,
             alpha: float=1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes loss.
        """
        loss_exp = self.loss_exp(x, y)
        loss_bc = self.loss_bc(x_bc, y_bc)  # Note: not used in the equation
        loss_pde = self.loss_pde(x_pde)
        return loss_exp + loss_pde * alpha, loss_exp, loss_pde


# Main training and evaluation loop

# pylint: disable=too-many-locals, too-many-arguments, invalid-name
def train_model(model: PINN,
                train_loader: DataLoader,
                validation_loader: DataLoader,
                test_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]=None,
                num_epochs: int=32,
                alpha: float = 0.05,
                save_strategy: Optional[str]="validation",
                save_path: Optional[str]=None) -> None:
    """
    Trains the model.
    :param model: Model
    :param train_loader: Train set
    :param validation_loader: Validation set
    :param test_loader: Test set
    :param optimizer: Controls and apply gradient descent
    :Optional[param] scheduler: Adaptive adjustment of learning rate
    :param num_epochs: Number of epochs to run the model
    :param alpha: ?? NO IDEA; Some relationship between PDE and EXP loss
    :param save_strategy: Saves the model based on performance on validation or test set
    :param save_path: Where the saved model is stored
    """

    model_state = None  # For saving the best model later on, if save strategy is specified
    if save_strategy:
        save_strategy = save_strategy.lower()
        assert save_strategy in {"test", "validation"}, \
            "Save strategy based on either 'validation' or 'test' set performance."
        assert save_path is not None, "Specify a save path of the model."

    print("epoch-----" +
          "PINNS Loss------EXP Loss-----PDE Loss-------" +
          "Val (EXP) Loss-----Validation R2-------" +
          "Test (EXP) Loss---Test R2")

    # PINNS, EXP, PDE, VALIDATION_EXP_LOSS, VALIDATION_R2
    res_with_best_r2 = [0.0, 0.0, 0.0, 0.0, 0.0]
    # TEST_EXP_LOSS, TEST_R2
    res_with_best_test_r2 = [0.0, 0.0]

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_exp, total_pde, batch_num = 0.0, 0.0, 0.0, 0
        for X_batch, y_batch in train_loader:  # mini-batch GD

            optimizer.zero_grad()  # resets gradient computation
            loss, loss_exp, loss_pde = model.loss(X_batch, y_batch,
                                                 X_batch, y_batch,
                                                 X_batch,
                                                 alpha=alpha)
            loss.backward()
            optimizer.step()

            # accumulate losses
            total_loss += loss.detach().cpu().item()
            total_exp += loss_exp.detach().cpu().item()
            total_pde += loss_pde.detach().cpu().item()
            batch_num += 1

        model.eval()
        with torch.no_grad(): # disable gradient computation
            validation_loss, validation_r2 = evaluate_model(model, validation_loader)
            curr_best_val_r2 = res_with_best_r2[4]

            if validation_r2 > curr_best_val_r2:  # found better r2 on val set, run on test set
                best_train_total = total_loss/batch_num
                best_train_exp = total_exp/batch_num
                best_train_pde = total_pde/batch_num
                res_with_best_r2 = [
                    best_train_total,
                    best_train_exp,
                    best_train_pde,
                    validation_loss,
                    validation_r2
                    ]
                # change alpha value based on condition
                if best_train_pde * alpha > best_train_exp * 0.5:  # WHY???? pde loss and exp loss
                    alpha *= 0.1

                # evalaute on test set
                test_loss, test_r2 = evaluate_model(model, test_loader)
                curr_best_test_loss = res_with_best_test_r2[0]
                curr_best_test_r2 = res_with_best_test_r2[1]
                # save better results
                if test_r2 > curr_best_test_r2:
                    res_with_best_test_r2 = [test_loss, test_r2]

                print(
                    f'{epoch+1:03d}-------' +
                    f'{best_train_total:.8f}-----' +
                    f'{best_train_exp:.8f}---' +
                    f'{best_train_pde:.8f}--------' +
                    f'{validation_loss:.8f}----------' +
                    f'{validation_r2:.5f}------------' +
                    f'{test_loss:.8f}------' +
                    f'{test_r2:.5f}'
                    )

                if save_strategy == "validation":
                    model_state = model.state_dict()
                elif save_strategy == "test":
                    if test_loss > curr_best_test_loss:
                        model_state = model.state_dict()
                else:
                    assert False, "This should not happen."
            else:
                print(
                    f'{epoch+1:03d}-------' +
                    f'{total_loss / batch_num:.8f}-----' +
                    f'{total_exp / batch_num:.8f}---' +
                    f'{total_pde / batch_num:.8f}--------' +
                    f'{validation_loss:.8f}----------' +
                    f'{validation_r2:.5f}'
                    )
   
        # adjust learning rate
        if scheduler:
            scheduler.step()

    # save model
    if save_strategy:
        assert model_state is not None and \
            save_path is not None, "Something went wrong while saving the model.."
        torch.save(model_state, save_path)


# pylint: disable=invalid-name
def evaluate_model(model: PINN,
                   data_loader: DataLoader) -> Tuple[float, float]:
    """
    Evaluate model on a dataset.
    :param model: model
    :param data_loader: Test set
    :return: loss, r2 score
    """
    model.eval()
    with torch.no_grad():  # disable gradient computation
        exp_loss, r2, total_count = 0.0, 0.0, 0.0
        for X, y in data_loader:
            # determine based on experimental loss; 
            # cant use total loss because involves PDE which requires gradient computation
            exp_loss += model.loss_exp(X, y).item()
            y_pred = model(X)
            r2 += r2_score(y_true=y.cpu().numpy(), y_pred=y_pred.cpu().numpy()) * len(y)
            total_count += len(y)
        return exp_loss / len(data_loader), r2 / total_count

# Main script
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Running on cuda..")
        device_ = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
    else:
        device_ = torch.device("cpu")

    train_set, val_set, test_set = prepare_data_for_training(TRAIN_PATH, TEST_PATH, 
                                                             device_, scaler_path=SCALER_PATH)

    F_hat = torch.zeros(128, 1)
    my_model = PINN([[11, 256, 64, 16, 3], [3, 128, 16, 3]], F_hat, device=device_)    # Try MSE
    # my_model = PINN([[11, 256, 64, 16, 3], [3, 128, 16, 3]], F_hat, device=device_,  # Try MAE
    #                 loss_function=nn.L1Loss(reduction='mean'))
    my_model = my_model.to(device_)

    my_optimizer = torch.optim.AdamW(my_model.parameters(),
                                     lr=0.001,
                                     betas=(0.9, 0.999))
    # for adaptive learning rate
    my_scheduler = torch.optim.lr_scheduler.MultiStepLR(my_optimizer,
                                                        milestones=[10, 20], gamma=0.1)

    train_model(
        model = my_model,
        train_loader = train_set,
        validation_loader = val_set,
        test_loader = test_set,
        optimizer = my_optimizer,
        scheduler = my_scheduler,
        save_strategy="test",
        save_path=MODEL_PATH
        )
