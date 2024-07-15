#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.utils.data import DataLoader
torch.set_default_dtype(torch.float)


# In[ ]:


import numpy as np
import pandas as pd


import torch
from torch import nn

import pickle
import time

import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')


# In[3]:


import os
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score # 1-mse/var


def load_data_forGridSearch(path, dataset_type):
    """
    Load training and testing dataset.
    Split data into features and labels.
    :param path: data file path
    :param dataset_type: type of dataset, "train" or "test"
    :return: feature and label numpy array depending on dataset_type
    """

    data = np.load(os.path.join(path, "{}.npy".format(dataset_type)))

    feature = data[:, :-3]
    label = data[:, -3:]

    assert feature.shape[1] == 11, "Imported features do not have the correct dimension"
    assert label.shape[1] == 3, "Imported labels do not have the correct dimension"

    return feature, label


def feature_scaling_forGridSearch(train, test):
    """
    Perform feature scaling based on the entire training dataset, and also apply it to the testing dataset.
    :param train: numpy array of training features
    :param test: numpy array of testing features
    :return: scaled numpy array of training and testing features, based on the max. and min. value in the training set
    """
    minmax_scale = MinMaxScaler().fit(train)
    train_transformed = minmax_scale.transform(train)
    test_transformed = minmax_scale.transform(test)

    assert train.shape == train_transformed.shape, "Feature dimension for the training set is changed due to scaling"
    assert test.shape == test_transformed.shape, "Feature dimension for the testing set is changed due to scaling"
    assert np.max(train_transformed) <= 1 + math.exp(-9), "The max. value of the training feature exceeds 1"
    assert np.min(train_transformed) >= 0, "The min. value of the training feature belows 0"

    return train_transformed, test_transformed


# In[4]:


timestr = time.strftime("%Y%m%d-%H%M%S")


# Want to solve magnetic field B of maxwell equation
# $$ \nabla \times B = 0 $$
# $$ \nabla \cdot B = 0 $$
# 
# The magnetic field B is based on current I:
# $$B=A\times I$$
# where A is independent of I
# 
# 
# The input of NN is x_1, x_2, x_3, I_1, I_2, ..., I_8<br>
# The output is B_1, B_2, B_3<br>
# 
# The deep learning will train this equation with the loss:
# $$L = L_{MSE} + \alpha L_{PINNS}$$
# $$L_{MSE} = \sum \|B^{exact}_i-B_i\|$$
# $$L_{PINNS} = \|\nabla \times B\| + \|\nabla \cdot B\|$$

# In[5]:


# load training and testing data
X_train, y_train = load_data_forGridSearch("/Users/andres/Desktop/ASTRI", "train")
X_test, y_test = load_data_forGridSearch("/Users/andres/Desktop/ASTRI", "test")


# In[6]:


# perform feature scaling
X_train_transformed, X_test_transformed = feature_scaling_forGridSearch(X_train, X_test)


# In[7]:


indices = np.random.permutation(X_train_transformed.shape[0])
train_len = int(X_train_transformed.shape[0]*0.7)
training_idx, val_idx = indices[:train_len], indices[train_len:]
X_val_transformed, y_val = X_train_transformed[val_idx,:], y_train[val_idx,:]
X_train_transformed, y_train =X_train_transformed[training_idx,:], y_train[training_idx,:]


# In[8]:


class FCN(nn.Module):
  #https://github.com/omniscientoctopus/Physics-Informed-Neural-Networks/tree/main/PyTorch/Burgers'%20Equation
    ##Neural Network
    def __init__(self,layers,f_hat, device):
        super().__init__() #call __init__ from parent class
        'activation function'
        self.activation = nn.ReLU()
        'loss function'
        self.loss_function = nn.MSELoss(reduction ='mean')
        'Initialise neural network as a list using nn.Modulelist'
        self.layers_b=layers[1]
        self.layers_a=layers[0]
        self.linears = nn.ModuleList([nn.Linear(self.layers_a[i], self.layers_a[i+1]) for i in range(len(self.layers_a)-1)])
        self.linears_b = nn.ModuleList([nn.Linear(self.layers_b[i], self.layers_b[i+1]) for i in range(len(self.layers_b)-1)])
        self.iter = 0 #For the Optimizer
        self.f_hat = f_hat.to(device) # xp source term, here is 0
        self.device = device
        'Xavier Normal Initialization'
        for i in range(len(self.layers_a)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)
        for i in range(len(self.layers_b)-1):
            nn.init.xavier_normal_(self.linears_b[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.linears_b[i].bias.data)
    'foward pass'
    def forward(self,x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)
        a = x[:,:]
        b = x[:,:3]
        for i in range(len(self.layers_a)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        for i in range(len(self.layers_b)-2):
            z = self.linears_b[i](b)
            b = self.activation(z)
        b = self.linears_b[-1](b)
        a = a*b
        return a

    'Loss Functions'
    #Loss BC
    def lossBC(self,x_BC,y_BC):
        loss_BC=self.loss_function(self.forward(x_BC),y_BC)
        return loss_BC

    # def lossBCN(self,x_BC):
    #     g=x_BC.clone()
    #     g.requires_grad=True #Enable differentiation
    #     f=self.forward(g)
    #     f_x_y = torch.autograd.grad(f, g, torch.ones([g.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
    #     f_x = f_x_y[:,[0]]
    #     f = f_x
    #     return self.loss_function(f,f_Nb)


    #Loss PDE
    def lossPDE(self,x_PDE):
        g=x_PDE.clone() # it is x
        g.requires_grad=True #Enable differentiation or g.requires_grad_()
        f=self.forward(g) # calculate the result from trainning model
#         f_x_y = autograd.grad(f,g,torch.ones([g.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0] #first derivative
#         f_xx_yy = autograd.grad(f_x_y,g,torch.ones(g.shape).to(device), create_graph=True)[0]#second derivative
        fx_x_y_z = torch.autograd.grad(f[:,0].reshape(-1,1), g, grad_outputs = torch.ones([g.shape[0], 1]).to(self.device), retain_graph = True, create_graph=True)[0] # create_graph for higher order derivative
        fy_x_y_z = torch.autograd.grad(f[:,1].reshape(-1,1), g, grad_outputs = torch.ones([g.shape[0], 1]).to(self.device), retain_graph = True, create_graph=True)[0]
        fz_x_y_z = torch.autograd.grad(f[:,2].reshape(-1,1), g, grad_outputs = torch.ones([g.shape[0], 1]).to(self.device), retain_graph = True, create_graph=True)[0]
        divergence = self.loss_function(fx_x_y_z[:,[0]]+fy_x_y_z[:,[1]]+fz_x_y_z[:,[2]], self.f_hat)
        curl_x = self.loss_function(fz_x_y_z[:,[1]]-fy_x_y_z[:,[2]], self.f_hat)
        curl_y = self.loss_function(fx_x_y_z[:,[2]]-fz_x_y_z[:,[0]], self.f_hat)
        curl_z = self.loss_function(fy_x_y_z[:,[0]]-fx_x_y_z[:,[1]], self.f_hat)
        return divergence + curl_x + curl_y + curl_z

    def loss(self,x_BC,y_BC,x_PDE,alpha=1.0):
        loss_bc=self.lossBC(x_BC,y_BC)
        loss_pde=self.lossPDE(x_PDE)
        #loss_bcn=self.lossBCN(x_BCN)
        return loss_bc+loss_pde*alpha,loss_bc,loss_pde

    #Optimizer              X_train_Nu,Y_train_Nu,X_train_Nf

    # def closure(self):
    #     optimizer.zero_grad()
    #     loss = self.loss(X_train_Nu,Y_train_Nu,X_train_Nf)
    #     loss.backward()
    #     self.iter += 1
    #     if self.iter % 100 == 0:
    #         loss2=self.lossBC(X_test,Y_test)
    #         print("Training Error:",loss.detach().cpu().numpy(),"---Testing Error:",loss2.detach().cpu().numpy())
    #     return loss


# In[9]:


#torch.set_default_dtype(torch.float)
# CUDA for PyTorch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")

max_epochs = 32

# Datasets
batch_size = 128
lr = 0.001
show_epochs = 1

# data_train = torch.utils.data.TensorDataset(X_train_transformed, y_train)
# train_loader = DataLoader(data_train, shuffle=True, batch_size=batch_size, drop_last=True)
train_loader = DataLoader(list(zip(X_train_transformed, y_train)), shuffle=True, batch_size=batch_size, drop_last=True)
val_loader = DataLoader(list(zip(X_val_transformed, y_val)), shuffle=False, batch_size=batch_size, drop_last=False)
test_loader = DataLoader(list(zip(X_test_transformed, y_test)), shuffle=False, batch_size=batch_size, drop_last=False)
#train_loader, val_loader = torch.utils.data.random_split(train_loader, [int(len(train_loader)*0.7), len(train_loader)-int(len(train_loader)*0.7)], generator=torch.Generator().manual_seed(42))


F_hat = torch.zeros(batch_size,1)
model = FCN([[11,128,64,16,3],[3,64, 16, 3]], F_hat, device = device)
model = model.to(device).float()
# optimizer = torch.optim.Adam(model.parameters(),lr = lr )

optimizer = torch.optim.AdamW(model.parameters(), lr = lr, betas=(0.9, 0.99)) # xp
milestone = [10, 20] # xp
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones= milestone, gamma=0.1) # xp

# In[10]:


loss_values={'loss':[],'loss_bc':[],'loss_pde':[],'test_loss':[]}
print("epoch-----Training Loss-----MSE Loss-----PINNS Loss-----Val Loss-----R2 value-----Test Loss-----R2 value")
best_R2 = [0,0,0,0,0,0]
alpha = 0.01
for epoch in range(max_epochs):
    total_loss, total_bc, total_pde, count_iter = 0.0, 0.0, 0.0, 0.0
    for X,Y in train_loader:
        X = X.to(device).float()
        Y = Y.to(device).float()
        optimizer.zero_grad()
        loss, loss_bc, loss_pde = model.loss(X, Y, X, alpha=alpha)
        #loss = model.lossBC(X, Y)
        #print('loss', loss.dtype, X.dtype, Y.dtype)
        loss.backward()
        optimizer.step()
        count_iter += 1
        total_loss += loss.detach().cpu().item()
        total_bc += loss_bc.detach().cpu().item()
        total_pde += loss_pde.detach().cpu().item()
    if epoch % show_epochs == 0:
        with torch.no_grad():
            model.eval()
            test_loss = 0.0
            R2 = 0.0
            total_count = 0.0
            count_iter_val = 0.0
            for X_test,Y_test in val_loader:
                X_test = X_test.to(device).float()
                Y_test = Y_test.to(device).float()
                test_loss += model.lossBC(X_test, Y_test).detach().cpu().item()
                Y_pred = model(X_test)
                R2 += r2_score(y_true = Y_test.detach().cpu().numpy(), y_pred = Y_pred.detach().cpu().numpy())*len(Y_test) # Y_test.shape[0]
                total_count += len(Y_test)
                count_iter_val += 1
            if R2/total_count>=best_R2[5]:
                best_R2 = [epoch, total_loss/count_iter, total_bc/count_iter, total_pde/count_iter, test_loss/count_iter_val, R2/total_count]
                if best_R2[3]*alpha > best_R2[2]*0.5:
                    alpha *= 0.1
                test_loss = 0.0
                R2 = 0.0
                total_count = 0.0
                count_iter_val = 0.0
                for X_test,Y_test in test_loader:
                    X_test = X_test.to(device).float()
                    Y_test = Y_test.to(device).float()
                    test_loss += model.lossBC(X_test, Y_test).detach().cpu().item()
                    Y_pred = model(X_test)
                    R2 += r2_score(y_true = Y_test.detach().cpu().numpy(), y_pred = Y_pred.detach().cpu().numpy())*len(Y_test)
                    total_count += len(Y_test)
                    count_iter_val += 1
                print('{:03d}-------{:.8f}--------{:.8f}---{:.8f}-----{:.8f}---{:.5f}------{:.8f}----{:.5f}'.format(best_R2[0],best_R2[1],best_R2[2],best_R2[3],best_R2[4],best_R2[5],test_loss/count_iter_val,R2/total_count))
            else:
            #print(epoch,'---',total_loss/count_iter,'---',total_bc/count_iter,'---',total_pde/count_iter,'---',test_loss/count_iter_val,'---',R2/total_count)
                print('{:03d}-------{:.8f}--------{:.8f}---{:.8f}-----{:.8f}---{:.5f}'.format(epoch, total_loss/count_iter,total_bc/count_iter,total_pde/count_iter,test_loss/count_iter_val,R2/total_count))
        model.train()

    scheduler.step() # xp




