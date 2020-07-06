import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import random
import gc
import os
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder
from torch.autograd import Variable

import math
import datetime
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader,Dataset
from torch.distributions.normal import Normal

device = torch.device('cuda')
m = Normal(torch.tensor([0.0], device='cuda'), torch.tensor([1.0], device='cuda'))

class M5Loader:

    def __init__(self, X, y, shuffle=True, batch_size=10000, cat_cols=[]):
        self.X_cont = X["dense"]
        self.X_cat = np.concatenate([X[k] for k in cat_cols], axis=1)
        self.y = y
        # self.y_lbl = y_lbl

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n_conts = self.X_cont.shape[1]
        self.len = self.X_cont.shape[0]
        n_batches, remainder = divmod(self.len, self.batch_size)

        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
        self.remainder = remainder  # for debugging

        self.idxes = np.array([i for i in range(self.len)])

    def __iter__(self):
        self.i = 0
        if self.shuffle:
            ridxes = self.idxes
            np.random.shuffle(ridxes)
            self.X_cat = self.X_cat[[ridxes]]
            self.X_cont = self.X_cont[[ridxes]]
            if self.y is not None:
                self.y = self.y[[ridxes]]
                # self.y_lbl = self.y_lbl[[ridxes]]

        return self

    def __next__(self):
        if self.i >= self.len:
            raise StopIteration

        if self.y is not None:
            y = torch.FloatTensor(self.y[self.i:self.i + self.batch_size].astype(np.float32))
            # y_lbl = torch.FloatTensor(self.y_lbl[self.i:self.i + self.batch_size].astype(np.float32))
        else:
            y = None
            # y_lbl = None

        xcont = torch.FloatTensor(self.X_cont[self.i:self.i + self.batch_size])
        xcat = torch.LongTensor(self.X_cat[self.i:self.i + self.batch_size])

        batch = (xcont, xcat, y)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class M5Net(nn.Module):
    def __init__(self, emb_dims, n_cont, device=device):
        super().__init__()
        self.device = device

        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        n_embs = sum([y for x, y in emb_dims])
        #t_embs = sum([y for x, y in emb_dims[7:]])

        self.n_embs = n_embs # + t_embs
        self.n_cont = n_cont
        inp_dim = n_embs + n_cont #+ t_embs
        self.inp_dim = inp_dim
        #print(n_embs, n_cont, t_embs)

        hidden_dim = 512

        self.fc0 = nn.Linear(inp_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, inp_dim)
        self.z_gate = nn.Sigmoid()

        self.fr0 = nn.Linear(inp_dim, hidden_dim)
        self.fr1 = nn.Linear(hidden_dim, 256)
        self.fr2 = nn.Linear(256, inp_dim)
        self.r_gate = nn.Sigmoid()

        self.fc3 = nn.Linear(inp_dim, hidden_dim)
        self.fc4= nn.Linear(hidden_dim, 256)
        self.fc5 = nn.Linear(256, inp_dim)
        self.tanh = nn.Tanh()

        self.out = nn.Linear(inp_dim, 1)
        self.output = nn.Softplus()

        # self.cl0 = nn.Linear(inp_dim, 256)
        # self.cl1 = nn.Linear(256, 128)
        # self.cl2 = nn.Linear(128, 1)
        # self.cl = nn.Sigmoid()

        self.fc0.apply(init_weights)
        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)
        self.z_gate.apply(init_weights)

        self.fr0.apply(init_weights)
        self.fr1.apply(init_weights)
        self.fr2.apply(init_weights)
        self.r_gate.apply(init_weights)

        self.fc3.apply(init_weights)
        self.fc4.apply(init_weights)
        self.fc5.apply(init_weights)
        self.tanh.apply(init_weights)

        self.out.apply(init_weights)
        self.output.apply(init_weights)

        # self.cl0.apply(init_weights)
        # self.cl1.apply(init_weights)
        # self.cl2.apply(init_weights)
        # self.cl.apply(init_weights)

    def encode_and_combine_data(self, cont_data, cat_data):
        xcat = [el(cat_data[:, k]) for k, el in enumerate(self.emb_layers)]
        xcat = torch.cat(xcat, 1)
        x = torch.cat([xcat, cont_data], 1)
        return x

    def forward(self, cont_data, cat_data):

        cont_data = cont_data.to(self.device)
        cat_data = cat_data.to(self.device)
        x = self.encode_and_combine_data(cont_data, cat_data)

        hz = self.fc0(x)
        hz = self.fc1(hz)
        hz = self.fc2(hz)
        z_gate = self.z_gate(hz)

        hr = self.fc0(x)
        hr = self.fc1(hr)
        hr = self.fc2(hr)
        r_gate = self.r_gate(hr)

        h1 = self.fc3(r_gate*x)
        h1 = self.fc4(h1)
        h1 = self.fc5(h1)
        update = self.tanh(h1)

        out = x*(1-z_gate) + z_gate*update

        out = self.out(out)
        out = self.output(out)

        # clh = self.cl0(x)
        # clh = self.cl1(clh)
        # clh = self.cl2(clh)
        # prob = self.cl(clh)
        #
        # out = torch.where(prob > 0.1, out, torch.tensor([0.0]).cuda())
        # out = out.squeeze()
        # prob = prob.squeeze()
        # h_adapt = self.adapt(h)
        # weight_adapt = self.adapt_weigth(h_adapt)

        #out = (weight_adapt * torch.exp(x)) / (1 + weight_adapt * (torch.exp(x)))

        return out #, prob

class M5Net_cl(nn.Module):
    def __init__(self, emb_dims, n_cont, device=device):
        super().__init__()
        self.device = device

        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        n_embs = sum([y for x, y in emb_dims])
        #t_embs = sum([y for x, y in emb_dims[7:]])

        self.n_embs = n_embs # + t_embs
        self.n_cont = n_cont
        inp_dim = n_embs + n_cont #+ t_embs
        self.inp_dim = inp_dim
        #print(n_embs, n_cont, t_embs)

        hidden_dim = 512

        self.fc0 = nn.Linear(inp_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, inp_dim)
        self.z_gate = nn.Sigmoid()

        self.fr0 = nn.Linear(inp_dim, hidden_dim)
        self.fr1 = nn.Linear(hidden_dim, 256)
        self.fr2 = nn.Linear(256, inp_dim)
        self.r_gate = nn.Sigmoid()

        self.fc3 = nn.Linear(inp_dim, hidden_dim)
        self.fc4= nn.Linear(hidden_dim, 256)
        self.fc5 = nn.Linear(256, inp_dim)
        self.tanh = nn.Tanh()

        self.out = nn.Linear(inp_dim, 1)
        self.output = nn.Sigmoid()

        self.fc0.apply(init_weights)
        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)
        self.z_gate.apply(init_weights)

        self.fr0.apply(init_weights)
        self.fr1.apply(init_weights)
        self.fr2.apply(init_weights)
        self.r_gate.apply(init_weights)

        self.fc3.apply(init_weights)
        self.fc4.apply(init_weights)
        self.fc5.apply(init_weights)
        self.tanh.apply(init_weights)

        self.out.apply(init_weights)
        self.output.apply(init_weights)


    def encode_and_combine_data(self, cont_data, cat_data):
        xcat = [el(cat_data[:, k]) for k, el in enumerate(self.emb_layers)]
        xcat = torch.cat(xcat, 1)
        x = torch.cat([xcat, cont_data], 1)
        return x

    def forward(self, cont_data, cat_data):

        cont_data = cont_data.to(self.device)
        cat_data = cat_data.to(self.device)
        x = self.encode_and_combine_data(cont_data, cat_data)

        hz = self.fc0(x)
        hz = self.fc1(hz)
        hz = self.fc2(hz)
        z_gate = self.z_gate(hz)

        hr = self.fc0(x)
        hr = self.fc1(hr)
        hr = self.fc2(hr)
        r_gate = self.r_gate(hr)

        h1 = self.fc3(r_gate*x)
        h1 = self.fc4(h1)
        h1 = self.fc5(h1)
        update = self.tanh(h1)

        out = x*(1-z_gate) + z_gate*update

        out = self.out(out)
        out = self.output(out)

        return out

class mixture(nn.Module):
    def __init__(self, emb_dims, n_cont, device=device):
        super().__init__()
        self.device = device

        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        n_embs = sum([y for x, y in emb_dims])


        self.n_embs = n_embs
        self.n_cont = n_cont
        inp_dim = n_embs + n_cont
        self.inp_dim = inp_dim

        hidden_dim = 256

        self.m1_1 = nn.Linear(inp_dim, hidden_dim)
        self.m1_2 = nn.Linear(256, 64)
        self.m1_3 = nn.Linear(64, 1)
        self.m1 = nn.Softplus()

        self.m2_1 = nn.Linear(inp_dim, hidden_dim)
        self.m2_2 = nn.Linear(256, 64)
        self.m2_3 = nn.Linear(64, 1)
        self.m2 = nn.Softplus()

        self.m3_1 = nn.Linear(inp_dim, hidden_dim)
        self.m3_2 = nn.Linear(256, 64)
        self.m3_3 = nn.Linear(64, 1)
        self.m3 = nn.Softplus()

        self.m4_1 = nn.Linear(inp_dim, hidden_dim)
        self.m4_2 = nn.Linear(256, 64)
        self.m4_3 = nn.Linear(64, 1)
        self.m4 = nn.Softplus()

        self.m5_1 = nn.Linear(inp_dim, hidden_dim)
        self.m5_2 = nn.Linear(256, 64)
        self.m5_3 = nn.Linear(64, 1)
        self.m5 = nn.Softplus()

        self.out = nn.Linear(5, 1)
        self.output = nn.Softplus()

        self.m1_1.apply(init_weights)
        self.m1_2.apply(init_weights)
        self.m1_3.apply(init_weights)
        self.m1.apply(init_weights)

        self.m2_1.apply(init_weights)
        self.m2_2.apply(init_weights)
        self.m2_3.apply(init_weights)
        self.m2.apply(init_weights)

        self.m3_1.apply(init_weights)
        self.m3_2.apply(init_weights)
        self.m3_3.apply(init_weights)
        self.m3.apply(init_weights)

        self.m4_1.apply(init_weights)
        self.m4_2.apply(init_weights)
        self.m4_3.apply(init_weights)
        self.m4.apply(init_weights)

        self.m5_1.apply(init_weights)
        self.m5_2.apply(init_weights)
        self.m5_3.apply(init_weights)
        self.m5.apply(init_weights)

        self.out.apply(init_weights)
        self.output.apply(init_weights)


    def encode_and_combine_data(self, cont_data, cat_data):
        xcat = [el(cat_data[:, k]) for k, el in enumerate(self.emb_layers)]
        xcat = torch.cat(xcat, 1)
        x = torch.cat([xcat, cont_data], 1)
        return x

    def forward(self, cont_data, cat_data):

        cont_data = cont_data.to(self.device)
        cat_data = cat_data.to(self.device)
        x = self.encode_and_combine_data(cont_data, cat_data)

        h1 = self.m1_1(x)
        h1 = self.m1_2(h1)
        h1 = self.m1_3(h1)
        out1 = self.m1(h1)

        h2 = self.m2_1(x)
        h2 = self.m2_2(h2)
        h2 = self.m2_3(h2)
        out2 = self.m2(h2)

        h3 = self.m3_1(x)
        h3 = self.m3_2(h3)
        h3 = self.m3_3(h3)
        out3 = self.m3(h3)

        h4 = self.m4_1(x)
        h4 = self.m4_2(h4)
        h4 = self.m4_3(h4)
        out4 = self.m4(h4)

        h5 = self.m5_1(x)
        h5 = self.m5_2(h5)
        h5 = self.m5_3(h5)
        out5 = self.m5(h5)

        out = torch.cat([out1, out2, out3, out4, out5], 1)
        out = self.out(out)
        out = self.output(out)

        return out1, out2, out3, out4, out5, out

class ZeroBalance_RMSE(nn.Module):
    def __init__(self, penalty=1.1):
        super().__init__()
        self.penalty = penalty

    def forward(self, y_pred, y_true):
        y_pred = y_pred.squeeze()
        y_true = torch.FloatTensor(y_true).to(device)
        sq_error = torch.where(y_true == 0, (y_true - y_pred) ** 2, self.penalty * (y_true - y_pred) ** 2)
        return torch.sqrt(torch.mean(sq_error))


class RMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        y_pred = y_pred.squeeze()
        y_true = torch.FloatTensor(y_true).to(device)
        return torch.sqrt(self.mse(y_pred, y_true))


class MSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        y_pred = y_pred.squeeze()
        y_true = torch.FloatTensor(y_true).to(device)
        return self.mse(y_pred, y_true)

def rmse_metric(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


class tweedie_loss(nn.Module):
    def __init__(self, penalty=1.1):
        super().__init__()
        self.penalty = penalty

    def forward(self, y_pred, y_true):

        y_pred = y_pred.squeeze()
        y_true = torch.FloatTensor(y_true).to(device)

        sq_error = torch.pow(y_pred, (-self.penalty)) * (
                ((y_pred * y_true) / (1 - self.penalty)) - ((torch.pow(y_pred, 2)) / (2 - self.penalty)))

        return torch.mean(-2*sq_error)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss