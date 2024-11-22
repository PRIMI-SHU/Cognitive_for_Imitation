# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 18:56:11 2021

@author: mrd
This the function for nueral dynamic planner
"""
import torch
import torch.nn as nn
import torch.autograd as autograd
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader,TensorDataset
import pandas as pd
from sklearn.model_selection import KFold
class Dynamic_Planner(nn.Module):
    '''A simple implementation of the multi-layer neural network'''
    def __init__(self, n_input=3, n_output=4, n_h=3, size_h=128):
        '''
        Specify the neural network architecture

        :param n_input: The dimension of the input
        :param n_output: The dimension of the output
        :param n_h: The number of the hidden layer
        :param size_h: The dimension of the hidden layer
        '''
        super(Dynamic_Planner, self).__init__()
        self.n_input = n_input
        self.fc_in = nn.Linear(n_input, size_h)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        assert n_h >= 1, "h must be integer and >= 1"
        self.fc_list = nn.ModuleList()
        for i in range(n_h - 1):
            self.fc_list.append(nn.Linear(size_h, size_h))
        self.fc_out = nn.Linear(size_h, n_output)
        # Initialize weight
        nn.init.uniform_(self.fc_in.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_out.weight, -0.1, 0.1)
        self.fc_list.apply(self.init_normal)
        self.criterion =nn.MSELoss(reduction='mean')
    def forward(self, x):
        out = x.view(-1, self.n_input)
        out = self.fc_in(out)
        out = self.tanh(out)
        for _, layer in enumerate(self.fc_list, start=0):
            out = layer(out)
            out = self.tanh(out)
        out = self.fc_out(out)
        return out

    def init_normal(self, m):
        if type(m) == nn.Linear:
            nn.init.uniform_(m.weight, -0.1, 0.1)
    def loss_function(self,x,y):
         loss = self.criterion(x, y)
         return loss