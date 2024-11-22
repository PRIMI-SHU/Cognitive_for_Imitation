#!/usr/bin/env python3
import sys
import torch
import numpy as np
import os
import torch
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))

# get the package root path
main_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
module_dir = os.path.join(main_dir)
sys.path.append(module_dir)


from util.helper import *
from util.models import *
import torch.optim as optim
from torch.autograd import Variable
class Encoder(nn.Module):
    def __init__(self,config):
        super(Encoder,self).__init__()
        self.ConvE=ConvEnc(config.chanel[2:],config.chanel[0],config.chanel[1])
    def forward(self,x):
        return self.ConvE(x)

class EmbeddingLeNet(nn.Module):
    def __init__(self,config):
        super(EmbeddingLeNet, self).__init__()


        mlp_dim=config.mlp_dim
        embedding_size=config.out_dim
        self.fc = nn.Sequential(
            nn.Linear(in_features=mlp_dim, out_features=512),
            nn.PReLU(),
            nn.Linear(512, 256),
            # nn.PReLU(),
            # nn.Linear(128, embedding_size)
        )

    def forward(self, x):
        output = self.fc(x)
        # output = F.normalize(output, p=2, dim=1)
        return output
class TripleNet(nn.Module):
    def __init__(self,config):
        super(TripleNet,self).__init__()
        self.encoder=Encoder(config)
        self.EmbeddingNet=EmbeddingLeNet(config)  
    def forward(self, i1, i2):
        i1=self.encoder(i1)
       
        i2=self.encoder(i2)
            
        dis1 = self.EmbeddingNet(i1)
        dis2 = self.EmbeddingNet(i2)
      
        return dis1, dis2

class Metric(nn.Module):
    def __init__(self,config):
        super(Metric,self).__init__()
        self.encoder=Encoder(config)
        self.EmbeddingNet=EmbeddingLeNet(config)  
    def forward(self, i1):
        i1=self.encoder(i1)      
        dis1 = self.EmbeddingNet(i1)
        return dis1