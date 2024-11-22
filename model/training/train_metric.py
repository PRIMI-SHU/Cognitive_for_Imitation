#!/usr/bin/env python3
#####Training function for other's encoder

import torch
import torchvision.models as models
import numpy as np
import os
import torch
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

# get the package root path
main_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
module_dir = os.path.join(main_dir)
sys.path.append(module_dir)
from util.models import *
from util.helper import *
from networks.metric import *

import torch.optim as optim
from torch.autograd import Variable

train_config=Hparams()
model=TripleNet(train_config)
print(model)
criterion = torch.nn.MarginRankingLoss(margin=train_config.margin)
optimizer = optim.Adam(params=model.parameters(), lr=train_config.lr)
dataset=Contrastive_DATA(train_config.embedding_data,train_config.compare_data,train_config.full_ann,'train')

train_loader=DataLoader(dataset,shuffle=True,batch_size=16)
device='cuda'

best_error=0.8
parent_d=train_config.metric_path
for epoch in range(1000):
    error=0
   
    for i,(x,y,z) in enumerate(train_loader):
        model=model.to(device)
        model.train()
        
        e2,e3=model.forward(y.to(device),z.to(device))
        
        
        
        
        dist_E1_E2 = F.pairwise_distance(x.to(device), e2, 2)
        dist_E1_E3 = F.pairwise_distance(x.to(device), e3, 2)

        target = torch.FloatTensor(dist_E1_E2.size()).fill_(-1)
        target = target.to(device)
        target = Variable(target)
        loss = criterion(dist_E1_E2, dist_E1_E3, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        error+=dist_E1_E2.mean()
    error/=(i+1)
    if error<best_error:
        torch.save(model.eval().cpu().state_dict(),train_config.metric_path) 
        best_error=error
        print('saved error:',best_error)
    
  
        

