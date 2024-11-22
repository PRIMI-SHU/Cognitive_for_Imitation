#!/usr/bin/env python
from trainer import Trainer
from trainer import Trainer
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from PIL import Image
from torch.utils.data import DataLoader,TensorDataset
import math
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
import random
# get the package root path
main_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
module_dir = os.path.join(main_dir)
sys.path.append(module_dir)

from util.helper import *
import cv2

from networks.dynamic_planner import Dynamic_Planner

###### Training functions for the Specific NN model

def forward_function(model,x,device):
    diff,action=x
    outputs=model(diff.to(device))
    return (action,outputs)
def loss_function(model,output,device):
    true_action,action=output
    error=model.loss_function(action.to(device),true_action.to(device))
    return error



#######Call back functions for training########
def test_function(trainer):
        trainer.model.eval()
        error_test=0
        for i, x in enumerate(trainer.test_data):
            output=trainer.forward_function(trainer.model,x,trainer.device)
            error=trainer.loss_function(trainer.model,output,trainer.device)
            error_test+=error.item()
        error_test/=(i+1)
        trainer.error_test=error_test
def print_error(trainer):
    print('Epoch:',trainer.epoch,'Train_error:',trainer.error_train,'Test_error:',trainer.error_test)

def save_model(trainer):
    
     torch.save(trainer.model.state_dict(), '/docker-ros/local_ws/catkin_ws/src/data_collection/dynamic1.ckpt')
    
if __name__=='__main__':
    train_config=Hparams()
    df = pd.read_csv(train_config.diff_data)
    df.head()
    X = df.values
    df=pd.read_csv(train_config.action_data)
    df.head()
    Y=df.values
    X=torch.tensor(X).float()
    Y=torch.tensor(Y).float()
    dataset=TensorDataset(X,Y)
    train_set, validation_set = torch.utils.data.random_split(dataset,[int(len(dataset)*0.9),len(dataset)-int(len(dataset)*0.9)])
    
    train_loader=DataLoader(dataset,shuffle=True,batch_size=32)
    test_loader=DataLoader(validation_set,shuffle=True,batch_size=32)
    
    model=Dynamic_Planner(3,5,3,128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    trainer=Trainer(model,optimizer,train_config,train_loader,test_loader,500,test_every=20,save_error=None)
    trainer.forward_function=forward_function
    trainer.loss_function=loss_function
    
    trainer.add_callback('on_batch_end',test_function)
    trainer.add_callback('on_batch_end',print_error)
    # trainer.add_callback('on_batch_end',plot_example)
    
    #Trainer run
    trainer.run()
    save_model(trainer)
    
    
    


