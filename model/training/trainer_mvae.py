#!/usr/bin/env python3
from trainer import Trainer
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from PIL import Image
from torch.utils.data import DataLoader
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

from networks.vae_blend import blend_vae
###### Training functions for the Specific NN model
def forward_function(model,x,device):
    image,joint,tip,image_true,joint_true,tip_true=x
    mu, out_mu, out_logstd=model.forward([image.to(device),joint.to(device),tip.to(device)],sample=True)
    return (mu, out_mu, out_logstd,image_true,joint_true,tip_true)

def loss_function(model,output,device):
    mu, out_mu, out_logstd,image_true,joint_true,tip_true=output
   
    error=model.loss_function([mu, out_mu, out_logstd],[image_true.to(device),joint_true.to(device),tip_true.to(device)],mse=True)
    return error
#####  Call back functions for training  
def save(trainer):
    torch.save(trainer.model.eval().state_dict(), train_config.mvae_path)
    print('saved model with error:',trainer.save_error)

  
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
        
def plot_example(trainer):
    
            trainer.model.eval()
            index=random.randint(0,len(dataset))
            im_test,joint,tip,_,joint_true,tip_true=dataset[index]
            im_test=torch.reshape(im_test,(1,1,128,128))
            joint_test=torch.reshape(joint,(1,5))
            tip_test=torch.reshape(tip,(1,3))
            
            _, out_mu1, _=trainer.model.forward([im_test.to(trainer.device),joint_test.to(trainer.device),tip_test.to(trainer.device)],sample=False)
            temp=out_mu1[0]
        
            temp=torch.reshape(temp,(1,128,128))
            temp=temp*255
            img=temp.detach().cpu().numpy()
            img=img.transpose(1, 2, 0)
            
            cv2.imwrite('out.jpg',img)
            
            print('Recon Joint:',out_mu1[1].detach().cpu().numpy(),'True Joint:',joint_true)
            print('Recon Tip:',out_mu1[2].detach().cpu().numpy(),'True Tip:',tip_true)
if __name__=='__main__':
    
    #Util set up
    train_config=Hparams()
    dataset=MVAE_DATA(train_config.babbling_data,train_config.blend_ann,'train')
    


    train_set, validation_set = torch.utils.data.random_split(dataset,[int(len(dataset)*0.9),len(dataset)-int(len(dataset)*0.9)]) 
   
    train_loader=DataLoader(dataset,shuffle=True,batch_size=256)
    test_loader=DataLoader(validation_set,shuffle=True,batch_size=256)
    model=blend_vae(train_config.in_chanels,train_config.out_chanels,train_config.in_shared,train_config.out_shared)
    optimizer = torch.optim.Adam(lr=0.0005, params=model.parameters(), amsgrad=True)
    
    #Trainer setup
    trainer=Trainer(model,optimizer,train_config,train_loader,test_loader,iter=500,test_every=5,save_error=0.001)
    trainer.forward_function=forward_function
    trainer.loss_function=loss_function
    
    trainer.add_callback('on_save',save)
    trainer.add_callback('on_batch_end',test_function)
    trainer.add_callback('on_batch_end',print_error)
    trainer.add_callback('on_batch_end',plot_example)
    
    #Trainer run
    trainer.run()
    
    
    
    












