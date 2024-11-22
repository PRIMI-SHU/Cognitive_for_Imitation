#!/usr/bin/env python3
import sys

import torch
import torch.nn as nn
import torch.optim as optim 
from torch.autograd import Variable
from networks.metric import *
from networks.vae_blend import *
from util.helper import *

import cv2

def noise(x):
    x_noise= - 2 * torch.ones_like(x,dtype=torch.float)
    return x_noise
class Agent():
    def __init__(self,vae,metric,dynamic,device):
        
        self.vae=vae.to(device)
        self.metric=metric.to(device)
        self.en_modes=self.vae.en_models
        self.map_encoder=nn.ModuleList([self.metric])
        self.dynamic_planner=dynamic.to(device)
        
        for i in range(1, len(self.vae.en_models)):
            self.map_encoder.append(self.vae.en_models[i])

        
        
        #AIF init 
        self.mu_attr=torch.zeros((1,5),dtype=torch.float).to(device)
        self.tip_attr=torch.zeros((1,3),dtype=torch.float).to(device)
        self.Im_attr=torch.zeros((1,1,128,128),dtype=torch.float).to(device)
        self.Im_hat=torch.zeros((1,1,128,128),dtype=torch.float).to(device)
        self.mu_hat=torch.zeros((1,5),dtype=torch.float).to(device)
        self.tip_hat=torch.zeros((1,3),dtype=torch.float).to(device)
        self.error_joints=[]
        
        # df=pd.read_csv('/docker-ros/local_ws/Metric_Internal_Model/image_std.csv')
        # self.Im_std=torch.tensor(df.values[:,1:]).reshape(1,1,128,128)
        # self.Im_std=self.Im_std.to(device)
        # df=pd.read_csv('/docker-ros/local_ws/Metric_Internal_Model/joint_std.csv')
        # self.mu_std=torch.tensor(df.values[:,1:]).reshape(1,3)
        # self.mu_std=self.mu_std.to(device)
        # df=pd.read_csv('/docker-ros/local_ws/Metric_Internal_Model/tip_std.csv')
        # self.tip_std=torch.tensor(df.values[:,1:]).reshape(1,3)
        # self.tip_std=self.tip_std.to(device)
        
        
        # self.SigmaP_yq0 = np.zeros((3, 3))
        # for i in range(3):
        #     self.SigmaP_yq0[i,i]=1/2.55
        self.act = np.zeros(3, float)
        
    #########Active Inference
    def set_goal(self,image):
        noise_joint=noise(self.mu_attr)
        noise_tip=noise(self.tip_hat)
        _, out_mu,_=self.map_recon([image,noise_joint,noise_tip])
        self.mu_attr=out_mu[1]
        self.Im_attr=out_mu[0]
        self.tip_attr=out_mu[2]
    
    def init_latent(self,image):
        
        self.Im_hat=image
        noise_joint=noise(self.mu_hat)
        noise_tip=noise(self.tip_hat)
        self.Im_hat,self.mu_hat,self.tip_hat,self.z1=self.vae.prediction([self.Im_hat,noise_joint,noise_tip])
        
    
   
    def perception(self):
        self.Im_hat,self.mu_hat,self.tip_hat,grad_Im,grad_mu,grad_Tip=self.vae.perception(self.z1,self.Im_attr,self.mu_attr,self.tip_attr)
        self.z1_dot=(0.2*grad_Im+0.6*grad_mu+0.2*grad_Tip)
        self.z1=self.z1_dot+self.z1
       
 
        return self.Im_hat,self.mu_hat,self.tip_hat
    
    def perception_std(self,joint):
        self.Im_hat,self.mu_hat,self.tip_hat,grad_Im,grad_mu,grad_Tip=self.vae.perception_std(self.z1,self.Im_attr,self.mu_attr,self.tip_attr,self.Im_std,self.mu_std,self.tip_std)
        self.z1_dot=(0.2*grad_Im+0.6*grad_mu+0.2*grad_Tip)
        self.z1=self.z1_dot+self.z1
        error=self.mu_hat-joint
        
        error=error.detach().cpu().numpy()[0]
        
        self.error_joints.append(error.tolist())
        
        
        return error,self.Im_hat,self.mu_hat
        
    def get_dz1(self,visual):
        self.Im_hat, self.mu_hat, grad_Im, grad_mu = self.vae.perception(self.z1,visual,self.mu_attr)

        return 0.1*grad_Im + 0.6*grad_mu
    
    def latent_energy(self):
        self.Im_hat,self.mu_hat,grad_Im,grad_mu=self.vae.perception(self.z1,self.Im_attr,self.mu_attr,self.im_std,self.mu_std)
        self.z1_dot=(0.1*grad_Im+0.6*grad_mu)
        self.z1=self.z1_dot+self.z1
        recon_joints=self.vae.recon(self.z1)
        
        return recon_joints  
    def minimiseE(self,visual,joints,dataset):
        z1_dot=self.get_dz1(self.Im_attr)+self.get_dz1(visual)
        self.z1=Variable(z1_dot+self.z1)
        
        recon_joints=self.vae.recon(self.z1)
        recon_joints=recon_joints.detach().cpu().numpy() 
        recon_joints=dataset.de_normalize(recon_joints[0])
        diff=(joints-recon_joints)
        
        self.act=diff*(1/2.55)
        return self.act
    ######### Mental Association   
    def translate_data_format(self,array):
        if torch.is_tensor(array):
            return array.detach().cpu().numpy()
        elif isinstance(array,np.ndarray):
            return torch.tensor(array,device='cuda',dtype=torch.float32)
    
           
    def self_recon(self,x):
        outs=[]
        out_slice=[]
        for x_m,model in zip(x,self.vae.en_models):
            o=model(x_m)
            
            outs.append(o)
            out_slice.append(o.shape[1])
        
        all=torch.cat(outs,dim=1)
        
        h=self.vae.shared_encoder(all)
        z=h
        h = self.vae.shared_decoder(z)
       
        begin = 0
        out_mu = []
        out_logstd = []
        for i, model in enumerate(self.vae.de_models):
            partial_h = h[:, begin:(begin+out_slice[i])]
            out = model(partial_h)
            d = out.shape[1]//2
            out_mu.append(out[:, :d])
            out_logstd.append(out[:, d:])
            begin += out_slice[i]
        return z, out_mu, out_logstd
    def map_recon(self,x):
        outs=[]
        
        out_slice=[]
        for x_m,model in zip(x,self.map_encoder):
            o=model(x_m)
            
            outs.append(o)
            out_slice.append(o.shape[1])
        
        all=torch.cat(outs,dim=1)
        
        h=self.vae.shared_encoder(all)
        z=h
        h = self.vae.shared_decoder(z)
       
        begin = 0
        out_mu = []
        out_logstd = []
        for i, model in enumerate(self.vae.de_models):
            partial_h = h[:, begin:(begin+out_slice[i])]
            out = model(partial_h)
            d = out.shape[1]//2
            out_mu.append(out[:, :d])
            out_logstd.append(out[:, d:])
            begin += out_slice[i]
        return z, out_mu, out_logstd
    def action(self,diff):
        action=self.dynamic_planner(diff)
        return action
