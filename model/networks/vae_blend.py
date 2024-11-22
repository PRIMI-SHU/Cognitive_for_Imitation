#!/usr/bin/env python3
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.autograd import Variable

current_dir = os.path.dirname(os.path.abspath(__file__))

# get the package root path
main_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
module_dir = os.path.join(main_dir)
sys.path.append(module_dir)


from util.models import *

class blend_vae(nn.Module):
    def __init__(self, in_chanel,out_chanel,in_shared,out_shared,init="xavier"):
        super(blend_vae,self).__init__()
        #Sperate Encoders for different modliality
        self.en_models=[]
        for chanel in in_chanel:
            if chanel[0]==-1: #MLP
                Mlp_E=MLP(chanel[1:],init=init)
                Mlp_E.layers.add_module(str(len(Mlp_E.layers)),nn.ReLU())
                self.en_models.append(Mlp_E)
            if chanel[0]==-2: #CNN
                ConvE=ConvEnc(chanel[3:],chanel[1],chanel[2])
                ConvE.add_module(str(len(ConvE.conv)+1),nn.ReLU())
                self.en_models.append(ConvE)
        self.en_models=nn.ModuleList(self.en_models)
        
        
        self.de_models=[]
        for chanel in out_chanel:
            if chanel[0]==-1:
                Mlp_D=MLP(chanel[1:], init=init)
                self.de_models.append(Mlp_D)
            if chanel[0]==-2:
                ConvD=ConvDec(chanel[3:],chanel[1],chanel[2])
                self.de_models.append(ConvD)
        self.de_models=nn.ModuleList(self.de_models)
        
        
        
        self.shared_encoder=nn.Sequential(MLP(in_shared,init=init),nn.Tanh())
        self.shared_encoder
        self.shared_decoder=nn.Sequential(MLP(out_shared,init=init),nn.ReLU())
        self.shared_decoder
        self.z_d=in_shared[-1]//2
        self.prior=torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))
        self.test_slice=[]
    def forward(self,x,sample=True):
        outs=[]
        out_slice=[]
        for x_m,model in zip(x,self.en_models):
            o=model(x_m)
            
            outs.append(o)
            out_slice.append(o.shape[1])
        
        all=torch.cat(outs,dim=1)
        
        h=self.shared_encoder(all)
        z=h
        # mu, logstd = h[:, :self.z_d], h[:, self.z_d:]
        # std = torch.exp(logstd)
        # if sample:
        #     dist = torch.distributions.Normal(mu, std)
        #     z = dist.rsample()
        # else:
        #     z = mu
        
        h = self.shared_decoder(z)
       
        begin = 0
        out_mu = []
        out_logstd = []
        for i, model in enumerate(self.de_models):
            partial_h = h[:, begin:(begin+out_slice[i])]
            out = model(partial_h)
            d = out.shape[1]//2
            out_mu.append(out[:, :d])
            out_logstd.append(out[:, d:])
            begin += out_slice[i]
        return z, out_mu, out_logstd
    def loss_function(self, x, y, lambd=1.0, beta=0, sample=True, reduce=True, mse=False):
        z_mu,o_mu,o_std=x[0],x[1],x[2]
        # z_std = torch.exp(z_logstd)
        # z_dist = torch.distributions.Normal(z_mu, z_std)
        # kl_loss = torch.distributions.kl_divergence(z_dist, self.prior)
        # if reduce:
        #     kl_loss = kl_loss.mean()
        # else:
        #     kl_loss = kl_loss.sum(dim=1).mean()
        recon_loss=0.0
        for x_m, x_s, y_m in zip(o_mu, o_std, y):
            x_m = x_m.reshape(x_m.shape[0], -1)
            x_s = x_s.reshape(x_s.shape[0], -1)
            y_m = y_m.reshape(y_m.shape[0], -1)
            if mse:
                modal_loss = torch.nn.functional.mse_loss(x_m, y_m, reduction="none")
            else:
                x_std = torch.exp(x_s)
                x_dist = torch.distributions.Normal(x_m, x_std)
                modal_loss = -x_dist.log_prob(y_m)

            if reduce:
                ll=torch.tensor([1/x_m.shape[1]]).to('cuda')
                recon_loss += ll*modal_loss.mean()
            else:
                recon_loss += modal_loss.sum(dim=1).mean()

        # recon_loss /= len(y)
        loss = lambd * recon_loss 
        return loss
        
    def produce_latent(self,x):
        outs=[]
        out_slice=[]
       
        for x_m,model in zip(x,self.en_models):
            o=model(x_m)
            outs.append(o)
            out_slice.append(o.shape[1])
        
        all=torch.cat(outs,dim=1)
        self.test_slice=out_slice
        h=self.shared_encoder(all)
        z=h
        return z
    def out_embedding(self,x):
        return self.en_models[0](x)
    def prediction(self,x):
        z=self.produce_latent(x)
        h = self.shared_decoder(z)
        
        begin = 0
        out_mu = []
        out_logstd = []
        for i, model in enumerate(self.de_models):
            partial_h = h[:, begin:(begin+self.test_slice[i])]
            
            out = model(partial_h)
            # print(out.shape)
            d = out.shape[1]//2
            out_mu.append(out[:, :d])
            out_logstd.append(out[:, d:])
            begin += self.test_slice[i]
        return out_mu[0],out_mu[1],out_mu[2],z
    def recon(self,z1):
       h = self.shared_decoder(z1)
       begin = 0
       out_mu = []
       out_logstd = []
       for i, model in enumerate(self.de_models):
            partial_h = h[:, begin:(begin+self.test_slice[i])]
            
            out = model(partial_h)
            
            d = out.shape[1]//2
            out_mu.append(out[:, :d])
            out_logstd.append(out[:, d:])
            begin += self.test_slice[i]
       
       Out_im, Out_mu = out_mu[0],out_mu[1]
       return Out_mu
    
    
    #####This is the compare code for AIF: borrowed from https://github.com/Cmeo97/MAIF?utm_source=catalyzex.com    
    def perception(self, z1, Im_attr, mu_attr,Tip_attr):
       
       z1 = Variable(z1, requires_grad=True)
       h = self.shared_decoder(z1)
       begin = 0
       out_mu = []
       out_logstd = []
       for i, model in enumerate(self.de_models):
            partial_h = h[:, begin:(begin+self.test_slice[i])]
            
            out = model(partial_h)
            
            d = out.shape[1]//2
            out_mu.append(out[:, :d])
            out_logstd.append(out[:, d:])
            begin += self.test_slice[i]
       
   
       # Prediction of Image and joints 
       Out_im, Out_mu,Out_tip = out_mu[0],out_mu[1],out_mu[2]
       # Initialization grad in the trees graph of the network  
       z1.grad = torch.zeros(z1.size(), dtype=torch.float, requires_grad=True).to('cuda')
       Out_mu.backward((mu_attr - Out_mu)/2, retain_graph=True)
       grad_mu = torch.clone(z1.grad) #to caculate the gradients dz while mutliply mu_std*(mu_attr - Out_mu)/0.001
       # Backward pass for both Image and joints 
       z1.grad = torch.zeros(z1.size(),  dtype=torch.float, requires_grad=True).to('cuda')
       Out_im.backward((Im_attr - Out_im)/10, retain_graph=True)# 0.005
       grad_Im = torch.clone(z1.grad)
       
       z1.grad = torch.zeros(z1.size(),  dtype=torch.float, requires_grad=True).to('cuda')
       Out_tip.backward((Tip_attr - Out_tip)/10, retain_graph=True)# 0.005
       grad_Tip = torch.clone(z1.grad)
       

       return Out_im, Out_mu, Out_tip,grad_Im, grad_mu,grad_Tip 
    def perception_std(self, z1, Im_attr, mu_attr,Tip_attr,Im_std,mu_std,Tip_std):
       #Very important
       z1 = Variable(z1, requires_grad=True)
       h = self.shared_decoder(z1)
       begin = 0
       out_mu = []
       out_logstd = []
       for i, model in enumerate(self.de_models):
            partial_h = h[:, begin:(begin+self.test_slice[i])]
            
            out = model(partial_h)
            
            d = out.shape[1]//2
            out_mu.append(out[:, :d])
            out_logstd.append(out[:, d:])
            begin += self.test_slice[i]
       
   
       # Prediction of Image and joints 
       Out_im, Out_mu,Out_tip = out_mu[0],out_mu[1],out_mu[2]
       # Initialization grad in the trees graph of the network  
       z1.grad = torch.zeros(z1.size(), dtype=torch.float, requires_grad=True).to('cuda')
       Out_mu.backward(mu_std*(mu_attr - Out_mu)/2, retain_graph=True)
       grad_mu = torch.clone(z1.grad) #to caculate the gradients dz while mutliply mu_std*(mu_attr - Out_mu)/0.001
       # Backward pass for both Image and joints 
       z1.grad = torch.zeros(z1.size(),  dtype=torch.float, requires_grad=True).to('cuda')
       Out_im.backward(Im_std*(Im_attr - Out_im)/10, retain_graph=True)# 0.005
       grad_Im = torch.clone(z1.grad)
       
       z1.grad = torch.zeros(z1.size(),  dtype=torch.float, requires_grad=True).to('cuda')
       Out_tip.backward(Tip_std*(Tip_attr - Out_tip)/10, retain_graph=True)# 0.005
       grad_Tip = torch.clone(z1.grad)
       

       return Out_im, Out_mu, Out_tip,grad_Im, grad_mu,grad_Tip 



       
        




