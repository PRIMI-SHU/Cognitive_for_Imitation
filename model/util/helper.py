#!/usr/bin/env python3
import pandas as pd
import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from PIL import Image
from torch.utils.data import DataLoader
import math
import cv2
import matplotlib.pyplot as plt
#Functions for data processing
class Hparams:
    def __init__(self):
        # Data path
        self.babbling_data='/docker-ros/local_ws/catkin_ws/src/data_collection/sawyer_data'
        self.anchor_data='/docker-ros/local_ws/catkin_ws/src/data_collection/human_sample'
        self.compare_data='/docker-ros/local_ws/catkin_ws/src/data_collection/human_sample'
        self.embedding_data='/docker-ros/local_ws/catkin_ws/src/data_collection/embed'
        self.diff_data='/docker-ros/local_ws/Cognative_for_Imitation/data/diff.csv'
        self.action_data='/docker-ros/local_ws/Cognative_for_Imitation/data/action.csv'
        #model path
        self.mvae_path='/docker-ros/local_ws/Cognative_for_Imitation/trained_model/model_sawyer.ckpt'
        self.metric_path='/docker-ros/local_ws/Cognative_for_Imitation/trained_model/model_metric_1.ckpt'
        self.dynamic_path='/docker-ros/local_ws/Cognative_for_Imitation/trained_model/dynamic.ckpt'
        #annotation path
        self.blend_ann='/docker-ros/local_ws/Cognative_for_Imitation/data/vae1.csv'
        self.full_ann='/docker-ros/local_ws/Cognative_for_Imitation/data/contrastive.csv'
        
        #mvae setting
        self.in_chanels=[[-2,1024,256, 1,32, 64, 64, 128, 128, 256], [-1, 5, 32, 64, 64, 128, 128, 256, 128],[-1, 3, 32, 64, 64, 128, 128, 256, 128]]   
        self.out_chanels=[[-2,256, 1024, 256, 256, 128, 128, 64, 64, 32], [-1, 128, 256, 128, 128, 64, 64, 32, 10],[-1, 128, 256, 128, 128, 64, 64, 32, 6]]    
        self.in_shared=[512,128]
        self.out_shared=[128,512] 
        
        
        self.auto='auto'
        #metric setting
        self.chanel=[1024,512, 1,32, 64, 64, 128, 128, 256]
        self.mlp_dim= 512
        self.out_dim=256
        self.margin=1
        self.lr = 0.0001# for ADAm only







def create_dataset_contrastive(path,save_path):

    train_df = pd.DataFrame(columns=["img_path","obs"])
    list_=os.listdir(path)
   
    png_list=[]
    for l_ in list_:
        if ".npy" in l_:
            # png_list.append(int(l_[0:-4]))
            png_list.append(l_[0:-4])
    # png_list.sort()
    # print(png_list)
    npy_list=png_list.copy()
        
    train_df["img_path"]=png_list
    train_df["obs"]=npy_list
    train_df.to_csv (save_path, index = False, header=True)
    
def create_dataset_blend(path,save_path):

    train_df = pd.DataFrame(columns=["obs_path","blend"])
    list_=os.listdir(path)
    png_list=[]
    npy_list=[]
    for l_ in list_:
        if ".npy" in l_:
            
            png_list.append(l_[0:-4])
            
    # png_list.sort()
    npy_list=png_list.copy()
    png_list1=png_list.copy()
    for p_,n_, in zip(png_list,npy_list):
        
        png_list1.append(p_)
        npy_list.append(-1)
        
        png_list1.append(-1)
        npy_list.append(n_)
        
   
        
    train_df["obs_path"]=png_list1
    train_df["blend"]=npy_list
    train_df.to_csv (save_path, index = False, header=True)


    
def imshow(im_test,show):
        img1=im_test*255
        im1=img1.detach().cpu().numpy()
        im1=im1.transpose(1, 2, 0)
        # cv2.imshow('simulated',im1)
        if show:
            plt.imshow(im1)
            plt.show()
        return im1
       
    
def noise(x):
    x_noise= - 2 * torch.ones_like(x,dtype=torch.float)
    return x_noise   

class MVAE_DATA(Dataset):
    def __init__(self,root_dir,annotation_file,train_type):
        self.root_dir=root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.file_name=None
        assert train_type in ['train','test'],'train type should be either: train or test'
        self.img_id=None
        self.train_type=train_type
        if self.train_type=='train':
            joint=[]
            tip=[]        
            for index in range(len(self.annotations)):
                obs_id=self.annotations.iloc[index,1]
                img_id=self.annotations.iloc[index,0]
              
                if obs_id!='-1' and img_id!='-1':
                    obs=np.load(os.path.join(self.root_dir,f'{obs_id}.npy'),allow_pickle=True)
                    obs=obs.tolist()
                        
                    if isinstance(obs,dict):
                            joint.append(np.array([obs['joints'][0],obs['joints'][2],obs['joints'][3],obs['joints'][4],obs['joints'][5]]))
                            
                            tip.append(np.array(obs['tip']))
            self.offset_joint=[]
            self.scale_joint=[]
            self.offset_tip=[]
            self.scale_tip=[]
            joint=np.array(joint)
            tip=np.array(tip)
            for i in range(len(joint[0])):
            
                self.offset_joint.append(joint[:,i].min())
                self.scale_joint.append(joint[:,i].max()-joint[:,i].min()+(1e-6))
                
            for i in range(len(tip[0])):    
                self.offset_tip.append(tip[:,i].min())
                self.scale_tip.append(tip[:,i].max()-tip[:,i].min()+(1e-6))
    def normalize(self,x,offset,scale):
        x_normed=[]
        for i,x_i in enumerate(x):
            x_n=((x_i - offset[i]) / scale[i]) * 2 - 1
            # x_n=(x_i-self.offset[i])/(self.scale[i])
            x_normed.append(x_n)
        return np.array(x_normed)    
    def de_normalize(self,x,offset,scale):
            x_de=[]
            for i,x_i in enumerate(x):
                x_d=(x_i+1)/2*scale[i]+offset[i]
                x_de.append(x_d)
            return np.array(x_de)
    def __len__(self):
        return len(self.annotations)
    def get_modolity(self,index):
        img_id = self.annotations.iloc[index, 0]
        obs_id=self.annotations.iloc[index,1]
        
        if obs_id!='-1' and img_id!='-1': 
            
            image=cv2.imread(os.path.join(self.root_dir, f'{img_id}.jpg'))
            assert image is not None,print(img_id)
            image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            image=cv2.resize(image,(128,128))
            image=image.astype("float32")/255
            image=torch.tensor(image,dtype=torch.float)
            img=torch.reshape(image,(1,128,128))
            
            obs=np.load(os.path.join(self.root_dir,f'{obs_id}.npy'),allow_pickle=True)
            obs=obs.tolist()
            joint=np.array([obs['joints'][0],obs['joints'][2],obs['joints'][3],obs['joints'][4],obs['joints'][5]])
            joint=self.normalize(joint,self.offset_joint,self.scale_joint)
            joint=torch.tensor(joint,dtype=torch.float)
            tip=obs['tip']
            tip=self.normalize(tip,self.offset_tip,self.scale_tip)
            tip=torch.tensor(tip,dtype=torch.float)
          
            return (img,joint,tip,img,joint,tip)
        if obs_id!='-1' and img_id=='-1':
            
            image=cv2.imread(os.path.join(self.root_dir, f'{obs_id}.jpg'))
            assert image is not None,print(img_id)
            image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            image=cv2.resize(image,(128,128))
            
            image=image.astype("float32")/255
            image=torch.tensor(image,dtype=torch.float)
            
            img=torch.reshape(image,(1,128,128))
            img_noise=noise(img)
            
            
            obs=np.load(os.path.join(self.root_dir,f'{obs_id}.npy'),allow_pickle=True)
            obs=obs.tolist()
            joint=np.array([obs['joints'][0],obs['joints'][2],obs['joints'][3],obs['joints'][4],obs['joints'][5]])
            joint=self.normalize(joint,self.offset_joint,self.scale_joint)
            joint=torch.tensor(joint,dtype=torch.float)
            tip=obs['tip']
            tip=self.normalize(tip,self.offset_tip,self.scale_tip)
            tip=torch.tensor(tip,dtype=torch.float)
            tip_noise=-2*torch.ones_like(tip,dtype=torch.float)
            
            return (img_noise, joint,tip_noise,img,joint,tip)
        if img_id!='-1' and obs_id=='-1':
            
            
            
            image=cv2.imread(os.path.join(self.root_dir, f'{img_id}.jpg'))
            assert image is not None,print(img_id)
            image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            image=cv2.resize(image,(128,128))
            
            image=image.astype("float32")/255
            image=torch.tensor(image,dtype=torch.float)
            img=torch.reshape(image,(1,128,128))
            
            
            
            obs=np.load(os.path.join(self.root_dir,f'{img_id}.npy'),allow_pickle=True)
            obs=obs.tolist()
            joint=np.array([obs['joints'][0],obs['joints'][2],obs['joints'][3],obs['joints'][4],obs['joints'][5]])
            joint=self.normalize(joint,self.offset_joint,self.scale_joint)
            joint=torch.tensor(joint,dtype=torch.float)
            tip=obs['tip']
            tip=self.normalize(tip,self.offset_tip,self.scale_tip)
            tip=torch.tensor(tip,dtype=torch.float)
            joint_noise=-2*torch.ones_like(joint,dtype=torch.float)
            tip_noise=-2*torch.ones_like(tip,dtype=torch.float)
        
            return (img, joint_noise,tip_noise,img,joint,tip)
    def get_embedding(self,index):
        img_id = self.annotations.iloc[index, 0]
        self.img_id=img_id
        image=cv2.imread(os.path.join(self.root_dir, f'{img_id}_robot.jpg'),cv2.IMREAD_GRAYSCALE)
        image=cv2.resize(image,(128,128))
        image=image.astype("float32")/255
        image=torch.tensor(image,dtype=torch.float)
        img=torch.reshape(image,(1,128,128))
        
        return img
    def __getitem__(self, index):
        if self.train_type=='train':
           return self.get_modolity(index)
        elif self.train_type=='test':
            return self.get_embedding(index)
        
class Contrastive_DATA(Dataset):
    def __init__(self,anchor_data,compare_data,annotation_file,mode):
       self.anchor_data=anchor_data
       self.compare_data=compare_data
       self.annotations = pd.read_csv(annotation_file)
       self.ann_list=list(self.annotations.values.flatten())
       self.mode=mode
       self.max_=[]
       self.min_=[]
       assert self.mode in ['train','test'],'mode should be either: train or test'
       if self.mode=='train':
           self.x=[[0]*3]*(max(self.ann_list)+1)
           for index in self.ann_list:
            
            obs=np.load(os.path.join(self.compare_data,f'{index}.npy'),allow_pickle=True)
            obs=obs.tolist()
           
            self.x[index]=obs['tip']
            
       if self.mode=='test':
          x=[]
          for index in range(len(self.annotations)):
            obs_id=self.annotations.iloc[index,0]
            if obs_id!=-1:
                obs=np.load(os.path.join(self.anchor_data,f'{obs_id}.npy'),allow_pickle=True)
                obs=obs.tolist()
                # x.append(np.array(obs['joints'][0][0:5]))
                x.append(np.array([obs['joints'][0],obs['joints'][2],obs['joints'][3],obs['joints'][4],obs['joints'][5]]))
                
          x=np.array(x)
          self.offset=[]
          self.scale=[]
          for i in range(len(x[0])):
                temp=x[:,i]
                self.offset.append(temp.min())
                self.scale.append(temp.max()-temp.min()+(1e-6))
                self.max_.append(temp.max())
                self.min_.append(temp.min())
    def normalize(self,x):
        x_normed=[]
        for i,x_i in enumerate(x):
            x_n=((x_i - self.offset[i]) / self.scale[i]) * 2 - 1
            x_normed.append(x_n)
        return x_normed    
    def de_normalize(self,x):
        x_de=[]
        for i,x_i in enumerate(x):
            x_d=(x_i+1)/2*self.scale[i]+self.offset[i]
            x_de.append(x_d)
        return x_de
    def toTensor(self,image):
        image=image.astype("float32")/255
        image=torch.tensor(image,dtype=torch.float)
        img=torch.reshape(image,(1,128,128))
        return img
    def __len__(self):
        return len(self.annotations)
    def get_train(self,index):
        img_id = self.annotations.iloc[index, 0]
        self.img_id=img_id
        result_list=self.ann_list.copy()
       
        obs=np.load(os.path.join(self.compare_data,f'{img_id}.npy'),allow_pickle=True)
        obs=obs.tolist()
        obs1=obs['tip']
        
        index=self.x.index(obs1)
        result_list.remove(index)
        np.random.shuffle(result_list)
        negative=result_list.pop()    
        
        
        obs=np.load(os.path.join(self.anchor_data,f'{img_id}.npy'),allow_pickle=True)
        obs=obs.tolist()
        anchor=obs['emed1'][0]
        
        anchor=torch.tensor(anchor,dtype=torch.float)
        anchor=anchor.squeeze(0)
        
        
        # anchor=cv2.imread(os.path.join(self.root_dir, f'{img_id}.jpg'),cv2.IMREAD_GRAYSCALE)
        pos=cv2.imread(os.path.join(self.compare_data, f'{img_id}_human.jpg'),cv2.IMREAD_GRAYSCALE)
        pos=cv2.resize(pos,(128,128))
        
        
        
        
        neg=cv2.imread(os.path.join(self.compare_data, f'{negative}_human.jpg'),cv2.IMREAD_GRAYSCALE)
        neg=cv2.resize(neg,(128,128))
        
        
        # joint=torch.tensor(joint,dtype=torch.float)
        return anchor,self.toTensor(pos),self.toTensor(neg)
    
    def get_test(self,index):
        img_id = self.annotations.iloc[index, 0]
        self.img_id=img_id
        
        obs=np.load(os.path.join(self.anchor_data,f'{img_id}.npy'),allow_pickle=True)
        obs=obs.tolist()
        
        
       
        joint=np.array([obs['joints'][0],obs['joints'][2],obs['joints'][3],obs['joints'][4],obs['joints'][5]])
        
        tip=obs['tip']
        
        joint=torch.tensor(joint,dtype=torch.float)
        tip=torch.tensor(tip,dtype=torch.float)
        
        anchor=cv2.imread(os.path.join(self.anchor_data, f'{img_id}_robot.jpg'),cv2.IMREAD_GRAYSCALE)
        anchor=cv2.resize(anchor,(128,128))
        pos=cv2.imread(os.path.join(self.compare_data, f'{img_id}_human.jpg'),cv2.IMREAD_GRAYSCALE)
        pos=cv2.resize(pos,(128,128))
        
        
        
        # joint=torch.tensor(joint,dtype=torch.float)
        return self.toTensor(anchor),self.toTensor(pos),joint,tip
    def __getitem__(self, index):
        if self.mode=='train':
           return self.get_train(index)
        if self.mode=='test':
            return self.get_test(index)

def get_std(config):
    
    config.full_ann='/docker-ros/local_ws/metric_robot/data1.csv'
    dataset=MVAE_DATA(config.babbling_data,config.full_ann,'train')
    joint=[]
    image=[]
    tip=[]
    for data in dataset:
        joint.append(data[1].numpy().tolist())
        image.append(data[0].numpy().tolist())
        tip.append(data[2].numpy().tolist())
        
    joint=np.array(joint)
    mu_std=np.std(joint,axis=0)
    mu_std=np.reshape(mu_std,(1,3))
    df=pd.DataFrame(mu_std)
    df.to_csv('/docker-ros/local_ws/Metric_Internal_Model/joint_std.csv')
    
    tip=np.array(tip)
    tip_std=np.std(tip,axis=0)
    tip_std=np.reshape(tip_std,(1,3))
    df=pd.DataFrame(tip_std)
    df.to_csv('/docker-ros/local_ws/Metric_Internal_Model/tip_std.csv')
    
    
    image=np.array(image)
    im_std=np.std(image,axis=0)
    im_std=np.reshape(im_std,(128,128))
    df=pd.DataFrame(im_std)
    df.to_csv('/docker-ros/local_ws/Metric_Internal_Model/image_std.csv')
    
        
if __name__=='__main__':
            
    # create_dataset_blend('/docker-ros/local_ws/catkin_ws/src/data_collection/sawyer_all2','/docker-ros/local_ws/catkin_ws/src/data_collection/vae1.csv')
          
    create_dataset_contrastive('/docker-ros/local_ws/catkin_ws/src/data_collection/test_marker','/docker-ros/local_ws/catkin_ws/src/data_collection/contrastive.csv')         
    