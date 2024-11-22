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
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

# get the package root path
main_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
module_dir = os.path.join(main_dir)
sys.path.append(module_dir)
from util.helper import *
from networks.vae_blend import *
train_config=Hparams()
model=blend_vae(train_config.in_chanels,train_config.out_chanels,train_config.in_shared,train_config.out_shared)    
dataset=MVAE_DATA(train_config.anchor_data,train_config.full_ann,'test')

model.load_state_dict(torch.load(train_config.mvae_path))

parent_d=train_config.embedding_data
model.eval()
for i in range(len(dataset)):
    embeddings={'emed1':[]}
    x=dataset[i]
   
   
    x=torch.reshape(x,(1,1,128,128))
   
    embeds=model.out_embedding(x)
    
    
    embeddings['emed1'].append(embeds)
    np.save(os.path.join(parent_d,f'{dataset.img_id}.npy'),embeddings)