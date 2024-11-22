import time
from collections import defaultdict
from torch.utils.data import DataLoader
import torch
from torch.utils.data.dataloader import DataLoader


class Trainer:
    def __init__(self,model,optimizer,config,train_data,test_data,iter,test_every,save_error):
        
        #model setup
        self.config=config
        
        if self.config.auto=='auto':
           device = 'cuda' if torch.cuda.is_available() else 'cpu'
           self.device=device
        else:
            self.device=self.config
        
        self.model=model.to(self.device)
        
        print("running on device", self.device)
        
        #data setup
        self.train_data=train_data
        self.test_data=test_data
        self.iter=iter
        self.error_train=0
        self.error_test=0
        self.epoch=0
        self.test_every=test_every
        #train function
        self.forward_function=None
        self.loss_function=None
        
        #call back function
        
        self.callbacks=defaultdict(list)
        self.optimizer=optimizer
        
        self.save_error=save_error
    
        
        
        
    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)
            
    
    def run(self):
        self.model.train()
        
        
        for epoch in range(self.iter):
            self.error_train=0
            self.error_test=0
            for i,x in enumerate(self.train_data):
                self.optimizer.zero_grad()
                output=self.forward_function(self.model,x,self.device)
                error=self.loss_function(self.model,output,self.device)
                
                error.backward()
                self.optimizer.step()
                self.error_train+=error.item()
                
            self.error_train/=(i+1)  
            if self.save_error:  
                if self.error_train<self.save_error:
                    self.save_error=self.error_train
                    self.trigger_callbacks('on_save')
                
            if self.test_every:
                if epoch%self.test_every==0:
                
                    self.epoch=epoch
                    
                    self.trigger_callbacks('on_batch_end')
            
            
                
        
        
        
        
        
        
        
            
        