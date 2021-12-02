# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 15:37:49 2019

@author: Knxie
"""

import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from scipy.io import loadmat
from torch.autograd import Variable as Var
import random



cuda = torch.cuda.is_available()
dtype = torch.FloatTensor

save_dir = './model'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model', default='deepRFF', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--train_data', default='data/Train400', type=str, help='path of train data')
parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=180, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
args = parser.parse_args()

class myRff(nn.Module):
    def __init__(self, input_dim=6, output_dim=200, sigma=1):
        super(myRff,self).__init__()
        net1 = []
        net1.append(nn.Linear(input_dim, output_dim))
        net1.append(nn.BatchNorm1d(output_dim))
        self.net1 = nn.Sequential(*net1)
        self.weight = Var(torch.from_numpy(np.sqrt(2*sigma)*np.random.randn(input_dim, output_dim)))
        self.linearNet = net1
        self._initialize_weights()
        
    def forward(self, x):
        x1 = self.net1(x)
        tmp1 = torch.cos(x1)
        tmp2 = torch.sin(x1)
        y1 = torch.cat((tmp1,tmp2),1)        
        
        return y1
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
           # elif isinstance(m, nn.Linear):
             #   m.weight = self.weight

class rffNet(nn.Module):
    def __init__(self, input_dim=6,output_dim=500, sigma=1):
        super(rffNet,self).__init__()
        self.net1 = myRff(input_dim,output_dim,sigma)
        self.net2 = myRff(2*output_dim,output_dim,sigma)
        self.net3 = myRff(2*output_dim,output_dim,sigma)
        self.net4 = myRff(2*output_dim,output_dim,sigma)
        self.net5 = myRff(2*output_dim,output_dim,sigma)
        self.output_net = nn.Linear(2*output_dim,1)
        
                
    def forward(self,x):
        out = self.net1(x)
        out = self.net2(out)
        out = self.net3(out)
        out = self.net4(out)
        out = self.net5(out)
        
        out = self.output_net(out)
        
        return out

def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)
    
    

    
    
    
if __name__ == '__main__':  

    log('process begins.')  
#data preparation
    train_flag = 1
    start_time = time.time()
    dataset_name = 'spambase'
    
    m = loadmat('./UCI/'+dataset_name+'.mat')
    
    flagtest = m["flagtest"]
    
    Xtrain  = np.array(m["X"])
    Ytrain = np.array(m["Y"])
    Xtest = Xtrain
    Ytest = Ytrain
    
    if flagtest==1:
        Xtrain  = m["X"]
        Ytrain = m["Y"]
        Xtest = m["X_test"]
        Ytest = m["Y_test"]
    else:
        rate = 0.5
        num = len(Ytrain)
        training_num = round(rate*num)
        
        testIdx = np.arange(num)
        
        trainIdx = np.random.choice(testIdx, training_num,False)
        testIdx = np.delete(testIdx, trainIdx)
        
        Xtrain = np.array(m["X"][trainIdx])
        Ytrain = np.array(m["Y"][trainIdx])
        
        Xtest = m["X"][testIdx]
        Ytest = m["Y"][testIdx]
    
    
    Xtrain = Var(torch.from_numpy(Xtrain).type(dtype))
    Ytrain = Var(torch.from_numpy(Ytrain).type(dtype))
    Xtest = Var(torch.from_numpy(Xtest).type(dtype))
    Ytest = Var(torch.from_numpy(Ytest).type(dtype))
    
    if train_flag:
        model = rffNet(Xtrain.shape[1])
        model.train()
        criterion = nn.MSELoss()
        
        if cuda:
            model = model.cuda()
            
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        epoch_loss = 0
    
            
        for i in range(1000):
            optimizer.zero_grad()
            if cuda:
                Xtrain, Ytrain = Xtrain.cuda(), Ytrain.cuda()
            loss = criterion(model(Xtrain), Ytrain)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('%4d  /  loss = %2.5f' % (i+1,  loss.item()))
                
        elapsed_time = time.time() - start_time
        torch.save(model, os.path.join(save_dir,dataset_name+'_model.pth')) 
        
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
    
        
        #test process
        
        torch.cuda.synchronize()
        start_time = time.time()
        
#        Xtest = Xtrain;
#        Ytest = Ytrain;
#        
        Xtest_ = Xtest.cuda()
        Y_ = model(Xtest_)  # inference
        
        Y = torch.sign(Y_).cpu()
        Ytest = Ytest.cpu()
      
        K = np.reshape((Y==Ytest).numpy(),(-1,1))
        
        k = np.sum(K)
        
        test_acc = k/K.shape[0]
        
        log('process ends.')       
        print('\ntest_acc: %2.3f'%(test_acc.item()))
    
    else:
    
        save_dir = './model'
        model_name = dataset_name+'_model.pth'
        model = []
        
        if not os.path.exists(os.path.join(save_dir, model_name)):
            print('--------Model does not exist?\n')
        else:
            # model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
            model = torch.load(os.path.join(save_dir, model_name))
            print('load trained model--'+model_name)
            
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
    
        
        #test process
        
        torch.cuda.synchronize()
        start_time = time.time()
        
#        Xtest = Xtrain;
#        Ytest = Ytrain;
#        
        Xtest_ = Xtest.cuda()
        Y_ = model(Xtest_)  # inference
        
        Y = torch.sign(Y_).cpu()
        
        K = np.reshape((Y==Ytest).numpy(),(-1,1))
        
        k = np.sum(K)
        
        test_acc = k/K.shape[0]
        
        log('process ends.')       
        print('\ntest_acc: %2.3f'%(test_acc.item()))
    
    
    
    