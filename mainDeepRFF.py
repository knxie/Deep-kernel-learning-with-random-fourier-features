# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 18:53:23 2019

main code after reconstruction

@author: Knxie
"""

import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from scipy.io import loadmat
from torch.autograd import Variable as Var
from collections import namedtuple
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
parser.add_argument('--cifar',default='/home/knight/桌面/deep_rff/pytorch_code/cifar10/cifar-10-batches-py',type=str,help='cifar_file path')
args = parser.parse_args()


#----self-defined loss functions ----begin
class hingeLoss(nn.Module):
    def __init__(self):
        super(myLoss,self).__init__()

    def forward(self, pred, truth):
      
        a=torch.max(1-pred.mul(truth),torch.zeros(pred.shape).cuda())
        
        return a.sum()


#----self-defined loss functions ----end
        
    
    
#----RFFnet ----begin
class myRff(nn.Module):
    def __init__(self, inputDim=6, outputDim=600, sigma=1):
        super(myRff,self).__init__()
        net1 = []
        net1.append(nn.Linear(inputDim, outputDimoutputDim))
        net1.append(nn.BatchNorm1d(outputDim))
#        net1.append(nn.Dropout(0.5))
        self.outputDim = outputDim
        self.net1 = nn.Sequential(*net1)
        self.sigma = sigma
        self.weight = Var(torch.from_numpy(np.sqrt(2*sigma)*np.random.randn(inputDim, outputDim)))
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
#            if isinstance(m, nn.Linear):
#                nn.init.normal_(m.weight,0,2/self.sigma)
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
    def __init__(self, inputDim=6, outputDim=500, resultDim=1, sigma=1, depth=3):
        super(rffNet,self).__init__()
        net = []
        net.append(myRff(inputDim,outputDim,sigma))
        for i in range(depth):
            net.append(myRff(2*outputDim,outputDim,sigma))
#        net.append(myRff(2*outputDim,500,sigma))
#        net.append(myRff(1000,20,sigma))
#        net.append(myRff(40,500,sigma))
#        net.append(myRff(1000,500,sigma))   
#        #mnist outputdim=10
            
        net.append(nn.Linear(2*outputDim,resultDim))
        self.net = nn.Sequential(*net)
        
        
        
                
    def forward(self,x):
        results=[]       
        out = self.net(x)
     
        return out    
    
#----RFFnet ----end

