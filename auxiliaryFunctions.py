# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 18:13:46 2019

--------auxiary functions





@author: Knxie
"""

import argparse
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from scipy.io import loadmat
from torch.autograd import Variable as Var
from collections import namedtuple 
import random


#----宏定义begin
cuda = torch.cuda.is_available()


#----宏定义end


#----loadDataset functions begin
def loadCifar10(file):
    import pickle
    
    X=[]
    Y=[]
    for i in range(5):
        with open(file+'/'+'data_batch_'+str(i+1),'rb') as fo:
            tmp=pickle.load(fo,encoding='bytes')
            if i==0:
                X=tmp[b'data']
                Y=tmp[b'labels']
            else:
                X=np.row_stack((X,tmp[b'data']))
                Y=np.append(Y,tmp[b'labels'],axis=0)
                
    with open(file+'/'+'test_batch','rb') as fo:
        tmp=pickle.load(fo,encoding='bytes')

        Xtest=tmp[b'data']
        Ytest=tmp[b'labels']
   
    return X,Y,Xtest,Ytest


def loadUCI(file):
    m = loadmat(file)
    flagtest = 0
    
    if ('flagtest' in m.keys()):
        flagtest = m["flagtest"]
    else:
        flagtest = 1
        
    if flagtest == 1:
        X = m["X_train"]
        Y = m["Y_train"]
        Xtest = m["X_test"]
        Ytest = m["Y_test"]
        return X,Y,Xtest,Ytest
    else:
        X = m["X"]
        Y = m["Y"]
        return X,Y
    
    
def loadIjcnn(fileTrain, fileTest):

    m=loadmat(fileTrain)
    m2=loadmat(fileTest)
    
    Xtrain=m2['X']
    Ytrain=m2['Y']
    Xtest=m['X']
    Ytest=m['Y']
    
    return Xtrain,Ytrain,Xtest,Ytest
        
        
#----loadDataset functions end
        
    
def splitDataset(X,Y,cvNum=5):
    rate = 1.0/cvNum
    
    num = len(Y)
    test_num = round(rate*num)
    
    Idx1  = [x for x in range(0,num)]
    Idx = [x for x in range(0,num)]
    random.shuffle(Idx)
    
    train_result=[];
    test_result=[];
    
    for i in range(cvNum):
        testIdx = Idx[i*test_num:max((i+1)*test_num-1,num-1)]
        if ((i+1)*test_num-1>=num):    
            testIdx = Idx[i*test_num:num-1] 
            
        trainIdx=np.delete(Idx1,testIdx)
    
        Xtrain = np.array(X[trainIdx])
        Ytrain = np.array(Y[trainIdx])
            
        
        Xtest = X[testIdx]
        Ytest = Y[testIdx] 
        
    return Xtrain,Ytrain,Xtest,Ytest




def classAcc(Xtrain,Ytrain,model,mseFlag=0):
    model.eval()
    acc=0
    flag=mseFlag
    
    
    train_acc=0
    if Xtrain.shape[0]<1000:
    
        Xtrain_ = Xtrain.cuda()
        output= model(Xtrain_) 
        
        if mseFlag == 1:    #use CrossEntropyLoss 
            predtrain = output.data.max(1, keepdim=True)[1]
            
            Ytrain=Ytrain.view(predtrain)
            predtrain=predtrain.cpu()
            Ytrain=Ytrain.cpu()
            K = np.reshape((predtrain==Ytrain).numpy(),(-1,1))
            k = np.sum(K)
            train_acc = k/K.shape[0]
            print('\nprocess_acc: %2.3f'%(train_acc.item()))
            
        elif mseFlag == 0:
            predtrain = output.data
            Ytrain=Ytrain.view(predtrain)
            predtrain=predtrain.cpu()
            Ytrain=Ytrain.cpu()
            K = np.reshape((torch.sign(predtrain)==torch.sign(Ytrain)).numpy(),(-1,1))
            k = np.sum(K)
            train_acc = k/K.shape[0]
            print('\nprocess_acc: %2.3f'%(train_acc.item()))
            
        
    else: 
        DDataset=TensorDataset(Xtrain,Ytrain)
        batch_size=64
    
        
        DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=False, batch_size=batch_size, shuffle=False)
        k=0
        a=0
        for n_count, batch_data in enumerate(DLoader):
          
            batch_x, batch_y=batch_data
            if cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            output=model(batch_x)
           
            if flag==1:
                predtrain = output.data
                predtrain=predtrain.cpu()
                batch_y=batch_y.view(predtrain.shape)
                batch_y=batch_y.cpu()
                 
                K = np.reshape((torch.sign(predtrain)==torch.sign(batch_y)).numpy(),(-1,1))
                k = k+ np.sum(K)
                a = a+K.shape[0]
            elif flag == 0:
                predtrain = output.data.max(1, keepdim=True)[1]
                predtrain=predtrain.cpu()
                batch_y=batch_y.view(predtrain.shape)
                batch_y=batch_y.cpu()
                K = np.reshape((predtrain==batch_y).numpy(),(-1,1))
                k =k+ np.sum(K)
                a = a+K.shape[0]
        
        
        train_acc = k/Ytrain.shape[0]
        print('\nprocess_acc: %2.3f'%(train_acc.item()))
         
        
    return train_acc    

    
    




