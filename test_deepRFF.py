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


class test(object):
    def __init__(self, v1,v2):
        self.v1 = v1
        self.v2 = v2
        print('initialize successfully.')
        
    def add(self):
        return self.v1+self.v2
    
    
    
if __name__ == '__main__':
    
    save_dir = './model'
    model_name = 'model.pth'
    model = []
    
    if not os.path.exists(os.path.join(save_dir, model_name)):
        print('--------Model does not exist?\n')
    else:
        # model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
        model = torch.load(os.path.join(save_dir, model_name))
        print('load trained model')
        
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()

    
   
    