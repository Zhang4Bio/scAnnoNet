# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:36:41 2023

@author: JL
"""

import torch.nn as nn
import torch
import torch.nn.init as init

def _weights_init(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)

class SE_block(nn.Module):
  def __init__(self,datat_dim,dropout):     
    super(SE_block,self).__init__()
    self.se_blcok = nn.Sequential(      
        nn.Linear(datat_dim,20),
        nn.Tanh(),
        nn.Dropout(p=dropout),
        nn.Linear(20,datat_dim),
        nn.Sigmoid()      
        )
  def forward(self, x):    
    x = self.se_blcok(x)
    return x

class GenexpNet(nn.Module):
  def __init__(self, input_dim,n_class,dropout):
    super(GenexpNet,self).__init__()   
    self.se_enc =  nn.Linear(input_dim, 20)
    self.se_tanh = nn.Tanh()
    self.se_drop = nn.Dropout(p=dropout)
    self.se_dec = nn.Linear(20,input_dim)
    self.se_sigmoid = nn.Sigmoid()      
    self.enc1 =  nn.Linear(input_dim, 128)
    self.relu1 = nn.ReLU()
    self.dropout1 = nn.Dropout(dropout)
    self.enc2 =  nn.Linear(128, 64)
    self.relu2 = nn.ReLU()
    self.dropout2 = nn.Dropout(dropout)
    self.enc3 =  nn.Linear(64, n_class)
    self.apply(_weights_init)
    
  def forward(self, x):

    input_mat = x    
    Amat = self.se_sigmoid(self.se_dec(self.se_drop(self.se_tanh(self.se_enc(x)))))
    Weighted_x = torch.mul(input_mat, Amat)
    x = self.dropout1(self.relu1(self.enc1(Weighted_x)))
    z = self.dropout2(self.relu2(self.enc2(x)))
    x = self.enc3(z)

    return x, Amat, Weighted_x, self.se_dec.weight.data, z 


