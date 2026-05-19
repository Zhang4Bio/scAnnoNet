# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:02:16 2025

@author: JL
"""
import os
import random
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def get_keys_from_value(dictionary, value):
    return [k for k, v in dictionary.items() if v == value]

def type_to_label_dict(types):
    # input -- types
    # output -- type_to_label dictionary
    dicts = {}
    all_type = list(set(types))
    for i in range(len(all_type)):
        dicts[all_type[i]] = i
    return dicts

# Convert types to labels
def convert_type_to_label(types, type_label_dict):
    # input -- list of types, and type_to_label dictionary
    # output -- list of labels
    type_list = list(types)
    labels = list()
    for t in type_list:
        labels.append(type_label_dict[t])
    return labels

def convert_label_to_type(types, type_label_dict):
    # input -- list of types, and type_to_label dictionary
    # output -- list of labels
    type_list = list(types)
    labels = list()
    for t in type_list:
        keys = get_keys_from_value(type_label_dict, t)
        labels.append(keys[0])
    return labels

def LDF_loss(attention,LDF_index,att_loss):
    loss = att_loss(attention.to(torch.float32),LDF_index.to(torch.float32))
    return loss

def CFC_loss(W):
    device = torch.device("cuda") if W.is_cuda else torch.device("cpu")

    n_row, n_col = W.shape
    WTW = torch.mm(W.T, W)
    
    loss = torch.abs(torch.mm(torch.ones(n_col,n_col).to(device), WTW).trace() - WTW.trace()) * (1/(n_row**2-len(torch.diag(WTW))))
    return loss

def sequence_scale(x):
  scaler = MinMaxScaler(feature_range=(0, 1))
  score= scaler.fit_transform(x)
  return score
    
def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

