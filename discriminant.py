# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 16:25:18 2023

@author: JL
"""

import numpy as np
from scipy.sparse import *
from sklearn.preprocessing import MinMaxScaler

from numpy import matrix



def sequence_scale(x):
  scaler = MinMaxScaler(feature_range=(0, 1))
  score= scaler.fit_transform(x)
  return score

def Discriminant_score(X, y, labda):
    """

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels       
    labda: {numpy array}, shape (1)
        input class labels

    Output
    ------
    score: {numpy array}, shape (,n_features)
        fisher score for each feature

    Reference
    ---------
    He, Xiaofei et al. "Laplacian Score for Feature Selection." NIPS 2005.
    Deng Cai et al. "Semi-supervised Discriminant Analysis." ICCV, 2007.
    """
    
    nSample, nDim = X.shape
    labels = np.unique(y)
    nClass = len(labels)
    total_mean = np.mean(X, axis=0)
    lbd = labda
    
    X_n = X - np.ones([nSample,nDim])*total_mean
     
    W = lil_matrix((nSample, nSample))
    for i in range(nClass):
        
        class_idx = (y == labels[i])
        class_idx_all = (class_idx[:, np.newaxis] & class_idx[np.newaxis, :]) 
        W[class_idx_all] = 1/np.sum(class_idx)
             
    D = np.array(W.sum(axis=1))
    D = diags(np.transpose(D),[0])
    L = D-W
    
    J = 2*np.dot(np.dot(np.transpose(X_n), L.todense()), X_n)
    
    D_prime = matrix(np.diagonal(np.dot(np.dot(np.transpose(X_n), D.todense()), X_n)))
    L_prime = matrix(np.diagonal(np.dot(np.dot(np.transpose(X_n), L.todense()), X_n)))
    
    fisher_numerator = D_prime - L_prime
    fisher_denominator_temp = L_prime
    fisher_denominator_temp[fisher_denominator_temp < 1e-12] = 1e5
    J__prime = np.diagonal(J)
    fisher_denominator = fisher_denominator_temp + lbd*J__prime
    
    
    score = np.array(fisher_numerator/fisher_denominator)[0,:]
    
    score = score.reshape(-1,1)
    score = sequence_scale(score).flatten()

    return np.transpose(score)

