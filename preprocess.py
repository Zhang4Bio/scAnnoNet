# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:06:41 2025

@author: JL
"""

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

from utils import set_seed


def preprocessing_sc_intra(data_name, copy=True, highly_genes=4000, min_cell_per_type=20,
                            filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    read_path_mat = f'./GenexpNet/datasets/intra/{data_name}/{data_name}.csv'
    read_path_lab = f'./GenexpNet/datasets/intra/{data_name}/Labels.csv'
    X = pd.read_csv(read_path_mat, delimiter=',', index_col=0).reset_index(drop=True)
    label = pd.read_csv(read_path_lab)
    label.columns = ['celltype']

    adata = ad.AnnData(X, obs=label, dtype='float64')
    adata.var['gene'] = list(X.columns)

    celltype_counts = adata.obs['celltype'].value_counts()
    valid_celltypes = celltype_counts[celltype_counts >= min_cell_per_type].index
    adata = adata[adata.obs['celltype'].isin(valid_celltypes)].copy()

    if copy:
        adata = adata.copy()

    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1000)
        sc.pp.filter_cells(adata, min_counts=3)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    if logtrans_input:
        sc.pp.log1p(adata)
        
    adata.layers["log1p"] = adata.X.copy()
    
    if highly_genes is not None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=highly_genes, subset=True)
    adata.raw = adata.copy() 
    if normalize_input:
        sc.pp.scale(adata, max_value=10)

    return adata


def preprocessing_sc_inter(data_name, copy=True, highly_genes=4000, min_cell_per_type=20,
                            size_factors=True, normalize_input=True, logtrans_input=True):
    read_path_mat = f'./GenexpNet/datasets/inter/{data_name}/{data_name}.csv'
    read_path_lab = f'./GenexpNet/datasets/inter/{data_name}/Labels.csv'
    X = pd.read_csv(read_path_mat, delimiter=',', index_col=0).reset_index(drop=True)
    label = pd.read_csv(read_path_lab)
    label.columns = ['celltype']

    adata = ad.AnnData(X, obs=label, dtype='float64')
    adata.var['gene'] = list(X.columns)

    celltype_counts = adata.obs['celltype'].value_counts()
    valid_celltypes = celltype_counts[celltype_counts >= min_cell_per_type].index
    adata = adata[adata.obs['celltype'].isin(valid_celltypes)].copy()

    if copy:
        adata = adata.copy()

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    if logtrans_input:
        sc.pp.log1p(adata)

    adata.layers["log1p"] = adata.X.copy()
    if highly_genes is not None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=highly_genes, subset=True)
    adata.raw = adata.copy() 
    if normalize_input:
        sc.pp.scale(adata, max_value=10)

    return adata


def preprocessing_sc_crobatch(data_name, copy=True, highly_genes=4000, min_cell_per_type=20,
                              filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    read_path_mat1 = f'./GenexpNet/datasets/cross_batch/{data_name}/batch1.csv'
    read_path_lab1 = f'./GenexpNet/datasets/cross_batch/{data_name}/batch1label.csv'
    read_path_mat2 = f'./GenexpNet/datasets/cross_batch/{data_name}/batch2.csv'
    read_path_lab2 = f'./GenexpNet/datasets/cross_batch/{data_name}/batch2label.csv'

    X1 = pd.read_csv(read_path_mat1, delimiter=',', index_col=0)
    X2 = pd.read_csv(read_path_mat2, delimiter=',', index_col=0)

    # 统一索引
    X1.index = ['data1_'+str(i) for i in range(X1.shape[0])]
    X2.index = ['data2_'+str(i) for i in range(X2.shape[0])]
    merged_X = pd.concat([X1, X2], axis=0)

    label1 = pd.read_csv(read_path_lab1, index_col=0)
    label2 = pd.read_csv(read_path_lab2, index_col=0)
    label1.columns = ['celltype']
    label2.columns = ['celltype']
    label1.index = X1.index
    label2.index = X2.index
    merged_label = pd.concat([label1, label2], axis=0)

    adata = ad.AnnData(merged_X, obs=merged_label)
    adata.var['gene'] = list(merged_X.columns)

    celltype_counts = adata.obs['celltype'].value_counts()
    valid_celltypes = celltype_counts[celltype_counts >= min_cell_per_type].index
    adata = adata[adata.obs['celltype'].isin(valid_celltypes)].copy()


    if copy:
        adata = adata.copy()

    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1000)
        sc.pp.filter_cells(adata, min_counts=3)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    if logtrans_input:
        sc.pp.log1p(adata)
        
    adata.layers["log1p"] = adata.X.copy()
    if highly_genes is not None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=highly_genes, subset=True)
    adata.raw = adata.copy() 
    if normalize_input:
        sc.pp.scale(adata, max_value=10)

    return adata

def write_text_matrix_sc_intra(data_name):
    
    adata = preprocessing_sc_intra(data_name) 
    
    save_path = f'./GenexpNet/datasets/intra/preprocessed/processed_{data_name}.h5ad'
    adata.write(save_path)
    


def write_text_matrix_sc_inter(data_name):
    

    adata = preprocessing_sc_inter(data_name) 
    
    save_path = f'./GenexpNet/datasets/inter/preprocessed/processed_{data_name}.h5ad'
    adata.write(save_path)
    
    
def write_text_matrix_sc_crobatch(data_name):
    

    adata = preprocessing_sc_crobatch(data_name) 
    
    save_path = f'./GenexpNet/datasets/cross_batch/preprocessed/processed_{data_name}.h5ad'
    adata.write(save_path)
    
    
    
if __name__ == "__main__":
    
    set_seed(2025)
    
    
    dataset_sc_intra = ['AMB', 'Baron Human','Segerstolpe', 'TM', 'Zheng 68K', 'Zheng sorted']
    
    dataset_sc_inter = ['10Xv2', '10Xv3', 'Drop-Seq', 'inDrop', 'Seq-Well']
    
    dataset_sc_crosbatch = ['Dendritic', 'Retina(5)', 'Retina(19)']
    

    for i in dataset_sc_intra:
        print(i)
        write_text_matrix_sc_intra(i)   
    for i in dataset_sc_inter:
        print(i)
        write_text_matrix_sc_inter(i)
    for i in dataset_sc_crosbatch:
        print(i)
        write_text_matrix_sc_crobatch(i)
    
    
    