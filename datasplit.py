# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:59:41 2025

@author: JL
"""

import scanpy as sc
import numpy as np
import json
from sklearn.model_selection import train_test_split
from utils import type_to_label_dict, convert_type_to_label, set_seed



def train_val_test_split(mat, labels, test_size):
    """
    Split data into training / validation / test sets.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    validation_size = test_size
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        mat, labels, test_size=test_size, random_state=0, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=validation_size / (1 - test_size), random_state=0, shuffle=True
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_data_split_intra_h5ad(name, min_cell_per_type=20):
    # Load preprocessed .h5ad
    h5ad_path = f'./GenexpNet/datasets/intra/preprocessed/processed_{name}.h5ad'
    adata = sc.read_h5ad(h5ad_path)
    
    adata.var_names = adata.var['gene'].astype(str)

    # Filter out small cell types
    celltype_counts = adata.obs['celltype'].value_counts()
    valid_celltypes = celltype_counts[celltype_counts >= min_cell_per_type].index
    adata = adata[adata.obs['celltype'].isin(valid_celltypes)].copy()

    # Generate numeric labels and store in obs
    label_dict = type_to_label_dict(adata.obs['celltype'])
    label_num = np.array(convert_type_to_label(adata.obs['celltype'], label_dict))
    adata.obs['label'] = label_num

    indices = np.arange(adata.n_obs)
    
    idx_train, idx_val, idx_test, y_train, y_val, y_test = train_val_test_split(
        mat=indices,      # Pass indices
        labels=label_num, # Pass labels
        test_size=0.1     # test_size parameter in the function
    )
    # ------------------------------------

    # Save path
    save_dir = f'./GenexpNet/datasets/intra/preprocessed/splited/splited_{name}/'

    # Physical slicing: adata[idx] automatically carries the raw layer and layers along
    dataset_info = {
        'Train': idx_train,
        'Val':   idx_val,
        'Test':  idx_test
    }

    for split_name, split_idx in dataset_info.items():
        adata_split = adata[split_idx].copy()
        # Write to file
        save_path = f'{save_dir}{split_name}_{name}.h5ad'
        adata_split.write(save_path, compression="gzip")
        print(f"Saved {split_name} set to {save_path}, shape: {adata_split.shape}")

    # Save label dictionary
    with open(f'{save_dir}dict_{name}.json', 'w') as f:
        json.dump(label_dict, f)


def get_data_split_inter_h5ad(name, idx, min_cell_per_type=20):
    # 1. Load preprocessed h5ad file
    h5ad_path = f'./GenexpNet/datasets/inter/preprocessed/processed_{name}.h5ad'
    adata = sc.read_h5ad(h5ad_path)
    
    adata.var_names = adata.var['gene'].astype(str)

    celltype_counts = adata.obs['celltype'].value_counts()
    valid_celltypes = celltype_counts[celltype_counts >= min_cell_per_type].index
    adata = adata[adata.obs['celltype'].isin(valid_celltypes)].copy()

    # 3. Build label mapping
    label_dict = type_to_label_dict(adata.obs['celltype'])
    label_num = np.array(convert_type_to_label(adata.obs['celltype'], label_dict))
    adata.obs['label'] = label_num # Store numeric labels back into obs

    # 4. Split by idx (Test = first idx rows, Train = remaining rows)
    # Note: Due to filtering small cell types, total rows may decrease; simple boundary protection
    total_obs = adata.n_obs
    current_idx = min(idx, total_obs)
    
    # Physical slicing: adata[0:idx] automatically carries raw and layers
    adata_test = adata[0:current_idx].copy()
    adata_train = adata[current_idx:].copy()

    # 5. Shuffle (shuffle training and test sets separately)
    def shuffle_adata(ad_obj):
        if ad_obj.n_obs == 0:
            return ad_obj
        shuf_indices = np.random.permutation(ad_obj.n_obs)
        return ad_obj[shuf_indices].copy()

    adata_train = shuffle_adata(adata_train)
    adata_test = shuffle_adata(adata_test)

    # 6. Save data
    save_dir = f'./GenexpNet/datasets/inter/preprocessed/splited/splited_{name}/'

    # Write h5ad with compression
    adata_train.write(f'{save_dir}Train_{name}.h5ad', compression="gzip")
    adata_test.write(f'{save_dir}Test_{name}.h5ad', compression="gzip")

    # Save label dictionary
    save_path_dict = f'{save_dir}dict_{name}.json'
    with open(save_path_dict, 'w') as f:
        json.dump(label_dict, f)
    
    print(f"Inter-dataset {name} split finished. idx: {current_idx}, Train: {adata_train.shape}, Test: {adata_test.shape}")


def get_data_split_crobatch_h5ad(name, min_cell_per_type=20):
    # 1. Load preprocessed h5ad
    h5ad_path = f'./GenexpNet/datasets/cross_batch/preprocessed/processed_{name}.h5ad'
    adata = sc.read_h5ad(h5ad_path)
    
    # Ensure gene names are the index
    adata.var_names = adata.var['gene'].astype(str)

    # 2. Distinguish batch1 and batch2 by index prefix
    # During preprocessing_sc_crobatch, indices were prefixed with 'data1_' and 'data2_'
    train_mask = adata.obs_names.str.startswith('data1')
    test_mask = adata.obs_names.str.startswith('data2')
    
    adata_train = adata[train_mask].copy()
    adata_test = adata[test_mask].copy()

    # 3. Define function to filter small cell types (ensures raw layer follows slicing)
    def filter_by_type(ad_obj, thresh):
        counts = ad_obj.obs['celltype'].value_counts()
        valid = counts[counts >= thresh].index
        return ad_obj[ad_obj.obs['celltype'].isin(valid)].copy()

    # Filter training and test sets separately
    adata_train = filter_by_type(adata_train, min_cell_per_type)
    adata_test = filter_by_type(adata_test, min_cell_per_type)

    # 4. Build label dictionary (based on training set)
    all_celltypes = sorted(
        set(adata_train.obs['celltype'].astype(str)) |
        set(adata_test.obs['celltype'].astype(str))
    )

    label_dict = {ct: i for i, ct in enumerate(all_celltypes)}
    
    # Store numeric labels in training set
    adata_train.obs['label'] = np.array(convert_type_to_label(adata_train.obs['celltype'], label_dict))
    
    # Store numeric labels in test set (if test set has unseen cell types, convert function handles it)
    adata_test.obs['label'] = np.array(convert_type_to_label(adata_test.obs['celltype'], label_dict))

    # 5. Shuffle
    def shuffle_adata(ad_obj):
        idx = np.random.permutation(ad_obj.n_obs)
        return ad_obj[idx].copy()

    adata_train = shuffle_adata(adata_train)
    adata_test = shuffle_adata(adata_test)

    # 6. Save data
    save_dir = f'./GenexpNet/datasets/cross_batch/preprocessed/splited/splited_{name}/'

    # Save Train and Test; physical slicing automatically preserves raw and layers
    adata_train.write(f'{save_dir}Train_{name}.h5ad', compression="gzip")
    adata_test.write(f'{save_dir}Test_{name}.h5ad', compression="gzip")

    # Save dictionary
    save_path_dict = f'{save_dir}dict_{name}.json'
    with open(save_path_dict, 'w') as f:
        json.dump(label_dict, f)
    
    print(f"Cross-batch {name} split finished. Train: {adata_train.shape}, Test: {adata_test.shape}")



if __name__ == "__main__":
    set_seed(2024)
    
    dataset_intra = ['AMB', 'Baron Human','Segerstolpe', 'TM', 'Zheng 68K', 'Zheng sorted']
    
    dataset_inter = ['10Xv2', '10Xv3', 'Drop-Seq', 'inDrop', 'Seq-Well']
    
    dataset_crobat = ['Dendritic', 'Retina(5)', 'Retina(19)']
    
    idx_list = [6444, 3222, 3222, 3222, 3176]
    
    for i, name in enumerate(dataset_intra):
        print('{}'.format(i))
  
        get_data_split_intra_h5ad(name)    
    
    for i, name in enumerate(dataset_inter):
        print('{}'.format(i))
 
        get_data_split_inter_h5ad(name, idx_list[i])    
        
    for i, name in enumerate(dataset_crobat):
        print('{}'.format(i))
     
        get_data_split_crobatch_h5ad(name)       
            
        
        
        
        
        
        

        
