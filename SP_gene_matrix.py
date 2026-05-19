# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 09:42:02 2025

@author: JL
"""


import numpy as np
import pandas as pd

import scanpy as sc
import json
from utils import set_seed

from scipy.stats import t
from numba import njit, prange

def compute_log2fc_matrix(avg_expr, pseudocount=1e-6):
    """
    Input: avg_expr is a c × m DataFrame (one cell type per row, one gene per column)
    Output: c × m log2FC matrix, log2 fold change for each cell type vs. all other types
    """
    
    cell_types = avg_expr.index
    log2fc_matrix = pd.DataFrame(index=cell_types, columns=avg_expr.columns)

    for cell_type in cell_types:
        expr_target = avg_expr.loc[cell_type] + pseudocount
        expr_others = avg_expr.drop(index=cell_type).mean(axis=0) + pseudocount
        log2fc = np.log2(expr_target / expr_others)
        log2fc_matrix.loc[cell_type] = log2fc

    return log2fc_matrix.astype(float)    




@njit(parallel=True)
def compute_mean_std(A):
    n, p = A.shape
    mean = np.empty(p)
    std = np.empty(p)
    for i in prange(p):
        col = A[:, i]
        m = 0.0
        for j in range(n):
            m += col[j]
        m /= n
        mean[i] = m

        s = 0.0
        for j in range(n):
            s += (col[j] - m) ** 2
        std[i] = np.sqrt(s / (n - 1))
    return mean, std

@njit(parallel=True)
def fast_pearsonr_only(A, B):
    n, p1 = A.shape
    _, p2 = B.shape

    A_mean, A_std = compute_mean_std(A)
    B_mean, B_std = compute_mean_std(B)

    corr = np.empty((p1, p2))

    for i in prange(p1):
        for j in prange(p2):
            num = 0.0
            for k in range(n):
                num += (A[k, i] - A_mean[i]) * (B[k, j] - B_mean[j])
            denom = (n - 1) * A_std[i] * B_std[j]
            corr[i, j] = num / denom
    return corr


def compute_p_values(corr, n):
    df = n - 2
    t_stat = corr * np.sqrt(df / (1 - corr ** 2))
    pval = 2 * t.sf(np.abs(t_stat), df)
    return pval



def prepare_data(data_name, datatype, copy=True, 
                 FCthreshold=1, p_thresh=0.001, top_n=20):
    # 1. Load training set
    train_path = f'./GenexpNet/datasets/{datatype}/preprocessed/splited/Splited_{data_name}/Train_{data_name}.h5ad'
    adata = sc.read_h5ad(train_path)
    adata = adata.copy() if copy else adata
    adata.var['gene'] = list(adata.var_names)  

    # 2. Load label dictionary
    label_dict_path = f'./GenexpNet/datasets/{datatype}/preprocessed/splited/Splited_{data_name}/dict_{data_name}.json'
    with open(label_dict_path) as f:
        label_dict = json.load(f)
    inv_label_dict = {v: k for k, v in label_dict.items()}
    
    # Extract raw data (non‑negative log1p‑transformed data)
    if adata.raw is not None:
        # Note: raw.X is often a sparse matrix
        raw_X = adata.raw.X.toarray() if hasattr(adata.raw.X, "toarray") else adata.raw.X
        # raw.var_names stores original gene names
        calc_matrix = pd.DataFrame(raw_X, columns=adata.raw.var_names)
    else:
        print("Warning: No raw layer found. Using X layer and forcing non‑negative values.")
        X_data = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        calc_matrix = pd.DataFrame(np.maximum(X_data, 0), columns=adata.var['gene'].values)

    cell_labels = [inv_label_dict[i] for i in adata.obs['label'].values]
    calc_matrix['cell_type'] = cell_labels
    
    # Compute mean expression matrix
    mean_matrix = calc_matrix.groupby('cell_type').mean()

    # 4. Compute Log2FC (based on non‑negative means)
    FC_mat = compute_log2fc_matrix(mean_matrix, pseudocount=1e-6)
    
    # 5. top_markers: fixed top_n
    top_markers = {}
    marker_rows = [] # For building DataFrame
    for cell_type in FC_mat.index:
        filtered_genes = FC_mat.loc[cell_type][FC_mat.loc[cell_type] > FCthreshold]
        genes_sorted = filtered_genes.sort_values(ascending=False).index
        top_genes = genes_sorted[:top_n] if len(genes_sorted) > top_n else genes_sorted
        top_markers[cell_type] = top_genes
        
        for g in top_genes:
            marker_rows.append({'celltype': cell_type, 'gene': g})

    top_markers_df = pd.DataFrame(marker_rows)

    # ----------------  Step 6: Co‑expression calculation ----------------
    # Use calc_matrix (from raw) with index as cell_labels
    calc_matrix.index = cell_labels
    
    # all_gene should come from calc_matrix columns, ensuring coverage of all markers
    all_gene = list(calc_matrix.columns)
    all_gene.remove('cell_type') # Remove the helper column
    
    all_marker = list({g for genes in top_markers.values() for g in genes})
    all_other_gene = [g for g in all_gene if g not in all_marker]

    selected_genes_by_type = {}
    results = []

    for cell_type, marker_genes in top_markers.items():
        marker_genes = list(marker_genes)
        if len(marker_genes) == 0: continue
        
        # Apply type_mask, compute correlations only within the current cell type
        type_mask = (calc_matrix.index == cell_type)
        
        # Small offset to prevent zero standard deviation
        X_marker = calc_matrix.loc[type_mask, marker_genes].values + 1e-5
        X_other = calc_matrix.loc[type_mask, all_other_gene].values + 1e-5
        n_cells = X_marker.shape[0]

        # If sample size is too small (<3), correlation is meaningless
        if n_cells < 3:
            selected_genes_by_type[cell_type] = []
            continue

        corr_mat = fast_pearsonr_only(X_other, X_marker)
        pval_mat = compute_p_values(corr_mat, n_cells)

        selected_genes = []
        for i in range(corr_mat.shape[0]):
            for j in range(corr_mat.shape[1]):
                if pval_mat[i,j] < p_thresh:
                    results.append({
                        'celltype': cell_type,
                        'marker_gene': marker_genes[j],
                        'coexpressed_gene': all_other_gene[i],
                        'pval': pval_mat[i,j]
                    })
                    selected_genes.append(all_other_gene[i])
        selected_genes_by_type[cell_type] = list(set(selected_genes))

    co_gene = list(set(sum(selected_genes_by_type.values(), [])))
    sp_co_df = pd.DataFrame(results)

    feature_list = list(set(co_gene + all_marker))
    filt_adata = adata[:, adata.var.gene.isin(feature_list)]

    # ---------------- Build a summary table of relevant genes (co‑expressed genes) ----------------
    stats_rows = []
    for ct, m_genes in top_markers.items():
        m_len = len(m_genes)
        co_len = len(selected_genes_by_type.get(ct, []))
        stats_rows.append({
            'cell_type': ct,
            'num_marker_genes': m_len,         # number of marker genes
            'num_coexpressed_genes': co_len,   # number of co‑expressed genes
            'num_features': m_len + co_len     # total features for this cell type
        })
    
    # Add a global deduplicated total row
    stats_rows.append({
        'cell_type': 'total_unique_features',
        'num_marker_genes': len(all_marker),
        'num_coexpressed_genes': len(co_gene),
        'num_features': len(feature_list)
    })
    
    stats_df = pd.DataFrame(stats_rows)

    return sp_co_df, pd.DataFrame(feature_list), filt_adata, stats_df, top_markers_df


def prepare_data_crobatch(data_name, datatype='cross_batch', copy=True, 
                           FCthreshold=1.5, p_thresh=0.001, top_n=20):
    """
    Cross‑batch training set feature calculation:
    1. Use adata.raw to compute Log2FC and extract markers.
    2. Compute Pearson correlation within each cell type to extract co‑expressed genes.
    """
    # 1. Load training set
    train_path = f'./GenexpNet/datasets/{datatype}/preprocessed/splited/Splited_{data_name}/Train_{data_name}.h5ad'
    adata = sc.read_h5ad(train_path)
    adata = adata.copy() if copy else adata
    # Ensure var_names are gene names as strings
    adata.var['gene'] = list(adata.var_names)

    # 2. Load label dictionary
    label_dict_path = f'./GenexpNet/datasets/{datatype}/preprocessed/splited/Splited_{data_name}/dict_{data_name}.json'
    with open(label_dict_path) as f:
        label_dict = json.load(f)
    inv_label_dict = {v: k for k, v in label_dict.items()}

    # ---------------- Extract raw data (log1p‑transformed non‑negative data) ----------------
    if adata.raw is not None:
        # Extract raw data matrix
        raw_X = adata.raw.X.toarray() if hasattr(adata.raw.X, "toarray") else adata.raw.X
        # raw.var_names stores original gene names
        calc_matrix = pd.DataFrame(raw_X, columns=adata.raw.var_names)
    else:
        print("Warning: No raw layer found. Using X layer and forcing non‑negative values.")
        X_data = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        calc_matrix = pd.DataFrame(np.maximum(X_data, 0), columns=adata.var['gene'].values)

    cell_labels = [inv_label_dict[i] for i in adata.obs['label'].values]
    calc_matrix['cell_type'] = cell_labels
    
    # 3. Compute mean expression matrix
    mean_matrix = calc_matrix.groupby('cell_type').mean()

    # 4. Compute Log2FC (based on non‑negative means)
    FC_mat = compute_log2fc_matrix(mean_matrix, pseudocount=1e-6)
    
    # 5. top_markers: fixed top_n
    top_markers = {}
    marker_rows = [] 
    for cell_type in FC_mat.index:
        filtered_genes = FC_mat.loc[cell_type][FC_mat.loc[cell_type] > FCthreshold]
        genes_sorted = filtered_genes.sort_values(ascending=False).index
        top_genes = genes_sorted[:top_n] if len(genes_sorted) > top_n else genes_sorted
        top_markers[cell_type] = top_genes
        
        for g in top_genes:
            marker_rows.append({'celltype': cell_type, 'gene': g})

    top_markers_df = pd.DataFrame(marker_rows)

    # ---------------- Corrected Step 6: Co‑expression calculation ----------------
    # Use calc_matrix (from raw) with index as cell_labels
    calc_matrix.index = cell_labels
    
    all_gene = list(calc_matrix.columns)
    if 'cell_type' in all_gene:
        all_gene.remove('cell_type') 
    
    all_marker = list({g for genes in top_markers.values() for g in genes})
    all_other_gene = [g for g in all_gene if g not in all_marker]

    selected_genes_by_type = {}
    results = []

    for cell_type, marker_genes in top_markers.items():
        marker_genes = list(marker_genes)
        if len(marker_genes) == 0: continue
        
        # Apply type_mask, compute correlations only within the current cell type
        type_mask = (calc_matrix.index == cell_type)
        
        # Use the non‑negative calc_matrix throughout; add offset to prevent zero standard deviation
        X_marker = calc_matrix.loc[type_mask, marker_genes].values + 1e-5
        X_other = calc_matrix.loc[type_mask, all_other_gene].values + 1e-5
        n_cells = X_marker.shape[0]

        # Check sample size
        if n_cells < 3:
            selected_genes_by_type[cell_type] = []
            continue

        corr_mat = fast_pearsonr_only(X_other, X_marker)
        pval_mat = compute_p_values(corr_mat, n_cells)

        selected_genes = []
        for i in range(corr_mat.shape[0]):
            for j in range(corr_mat.shape[1]):
                if pval_mat[i,j] < p_thresh:
                    results.append({
                        'celltype': cell_type,
                        'marker_gene': marker_genes[j],
                        'coexpressed_gene': all_other_gene[i],
                        'pval': pval_mat[i,j]
                    })
                    selected_genes.append(all_other_gene[i])
        selected_genes_by_type[cell_type] = list(set(selected_genes))

    co_gene = list(set(sum(selected_genes_by_type.values(), [])))
    sp_co_df = pd.DataFrame(results)

    # Aggregate feature list
    feature_list = list(set(co_gene + all_marker))
    filt_adata = adata[:, adata.var.gene.isin(feature_list)]

    # ---------------- Generate summary table stats_df ----------------
    stats_rows = []
    for ct, m_genes in top_markers.items():
        m_len = len(m_genes)
        co_len = len(selected_genes_by_type.get(ct, []))
        stats_rows.append({
            'cell_type': ct,
            'num_marker_genes': m_len,
            'num_coexpressed_genes': co_len,
            'num_features': m_len + co_len
        })
    
    stats_rows.append({
        'cell_type': 'total_unique_features',
        'num_marker_genes': len(all_marker),
        'num_coexpressed_genes': len(co_gene),
        'num_features': len(feature_list)
    })
    
    stats_df = pd.DataFrame(stats_rows)

    return sp_co_df, pd.DataFrame(feature_list), filt_adata, stats_df, top_markers_df


def write_text_matrix_sc_intra(data_name, top_gene=20):
    
    sp_co_df, feature_list, adata, stats_df, top_markers_df = prepare_data(data_name, 'intra',top_n=top_gene, FCthreshold=1.5, p_thresh=0.00001)

    marker_save_path = f'./GenexpNet/datasets/intra/preprocessed/top_marker_{data_name}_{top_gene}.csv'
    df_save_path = f'./GenexpNet/datasets/intra/preprocessed/gene_list_{data_name}_{top_gene}.csv'
    feature_save_path = f'/GenexpNet/datasets/intra/preprocessed/feature_list_{data_name}_{top_gene}.csv'
    stats_save_path = f'./GenexpNet/datasets/intra/preprocessed/stats_{data_name}_{top_gene}.csv'

    sp_co_df.to_csv(df_save_path, index=False, header=True)
    feature_list.to_csv(feature_save_path, index=False, header=True)
    top_markers_df.to_csv(marker_save_path, index=False, header=True)
    stats_df.to_csv(stats_save_path, index=False)


def write_text_matrix_sc_inter(data_name, top_gene=20):
    sp_co_df, feature_list, adata, stats_df, top_markers_df = prepare_data(data_name, 'inter', top_n=top_gene, FCthreshold=1.5, p_thresh=0.00001)
    
    marker_save_path = f'./GenexpNet/datasets/inter/preprocessed/top_marker_{data_name}_{top_gene}.csv'
    df_save_path = f'./GenexpNet/datasets/inter/preprocessed/gene_list_{data_name}_{top_gene}.csv'
    feature_save_path = f'./GenexpNet/datasets/inter/preprocessed/feature_list_{data_name}_{top_gene}.csv'
    stats_save_path = f'./GenexpNet/datasets/inter/preprocessed/stats_{data_name}_{top_gene}.csv'

    sp_co_df.to_csv(df_save_path, index=False, header=True)
    feature_list.to_csv(feature_save_path, index=False, header=True)
    top_markers_df.to_csv(marker_save_path, index=False, header=True)
    stats_df.to_csv(stats_save_path, index=False)


def write_text_matrix_sc_crobatch(data_name, top_gene=20):
    sp_co_df, feature_list, adata, stats_df, top_markers_df = prepare_data_crobatch(data_name, top_n=top_gene, FCthreshold=1.5, p_thresh=0.00001)
    
    
    marker_save_path = f'./GenexpNet/datasets/cross_batch/preprocessed/top_marker_{data_name}_{top_gene}.csv'
    df_save_path = f'./GenexpNet/datasets/cross_batch/preprocessed/gene_list_{data_name}_{top_gene}.csv'
    feature_save_path = f'./GenexpNet/datasets/cross_batch/preprocessed/feature_list_{data_name}_{top_gene}.csv'
    stats_save_path = f'./GenexpNet/datasets/cross_batch/preprocessed/stats_{data_name}_{top_gene}.csv'

    sp_co_df.to_csv(df_save_path, index=False, header=True)
    feature_list.to_csv(feature_save_path, index=False, header=True)
    top_markers_df.to_csv(marker_save_path, index=False, header=True)
    stats_df.to_csv(stats_save_path, index=False)
    
if __name__ == "__main__":
    
    set_seed(2025)
    
    dataset_sc_intra = ['AMB', 'Baron Human','Segerstolpe', 'TM', 'Zheng 68K', 'Zheng sorted']
    
    dataset_sc_inter = ['10Xv2', '10Xv3', 'Drop-Seq', 'inDrop', 'Seq-Well']
    
    dataset_sc_crosbatch = ['Dendritic', 'Retina(5)', 'Retina(19)']
    

    
    for gene_num in  [10, 30, 50]:
        
        for i in dataset_sc_intra:
            print(i)
            write_text_matrix_sc_intra(i, top_gene=gene_num)   
            
        for i in dataset_sc_inter:
            print(i)
            write_text_matrix_sc_inter(i, top_gene=gene_num)
           
        for i in dataset_sc_crosbatch:
            print(i)
            write_text_matrix_sc_crobatch(i, top_gene=gene_num)
        
        
    
