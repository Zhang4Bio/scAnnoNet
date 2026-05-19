# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 21:50:49 2024

@author: JL
"""

import time
import argparse

import json

import pandas as pd
import numpy as np
import scanpy as sc

from utils import set_seed, LDF_loss, CFC_loss
from discriminant import Discriminant_score
from model import GenexpNet

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from imblearn.metrics import sensitivity_score, specificity_score


import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import WeightedRandomSampler


import warnings
warnings.filterwarnings('ignore')

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument("--n_epochs_pre", type=int, default=20,
                        help='Number of epochs for pre-training.')
    parser.add_argument("--n_epochs_cla", type=int, default=100,
                        help='Number of epochs for classificaiton training.')  
    parser.add_argument("--batch_size", default=256, type=int,
                        help="batch size for training ",
                        )

    # optimization
    
    parser.add_argument('--lr_rec', type=float, default=0.1, 
                        help='pretraining learning rate.')
    parser.add_argument('--lr_cla', type=float, default=0.0001, 
                        help='classification learning rate.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate.')
    parser.add_argument('--beta', type=float, default=1,
                        help='Regularization rate of discriminant score.') 
    parser.add_argument('--w1', type=float, default=1000,
                        help='Weight of LDF loss.') 
    parser.add_argument('--w2', type=float, default=1,
                        help='Weight of CFC loss.') 
    parser.add_argument('--lr_decay_steps', type=int, default=20,
                        help='Decay the learning rate interval step size')
    parser.add_argument('--lr_decay_rate', type=float, default=0.99,
                        help='Decay the learning rate')
    
    # model dataset
    parser.add_argument('--oversampling', type=bool, default=True,
                        help='Apply oversampling or not')
    
    parser.add_argument('--data_type', type=str, default='intra',
                        choices=[
                                'intra',                 
                                'inter',
                                'cross_batch'
                                 ],
                        help='dataset name')
    parser.add_argument('--dataset', type=str, default='AMB',
                        choices=[
                                'AMB',            
                                'Baron Human',   
                                'Segerstolpe',
                                'TM',            
                                'Zheng 68K',     
                                'Zheng sorted',  
                                
                                '10Xv2',
                                '10Xv3',
                                'Drop-Seq',
                                'inDrop',
                                'Seq-Well',
                                
                                'Dendritic',
                                'Retina(5)',
                                'Retina(19)',
                                 ],
                        help='dataset name')
    
    # other setting
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of iterations')
    parser.add_argument('--seed', type=int, default=2025, 
                        help='Random seed.')

    args = parser.parse_args()
    
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


#Pretrain the attention module.
def Attention_PREtrain(model, train_loader, optimazer_pre, scheduler_pre, LDF_index, ATT_loss, args):
    model.train()
    best_epoch = 0
    best_loss = float("inf")
    history_avg = []  
    train_start =  time.time()
    
    for epoch in range(args.n_epochs_pre):
    
        All_A = []
        train_loss = 0.0   
        times = 0
        epoch_start = time.time()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            
            times=times+1
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            _,A,_,_,_ = model(inputs)
            All_A.append(A)
            optimazer_pre.zero_grad()      
            loss = LDF_loss(A,torch.from_numpy(LDF_index).to(args.device), ATT_loss)
            loss.backward()
            optimazer_pre.step()
            scheduler_pre.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / times
        history_avg.append([epoch, avg_train_loss])
        epoch_end = time.time()
        if (epoch+1)%5 == 0:
            print('Train:[{}/{}]\n' 
                  'Training Loss: {:.8f}\n'
                  'Time: {:.4f}'.format(
                      epoch+1, args.n_epochs_pre, loss.item(), epoch_end-epoch_start))
            
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            best_epoch = epoch
            
    All_A = torch.cat((All_A),0)
    train_end = time.time()           
    print("---Epoch:{}/{}--- \n Training: Loss: {:.4f}\n\t Best Loss: {:.4f} at epoch {:03d}\n Time: {:.4f}s".format(
        epoch + 1, args.n_epochs_pre, avg_train_loss, best_loss, best_epoch, train_end-train_start))
     
    return history_avg, All_A        

def CLA_train(model, train_loader, val_loader, optimizer, scheduler_cla, cla_loss, LDF_index, ATT_loss, args):
    model.train()   
    best_epoch = 0
    best_loss = float("inf")
    
    history_avg = []
    
    for epoch in range(args.n_epochs_cla):  
        epoch_start = time.time()
        
        All_A = []
        W_x = []

        times = 0
        total_batch = 0
        train_loss,train_correct = 0.0,0
     
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            
        
            times=times+1
            inputs, labels = inputs.to(args.device), labels.type(torch.LongTensor).to(args.device)
        
            outputs,A,Weighted_X,sedec_w,z = model(inputs)
            All_A.append(A)
            W_x.append(Weighted_X)
                       
            optimizer.zero_grad()
            
            loss1 = cla_loss(outputs.to(torch.float32).to(args.device),labels.type(torch.LongTensor).to(args.device))
            loss2 = LDF_loss(A,torch.from_numpy(LDF_index).to(args.device),ATT_loss)  
            loss = loss1 + args.w1*loss2 + args.w2*CFC_loss(z)/len(inputs)
            loss.backward()
            optimizer.step()
            scheduler_cla.step()

            train_loss += loss.item()
            
            predictions = torch.max(F.softmax(outputs.data,dim = 1), 1)[1]
            correct_batch = (predictions == labels).sum().item()

            temp_batch = labels.size(0)
            total_batch += temp_batch
            train_correct += correct_batch
            
        avg_train_loss = train_loss / times
        avg_train_acc = train_correct / total_batch
        
        avg_val_loss, avg_val_acc, _, _, _, _, _, _ = CLA_val(model, val_loader, cla_loss, args)
        
        history_avg.append([epoch, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc]) 
        
        All_A = torch.cat((All_A),0)
        new_x = torch.cat((W_x),0)
            
        epoch_end = time.time()
        
        if (epoch+1)%5 == 0:
            print("Epoch:[{}]/[{}] \n Train Loss: {:.4f} \t Train acc: {:.2f}%\n Val Loss: {:.4f} \t Val Acc: {:.2f}% \n Time: {:.4f}s".format(
                epoch+1, args.n_epochs_cla, avg_train_loss, avg_train_acc*100,
                avg_val_loss, avg_val_acc*100, epoch_end-epoch_start))

            print(
                   'cla lr: {:.8f}\n'                
                   'Time: {:.4f}'.format(
                       optimizer.param_groups[0]['lr'],  epoch_end-epoch_start))
            print("Best Loss: {:.4f} at epoch {}".format(best_loss, best_epoch))
          
    print("------Finished Training------")
    
    return history_avg,best_epoch,All_A,new_x 
    
def CLA_val(model, val_loader, criterion, args):
    
    model.eval()
    val_loss, val_correct = 0.0, 0
    acc, f1, recall, precision, sensitivity, specificity= 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    total_batch = 0
    times = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            times = times + 1
            n_row,n_col = inputs.size()
            inputs, labels = inputs.to(args.device), labels.type(torch.LongTensor).to(args.device)

            output,_,_,_,_ = model(inputs)
            loss = criterion(output, labels)
            val_loss += loss.item()
            
            predictions = torch.max(F.softmax(output.data,dim = 1), 1)[1]

            correct_batch = (predictions == labels).sum().item()
            
            temp_batch = labels.size(0)
            total_batch += temp_batch
            val_correct += correct_batch
            
            acc += accuracy_score(y_true = labels.cpu(), y_pred = predictions.cpu())
            f1 += f1_score(y_true=labels.cpu(), y_pred = predictions.cpu(), average='weighted')
            recall += recall_score(y_true=labels.cpu(), y_pred = predictions.cpu(), average='weighted')
            precision += precision_score(y_true=labels.cpu(), y_pred = predictions.cpu(), average='weighted')
            sensitivity += sensitivity_score(y_true=labels.cpu().data, y_pred = predictions.cpu().data, average='weighted')
            specificity += specificity_score(y_true=labels.cpu().data, y_pred = predictions.cpu().data, average='weighted')   
            
    avg_val_loss = val_loss / times
    
    return avg_val_loss, acc/times, f1/times, recall/times, precision/times, sensitivity/times, specificity/times, predictions 
    


def set_loader_Splited_scRNA(args):

    # 1. Load training, validation, and test data
    train_data = sc.read_h5ad(
        f'./GenexpNet/datasets/{args.data_type}/preprocessed/splited/splited_{args.dataset}/Train_{args.dataset}.h5ad'
    )
    val_data = sc.read_h5ad(
        f'./GenexpNet/datasets/{args.data_type}/preprocessed/splited/splited_{args.dataset}/Test_{args.dataset}.h5ad'
    )
    test_data = sc.read_h5ad(
        f'./GenexpNet/datasets/{args.data_type}/preprocessed/splited/splited_{args.dataset}/Test_{args.dataset}.h5ad'
    )

    # 2. Load label dictionary
    label_dict_path = f'./GenexpNet/datasets/{args.data_type}/preprocessed/splited/Splited_{args.dataset}/dict_{args.dataset}.json'
    with open(label_dict_path) as file:
        label_dict = json.load(file)

    # 3. Load feature index (marker / co‑expressed genes), ignoring column names
    feature_index_path = f'./GenexpNet/datasets/{args.data_type}/preprocessed/feature_list_{args.dataset}_{args.t_gene}.csv'
    feature_index = pd.read_csv(feature_index_path, header=None)[0].tolist()  # First column contains gene names

    feature_index = [str(x) for x in feature_index]

    # 4. Filter the columns corresponding to feature_index in train/val/test sets
    train_gene_list = list(train_data.var_names)  # Use var_names to ensure gene name strings
    feature_cols = [i for i, g in enumerate(train_gene_list) if g in feature_index]

    X_train = train_data.X[:, feature_cols]
    X_val   = val_data.X[:, feature_cols]
    X_test  = test_data.X[:, feature_cols]

    # 5. Labels
    y_train = train_data.obs['label'].values.reshape(-1)
    y_val   = val_data.obs['label'].values.reshape(-1)
    y_test  = test_data.obs['label'].values.reshape(-1)

    # 6. Sample count and data dimensions
    n_row, n_col = X_train.shape
    print(f"Number of samples: {n_row}, Number of features: {n_col}")
    args.data_dim = n_col

    # 7. Convert to tensors
    tensor_TrainValues = torch.FloatTensor(np.array(X_train)).float()
    tensor_ValValues   = torch.FloatTensor(np.array(X_val)).float()
    tensor_TestValues  = torch.FloatTensor(np.array(X_test)).float()

    tensor_TrainLabels = torch.FloatTensor(y_train).float()
    tensor_ValLabels   = torch.FloatTensor(y_val).float()
    tensor_TestLabels  = torch.FloatTensor(y_test).float()

    args.n_class = len(set(np.concatenate([y_train, y_val, y_test])))

    return tensor_TrainValues, tensor_ValValues, tensor_TestValues, \
           tensor_TrainLabels, tensor_ValLabels, tensor_TestLabels, label_dict

def set_loader_Splited_scRNA_OverSample(args, alpha=0.5, threshold=100):

    # 1. Load training/validation/test sets
    train_data = sc.read_h5ad(
        f'E./GenexpNet/datasets/{args.data_type}/preprocessed/splited/splited_{args.dataset}/Train_{args.dataset}.h5ad'
    )
    val_data = sc.read_h5ad(
        f'./GenexpNet/datasets/{args.data_type}/preprocessed/splited/splited_{args.dataset}/Test_{args.dataset}.h5ad'
    )
    test_data = sc.read_h5ad(
        f'./GenexpNet/datasets/{args.data_type}/preprocessed/splited/splited_{args.dataset}/Test_{args.dataset}.h5ad'
    )

    # 2. Load label dictionary
    label_dict_path = f'./GenexpNet/datasets/{args.data_type}/preprocessed/splited/Splited_{args.dataset}/dict_{args.dataset}.json'
    with open(label_dict_path) as file:
        label_dict = json.load(file)

    # 3. Load feature index (marker/co‑expressed genes)
    feature_index_path = f'./GenexpNet/datasets/{args.data_type}/preprocessed/feature_list_{args.dataset}_{args.t_gene}.csv'
    feature_index = pd.read_csv(feature_index_path, header=None)[0].tolist()  # First column contains gene names
    feature_index = [str(x) for x in feature_index]

    # 4. Filter the columns corresponding to feature_index in train/val/test sets
    train_gene_list = list(train_data.var_names)  # Use var_names to ensure gene name strings
    feature_cols = [i for i, g in enumerate(train_gene_list) if g in feature_index]

    X_train = train_data.X[:, feature_cols]
    X_val   = val_data.X[:, feature_cols]
    X_test  = test_data.X[:, feature_cols]

    # Convert sparse matrices to dense numpy arrays for further processing
    if hasattr(X_train, "toarray"): X_train = X_train.toarray()
    if hasattr(X_val, "toarray"): X_val = X_val.toarray()
    if hasattr(X_test, "toarray"): X_test = X_test.toarray()

    # 5. Labels
    y_train = train_data.obs['label'].values.reshape(-1)
    y_val   = val_data.obs['label'].values.reshape(-1)
    y_test  = test_data.obs['label'].values.reshape(-1)

    # 6. Sample count and data dimensions
    n_row, n_col = X_train.shape
    print(f"Number of samples: {n_row}, Number of features: {n_col}")
    args.data_dim = n_col
    args.n_class = len(set(np.concatenate([y_train, y_val, y_test])))

    # 7. Convert to tensors
    tensor_TrainValues = torch.FloatTensor(np.array(X_train)).float()
    tensor_ValValues   = torch.FloatTensor(np.array(X_val)).float()
    tensor_TestValues  = torch.FloatTensor(np.array(X_test)).float()

    tensor_TrainLabels = torch.as_tensor(y_train, dtype=torch.long)
    tensor_ValLabels   = torch.as_tensor(y_val, dtype=torch.long)
    tensor_TestLabels  = torch.as_tensor(y_test, dtype=torch.long)

    # ==========================================
    # 8. Over‑Sampling mechanism
    # ==========================================
    
    # Count samples per class in the training set, clamp_min(1) to avoid division by zero
    class_counts = torch.bincount(tensor_TrainLabels, minlength=args.n_class).clamp_min(1)
    
    # Compute maximum imbalance ratio
    imbalance_ratio = class_counts.max().item() / class_counts.min().item()
    
    # Determine weighting strategy based on imbalance severity
    if imbalance_ratio > threshold:
        # If severely imbalanced, apply smoothing alpha to prevent extreme penalties
        class_weights = 1.0 / (class_counts.float() ** alpha)
    else:
        # Normal inverse weighting
        class_weights = 1.0 / class_counts.float()  
        
    # Normalize weights
    class_weights = class_weights / class_weights.mean()
    
    # Assign weight to each training sample according to its class
    sample_weights = class_weights[tensor_TrainLabels]
    
    # Initialize WeightedRandomSampler
    samplers = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )

    return tensor_TrainValues, tensor_ValValues, tensor_TestValues, \
           tensor_TrainLabels, tensor_ValLabels, tensor_TestLabels, label_dict, samplers

def set_model(args):
    
    model = GenexpNet(input_dim = args.data_dim, n_class = args.n_class, dropout = args.dropout)
    para_cla = model.parameters()
    para_att = [
              {'params': model.se_enc.parameters()},
              {'params': model.se_tanh.parameters()},
              {'params': model.se_drop.parameters()},
              {'params': model.se_dec.parameters()},
              {'params': model.se_sigmoid.parameters()}
                ]


    optimazer_pre = torch.optim.Adam(para_att,lr=args.lr_rec)
    optimizer_cla = torch.optim.Adam(para_cla,lr=args.lr_cla)

    
    scheduler_1 = StepLR(optimazer_pre, step_size=args.lr_decay_steps, gamma=args.lr_decay_rate)
    
    scheduler_2 = StepLR(optimizer_cla, step_size=args.lr_decay_steps, gamma=args.lr_decay_rate)
    
    cla_loss = torch.nn.CrossEntropyLoss()
    ATT_loss = torch.nn.MSELoss()
    
    
    model.to(args.device)
    cla_loss.to(args.device)
    ATT_loss.to(args.device)

    return model, cla_loss, ATT_loss, optimazer_pre, optimizer_cla, scheduler_1, scheduler_2



def main():
    
        args = parse_args()
        set_seed(args.seed)
        
        if args.oversampling:
            Train_data, Val_data, Test_data, Train_label, Val_label, Test_label, label_dict, samplers = set_loader_Splited_scRNA_OverSample(args)
        else:
            Train_data, Val_data, Test_data, Train_label, Val_label, Test_label, label_dict = set_loader_Splited_scRNA(args)
                            
        LDF_index = Discriminant_score(Train_data.numpy(), Train_label.numpy(),args.beta)
            
        Train_dataset = Data.TensorDataset(Train_data, Train_label)
        Val_dataset = Data.TensorDataset(Val_data, Val_label)
        Test_dataset = Data.TensorDataset(Test_data, Test_label)
        
        if args.oversampling:
            
            train_loader = torch.utils.data.DataLoader(
                        Train_dataset,
                        batch_size=args.batch_size,
                        sampler=samplers,
                        shuffle=False,
                        drop_last=False
                    )
                
        else:
            train_loader = torch.utils.data.DataLoader(
                        Train_dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        drop_last=False
                    )
            
            
            
        val_loader = torch.utils.data.DataLoader(
                    Val_dataset,
                    batch_size=100000000,
                    shuffle=False,
                    drop_last=False
                )
            
        test_loader = torch.utils.data.DataLoader(
                    Test_dataset,
                    batch_size=100000000,
                    shuffle=False,
                    drop_last=False
                )
                  
        acc_list = list()
        f1_list = list()
        recall_list = list()
        precision_list = list()
        sensitivity_list = list()
        specificity_list = list()
        times_list = list()
        
            
        for i in range(args.iterations):
                print(args.dataset)
                print(i)
                print(args.g_num)
                print("iteration: {}".format(i+1))
                time_stratime = time.time()
                
                model, cla_loss, ATT_loss, optimazer_pre, optimizer_cla, scheduler_pre, scheduler_cla = set_model(args)
                
                print("==> Pre_training..")
                history_avg, All_A  = Attention_PREtrain(model, train_loader, optimazer_pre, scheduler_pre, LDF_index, ATT_loss, args)
                print("==> classification learning..")
                history_avg,best_epoch,All_A,new_x  = CLA_train(model, train_loader, val_loader, optimizer_cla, scheduler_cla, cla_loss, LDF_index, ATT_loss, args)
                test_loss, test_acc, test_f1, test_recall, test_precision, test_sensitivity, test_specificity, predictions = CLA_val(model, test_loader, cla_loss, args)
                
                time_endtime = time.time()
                time_cost = time_endtime-time_stratime
                
                acc_list.append(test_acc)
                f1_list.append(test_f1)
                recall_list.append(test_recall)
                precision_list.append(test_precision)
                sensitivity_list.append(test_sensitivity)
                specificity_list.append(test_specificity)
                times_list.append(time_cost)
                
                print("==> Test.. \n  Test acc: {:.6f} | Test f1: {:.6f} | Test recall: {:.6f} | Test precision: {:.6f} | Test sensitivity: {:.6f} | Test specificity: {:.6f} | Time: {:.2f}".format(
                     test_acc*100, test_f1*100, test_recall*100, test_precision*100, test_sensitivity*100, test_specificity*100, time_cost))
                
        print('all iterations')    
        print(np.mean((np.array(acc_list))))
        print(np.mean((np.array(f1_list))))
        print(np.mean((np.array(recall_list))))
        print(np.mean((np.array(precision_list))))
        print(np.mean((np.array(sensitivity_list))))
        print(np.mean((np.array(specificity_list))))
        print(np.mean((np.array(times_list))))
            
        contact_list = {"acc" : acc_list,
                        "f1" : f1_list,
                        "recall" : recall_list,
                        "precision" : precision_list,
                        "sensitivity" : sensitivity_list,
                        "specificity" : specificity_list,
                        "time_cost" : times_list   
            }
        
     
            
        result =  pd.DataFrame(contact_list)
        
        result.to_csv(
            f'./GenexpNet/datasets/result/{args.data_type}/{args.dataset}/Result_GenexpNet_{args.dataset}_{args.t_gene}.csv',
            encoding='gbk',
            index=False)
                
if __name__ == "__main__":
    main()        
    
    
    
    
