import os
import logging
from time import time
import numpy as np 
import pickle
import random

import torch
import torch.utils.data as utils_data
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import  roc_curve, auc, roc_auc_score
import matplotlib
import matplotlib.pyplot as plt

import Dataset 
import models 
import params
import training_functions

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


import matplotlib.colors as mcolors

def test_model_dict(p):
    # Set Random Seeds:
    if torch.cuda.is_available() and p.CUDA:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            torch.cuda.manual_seed_all(0)
    else:
        device = torch.device("cpu")
            
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    np.random.seed(1)
    random.seed(1)

    # Instantiate Model:
    
    model = p.model_dictionary['ref'](p.BATCH_SIZE, device, p.model_dictionary['hyperparams'], p)
    optimizer = p.model_dictionary['optimizer'](params = model.parameters(), lr = p.LR)
    lc_loss_func = p.model_dictionary['lc loss function']()
    if p.model_dictionary['hyperparams']['probabilistic output']:
        traj_loss_func = training_functions.NLL_loss
    else:
        traj_loss_func = p.model_dictionary['traj loss function']()
    # Instantiate Dataset: 
    tr_dataset = Dataset.LCDataset(p.TRAIN_DATASET_DIR, p.TR_DATA_FILES, 
        in_seq_len = p.IN_SEQ_LEN,
        out_seq_len = p.TGT_SEQ_LEN,
        end_of_seq_skip_len = p.SKIP_SEQ_LEN,
        data_type = p.model_dictionary['data type'], 
        state_type = p.model_dictionary['state type'], 
        keep_plot_info= False, 
        unbalanced = p.UNBALANCED,
        force_recalc_start_indexes = False)
    #val_dataset = Dataset.LCDataset(p.TRAIN_DATASET_DIR, p.VAL_DATA_FILES,  data_type = p.model_dictionary['data type'], state_type = p.model_dictionary['state type'], keep_plot_info= False, states_min = tr_dataset.states_min, states_max = tr_dataset.states_max, output_states_min = tr_dataset.output_states_min, output_states_max = tr_dataset.output_states_max)
    
    te_dataset = Dataset.LCDataset(p.TEST_DATASET_DIR, p.TE_DATA_FILES,
        in_seq_len = p.IN_SEQ_LEN,
        out_seq_len = p.TGT_SEQ_LEN,
        end_of_seq_skip_len = p.SKIP_SEQ_LEN,
        data_type = p.model_dictionary['data type'], 
        state_type = p.model_dictionary['state type'], 
        keep_plot_info= True, 
        import_states = True,
        unbalanced = p.UNBALANCED,
        force_recalc_start_indexes = False,
        states_min = tr_dataset.states_min, 
        states_max = tr_dataset.states_max,
        output_states_min = tr_dataset.output_states_min, 
        output_states_max = tr_dataset.output_states_max)
    
    # Evaluate:
    te_result_dic, traj_df = training_functions.eval_top_func(p, model, lc_loss_func, traj_loss_func, te_dataset, device)
    

if __name__ == '__main__':

   
    #torch.cuda.empty_cache()
    p = params.ParametersHandler('Constant_Parameter.yaml', 'highD.yaml', './config')
    #p = params.ParametersHandler('ManouvreTransformerTraj.yaml', 'highD.yaml', './config')
    # Do Not import experiment file for constant parameter models
    #experiment_file = 'experiments/ManouvreTransformerTraj_highD_2022-06-28 14:55:44.049101'
    #p.import_experiment(experiment_file)
    test_model_dict(p)
    
