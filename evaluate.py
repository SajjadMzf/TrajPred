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
import models_dict as m
import training_functions

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


import matplotlib.colors as mcolors

def test_model_dict(model_dict, p):
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
    
    model = model_dict['ref'](p.BATCH_SIZE, device, model_dict['hyperparams'], p)
    optimizer = model_dict['optimizer'](params = model.parameters(), lr = p.LR)
    lc_loss_func = model_dict['lc loss function']()
    if model_dict['hyperparams']['probabilistic output']:
        traj_loss_func = training_functions.NLL_loss
    else:
        traj_loss_func = model_dict['traj loss function']()
    ttlc_loss_func = model_dict['ttlc loss function']()
    task = model_dict['hyperparams']['task']
    # Instantiate Dataset: 
    tr_dataset = Dataset.LCDataset(p.TRAIN_DATASET_DIR, p.TR_DATA_FILES, 
        in_seq_len = p.IN_SEQ_LEN,
        out_seq_len = p.TGT_SEQ_LEN,
        end_of_seq_skip_len = p.SKIP_SEQ_LEN,
        data_type = model_dict['data type'], 
        state_type = model_dict['state type'], 
        keep_plot_info= False, 
        unbalanced = p.UNBALANCED,
        force_recalc_start_indexes = False,
        traj_output = (task==params.TRAJECTORYPRED))
    #val_dataset = Dataset.LCDataset(p.TRAIN_DATASET_DIR, p.VAL_DATA_FILES,  data_type = model_dict['data type'], state_type = model_dict['state type'], keep_plot_info= False, traj_output = (task==params.TRAJECTORYPRED), states_min = tr_dataset.states_min, states_max = tr_dataset.states_max, output_states_min = tr_dataset.output_states_min, output_states_max = tr_dataset.output_states_max)
    
    te_dataset = Dataset.LCDataset(p.TEST_DATASET_DIR, p.TE_DATA_FILES,
        in_seq_len = p.IN_SEQ_LEN,
        out_seq_len = p.TGT_SEQ_LEN,
        end_of_seq_skip_len = p.SKIP_SEQ_LEN,
        data_type = model_dict['data type'], 
        state_type = model_dict['state type'], 
        keep_plot_info= True, 
        traj_output = (task==params.TRAJECTORYPRED), 
        import_states = True,
        unbalanced = p.UNBALANCED,
        force_recalc_start_indexes = False,
        states_min = tr_dataset.states_min, 
        states_max = tr_dataset.states_max,
        output_states_min = tr_dataset.output_states_min, 
        output_states_max = tr_dataset.output_states_max)
    '''
    print('training')
    print(tr_dataset.states_max)
    print(tr_dataset.states_min)
    print('validation')
    print(val_dataset.states_max)
    print(val_dataset.states_min)
    print('test')
    print(te_dataset.states_max)
    print(te_dataset.states_min)
    exit()
    '''
    # Evaluate:
    te_result_dic, traj_df = training_functions.eval_top_func(p, model, lc_loss_func, traj_loss_func, task, te_dataset, device, model_tag = model_dict['tag'])
    

if __name__ == '__main__':

    '''
    p = params.Parameters(SELECTED_MODEL = 'CONSTANT_PARAMETER', SELECTED_DATASET = 'HIGHD', UNBALANCED = False, ABLATION = False)

    model_dict = m.MODELS[p.SELECTED_MODEL]
    model_dict['hyperparams']['task'] = params.TRAJECTORYPRED
    model_dict['state type'] = 'ours' #it has to be ours for constant parameter model
    model_dict['tag'] = training_functions.update_tag(model_dict)
    test_model_dict(model_dict, p)
    '''
    #torch.cuda.empty_cache()
    p = params.Parameters(SELECTED_MODEL = 'TRANSFORMER_TRAJ', SELECTED_DATASET = 'HIGHD', UNBALANCED = False, ABLATION = False)

    model_dict = m.MODELS[p.SELECTED_MODEL]
    model_dict['hyperparams']['task'] = params.TRAJECTORYPRED
    model_dict['hyperparams']['multi modal'] = False
    model_dict['state type'] = 'ours'
    model_dict['tag'] = training_functions.update_tag(model_dict)
    test_model_dict(model_dict, p)
    
