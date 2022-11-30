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
import model_specific_training_functions as mstf

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import kpis
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
    man_loss_func = p.model_dictionary['man loss function']
    model_eval_func = p.model_dictionary['model evaluation function']
    model_kpi_func = p.model_dictionary['model kpi function']
    if p.model_dictionary['hyperparams']['probabilistic output']:
        traj_loss_func = kpis.NLL_loss
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
    print(p.TE_DATA_FILES)
    #exit()
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
    kpi_dict = training_functions.eval_top_func(p, model_eval_func, model_kpi_func, model, (traj_loss_func, man_loss_func), te_dataset, device)
    p.export_evaluation(kpi_dict)

if __name__ == '__main__':

   
    #torch.cuda.empty_cache()
    #p = params.ParametersHandler('Constant_Parameter.yaml', 'highD.yaml', './config')
    p = params.ParametersHandler('SMTP.yaml', 'highD.yaml', './config')
    # Do Not import experiment file for constant parameter models
    #experiment_file = 'experiments/MTPMTT_highD_2022-08-22 15:47:03.709155'#MTPMTT_highD_2022-08-22 15:47:03.709155' #MMnTP_highD_2022-10-26 17:52:20.822886' #MMnTP_highD_2022-10-20 18:26:31.377915'
    #experiment_file = 'experiments/DMTP_highD_2022-10-30 15:56:20.089368'
    #experiment_file = 'experiments/MMnTP_highD_2022-11-03 11:22:06.670845'#MMnTP_highD_2022-11-02 01:47:35.849718'#'experiments/MMnTP_highD_2022-11-01 13:35:55.866717'#DMTP_highD_2022-10-30 15:56:20.089368'
    # Changes w.r.t. Training Hyperparameters
    experiment_file = 'experiments/SMTP_highD_2022-11-29 09:07:25.977406' # experiments/DMTP_highD_2022-11-29 13:21:03.655754'#'experiments/DMTP_highD_2022-11-29 13:21:03.655754'
    p.import_experiment(experiment_file)
    p.hyperparams['experiment']['debug_mode'] = False
    p.hyperparams['dataset']['balanced'] = True
    p.hyperparams['training']['batch_size'] = 64
    p.hyperparams['experiment']['multi_modal_eval'] = True
    # make sure to use following function to update hyperparameters
    p.match_parameters()
    test_model_dict(p)
    
