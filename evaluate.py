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
import TPMs 
import params
import top_functions

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import kpis
import matplotlib.colors as mcolors
import TPMs
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
    tr_dataset = Dataset.LCDataset(p.TR.DATASET_DIR, p.TR.DATA_FILES, p.TR.MAP_INDEX, p.TR.MAP_DIRS,
        index_file = Dataset.get_index_file(p, p.TR,  'Tr'),
        data_type = p.model_dictionary['data type'], 
        state_type = p.model_dictionary['state type'], 
        use_map_features = p.hyperparams['model']['use_map_features'],
        keep_plot_info= False, 
        force_recalc_start_indexes = False)
    #val_dataset = Dataset.LCDataset(p.TRAIN_DATASET_DIR, p.VAL_DATA_FILES,  data_type = p.model_dictionary['data type'], state_type = p.model_dictionary['state type'], keep_plot_info= False, states_min = tr_dataset.states_min, states_max = tr_dataset.states_max, output_states_min = tr_dataset.output_states_min, output_states_max = tr_dataset.output_states_max)
    
    #exit()
    te_dataset = Dataset.LCDataset(p.TE.DATASET_DIR, p.TE.DATA_FILES, p.TE.MAP_INDEX, p.TE.MAP_DIRS,
        index_file = Dataset.get_index_file(p,p.TE,  'Te'),
        data_type = p.model_dictionary['data type'], 
        state_type = p.model_dictionary['state type'], 
        use_map_features = p.hyperparams['model']['use_map_features'],
        keep_plot_info= True, 
        import_states = True,
        force_recalc_start_indexes = False,
        states_min = tr_dataset.states_min, 
        states_max = tr_dataset.states_max,
        output_states_min = tr_dataset.output_states_min, 
        output_states_max = tr_dataset.output_states_max)
    
    # Evaluate:
    kpi_dict = top_functions.eval_top_func(p, model_eval_func, model_kpi_func, model, (traj_loss_func, man_loss_func), te_dataset, device)
    p.export_evaluation(kpi_dict)

if __name__ == '__main__':

   
    #torch.cuda.empty_cache()
    #p = params.ParametersHandler('Constant_Parameter.yaml', 'highD.yaml', './config')
    p = params.ParametersHandler('DMTP.yaml', 'exid_train.yaml', './config',seperate_test_dataset='exid_test.yaml')
  
    experiment_file = 'experiments/MMnTP_exid_train_2023-02-22 16:23:16.871626'#'experiments/DMTP_exid_train_2023-02-22 11:38:01.533814' # DMTP_exid_train_2023-02-21 18:56:42.572922' # mode 1
    p.import_experiment(experiment_file)
    p.hyperparams['experiment']['debug_mode'] = False
    p.hyperparams['dataset']['balanced'] = True
    p.hyperparams['training']['batch_size'] = 64
    p.hyperparams['experiment']['multi_modal_eval'] = True
    if p.hyperparams['model']['multi_modal'] == False:
        p.hyperparams['experiment']['multi_modal_eval'] = False
    # make sure to use following function to update hyperparameters
    p.match_parameters()
    test_model_dict(p)



