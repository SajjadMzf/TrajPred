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
import pdb 
import kpis
import matplotlib.colors as mcolors
import TPMs
def test_model_dict(p):
    # Set Random Seeds:
    if False:#torch.cuda.is_available() and p.CUDA:
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
    tr_dataset = Dataset.LCDataset(p.TR.DATASET_DIR, p.TR.DATA_FILES, 
        index_file = Dataset.get_index_file(p, p.TR,  'Tr'),
        data_type = p.model_dictionary['data type'], 
        state_type = p.model_dictionary['state type'], 
        use_map_features = p.hyperparams['model']['use_map_features'],
        keep_plot_info= False, 
        force_recalc_start_indexes = False)
    #val_dataset = Dataset.LCDataset(p.TRAIN_DATASET_DIR, p.VAL_DATA_FILES,  data_type = p.model_dictionary['data type'], state_type = p.model_dictionary['state type'], keep_plot_info= False, states_min = tr_dataset.states_min, states_max = tr_dataset.states_max, output_states_min = tr_dataset.output_states_min, output_states_max = tr_dataset.output_states_max)
    
    #exit()
    te_dataset = Dataset.LCDataset(p.TE.DATASET_DIR, p.TE.DATA_FILES,
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

    '''
    p = params.ParametersHandler('Constant_Parameter.yaml', 'm40_train.yaml', './config', # POVL_SM.yaml
                                  seperate_test_dataset='m40_test.yaml',
                                  seperate_deploy_dataset='m40_deploy.yaml')
    
    #p = params.ParametersHandler('Constant_Parameter.yaml', 'highD.yaml', './config')
    
    p.hyperparams['experiment']['multi_modal_eval'] = False
    p.hyperparams['model']['multi_modal'] = False
    p.hyperparams['model']['man_dec_out'] = False
    p.hyperparams['training']['batch_size'] = 2000
    
    
    
    p.match_parameters()
    test_model_dict(p)

    
    exit()
    '''
    
    p = params.ParametersHandler('POVL_SM.yaml', 'm40_train.yaml', './config', # POVL_SM.yaml
                                  seperate_test_dataset='m40_test.yaml',
                                  seperate_deploy_dataset='m40_deploy.yaml')
    
    experiment_file = 'experiments/POVL_SM_m40_train_2024-02-04 12:56:38.408576'
    p.import_experiment(experiment_file)
    p.hyperparams['experiment']['debug_mode'] = True
    p.hyperparams['dataset']['balanced'] = False
    p.hyperparams['training']['batch_size'] = 2000
    p.hyperparams['experiment']['multi_modal_eval'] = False
    p.match_parameters()
    print(p.TE.DATASET_DIR)
    test_model_dict(p)


