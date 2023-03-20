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
import export
import TPMs
def deploy_model_dict(p):
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
    model_deploy_func = p.model_dictionary['model deploy function']
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
    de_dataset = Dataset.LCDataset(p.DE.DATASET_DIR, p.DE.DATA_FILES, p.DE.MAP_INDEX, p.DE.MAP_DIRS,
        index_file = Dataset.get_index_file(p,p.DE,  'De'),
        data_type = p.model_dictionary['data type'], 
        state_type = p.model_dictionary['state type'], 
        use_map_features = p.hyperparams['model']['use_map_features'],
        keep_plot_info= True, 
        import_states = True,
        force_recalc_start_indexes = False,
        deploy_data = True,
        states_min = tr_dataset.states_min, 
        states_max = tr_dataset.states_max,
        output_states_min = tr_dataset.output_states_min, 
        output_states_max = tr_dataset.output_states_max)
    
    export_dict = top_functions.deploy_top_func(p, model_deploy_func, model, de_dataset, device)
    export.export_results_SM(export_dict)
if __name__ == '__main__':

   
    p = params.ParametersHandler('POVL_SM.yaml', 'exid_train.yaml', './config', seperate_test_dataset='exid_test.yaml',seperate_deploy_dataset='exid_deploy.yaml')
    experiment_file = 'experiments/POVL_SM_exid_train_2023-03-11 13:42:13.664618'
    p.import_experiment(experiment_file)
    p.hyperparams['experiment']['debug_mode'] = False
    p.hyperparams['dataset']['balanced'] = False
    p.hyperparams['training']['batch_size'] = 64
    p.hyperparams['experiment']['multi_modal_eval'] = False
    p.hyperparams['problem']['SKIP_SEQ_LEN'] = 0 # TODO: remove (already set in .yaml), also uncomment in import experiment line 5
    # make sure to use following function to update hyperparameters
    p.match_parameters()
    deploy_model_dict(p)


