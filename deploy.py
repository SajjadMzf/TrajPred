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

import kpis
import matplotlib.colors as mcolors
import export
import TPMs
def deploy_model_dict(p, export_file_name):
    # Set Random Seeds:
    if torch.cuda.is_available() and p.CUDA:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            torch.cuda.manual_seed_all(0)
    else:
        device = torch.device("cpu")
    print(device)        
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    np.random.seed(1)
    random.seed(1)

    # Instantiate Model:
    
    model = p.model_dictionary['ref'](p.BATCH_SIZE, device, 
                                      p.model_dictionary['hyperparams'], p)
    model_deploy_func = p.model_dictionary['model deploy function']
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
    de_dataset = Dataset.LCDataset(p.DE.DATASET_DIR, p.DE.DATA_FILES, 
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
    
    te_dataset = Dataset.LCDataset(p.TE.DATASET_DIR, p.TE.DATA_FILES, 
        index_file = Dataset.get_index_file(p,p.TE,  'Te'),
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
    


    #de_export_dict = top_functions.deploy_top_func(p, model_deploy_func, model,
    #                                              de_dataset, device)
    
    te_export_dict = top_functions.deploy_top_func(p, model_deploy_func, model,
                                                 te_dataset, device)
    if p.MULTI_MODAL:
        #export.export_results(export_file_name, de_export_dict, 'De')     
        export.export_results(export_file_name, te_export_dict, 'Te')     
    
    else:
        #export.export_results_SM(export_file_name, de_export_dict, 'De')
        export.export_results_SM(export_file_name, te_export_dict, 'Te')
        
if __name__ == '__main__':

    '''
    p = params.ParametersHandler('Constant_Parameter.yaml', 'exid_train.yaml', './config',
                                  seperate_test_dataset='exid_test.yaml',
                                  seperate_deploy_dataset='exid_deploy.yaml')
    p.hyperparams['training']['batch_size'] = 1 # ConstantX doesnt support variable batch size
    p.hyperparams['experiment']['debug_mode'] = False
    p.hyperparams['dataset']['ablation'] = False
    p.hyperparams['model']['multi_modal'] = False
    p.hyperparams['model']['man_dec_out'] = False
    p.hyperparams['experiment']['multi_modal_eval'] = False
    p.hyperparams['model']['use_map_features'] = False
    p.hyperparams['dataset']['balanced'] = False
    # make sure to use following function to update hyperparameters
    p.match_parameters()
    deploy_model_dict(p)
    
    exit()
    '''
    p = params.ParametersHandler('POVL_SM.yaml', 'exid_train.yaml', './config',
                                  seperate_test_dataset='m40_test.yaml',
                                  seperate_deploy_dataset='m40_deploy.yaml')
    experiment_file = 'experiments/POVL_SM_exid_train_2024-01-07 15:11:00.606801'
    export_file_name = 'NoTL_POVL_SM_M40_train'
    #experiment_file = 'experiments/POVL_SM_exid_train_2023-05-04 10:49:00.060446'
    '''
    Constant Parameters
    'experiments/POVL_SM_exid_train_2023-05-04 10:49:00.060446'
    'experiments/POVL_exid_train_2023-05-09 12:48:30.344067'
    'experiments/DMT_POVL_exid_train_2023-05-05 12:23:47.848009'

    '''
    p.import_experiment(experiment_file)
    p.hyperparams['experiment']['debug_mode'] = False
    p.hyperparams['dataset']['balanced'] = False
    p.hyperparams['training']['batch_size'] = 1000
    p.hyperparams['experiment']['multi_modal_eval'] = True
    p.hyperparams['model']['multi_modal'] = True
    # make sure to use following function to update hyperparameters
    p.match_parameters()
    deploy_model_dict(p, export_file_name)
    
   

