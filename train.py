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

from torch.utils.tensorboard import SummaryWriter

import torch.multiprocessing
import matplotlib.colors as mcolors

import Dataset
import params
import kpis
import top_functions
from evaluate import test_model_dict

import TPMs



def train_model_dict(p):
    # Set Random Seeds:
    if torch.cuda.is_available() and p.CUDA:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            torch.cuda.manual_seed_all(0)
            print('Running on:', device)

    else:
        print('Running on CPU!!!')
        exit()
        device = torch.device("cpu")
            
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    np.random.seed(1)
    random.seed(1)

    # Instantiate Model:
   
    model = p.model_dictionary['ref'](p.BATCH_SIZE, device, p.model_dictionary['hyperparams'], p)
    model = model.to(device)
    optimizer = p.model_dictionary['optimizer'](params = model.parameters(), lr = p.LR)
    man_loss_func = p.model_dictionary['man loss function']
    if p.model_dictionary['hyperparams']['probabilistic output']:
        traj_loss_func = kpis.NLL_loss
    else:
        traj_loss_func = p.model_dictionary['traj loss function']()
    model_train_func = p.model_dictionary['model training function']
    model_eval_func = p.model_dictionary['model evaluation function']
    model_kpi_func = p.model_dictionary['model kpi function']
    
    tr_dataset = Dataset.LCDataset(p.TR.DATASET_DIR, p.TR.DATA_FILES, p.TR.MAP_INDEX, p.TR.MAP_DIRS,
        index_file = Dataset.get_index_file(p,p.TR, 'Tr'),
        data_type = p.model_dictionary['data type'], 
        state_type = p.model_dictionary['state type'], 
        use_map_features = p.hyperparams['model']['use_map_features'],
        keep_plot_info= False, 
        force_recalc_start_indexes = False)
  
    val_dataset = Dataset.LCDataset(p.TR.DATASET_DIR, p.TR.DATA_FILES, p.TR.MAP_INDEX, p.TR.MAP_DIRS,
        index_file = Dataset.get_index_file(p,p.TR,  'Val'),
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
    
    # Train/Evaluate:
    if p.DEBUG_MODE:
        tb_log_dir = "runs(debugging)/{}/{}".format(p.experiment_group,p.experiment_file)
    else:
        tb_log_dir = "runs/{}/{}".format(p.experiment_group,p.experiment_file)
    tb = SummaryWriter(log_dir= tb_log_dir)
    val_result_dic = top_functions.train_top_func(p,model_train_func, model_eval_func,
                                                   model_kpi_func, model, (traj_loss_func, man_loss_func),
                                                     optimizer, tr_dataset, val_dataset,device, tensorboard = tb)    
    
    # Save results:
    if p.parameter_tuning_experiment:
        log_file_dir = os.path.join(p.TABLES_DIR,p.tuning_experiment_name + '.csv')  
        log_columns = [key for key in p.log_dict]
        log_columns = ', '.join(log_columns) + '\n'
        result_line = [str(p.log_dict[key]) for key in p.log_dict]
        result_line = ', '.join(result_line) + '\n'
        if os.path.exists(log_file_dir) == False:
            result_line = log_columns + result_line
        with open(log_file_dir, 'a') as f:
            f.write(result_line)

   
    tb.close()

if __name__ == '__main__':
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    '''
    p = params.ParametersHandler('MMnTP.yaml', 'highD.yaml', './config')
    p.hyperparams['experiment']['debug_mode'] = True
    p.match_parameters()
    #1
    train_model_dict(p)
    '''
    # with map MM
    p = params.ParametersHandler('DMT_POVL.yaml', 'exid_train.yaml', './config', seperate_test_dataset='exid_test.yaml',seperate_deploy_dataset='exid_deploy.yaml')
    p.hyperparams['experiment']['group'] = 'dmt_povl'
    p.hyperparams['training']['batch_size'] = 1000
    p.hyperparams['experiment']['debug_mode'] = False
    p.hyperparams['dataset']['ablation'] = False
    p.hyperparams['experiment']['multi_modal_eval'] = False
    p.hyperparams['model']['use_map_features'] = False
    p.hyperparams['dataset']['balanced'] = True
    p.match_parameters()
    p.export_experiment()
    #1
    train_model_dict(p)
    p.hyperparams['experiment']['multi_modal_eval'] = True
    p.hyperparams['dataset']['balanced'] = False
    p.match_parameters()
    test_model_dict(p)
    
    