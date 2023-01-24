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
import kpis
import training_functions
import model_specific_training_functions as mstf

from torch.utils.tensorboard import SummaryWriter

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')



import matplotlib.colors as mcolors



def train_model_dict(p):
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
    
    tr_dataset = Dataset.LCDataset(p.DATASET_DIR, p.DATA_FILES,
        index_file = Dataset.get_index_file(p,  'Tr'),
        data_type = p.model_dictionary['data type'], 
        state_type = p.model_dictionary['state type'], 
        keep_plot_info= False, 
        force_recalc_start_indexes = False)
  
    val_dataset = Dataset.LCDataset(p.DATASET_DIR, p.DATA_FILES,
        index_file = Dataset.get_index_file(p,  'Val'),
        data_type = p.model_dictionary['data type'], 
        state_type = p.model_dictionary['state type'], 
        keep_plot_info= True, 
        import_states = True,
        force_recalc_start_indexes = False,
        states_min = tr_dataset.states_min, 
        states_max = tr_dataset.states_max, 
        output_states_min = tr_dataset.output_states_min,
        output_states_max = tr_dataset.output_states_max)
    
    te_dataset = Dataset.LCDataset(p.DATASET_DIR, p.DATA_FILES,
        index_file = Dataset.get_index_file(p,  'Te'),
        data_type = p.model_dictionary['data type'],
        state_type = p.model_dictionary['state type'], 
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
    val_result_dic = training_functions.train_top_func(p,model_train_func, model_eval_func, model_kpi_func, model, (traj_loss_func, man_loss_func), optimizer, tr_dataset, val_dataset,device, tensorboard = tb)    
    #kpi_dic = training_functions.eval_top_func(p, model_eval_func, model_kpi_func, model, (traj_loss_func, man_loss_func), te_dataset, device,  tensorboard = tb)
    #print('x')
    p.export_experiment()
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
    
    '''
    p = params.ParametersHandler('MMnTP.yaml', 'highD.yaml', './config')
    p.hyperparams['experiment']['debug_mode'] = True
    p.match_parameters()
    #1
    train_model_dict(p)
    '''
    
    p = params.ParametersHandler('MMnTP.yaml', 'NGSIM.yaml', './config')
    p.hyperparams['experiment']['group'] = 'lrwub32'
    p.hyperparams['training']['batch_size'] = 32
    p.hyperparams['experiment']['debug_mode'] = False
    p.hyperparams['dataset']['ablation'] = False
    p.hyperparams['experiment']['multi_modal_eval'] = False
    p.model['hyperparams']['number of modes'] = 10
    p.match_parameters()
    #1
    train_model_dict(p)
    
    
   


    