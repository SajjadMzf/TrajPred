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
    #print('x')
    model = p.model_dictionary['ref'](p.BATCH_SIZE, device, p.model_dictionary['hyperparams'], p)
    optimizer = p.model_dictionary['optimizer'](params = model.parameters(), lr = p.LR)
    lc_loss_func = p.model_dictionary['lc loss function'](ignore_index=-1)
    if p.model_dictionary['hyperparams']['probabilistic output']:
        traj_loss_func = training_functions.NLL_loss
    else:
        traj_loss_func = p.model_dictionary['traj loss function']()
    #print('x')
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
    #print(tr_dataset.states_max-tr_dataset.states_min)
    #assert np.all((tr_dataset.states_max-tr_dataset.states_min)>0)
    #print('output state min: {}, output state max: {}'.format(tr_dataset.output_states_min, tr_dataset.output_states_max))
    #exit()
    #print('x')
    val_dataset = Dataset.LCDataset(p.TRAIN_DATASET_DIR, p.VAL_DATA_FILES,
        in_seq_len = p.IN_SEQ_LEN,
        out_seq_len = p.TGT_SEQ_LEN,
        end_of_seq_skip_len = p.SKIP_SEQ_LEN,
        data_type = p.model_dictionary['data type'], 
        state_type = p.model_dictionary['state type'], 
        keep_plot_info= False, 
        import_states = True,
        unbalanced = p.UNBALANCED,
        force_recalc_start_indexes = False,
        states_min = tr_dataset.states_min, 
        states_max = tr_dataset.states_max, 
        output_states_min = tr_dataset.output_states_min,
        output_states_max = tr_dataset.output_states_max)
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
    #print(tr_dataset.__len__())
    #print(val_dataset.__len__())
    #print(te_dataset.__len__())
    #exit()
    # Train/Evaluate:
    tb = SummaryWriter()
    val_result_dic = training_functions.train_top_func(p, model, optimizer, lc_loss_func, traj_loss_func, tr_dataset, val_dataset,device, tensorboard = tb)    
    te_result_dic, traj_df = training_functions.eval_top_func(p, model, lc_loss_func, traj_loss_func, te_dataset, device,  tensorboard = tb)
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
    
    #        'layer number': 3,
    #        'model dim':512,
    #        'feedforward dim': 128,
    #        'classifier dim': 128,
    #        'head number': 8,
    #torch.cuda.empty_cache()
    '''
    tuning_experiment_name = 'different model parameters for transformer traj non man_based'
    selected_params = ['experiment_tag', 'DEBUG_MODE', 
                        'MAN_DEC_IN', 
                        'MAN_DEC_OUT', 
                        'model[\'hyperparams\'][\'layer number\']',
                        'model[\'hyperparams\'][\'model dim\']',
                        'model[\'hyperparams\'][\'head number\']',
                         ]
    selected_metrics = ['FDE_table', 'RMSE_table']
    '''
    p = params.ParametersHandler('ManouvreTransformerTraj.yaml', 'highD.yaml', './config')
    #1
    train_model_dict(p)
    
    '''
    #2
    p.model['hyperparams']['layer number'] = 1
    p.model['hyperparams']['model dim'] = 128
    p.model['hyperparams']['head number'] = 2
    p.new_experiment()
    p.tune_params(tuning_experiment_name, selected_params, selected_metrics)
    train_model_dict(p)
    #3
    p.model['hyperparams']['layer number'] = 2
    p.model['hyperparams']['model dim'] = 128
    p.model['hyperparams']['head number'] = 4
    p.new_experiment()
    p.tune_params(tuning_experiment_name, selected_params, selected_metrics)
    train_model_dict(p)
    
    #4
    p.model['hyperparams']['layer number'] = 3
    p.model['hyperparams']['model dim'] = 128
    p.model['hyperparams']['head number'] = 8
    p.new_experiment()
    p.tune_params(tuning_experiment_name, selected_params, selected_metrics)
    train_model_dict(p)
    #5
    p.model['hyperparams']['layer number'] = 1
    p.model['hyperparams']['model dim'] = 512
    p.model['hyperparams']['head number'] = 2
    p.new_experiment()
    p.tune_params(tuning_experiment_name, selected_params, selected_metrics)
    train_model_dict(p)
    #6
    p.model['hyperparams']['layer number'] = 1
    p.model['hyperparams']['model dim'] = 512
    p.model['hyperparams']['head number'] = 8
    p.new_experiment()
    p.tune_params(tuning_experiment_name, selected_params, selected_metrics)
    train_model_dict(p)
    '''
    
    
    
   


    