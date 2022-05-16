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
import utils


from torch.utils.tensorboard import SummaryWriter

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')



import matplotlib.colors as mcolors



def train_model_dict(model_dict, p):
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
        traj_loss_func = utils.NLL_loss
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
    #print(tr_dataset.states_max-tr_dataset.states_min)
    #assert np.all((tr_dataset.states_max-tr_dataset.states_min)>0)
    #print('output state min: {}, output state max: {}'.format(tr_dataset.output_states_min, tr_dataset.output_states_max))
    #exit()
    val_dataset = Dataset.LCDataset(p.TRAIN_DATASET_DIR, p.VAL_DATA_FILES,
        in_seq_len = p.IN_SEQ_LEN,
        out_seq_len = p.TGT_SEQ_LEN,
        end_of_seq_skip_len = p.SKIP_SEQ_LEN,
        data_type = model_dict['data type'], 
        state_type = model_dict['state type'], 
        keep_plot_info= False, 
        traj_output = (task==params.TRAJECTORYPRED), 
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
    #print(tr_dataset.__len__())
    #print(val_dataset.__len__())
    #print(te_dataset.__len__())
    #exit()
    # Train/Evaluate:
    tb = SummaryWriter()
    val_result_dic = utils.train_top_func(p, model, optimizer, lc_loss_func, traj_loss_func, task, tr_dataset, val_dataset,device, model_tag = model_dict['tag'], tensorboard = tb)    
    te_result_dic, traj_df = utils.eval_top_func(p, model, lc_loss_func, traj_loss_func, task, te_dataset, device, model_tag = model_dict['tag'], tensorboard = tb)
    
    
    # Save results:
    log_file_dir = p.TABLES_DIR + p.SELECTED_DATASET + '_' + model_dict['name'] + '.csv'  
    log_dict = model_dict['hyperparams'].copy()
    log_dict['state type'] = model_dict['state type']
    tb_hp_dic = log_dict.copy()
    log_dict.update(val_result_dic)
    log_dict.update(te_result_dic)
    log_columns = [key for key in log_dict]
    log_columns = ', '.join(log_columns) + '\n'
    result_line = [str(log_dict[key]) for key in log_dict]
    result_line = ', '.join(result_line) + '\n'
    if os.path.exists(log_file_dir) == False:
        result_line = log_columns + result_line
    with open(log_file_dir, 'a') as f:
        f.write(result_line)

    tb.add_hparams(
            tb_hp_dic,
            te_result_dic
        )
    tb.close()

if __name__ == '__main__':
    
    #        'layer number': 3,
    #        'model dim':512,
    #        'feedforward dim': 128,
    #        'classifier dim': 128,
    #        'head number': 8,
    torch.cuda.empty_cache()
    print('---------------------------------------------------------------------------------------')
    print('---------------------------------------------------------------------------------------')
    p = params.Parameters(SELECTED_MODEL = 'NOVEL_TRANSFORMER_TRAJ', SELECTED_DATASET = 'HIGHD', UNBALANCED = False, ABLATION = False)

    #1
    model_dict = m.MODELS[p.SELECTED_MODEL]

    model_dict['hyperparams']['task'] = params.TRAJECTORYPRED
    model_dict['hyperparams']['multi modal'] = False
    model_dict['hyperparams']['layer number'] = 3
    model_dict['state type'] = 'wirth'
    model_dict['tag'] = utils.update_tag(model_dict)

    train_model_dict(model_dict, p)
