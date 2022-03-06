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
    ttlc_loss_func = model_dict['ttlc loss function']()
    task = model_dict['hyperparams']['task']
    # Instantiate Dataset: 
    tr_dataset = Dataset.LCDataset(p.TRAIN_DATASET_DIR, p.TR_DATA_FILES, data_type = model_dict['data type'], state_type = model_dict['state type'], keep_plot_info= False)
    te_dataset = Dataset.LCDataset(p.TEST_DATASET_DIR, p.TE_DATA_FILES,  data_type = model_dict['data type'], state_type = model_dict['state type'], keep_plot_info= True, states_min = tr_dataset.states_min, states_max = tr_dataset.states_max)

    # Evaluate:
    te_result_dic = utils.eval_top_func(p, model, lc_loss_func, ttlc_loss_func, task, te_dataset, device, model_tag = model_dict['tag'])
    

if __name__ == '__main__':
    
    p = params.Parameters(SELECTED_MODEL = 'REGIONATTCNN3', SELECTED_DATASET = 'HIGHD', UNBALANCED = False, ABLATION = False)

    model_dict = m.MODELS[p.SELECTED_MODEL]
    model_dict['hyperparams']['task'] = params.DUAL
    model_dict['hyperparams']['curriculum loss'] = True
    model_dict['hyperparams']['curriculum seq'] = True
    model_dict['hyperparams']['curriculum virtual'] = False
    model_dict['tag'] = utils.update_tag(model_dict)
    test_model_dict(model_dict, p)