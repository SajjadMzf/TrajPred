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
from train import train_model_dict
import TPMs


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
    p = params.ParametersHandler('POVL_SM.yaml', 'm40_train.yaml', './config', 
                                 seperate_test_dataset='m40_test.yaml',
                                 seperate_deploy_dataset='m40_deploy.yaml')
    #p = params.ParametersHandler('POVL_SM.yaml', 'highD.yaml', './config')
    
    p.hyperparams['experiment']['group'] = 'povl_sm'
    p.hyperparams['experiment']['transfer_learning'] = 'OnlyFC'
    p.hyperparams['experiment']['debug_mode'] = False
    p.hyperparams['experiment']['multi_modal_eval'] = False
    p.hyperparams['dataset']['balanced'] = False
    p.match_parameters()
    p.export_experiment()
    #1
    prev_best_model = p.WEIGHTS_DIR + 'POVL_SM_exid_train_2024-01-07 15:11:00.606801.pt'
    train_model_dict(p, prev_best_model =prev_best_model)
    p.hyperparams['experiment']['multi_modal_eval'] = False
    p.hyperparams['dataset']['balanced'] = False
    p.match_parameters()
    test_model_dict(p)
