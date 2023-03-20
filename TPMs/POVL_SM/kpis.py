import os
import random
import numpy as np 
import pickle
import torch
import torch.nn as nn
import torch.utils.data as utils_data
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import torch.nn.functional as F
import logging
import math
from time import time
from sklearn import metrics
import pandas as pd
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

font = {'size'   : 22}
matplotlib.rcParams['figure.figsize'] = (18, 12)
matplotlib.rc('font', **font)

import pdb


# 

def POVL_SM_kpis(p, kpi_input_dict, traj_min, traj_max, figure_name):
    '''
    1. NLL (TBD)
    2. error based 
    Miss Rate (MR):  The number of scenarios where none of the forecasted trajectories are within 2.0 meters of ground truth according to endpoint error.
    Minimum Final Displacement Error  (minFDE): The L2 distance between the endpoint of the best forecasted trajectory and the ground truth.  The best here refers to the trajectory that has the minimum endpoint error.
    Minimum Average Displacement Error  (minADE): The average L2 distance between the best forecasted trajectory and the ground truth.  The best here refers to the trajectory that has the minimum endpoint error.
    Probabilistic minimum Final Displacement Error  (p-minFDE): This is similar to minFDE. The only difference is we add min(-log(p), -log(0.05)) to the endpoint L2 distance, where p corresponds to the probability of the best forecasted trajectory.
    Probabilistic minimum Average Displacement Error  (p-minADE): This is similar to minADE. The only difference is we add min(-log(p), -log(0.05)) to the average L2 distance, where p corresponds to the probability of the best forecasted trajectory.
    3. N accident, N road violation (not here)
    '''
    '''
     batch_kpi_input_dict = {    
        'data_file': data_file,
        'tv': tv_id.numpy(),
        'frames': frames.numpy(),
        'traj_min': dataset.output_states_min,
        'traj_max': dataset.output_states_max,  
        'input_features': feature_data.cpu().data.numpy(),
        'traj_gt': traj_gt.cpu().data.numpy(),
        'traj_dist_preds': BM_predicted_data_dist.cpu().data.numpy(),
        'man_gt': man_gt.cpu().data.numpy(),
        'man_preds': man_vectors.cpu().data.numpy(),
        'mode_prob': mode_prob.detach().cpu().data.numpy(),
    }
    '''
    input_padding_mask = np.concatenate(kpi_input_dict['input padding mask'], axis = 0)
    input_padding_mask = p.MAX_IN_SEQ_LEN - np.argmin(input_padding_mask, axis = -1)
    ovl_index = [None]*p.MAX_IN_SEQ_LEN
    for o_len in range(1, p.MAX_IN_SEQ_LEN+1):
        ovl_index[o_len-1] = input_padding_mask==o_len
    
    
    dtraj_gt = np.concatenate(kpi_input_dict['traj_gt'], axis = 0)
    
    dtraj_pred = np.concatenate(kpi_input_dict['traj_dist_preds'], axis = 0)
    
    traj_max = kpi_input_dict['traj_max'][0]
    traj_min = kpi_input_dict['traj_min'][0]
    
    #denormalise
    dtraj_pred[:,:,:2] = dtraj_pred[:,:,:2]*(traj_max-traj_min) + traj_min
    dtraj_pred[:,:,2:4] = dtraj_pred[:,:,2:4]*(traj_max-traj_min)
    
    dtraj_gt = dtraj_gt*(traj_max-traj_min) + traj_min
    #from diff to actual
    traj_pred = np.cumsum(dtraj_pred[:,:,:2], axis = 1)
    traj_gt = np.cumsum(dtraj_gt, axis = 1)
    
    rmse_ol, n_samples_ovl = calc_ovl_rmse(p, traj_pred, traj_gt, ovl_index)
    rmse_time, rmse = calc_rmse(p, traj_pred, traj_gt)
    return {
        'ol_rmse_table': rmse_ol,
        'time_rmse_list':rmse_time,
        'n_samples_ovl_list': n_samples_ovl,
        'rmse': rmse # minRMSE K=1 max pred horizon
    }


def calc_ovl_rmse(p, traj_pred, traj_gt, ovl_index):
    seq_len = traj_pred.shape[1]
    rmse_table = np.zeros((1, p.MAX_IN_SEQ_LEN))
    n_sample_ovl = np.zeros((p.MAX_IN_SEQ_LEN))
    for ol in range(p.MAX_IN_SEQ_LEN):
        cur_n_samples = np.sum(ovl_index[ol])
        if cur_n_samples==0:
            continue
        n_sample_ovl[ol] = cur_n_samples
        cur_traj_pred = traj_pred[ovl_index[ol]]
        cur_traj_gt = traj_gt[ovl_index[ol]]
        rmse_table[0,ol] = np.sqrt(
            np.sum((cur_traj_pred-cur_traj_gt)**2)/(cur_n_samples*seq_len)
            )
    return rmse_table, n_sample_ovl



def calc_rmse(p, traj_pred, traj_gt):
    n_samples = traj_pred.shape[0]
    prediction_ts = int(p.TGT_SEQ_LEN/p.FPS)
    if (p.TGT_SEQ_LEN/p.FPS) % 1 != 0:
        raise(ValueError('Target sequence length not dividable by FPS'))
    rmse_table = np.zeros((prediction_ts))
    for ts in range(prediction_ts):
        ts_index = (ts+1)*p.FPS
        cur_n_samples = n_samples * ts_index
        cur_traj_pred = traj_pred[:,:ts_index]
        cur_traj_gt = traj_gt[:,:ts_index]
        rmse_table[ts] = np.sqrt(
            np.sum((cur_traj_pred-cur_traj_gt)**2)/cur_n_samples
            )
    
    return rmse_table, rmse_table[-1]
