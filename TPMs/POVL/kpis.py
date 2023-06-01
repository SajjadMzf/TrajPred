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
    rmse_time = calc_rmse(p, traj_pred, traj_gt)
    return {
        'rmseVSol': rmse_ol,
        'rmseVStime':rmse_time,
        'n_samples_ovl_list': n_samples_ovl,
        'rmse': rmse # minRMSE K=1 max pred horizon
    }




def POVL_kpis(p, kpi_input_dict, traj_min, traj_max, figure_name):
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
    
    time_bar_gt = np.concatenate(kpi_input_dict['time_bar_gt'], axis = 0)
    time_bar_preds = np.concatenate(kpi_input_dict['time_bar_preds'], axis = 0)
    
    man_gt = np.concatenate(kpi_input_dict['man_gt'], axis = 0)
    man_preds = np.concatenate(kpi_input_dict['man_preds'], axis = 0)
    mode_prob = np.concatenate(kpi_input_dict['mode_prob'], axis = 0)
    total_samples = mode_prob.shape[0]
    hp_mode = np.argmax(mode_prob, axis = 1)
    (unique_modes, mode_freq) = np.unique(hp_mode, return_counts = True)
    sorted_args = np.argsort(unique_modes)
    unique_modes = unique_modes[sorted_args]
    mode_freq = mode_freq[sorted_args]
    
    dtraj_gt = np.concatenate(kpi_input_dict['traj_gt'], axis = 0)
    
    dtraj_pred = np.concatenate(kpi_input_dict['traj_dist_preds'], axis = 0)
    
    traj_max = kpi_input_dict['traj_max'][0]
    traj_min = kpi_input_dict['traj_min'][0]
    
    #denormalise
    dtraj_pred[:,:,:,:2] = dtraj_pred[:,:,:,:2]*(traj_max-traj_min) + traj_min
    dtraj_pred[:,:,:,2:4] = dtraj_pred[:,:,:,2:4]*(traj_max-traj_min)
    
    dtraj_gt = dtraj_gt*(traj_max-traj_min) + traj_min
    #from diff to actual
    traj_pred = np.cumsum(dtraj_pred[:,:,:,:2], axis = 2)
    traj_gt = np.cumsum(dtraj_gt, axis = 1)
    hp_traj_pred = traj_pred[np.arange(total_samples), hp_mode]
    
    mnlld, mnll = calc_meanNLL(p, dtraj_pred, dtraj_gt, traj_gt, mode_prob)
    #minFDE = {}
    minRMSE = {}
    minRMSE_ovl ={}
    #minMR = {}
    maxACC = {}
    #minTimeACC = {}
    minTimeMAE = {}
    rmse = {}

    for K in range(1,min(10,mode_prob.shape[1])+1):
        if K>1 and p.MULTI_MODAL_EVAL == False:
            break
        key = 'K={}'.format(K)
        kbest_modes = np.argpartition(mode_prob, -1*K, axis = 1)[:,-1*K:]
        index_array = np.repeat(np.arange(total_samples),K).reshape(total_samples, K)
        

        kbest_traj_pred = traj_pred[index_array, kbest_modes]
        kbest_modes_probs = mode_prob[index_array, kbest_modes]
        kbest_modes_probs = np.divide(kbest_modes_probs, np.sum(kbest_modes_probs, axis = 1).reshape(total_samples,1)) 
        kbest_man_preds = man_preds[index_array, kbest_modes]
        maxACC[key] = calc_man_acc(p, kbest_man_preds, man_gt)
        
        minRMSE_ovl[key],rmse_table, rmse[key], n_samples_ovl = calc_ovl_minRMSE(p, kbest_traj_pred, kbest_modes_probs, traj_gt, ovl_index)
    
    return {
        'mnlld': mnlld,
        'mnll': mnll,
        'maxACC':maxACC,
        'minRMSE_ovl': minRMSE_ovl,
        'rmse_table':rmse_table,
        'n_samples_ovl_list': n_samples_ovl,
        'rmse': rmse['K=1'] # minRMSE K=1 max pred horizon
    }



def MMnTP_kpis(p, kpi_input_dict, traj_min, traj_max, figure_name):
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
    time_bar_gt = np.concatenate(kpi_input_dict['time_bar_gt'], axis = 0)
    time_bar_preds = np.concatenate(kpi_input_dict['time_bar_preds'], axis = 0)
    
    man_gt = np.concatenate(kpi_input_dict['man_gt'], axis = 0)
    man_preds = np.concatenate(kpi_input_dict['man_preds'], axis = 0)
    mode_prob = np.concatenate(kpi_input_dict['mode_prob'], axis = 0)
    total_samples = mode_prob.shape[0]
    hp_mode = np.argmax(mode_prob, axis = 1)
    (unique_modes, mode_freq) = np.unique(hp_mode, return_counts = True)
    sorted_args = np.argsort(unique_modes)
    unique_modes = unique_modes[sorted_args]
    mode_freq = mode_freq[sorted_args]
    
    dtraj_gt = np.concatenate(kpi_input_dict['traj_gt'], axis = 0)
    
    dtraj_pred = np.concatenate(kpi_input_dict['traj_dist_preds'], axis = 0)
    
    traj_max = kpi_input_dict['traj_max'][0]
    traj_min = kpi_input_dict['traj_min'][0]
    
    #denormalise
    dtraj_pred[:,:,:,:2] = dtraj_pred[:,:,:,:2]*(traj_max-traj_min) + traj_min
    dtraj_pred[:,:,:,2:4] = dtraj_pred[:,:,:,2:4]*(traj_max-traj_min)
    
    dtraj_gt = dtraj_gt*(traj_max-traj_min) + traj_min
    #from diff to actual
    traj_pred = np.cumsum(dtraj_pred[:,:,:,:2], axis = 2)
    traj_gt = np.cumsum(dtraj_gt, axis = 1)
    hp_traj_pred = traj_pred[np.arange(total_samples), hp_mode]
    
    mnlld, mnll = calc_meanNLL(p, dtraj_pred, dtraj_gt, traj_gt, mode_prob)
    #minFDE = {}
    minRMSE = {}
    #minMR = {}
    minACC = {}
    #minTimeACC = {}
    minTimeMAE = {}
    rmse = {}
    for K in range(1,min(10,mode_prob.shape[1])+1):
        if K>1 and p.MULTI_MODAL_EVAL == False:
            break
        key = 'K={}'.format(K)
        kbest_modes = np.argpartition(mode_prob, -1*K, axis = 1)[:,-1*K:]
        index_array = np.repeat(np.arange(total_samples),K).reshape(total_samples, K)
        

        kbest_traj_pred = traj_pred[index_array, kbest_modes]
        kbest_modes_probs = mode_prob[index_array, kbest_modes]
        kbest_modes_probs = np.divide(kbest_modes_probs, np.sum(kbest_modes_probs, axis = 1).reshape(total_samples,1)) 
        kbest_man_preds = man_preds[index_array, kbest_modes]
        kbest_time_bar_preds = time_bar_preds[index_array, kbest_modes]
        #print(kbest_modes_probs)
        #print(np.sum(kbest_modes_probs, axis = 1))
        #exit()
        minACC[key] = calc_man_acc(p, kbest_man_preds, man_gt)
        #minTimeACC[key], minTimeACC['min'] = calc_time_acc(p, kbest_time_bar_preds, time_bar_gt)
        minTimeMAE[key], minTimeMAE['random'] = calc_time_mae(p, kbest_time_bar_preds, time_bar_gt)
        #sm_metrics_df, rmse = calc_sm_metric_df(p, hp_traj_pred, traj_gt)
    
        #minFDE[key] = calc_minFDE(p,kbest_traj_pred, kbest_modes_probs, traj_gt)
        minRMSE[key], rmse[key] = calc_minRMSE(p,kbest_traj_pred, kbest_modes_probs, traj_gt)
        
    
    return {
        'activated modes group': unique_modes,
        'activated modes percentage group': 100*mode_freq/sum(mode_freq),
        'high prob mode histogram':hp_mode,
        'time pr histogram': time_bar_preds[:,:,0],
        'time gt histogram': time_bar_gt[:,0],
        'minACC': minACC,
        #'minTimeACC': minTimeACC,
        'minTimeMAE': minTimeMAE,
        #'minFDE': minFDE, 
        'minRMSE': minRMSE,
        'mnlld': mnlld,
        'mnll': mnll,
        'rmse': rmse['K=1'] # minRMSE K=1 max pred horizon
    }



def calc_man_acc(p, man_pred, man_gt ):
    # takes maneouvre vector as input
    total_samples = man_pred.shape[0]
    n_mode = man_pred.shape[1]
    tgt_seq_len = man_pred.shape[2]
    acc = np.zeros((total_samples, n_mode))
    for i in range(n_mode):
        acc[:,i] = np.sum(man_pred[:,i] == man_gt, axis = -1)/tgt_seq_len
    best_mode = np.argmax(acc, axis = 1)
    acc = acc[np.arange(total_samples), best_mode]
    minACC = np.sum(acc)/total_samples
    return minACC

def calc_time_acc(p, time_pred, time_gt ):
    # takes maneouvre vector as input
    
    total_samples = time_pred.shape[0]
    n_mode = time_pred.shape[1]
    change_times = time_pred.shape[2] +1
    acc = np.zeros((total_samples, n_mode))
    #n_invalid_times = np.sum(time_gt==-1)
    for i in range(n_mode):
        acc[:,i] = np.sum(np.logical_or(np.around(time_pred[:,i]) == time_gt, time_gt==-1), axis = -1)
    best_mode = np.argmax(acc, axis = 1)
    acc = acc[np.arange(total_samples), best_mode]
    minACC = np.sum(acc)/(total_samples*change_times)
    zeroPredACC = np.sum(time_gt==-1)/(total_samples*change_times)
        
    return minACC, zeroPredACC

def calc_time_mae(p,time_pred, time_gt):
    total_samples = time_pred.shape[0]
    n_mode = time_pred.shape[1]
    change_times = time_pred.shape[2]
    mae = np.zeros((total_samples, n_mode))
    n_valid_times = np.sum(time_gt!=-1)
    random_time = np.random.randint(8, size = (total_samples, change_times))
    for i in range(n_mode):
        mae[:,i] = np.sum(np.multiply(np.abs(time_pred[:,i]- time_gt), time_gt!=-1), axis=-1)
    random_mae = np.sum(np.multiply(np.abs(random_time- time_gt), time_gt!=-1), axis=-1)
    best_mode = np.argmin(mae, axis = 1)
    
    mae = mae[np.arange(total_samples), best_mode]
    minMAE = np.sum(mae)/n_valid_times
    randomMAE = np.sum(random_mae)/n_valid_times

    return minMAE, randomMAE



def calc_man_acc(p, man_pred, man_gt ):
    # takes maneouvre vector as input
    total_samples = man_pred.shape[0]
    n_mode = man_pred.shape[1]
    tgt_seq_len = man_pred.shape[2]
    acc = np.zeros((total_samples, n_mode))
    for i in range(n_mode):
        acc[:,i] = np.sum(man_pred[:,i] == man_gt, axis = -1)/tgt_seq_len
    best_mode = np.argmax(acc, axis = 1)
    acc = acc[np.arange(total_samples), best_mode]
    minACC = np.sum(acc)/total_samples
    
        
    return minACC

def calc_man_acc_torch( man_pred, man_gt, device = torch.device("cpu")):
    total_samples = man_pred.shape[0]
    n_mode = man_pred.shape[1]
    tgt_seq_len = man_pred.shape[2]
    acc = torch.zeros((total_samples, n_mode), device = device)
    for i in range(n_mode):
        acc[:,i] = torch.sum(man_pred[:,i] == man_gt, dim = -1)/tgt_seq_len
    return acc




def calc_ovl_minRMSE(p, traj_pred, mode_prob, traj_gt, ovl_index):
    n_samples = traj_pred.shape[0]
    seq_len = traj_pred.shape[2]
    n_mode = traj_pred.shape[1]
    columns = ['<{} fr'.format(ol+1) for ol in range(p.MAX_IN_SEQ_LEN)]
    index = ['RMSE_OVL']
    rmse_table = np.zeros((1, p.MAX_IN_SEQ_LEN))
    n_sample_ovl = np.zeros((p.MAX_IN_SEQ_LEN))
    for ol in range(p.MAX_IN_SEQ_LEN):
        cur_n_samples = np.sum(ovl_index[ol])
        if cur_n_samples==0:
            continue
        n_sample_ovl[ol] = cur_n_samples
        cur_traj_pred = traj_pred[ovl_index[ol]]
        cur_traj_gt = traj_gt[ovl_index[ol]]
        rmse = np.zeros((cur_n_samples, n_mode))
        for i in range(n_mode):
            rmse[:,i] = np.sqrt(np.sum(np.sum((cur_traj_pred[:,i]-cur_traj_gt)**2, axis=-1), axis = -1)/(cur_n_samples*seq_len))
        
        best_mode = np.argmin(rmse, axis = 1)
        cur_best_traj_pred = cur_traj_pred[np.arange(cur_n_samples), best_mode]
        rmse_table[0,ol] = np.sqrt(
            np.sum((cur_best_traj_pred-cur_traj_gt)**2)/(cur_n_samples*seq_len)
            )
    rmse_df = pd.DataFrame(data= rmse_table, columns = columns, index = index)
    
    rmse = np.zeros((n_samples, n_mode))
    for i in range(n_mode):
        rmse[:,i] = np.sqrt(np.sum(np.sum((traj_pred[:,i]-traj_gt)**2, axis=-1), axis = -1)/(n_samples*seq_len))
    best_mode = np.argmin(rmse, axis = 1)
    best_traj_pred = traj_pred[np.arange(n_samples), best_mode]
    rmse = np.sqrt(np.sum((best_traj_pred-traj_gt)**2)/(n_samples*seq_len))
    pdb.set_trace()
    return rmse_df, rmse_table, rmse, n_sample_ovl



def calc_minRMSE(p, traj_pred, mode_prob, traj_gt):
    n_samples = traj_pred.shape[0]
    seq_len = traj_pred.shape[2]
    n_mode = traj_pred.shape[1]
    prediction_ts = int(p.TGT_SEQ_LEN/p.FPS)
    if (p.TGT_SEQ_LEN/p.FPS) % 1 != 0:
        raise(ValueError('Target sequence length not dividable by FPS'))
    columns = ['<{} sec'.format(ts+1) for ts in range(prediction_ts)]
    index = ['RMSE']
    rmse_table = np.zeros((1, prediction_ts))
    for ts in range(prediction_ts):
        ts_index = (ts+1)*p.FPS
        cur_n_samples = n_samples * ts_index
        cur_traj_pred = traj_pred[:,:,:ts_index]
        cur_traj_gt = traj_gt[:,:ts_index]
        rmse = np.zeros((n_samples, n_mode))
        for i in range(n_mode):
            rmse[:,i] = np.sqrt(np.sum(np.sum((cur_traj_pred[:,i,:,:]-cur_traj_gt[:,:,:])**2, axis=-1), axis = -1)/ts_index)
        
        best_mode = np.argmin(rmse, axis = 1)
        cur_best_traj_pred = cur_traj_pred[np.arange(n_samples), best_mode]
        rmse_table[0,ts] = np.sqrt(
            np.sum((cur_best_traj_pred-cur_traj_gt)**2)/cur_n_samples
            )
    rmse_df = pd.DataFrame(data= rmse_table, columns = columns, index = index)
    
    return rmse_df, rmse_table[0,-1]

def calc_rmse_vs_time(p,traj_pred, traj_gt):
    total_samples = traj_gt.shape[0]
    prediction_ts = int(p.TGT_SEQ_LEN/p.FPS)
    if (p.TGT_SEQ_LEN/p.FPS) % 1 != 0:
        raise(ValueError('Target sequence length not dividable by FPS'))
    columns = ['<{} sec'.format(ts+1) for ts in range(prediction_ts)]
    index = ['RMSE_lat', 'RMSE_long', 'RMSE']
    data = np.zeros((3, prediction_ts))
    for ts in range(prediction_ts):
        
        ts_index = (ts+1)*p.FPS
        current_total_samples = total_samples * ts_index
        #rmse
        data[0,ts] = np.sqrt(
            np.sum((traj_pred[:,:ts_index,0]-traj_gt[:,:ts_index,0])**2)/current_total_samples
            )
        data[1,ts] = np.sqrt(
            np.sum((traj_pred[:,:ts_index,1]-traj_gt[:,:ts_index,1])**2)/current_total_samples
            )
        data[2,ts] = np.sqrt(
            np.sum((traj_pred[:,:ts_index,:]-traj_gt[:,:ts_index,:])**2)/current_total_samples
            )
    #FDE_table = ''.join(['{}:{:.4f}'.format(columns[ts], data[2,ts]) for ts in range(prediction_ts)])
    #RMSE_table = ''.join(['{}:{:.4f}'.format(columns[ts], data[5,ts]) for ts in range(prediction_ts)])
    rmse = data[2,prediction_ts-1]
    rmse_df = pd.DataFrame(data= data, columns = columns, index = index)
    
    return rmse_df, rmse

def calc_meanNLL(p, dy_pred, dy_gt, y_gt, mode_prob):
    n_mode = dy_pred.shape[1]
    n_sample = dy_pred.shape[0]
    prediction_ts = int(p.TGT_SEQ_LEN/p.FPS)
    if (p.TGT_SEQ_LEN/p.FPS) % 1 != 0:
            raise(ValueError('Target sequence length not dividable by FPS'))
    columns = ['<{} sec'.format(ts+1) for ts in range(prediction_ts)]
    index = ['RMSE_lat', 'RMSE_long', 'RMSE']
    if n_mode ==1 or p.MULTI_MODAL==False:
        nlld, nll = log_likelihood_numpy(p, dy_pred[:,0], dy_gt, y_gt)
        nlld = -1*nlld
        nll = -1*nll
        #print(nll)
    else:
        lld = np.zeros((n_sample, n_mode, p.TGT_SEQ_LEN))
        ll = np.zeros((n_sample, n_mode, p.TGT_SEQ_LEN))
        
        for i in range(n_mode):
            lld[:,i], ll[:,i] = log_likelihood_numpy(p, dy_pred[:,i], dy_gt, y_gt)
        
        # computing lower bound for log likelihood: min(x1,x2)<=log(exp(p1x1+p2x2)) if p1+p2 =1
        c = np.amin(ll, axis = 1)
        cd = np.amin(lld, axis = 1)
        mode_probe_tiled = np.tile(mode_prob, [p.TGT_SEQ_LEN,1,1])
        #pdb.set_trace()
        mode_probe_tiled = np.transpose(mode_probe_tiled, [1,2,0])
        
        lld_mean = np.multiply(np.exp(lld),mode_probe_tiled)
        lld_mean = np.sum(lld_mean, axis =1)
        nlld = -1*np.log(lld_mean)
        nlld[np.isinf(nlld)] =-1*cd[np.isinf(nlld)]
        
        ll_mean = np.multiply(np.exp(ll),mode_probe_tiled)
        ll_mean = np.sum(ll_mean, axis =1)
        nll = -1*np.log(ll_mean)
        nll[np.isinf(nll)] =-1*c[np.isinf(nll)]
        
    prediction_ts = int(p.TGT_SEQ_LEN/p.FPS)
    n_sample = dy_pred.shape[0]
    
    mnll = np.zeros((n_sample, prediction_ts))
    mnlld = np.zeros((n_sample, prediction_ts))
    
    for ts in range(prediction_ts):
        ts_index = (ts+1)*p.FPS
        mnll[:, ts] = np.sum(nll[:, :ts_index], axis = 1)/ts_index
        mnlld[:, ts] = np.sum(nlld[:, :ts_index], axis = 1)/ts_index
    
    mnlld = np.sum(mnlld, axis = 0)/n_sample
    mnlld = mnlld.reshape(1,5)
    mnlld_df = pd.DataFrame(data= mnlld, columns = columns, index = ['Mean_NLLD'])
    
    mnll = np.sum(mnll, axis = 0)/n_sample
    mnll = mnll.reshape(1,5)
    mnll_df = pd.DataFrame(data= mnll, columns = columns, index = ['Mean_NLL'])
    return mnlld_df, mnll_df


def log_likelihood_numpy(p,dy_pred, dy_gt, y_gt):
        #print('y_pred', y_pred.shape)
        mudY = dy_pred[:,:,0]
        mudX = dy_pred[:,:,1]
        sigdY = dy_pred[:,:,2] # sigY = standard deviation of y
        sigdX = dy_pred[:,:,3] # sigX = standard deviation of x
        rhodXY = dy_pred[:,:,4]
        ohrdXY = np.power(1-np.power(rhodXY,2),-0.5)
        dy = dy_gt[:,:, 0]
        dx = dy_gt[:,:, 1]

        zd = np.power(sigdX,-2)*np.power(dx-mudX,2) + np.power(sigdY,-2)*np.power(dy-mudY,2) - 2*rhodXY*np.power(sigdX,-1)*np.power(sigdY, -1)*(dx-mudX)*(dy-mudY)
        
        denomd = np.log((1/(2*math.pi))*np.power(sigdX,-1)*np.power(sigdY,-1)*ohrdXY)
        
        lld = (denomd - 0.5*np.power(ohrdXY,2)*zd)
        
        y = y_gt[:,:, 0]
        x = y_gt[:,:, 1]


        muY = np.cumsum(mudY, axis = 1)
        muX = np.cumsum(mudX, axis = 1)
        sigX = np.sqrt(np.cumsum(np.power(sigdX,2), axis = 1))
        sigY = np.sqrt(np.cumsum(np.power(sigdY,2), axis = 1))
        rho_denom = np.multiply(sigX,sigY)
        rho_nom = np.multiply(rhodXY, np.multiply(sigdX, sigdY))
        rhoXY = np.divide(np.cumsum(rho_nom, axis = 1), rho_denom)
        ohrXY = np.power(1-np.power(rhoXY,2),-0.5)
        z = np.power(sigX,-2)*np.power(x-muX,2) + np.power(sigY,-2)*np.power(y-muY,2) - 2*rhoXY*np.power(sigX,-1)*np.power(sigY, -1)*(x-muX)*(y-muY)
        
        denom = np.log((1/(2*math.pi))*np.power(sigX,-1)*np.power(sigY,-1)*ohrXY)
        
        ll = (denom - 0.5*np.power(ohrXY,2)*z)
        
        return lld, ll
   





