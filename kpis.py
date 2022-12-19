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
from debugging_utils import *
font = {'size'   : 22}
matplotlib.rcParams['figure.figsize'] = (18, 12)
matplotlib.rc('font', **font)

import models_functions as mf


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
    
    mnll = calc_meanNLL(p, dtraj_pred, dtraj_gt, mode_prob)
    minFDE = {}
    minRMSE = {}
    minMR = {}
    minACC = {}
    minTimeACC = {}
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
        minTimeACC[key], minTimeACC['min'] = calc_time_acc(p, kbest_time_bar_preds, time_bar_gt)
        minTimeMAE[key], minTimeMAE['random'] = calc_time_mae(p, kbest_time_bar_preds, time_bar_gt)
        #sm_metrics_df, rmse = calc_sm_metric_df(p, hp_traj_pred, traj_gt)
    
        minFDE[key] = calc_minFDE(p,kbest_traj_pred, kbest_modes_probs, traj_gt)
        minRMSE[key], rmse[key] = calc_minRMSE(p,kbest_traj_pred, kbest_modes_probs, traj_gt)
        
    
    return {
        'activated modes group': unique_modes,
        'activated modes percentage group': 100*mode_freq/sum(mode_freq),
        'high prob mode histogram':hp_mode,
        'time pr histogram': time_bar_preds[:,:,0],
        'time gt histogram': time_bar_gt[:,0],
        'minACC': minACC,
        'minTimeACC': minTimeACC,
        'minTimeMAE': minTimeMAE,
        'minFDE': minFDE, 
        'minRMSE': minRMSE,
        'mnll': mnll,
        'rmse': rmse['K=1'] # minRMSE K=1 max pred horizon
    }


def XMTP_kpis(p, kpi_input_dict, traj_min, traj_max, figure_name):
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
        'traj_dist_pred': BM_predicted_data_dist.cpu().data.numpy(),
        'man_gt': man_gt.cpu().data.numpy(),
        'man_preds': man_vectors.cpu().data.numpy(),
        'mode_prob': mode_prob.detach().cpu().data.numpy(),
    }
    '''
    
    
    mode_prob = np.concatenate(kpi_input_dict['mode_prob'], axis = 0)
    total_samples = mode_prob.shape[0]
    hp_mode = np.argmax(mode_prob, axis = 1)
    (unique_modes, mode_freq) = np.unique(hp_mode, return_counts = True)
    sorted_args = np.argsort(unique_modes)
    unique_modes = unique_modes[sorted_args]
    mode_freq = mode_freq[sorted_args]
    

    traj_gt = np.concatenate(kpi_input_dict['traj_gt'], axis = 0)
    
    traj_pred = np.concatenate(kpi_input_dict['traj_dist_preds'], axis = 0)
    traj_pred = traj_pred[:,:,:,:2]
    traj_max = kpi_input_dict['traj_max'][0]
    traj_min = kpi_input_dict['traj_min'][0]
    
    #denormalise
    traj_pred = traj_pred*(traj_max-traj_min) + traj_min
    traj_gt = traj_gt*(traj_max-traj_min) + traj_min
    #from diff to actual
    traj_pred = np.cumsum(traj_pred, axis = 2)
    traj_gt = np.cumsum(traj_gt, axis = 1)
    hp_traj_pred = traj_pred[np.arange(total_samples), hp_mode]


    minFDE = {}
    minRMSE = {}
    minMR = {}
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
        
        kbest_modes = np.argpartition(mode_prob, -1*K, axis = 1)[:,-1*K:]
        index_array = np.repeat(np.arange(total_samples),K).reshape(total_samples, K)
            
        minFDE[key] = calc_minFDE(p,kbest_traj_pred, kbest_modes_probs, traj_gt)
        minRMSE[key], rmse[key] = calc_minRMSE(p,kbest_traj_pred, kbest_modes_probs, traj_gt)



    return {
        'activated modes group': unique_modes,
        'activated modes percentage group': 100*mode_freq/sum(mode_freq),
        'high prob mode histogram':hp_mode,
        'minFDE': minFDE, 
        'minRMSE': minRMSE,
        'rmse': rmse['K=1'] # minRMSE K=1 max pred horizon
    }



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

def calc_fde_vs_time(p,traj_pred, traj_gt, mr_thr):
    # fde table
    
    total_samples = traj_gt.shape[0]
    prediction_ts = int(p.TGT_SEQ_LEN/p.FPS)
    if (p.TGT_SEQ_LEN/p.FPS) % 1 != 0:
        raise(ValueError('Target sequence length not dividable by FPS'))
    columns = ['<{} sec'.format(ts+1) for ts in range(prediction_ts)]
    index = ['FDE_lat', 'FDE_long', 'FDE', 'MR']
    data = np.zeros((4, prediction_ts))
    for ts in range(prediction_ts):
        
        ts_index = (ts+1)*p.FPS
        #fde
        data[0,ts] = np.sum(np.absolute(traj_pred[:,ts_index-1,0]-traj_gt[:,ts_index-1,0]))/total_samples # 0 is laterel, 1 is longitudinal
        data[1,ts] = np.sum(np.absolute(traj_pred[:,ts_index-1,1]-traj_gt[:,ts_index-1,1]))/total_samples # 0 is laterel, 1 is longitudinal
        data[2,ts] = np.sum(
            np.sqrt(np.sum((traj_pred[:,ts_index-1,:]-traj_gt[:,ts_index-1,:])**2, axis = -1))
        )/total_samples # 0 is laterel, 1 is longitudinal
        data[3,ts] = np.sum(np.sqrt(np.sum((traj_pred[:,ts_index-1,:]-traj_gt[:,ts_index-1,:])**2, axis = -1))>mr_thr)/total_samples  
    fde_df = pd.DataFrame(data= data, columns = columns, index = index)
    
    return fde_df


def calc_minFDE(p, traj_pred, mode_prob, traj_gt, mr_thr = 2):
    total_samples = traj_pred.shape[0]
    n_mode = traj_pred.shape[1]
    fde = np.zeros((total_samples, n_mode))
    for i in range(n_mode):
        fde[:,i] = np.sum(np.absolute(traj_pred[:,i,-1,:]-traj_gt[:,-1,:]), axis=-1)
    best_mode = np.argmin(fde, axis = 1)
    
    fde = fde[np.arange(total_samples), best_mode]
    minFDE = calc_fde_vs_time(p,traj_pred[np.arange(total_samples), best_mode], traj_gt, mr_thr)
    
    return minFDE

def calc_minRMSE(p, traj_pred, mode_prob, traj_gt):
    total_samples = traj_pred.shape[0]
    seq_len = traj_pred.shape[2]
    n_mode = traj_pred.shape[1]
    rmse = np.zeros((total_samples, n_mode))
    for i in range(n_mode):
        rmse[:,i] = np.sqrt(np.sum(np.sum((traj_pred[:,i,:,:]-traj_gt[:,:,:])**2, axis=-1), axis = -1)/seq_len)
    
    best_mode = np.argmin(rmse, axis = 1)
    minRMSE, rmse = calc_rmse_vs_time(p,traj_pred[np.arange(total_samples), best_mode], traj_gt)
    
    return minRMSE, rmse


def calc_meanNLL(p, y_pred, y_gt, mode_prob):
    n_mode = y_pred.shape[1]
    n_sample = y_pred.shape[0]
    prediction_ts = int(p.TGT_SEQ_LEN/p.FPS)
    if (p.TGT_SEQ_LEN/p.FPS) % 1 != 0:
            raise(ValueError('Target sequence length not dividable by FPS'))
    columns = ['<{} sec'.format(ts+1) for ts in range(prediction_ts)]
    index = ['RMSE_lat', 'RMSE_long', 'RMSE']
    if n_mode ==1:
        mnll = -1*log_likelihood_numpy(p, y_pred[:,0], y_gt)
        mnll = np.sum(mnll, axis = 0)/n_sample
    else:
        likelihood_per_mode = np.zeros((n_sample,n_mode , prediction_ts))
        for i in range(n_mode):
            log_likelihood = log_likelihood_numpy(p, y_pred[:,i], y_gt)
            for j in range(prediction_ts):
                likelihood_per_mode[:,i,j] = np.multiply(mode_prob[:,i], np.exp(log_likelihood[:,j]))
        mnll = -1*np.log(np.sum(likelihood_per_mode, axis = 1))
        mnll = np.sum(mnll, axis = 0)/n_sample
    mnll = mnll.reshape(1,5)
    mnll_df = pd.DataFrame(data= mnll, columns = columns, index = ['Mean_NLL'])
    return mnll_df


def log_likelihood_numpy(p,y_pred, y_gt):
        #print_shape('y_pred', y_pred)
        muY = y_pred[:,:,0]
        muX = y_pred[:,:,1]
        sigY = y_pred[:,:,2] # sigY = standard deviation of y
        sigX = y_pred[:,:,3] # sigX = standard deviation of x
        rho = y_pred[:,:,4]
        ohr = np.power(1-np.power(rho,2),-0.5)
        y = y_gt[:,:, 0]
        x = y_gt[:,:, 1]
       
        z = np.power(sigX,-2)*np.power(x-muX,2) + np.power(sigY,-2)*np.power(y-muY,2) - 2*rho*np.power(sigX,-1)*np.power(sigY, -1)*(x-muX)*(y-muY)
        
        denom = np.log((1/(2*math.pi))*np.power(sigX,-1)*np.power(sigY,-1)*ohr)
        
        ll = (denom - 0.5*np.power(ohr,2)*z)
        #print(nll)
        prediction_ts = int(p.TGT_SEQ_LEN/p.FPS)
        n_sample = y_pred.shape[0]
        ll_seq = np.zeros((n_sample, prediction_ts))
        for ts in range(prediction_ts):
            ts_index = (ts+1)*p.FPS
            ll_seq[:, ts] = np.sum(ll[:, :ts_index], axis = 1)/ts_index
        return ll_seq
    

def NLL_loss(y_pred, y_gt):
        #print_shape('y_pred', y_pred)
        muY = y_pred[:,:,0]
        muX = y_pred[:,:,1]
        sigY = y_pred[:,:,2] # sigY = standard deviation of y
        sigX = y_pred[:,:,3] # sigX = standard deviation of x
        rho = y_pred[:,:,4]
        ohr = torch.pow(1-torch.pow(rho,2),-0.5)
        y = y_gt[:,:, 0]
        x = y_gt[:,:, 1]
        #print_shape('y_pred',y_pred)
        #print_shape('y_gt',y_gt)
        z = torch.pow(sigX,-2)*torch.pow(x-muX,2) + torch.pow(sigY,-2)*torch.pow(y-muY,2) - 2*rho*torch.pow(sigX,-1)*torch.pow(sigY, -1)*(x-muX)*(y-muY)
        #if torch.sum(z)<0:
        #    print_value('z',z)
        #    exit()
        
        denom = torch.log((1/(2*math.pi))*torch.pow(sigX,-1)*torch.pow(sigY,-1)*ohr)
        #print(denom)
        #if torch.sum(denom)>0:
        #    print_value('denom',denom)
        nll = -1*(denom - 0.5*torch.pow(ohr,2)*z)
        #print(nll)
        lossVal = torch.sum(nll)/np.prod(y.shape)
       
        #print_value('lossval',lossVal)
        #exit()
        return lossVal
    


def MTPM_loss(p, man_pred, man_vec_gt, n_mode, man_per_mode, device, test_phase = False, time_reg = True):
    # man pred: [batch_size, (1+3*man_per_mode + tgt_seq_len)*modes]
    # man_gt: [batch_size, tgt_seq_len]
    
    tgt_seq_len = man_vec_gt.shape[1]
    w_ind = mf.divide_prediction_window(tgt_seq_len, man_per_mode)
    man_gt, time_gt = mf.man_vector2man_n_timing(man_vec_gt, man_per_mode, w_ind)
    man_gt = man_gt.to(device).type(torch.long)
    time_gt = time_gt.to(device).type(torch.long)
    man_vec_gt = man_vec_gt.to(device).type(torch.long)
    batch_size = man_pred.shape[0]
    #mode probabilities
    mode_pr = man_pred[:, 0:n_mode] # mode prediction: probability of modes
    man_pr = man_pred[:,n_mode:n_mode+ n_mode*3*man_per_mode] # manouvre prediction: order of manouvres 
    time_pr = man_pred[:,n_mode+ n_mode*3*man_per_mode:] # timing of the manouvre
    
    

    man_pr = man_pr.reshape(batch_size, n_mode, man_per_mode, 3)
    man_pr_class = torch.argmax(man_pr, dim = -1)
    man_pr = torch.permute(man_pr,(0,1,3,2))
    if time_reg:
        time_pr = time_pr.reshape(batch_size, n_mode, man_per_mode-1).clone()
        for i in range(man_per_mode-1):
            time_pr[:,:,i] = (time_pr[:,:,i]/2 + 0.5)*(w_ind[i,1]-w_ind[i,0]) 
        time_bar_pred = time_pr
        
    else:
        time_pr = time_pr.reshape(batch_size, n_mode, tgt_seq_len)
    
        time_pr_list = []
        for i in range(len(w_ind)):
            time_pr_list.append(time_pr[:,:,w_ind[i,0]:w_ind[i,1]])
        
    
    loss_func = torch.nn.CrossEntropyLoss(ignore_index = -1)
    loss_func_no_r = torch.nn.CrossEntropyLoss(ignore_index = -1, reduction = 'none')
    reg_loss_func_no_r = torch.nn.MSELoss(reduction = 'none')
    man_loss_list = []
    time_loss_list = []
    for mode_itr in range(n_mode):
        
        man_loss_list.append(torch.sum(loss_func_no_r(man_pr[:,mode_itr], man_gt), dim = 1)) 
        if time_reg:
            mode_time_loss = torch.sum(torch.mul(reg_loss_func_no_r(time_pr[:,mode_itr], time_gt.float()), (time_gt!=-1).float()), dim = -1)
        else:
            mode_time_loss = 0
            for i, mode_time_pr in enumerate(time_pr_list):
                mode_time_loss += loss_func_no_r(mode_time_pr[:,mode_itr], time_gt[:,i])
        time_loss_list.append(mode_time_loss)
    man_losses = torch.stack(man_loss_list, dim = 1)
    time_losses = torch.stack(time_loss_list, dim = 1)
    
    man_pr = torch.permute(man_pr, (0,1,3,2))
    man_pr = man_pr.reshape(batch_size*n_mode, man_per_mode,3)
    man_pr_argmax = torch.argmax(man_pr, dim = -1)
    #time_pr = time_pr.reshape(batch_size*n_mode, tgt_seq_len)
    # Un comment for man acc loss calculation
    '''
    time_pr_arg_list = []
    for i in range(len(w_ind)):
        time_pr_arg_list.append(torch.argmax(time_pr[:,w_ind[i,0]:w_ind[i,1]], dim = -1))
    
    man_vec_pr = mf.man_n_timing2man_vector(man_pr_argmax, time_pr_arg_list, tgt_seq_len, w_ind, device)
    man_vec_pr = man_vec_pr.reshape(batch_size, n_mode, tgt_seq_len)
    #print(man_vec_pr[:,0].shape)
    man_acc = calc_man_acc_torch( man_vec_pr, man_vec_gt, device)
    '''
    if test_phase:
        winning_mode = torch.argmax(mode_pr, dim=1)
    else:
        #winning_mode = torch.argmin(man_vec_loss, dim = 1)
        winning_mode = torch.argmin(man_losses, dim = 1)
        #winning_mode = torch.argmin(man_acc, dim = 1)

    mode_loss = loss_func(mode_pr, winning_mode)
    man_loss = torch.mean(man_losses[np.arange(batch_size), winning_mode])
    time_loss = torch.mean(time_losses[np.arange(batch_size), winning_mode])
    lossVal = mode_loss + man_loss + time_loss 
    return lossVal, mode_loss, man_loss, time_loss, time_bar_pred



