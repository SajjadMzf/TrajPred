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
        'traj_dist_pred': BM_predicted_data_dist.cpu().data.numpy(),
        'man_gt': man_gt.cpu().data.numpy(),
        'man_preds': man_vectors.cpu().data.numpy(),
        'mode_prob': mode_prob.detach().cpu().data.numpy(),
    }
    '''
    K = 6

    mode_prob = np.concatenate(kpi_input_dict['mode_prob'], axis = 0)
    total_samples = mode_prob.shape[0]

    hp_mode = np.argmax(mode_prob, axis = 1)
    kbest_modes = np.argpartition(mode_prob, -1*K, axis = 1)[:,-1*K:]
    traj_gt = np.concatenate(kpi_input_dict['traj_gt'], axis = 0)
    
    traj_pred = np.concatenate(kpi_input_dict['traj_dist_pred'], axis = 0)
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
    index_array = np.repeat(np.arange(total_samples),K).reshape(total_samples, K)
    #print(mode_prob)
    #print(hp_mode)
    #print(kbest_modes)
    #print(kbest_modes[0])
    #print(index_array[0])
    kbest_traj_pred = traj_pred[index_array, kbest_modes]
    kbest_modes_probs = mode_prob[index_array, kbest_modes]
    kbest_modes_probs = np.divide(kbest_modes_probs, np.sum(kbest_modes_probs, axis = 1).reshape(total_samples,1)) 
    #print(kbest_modes_probs)
    #print(np.sum(kbest_modes_probs, axis = 1))
    #exit()
    sm_metrics_df, rmse = calc_sm_metric_df(p, hp_traj_pred, traj_gt)
    
    minFDE, p_minFDE, brier_minFDE, MR = calc_minFDE(kbest_traj_pred, kbest_modes_probs, traj_gt)
    minADE, p_minADE, brier_minADE = calc_minADE(kbest_traj_pred, kbest_modes_probs, traj_gt)
    
    return {
        'high prob mode histogram':hp_mode,
        'single modal metric group:\n': sm_metrics_df,
        'minFDE': minFDE, 
        'p_minFDE':p_minFDE,
        'brier_minFDE':brier_minFDE,
        'missrate': MR,
        'minADE': minADE,
        'p_minADE': p_minADE, 
        'brier_min_ADE':brier_minADE,
        'rmse': rmse
    }


def calc_sm_metric_df(p,traj_pred, traj_gt):
    # fde, rmse table
    
    total_samples = traj_gt.shape[0]
    prediction_ts = int(p.TGT_SEQ_LEN/p.FPS)
    if (p.TGT_SEQ_LEN/p.FPS) % 1 != 0:
        raise(ValueError('Target sequence length not dividable by FPS'))
    columns = ['<{} sec'.format(ts+1) for ts in range(prediction_ts)]
    index = ['FDE_lat', 'FDE_long', 'FDE', 'RMSE_lat', 'RMSE_long', 'RMSE']
    data = np.zeros((6, prediction_ts))
    for ts in range(prediction_ts):
        
        ts_index = (ts+1)*p.FPS
        current_total_samples = total_samples * ts_index
        #fde
        data[0,ts] = np.sum(np.absolute(traj_pred[:,ts_index-1,0]-traj_gt[:,ts_index-1,0]))/total_samples # 0 is laterel, 1 is longitudinal
        data[1,ts] = np.sum(np.absolute(traj_pred[:,ts_index-1,1]-traj_gt[:,ts_index-1,1]))/total_samples # 0 is laterel, 1 is longitudinal
        data[2,ts] = np.sum(np.absolute(traj_pred[:,ts_index-1,:]-traj_gt[:,ts_index-1,:]))/total_samples # 0 is laterel, 1 is longitudinal
        #rmse
        data[3,ts] = np.sqrt(
            np.sum((traj_pred[:,:ts_index,0]-traj_gt[:,:ts_index,0])**2)/current_total_samples
            )
        data[4,ts] = np.sqrt(
            np.sum((traj_pred[:,:ts_index,1]-traj_gt[:,:ts_index,1])**2)/current_total_samples
            )
        data[5,ts] = np.sqrt(
            np.sum((traj_pred[:,:ts_index,:]-traj_gt[:,:ts_index,:])**2)/current_total_samples
            )
    #FDE_table = ''.join(['{}:{:.4f}'.format(columns[ts], data[2,ts]) for ts in range(prediction_ts)])
    #RMSE_table = ''.join(['{}:{:.4f}'.format(columns[ts], data[5,ts]) for ts in range(prediction_ts)])
    rmse = data[5,prediction_ts-1]
    result_df = pd.DataFrame(data= data, columns = columns, index = index)
    
    return result_df, rmse

def calc_minFDE(traj_pred, mode_prob, traj_gt, mr_thr = 2):
    total_samples = traj_pred.shape[0]
    n_mode = traj_pred.shape[1]
    fde = np.zeros((total_samples, n_mode))
    for i in range(n_mode):
        fde[:,i] = np.sum(np.absolute(traj_pred[:,i,-1,:]-traj_gt[:,-1,:]), axis=-1)
    
    best_mode = np.argmin(fde, axis = 1)
    best_mode_prob = mode_prob[np.arange(total_samples), best_mode]
    b_fde_prob = np.power((1-best_mode_prob),2)
    p_fde_prob = -1*np.log(best_mode_prob)
    p_fde_prob[p_fde_prob<-1*np.log(0.05)] = -1*np.log(0.05) 
    fde = fde[np.arange(total_samples), best_mode]
    minFDE = np.sum(fde)/total_samples
    p_minFDE = np.sum(p_fde_prob*fde)/total_samples
    brier_minFDE = np.sum(b_fde_prob*fde)/total_samples
    MR = np.sum(fde>mr_thr)/total_samples
    return minFDE, p_minFDE, brier_minFDE, MR

def calc_minADE(traj_pred, mode_prob, traj_gt):
    total_samples = traj_pred.shape[0]
    seq_len = traj_pred.shape[2]
    n_mode = traj_pred.shape[1]
    ade = np.zeros((total_samples, n_mode))
    for i in range(n_mode):
        ade[:,i] = np.sum(np.sum(np.absolute(traj_pred[:,i,:,:]-traj_gt[:,:,:]), axis=-1), axis = -1)
    
    best_mode = np.argmin(ade, axis = 1)
    best_mode_prob = mode_prob[np.arange(total_samples), best_mode]
    b_ade_prob = np.power((1-best_mode_prob),2)
    p_ade_prob = -1*np.log(best_mode_prob)
    p_ade_prob[p_ade_prob<-1*np.log(0.05)] = -1*np.log(0.05) 
    ade = ade[np.arange(total_samples), best_mode]
    minADE = np.sum(ade)/(total_samples*seq_len)
    p_minADE = np.sum(p_ade_prob*ade)/(total_samples*seq_len)
    brier_minADE = np.sum(b_ade_prob*ade)/(total_samples*seq_len)
    return minADE, p_minADE, brier_minADE



   

def calc_roc_n_prc(p, all_lc_preds, all_labels, num_samples, figure_name, thr_type, eval_type):
    if thr_type == 'thr':
        
        thr_range = np.arange(0,101,1)/100
        precision_vs_thr = np.zeros_like(thr_range)
        recall_vs_thr = np.zeros_like(thr_range)
        fpr_vs_thr = np.zeros_like(thr_range)
        for i,thr in enumerate(thr_range):

            all_lc_preds_thr = all_lc_preds>=thr
            all_lc_preds_thr[:,:,0] = np.logical_not(np.logical_or(all_lc_preds_thr[:,:,1],all_lc_preds_thr[:,:,2]))
            all_pred = []
            all_pred.append(all_lc_preds_thr[:,:,0] * all_lc_preds[:,:,0]+ np.logical_not(all_lc_preds_thr[:,:,0]) * -1)# -1 is to make sure when thr is 0 non of lc is selected in argemax
            all_pred.append(all_lc_preds_thr[:,:,1] * all_lc_preds[:,:,1])
            all_pred.append(all_lc_preds_thr[:,:,2] * all_lc_preds[:,:,2])
            all_pred = np.stack(all_pred, axis = -1)
            all_pred = np.argmax(all_pred, axis =-1)
            precision_vs_thr[i], recall_vs_thr[i], fpr_vs_thr[i] = calc_prec_recall_fpr(p, all_pred, all_labels, prediction_seq, num_samples)
    else:
        raise('Unknown thr type')

    recall_vs_thr = np.flip(recall_vs_thr, axis = 0)
    fpr_vs_thr = np.flip(fpr_vs_thr, axis = 0)
    precision_vs_thr = np.flip(precision_vs_thr, axis = 0)

    if eval_type == 'Test':
        # Creating dirs
        figure_dir = os.path.join(p.FIGS_DIR, 'roc')
        if not os.path.exists(figure_dir):
            os.mkdir(figure_dir)
        plot_dir = os.path.join(figure_dir, figure_name +p.unblanaced_ext+ '.png')
        fig_obs_dir = os.path.join(figure_dir, figure_name +p.unblanaced_ext+  '.pickle')

        recall_ax = plt.figure()

        # Creating Figs
        plt.plot(fpr_vs_thr, recall_vs_thr)
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.grid()
        recall_ax.savefig(plot_dir)
        with open(fig_obs_dir, 'wb') as fid:
            pickle.dump(recall_ax, fid)

        # Creating dirs
        figure_dir = os.path.join(p.FIGS_DIR, 'prec_recall_curv')
        if not os.path.exists(figure_dir):
            os.mkdir(figure_dir)
        plot_dir = os.path.join(figure_dir, figure_name +p.unblanaced_ext+ '.png')
        fig_obs_dir = os.path.join(figure_dir, figure_name +p.unblanaced_ext+  '.pickle')

        recall_ax = plt.figure()

        # Creating Figs
        plt.plot(recall_vs_thr, precision_vs_thr)
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid()
        recall_ax.savefig(plot_dir)
        with open(fig_obs_dir, 'wb') as fid:
            pickle.dump(recall_ax, fid)
        
    auc = metrics.auc(fpr_vs_thr, recall_vs_thr)
    #auc = 0
    max_j = max(recall_vs_thr-fpr_vs_thr)

    return auc, max_j

def calc_prec_recall_fpr(p, all_preds, all_labels, prediction_seq, num_samples):
    # metrics with thr
    all_hits = np.zeros_like(all_preds)
    
    recall_vs_TTLC = np.zeros((prediction_seq))
    
    acc_vs_SEQ = np.zeros((prediction_seq))
    precision_vs_SEQ = np.zeros((prediction_seq))
    FPR_vs_SEQ = np.zeros((prediction_seq))
    FN_vs_TTLC = np.zeros((prediction_seq))
    
    TP_vs_TTLC = np.zeros((prediction_seq))
    FP_vs_SEQ = np.zeros((prediction_seq))
    TN_vs_SEQ = np.zeros((prediction_seq))

    for t in range(prediction_seq):
        all_hits[:,t] = (all_preds[:,t] == all_labels)
        TP_vs_TTLC[t] = np.sum(np.logical_and(all_hits[:,t], (all_labels!=0)))/num_samples
        FN_vs_TTLC[t] = np.sum(np.logical_and((all_hits[:,t]==False), (all_labels!=0)))/num_samples
        FP_vs_SEQ[t] = np.sum(
            np.logical_or(
                np.logical_and((all_hits[:,t]==False), (all_labels==0)), 
                np.logical_and(
                    np.logical_and((all_hits[:,t]==False), (all_labels!=0)), 
                    all_preds[:,t]!=0)
                    )
                    )/num_samples
        TN_vs_SEQ[t] = np.sum(np.logical_and((all_hits[:,t]==True), (all_labels==0)))/num_samples
        recall_vs_TTLC[t] = TP_vs_TTLC[t]/(TP_vs_TTLC[t] + FN_vs_TTLC[t])
        precision_vs_SEQ[t] = (TP_vs_TTLC[t] + 1e-14)/(TP_vs_TTLC[t] + FP_vs_SEQ[t] + 1e-14)
        FPR_vs_SEQ[t] = FP_vs_SEQ[t]/(FP_vs_SEQ[t] + TN_vs_SEQ[t])
        
    
    
    precision = np.mean(precision_vs_SEQ)
    FPR = np.mean(FPR_vs_SEQ)
    recall = np.mean(recall_vs_TTLC)
    return precision, recall,FPR
def calc_classification_metrics(p, all_preds, all_labels, prediction_seq, num_samples, eval_type, figure_name):
    # metrics with thr
    all_hits = np.zeros_like(all_preds)
    all_TPs = np.zeros_like(all_preds)
    FP_index = np.zeros_like(all_preds, dtype= np.bool)
    
    FP_TTLC = np.zeros((all_preds.size))
    cur_FP_index = 0
    recall_vs_TTLC = np.zeros((prediction_seq))
    
    acc_vs_SEQ = np.zeros((prediction_seq))
    precision_vs_SEQ = np.zeros((prediction_seq))
    FPR_vs_SEQ = np.zeros((prediction_seq))
    FN_vs_TTLC = np.zeros((prediction_seq))
    
    TP_vs_TTLC = np.zeros((prediction_seq))
    FP_vs_SEQ = np.zeros((prediction_seq))
    TN_vs_SEQ = np.zeros((prediction_seq))

    for t in range(prediction_seq):
        all_hits[:,t] = (all_preds[:,t] == all_labels)
        all_TPs[:,t] = np.logical_and(all_hits[:,t], (all_labels!=0))
        TP_vs_TTLC[t] = np.sum(np.logical_and(all_hits[:,t], (all_labels!=0)))/num_samples
        FN_vs_TTLC[t] = np.sum(np.logical_and((all_hits[:,t]==False), (all_labels!=0)))/num_samples
        FP_vs_SEQ[t] = np.sum(
            np.logical_or(
                np.logical_and((all_hits[:,t]==False), (all_labels==0)), 
                np.logical_and(
                    np.logical_and((all_hits[:,t]==False), (all_labels!=0)), 
                    all_preds[:,t]!=0)
                    )
                    )/num_samples
        TN_vs_SEQ[t] = np.sum(np.logical_and((all_hits[:,t]==True), (all_labels==0)))/num_samples
        prev_FP_index = cur_FP_index
        FP_index[:,t] =np.logical_or(
                np.logical_and((all_hits[:,t]==False), (all_labels==0)), 
                np.logical_and(
                    np.logical_and((all_hits[:,t]==False), (all_labels!=0)), 
                    all_preds[:,t]!=0)
                    )
        cur_FP_index += np.sum(FP_index[:,t])
        

        recall_vs_TTLC[t] = TP_vs_TTLC[t]/(TP_vs_TTLC[t] + FN_vs_TTLC[t])
        precision_vs_SEQ[t] = TP_vs_TTLC[t]/(TP_vs_TTLC[t] + FP_vs_SEQ[t])
        FPR_vs_SEQ[t] = FP_vs_SEQ[t]/(FP_vs_SEQ[t] + TN_vs_SEQ[t])
        
        acc_vs_SEQ[t] = sum(all_hits[:,t])/num_samples
    
    
    #print(recall_vs_TTLC)
    accuracy = np.mean(acc_vs_SEQ)
    precision = np.mean(precision_vs_SEQ)
    FPR = np.mean(FPR_vs_SEQ)
    recall = np.mean(recall_vs_TTLC)
    f1 = 2*(precision*recall)/(precision+recall)
    #cumul_FP_TTLC = 0
    if eval_type == 'Test':
        # Creating dirs
        figure_dir = os.path.join(p.FIGS_DIR, 'recall_vs_TTLC')
        if not os.path.exists(figure_dir):
            os.mkdir(figure_dir)
        plot_dir = os.path.join(figure_dir, figure_name +p.unblanaced_ext+ '.png')
        fig_obs_dir = os.path.join(figure_dir, figure_name +p.unblanaced_ext+  '.pickle')

        recall_ax = plt.figure()
        ttlc_seq = (prediction_seq-np.arange(prediction_seq))/p.FPS

        # Creating Figs
        plt.plot(ttlc_seq, recall_vs_TTLC*100)
        plt.ylim(0,100)
        plt.xlim(ttlc_seq[0], ttlc_seq[-1])
        plt.xlabel('Time to lane change (TTLC) (s)')
        plt.ylabel('Recall (%)')
        plt.grid()
        recall_ax.savefig(plot_dir)
        with open(fig_obs_dir, 'wb') as fid:
            pickle.dump(recall_ax, fid)
        
        

    

    return accuracy, precision, recall, f1, FPR, all_TPs


def calc_avg_pred_time(p, all_TPs, all_labels, prediction_seq, num_samples):
    all_TPs_LC = all_TPs[all_labels!=0,:]
    num_lc = all_TPs_LC.shape[0]
    seq_len = all_TPs_LC.shape[1]
    all_TPs_LC_r = np.flip(all_TPs_LC, 1)

    first_false_preds = np.ones((num_lc))*prediction_seq
    last_true_preds = np.zeros((num_lc))
    for i in range(num_lc):
        if np.any(all_TPs_LC_r[i]) == True:
            last_true_preds[i] = np.nonzero(all_TPs_LC_r[i])[0][-1] + 1
        for seq_itr in range(seq_len):
            end_lim = min(seq_len, seq_itr + p.ACCEPTED_GAP+1)
            if np.any(all_TPs_LC_r[i, seq_itr:end_lim]) == False:
                first_false_preds[i] = seq_itr
                break
        

    robust_pred_time = np.sum(first_false_preds)/(p.FPS*num_lc)
    pred_time = np.sum(last_true_preds)/(p.FPS*num_lc)
    
    return robust_pred_time, pred_time


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
    


def MTPM_loss(man_pred, man_gt, n_mode, man_per_mode, device, alpha = 1, beta = 1, test_phase = False):
    # man pred: [batch_size, (1+3*man_per_mode + tgt_seq_len)*modes]
    # man_gt: [batch_size, tgt_seq_len]
    
    tgt_seq_len = man_gt.shape[1]
    w_ind = mf.divide_prediction_window(tgt_seq_len, man_per_mode)
    man_gt, time_gt = mf.man_vector2man_n_timing(man_gt, man_per_mode, w_ind)
    man_gt = man_gt.to(device).type(torch.long)
    time_gt = time_gt.to(device).type(torch.long)
    batch_size = man_pred.shape[0]
    #mode probabilities
    mode_pr = man_pred[:, 0:n_mode]
    man_pr = man_pred[:,n_mode:n_mode+ n_mode*3*man_per_mode]
    time_pr = man_pred[:,n_mode+ n_mode*3*man_per_mode:]
    
    

    man_pr = man_pr.reshape(batch_size, n_mode, man_per_mode, 3)
    time_pr = time_pr.reshape(batch_size, n_mode, tgt_seq_len)
    
    time_pr_list = []
    for i in range(len(w_ind)):
        time_pr_list.append(time_pr[:,:,w_ind[i,0]:w_ind[i,1]])
    
    loss_func = torch.nn.CrossEntropyLoss(ignore_index = -1)
    loss_func_no_r = torch.nn.CrossEntropyLoss(ignore_index = -1, reduction = 'none')

    man_loss_list = []
    time_loss_list = []
    for mode_itr in range(n_mode):
        man_loss_list.append(torch.sum(loss_func_no_r(man_pr[:,mode_itr], man_gt), dim = 1)) 
        mode_time_loss = 0
        for i, mode_time_pr in enumerate(time_pr_list):
            mode_time_loss += loss_func_no_r(mode_time_pr[:,mode_itr], time_gt[:,i])
        time_loss_list.append(mode_time_loss)
    man_losses = torch.stack(man_loss_list)
    time_losses = torch.stack(time_loss_list)
    
    if test_phase:
        winning_mode = torch.argmax(mode_pr, dim=1)
    else:
        winning_mode = mf.find_winning_mode(man_losses, time_losses)
    
    mode_loss = loss_func(mode_pr, winning_mode)
    man_loss = torch.mean(man_losses[winning_mode, np.arange(batch_size)])
    time_loss = torch.mean(time_losses[winning_mode, np.arange(batch_size)])
    lossVal = mode_loss + alpha*man_loss + beta*time_loss 
    return lossVal, mode_loss, man_loss, time_loss



