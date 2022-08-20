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

def calc_metric(p, all_traj_preds, all_traj_labels, man_preds, man_labels, traj_label_min, traj_label_max, epoch=None, eval_type = 'Test', figure_name= None):
    
    traj_metrics, man_metrics, traj_df, RMSE_table, FDE_table = calc_traj_metrics(p, all_traj_preds, all_traj_labels, man_preds, man_labels, traj_label_min, traj_label_max)
    if eval_type == 'Test' and p.parameter_tuning_experiment:
        for metric_str in p.selected_metrics:
            p.log_dict[metric_str] = eval(metric_str)
    
    return traj_metrics, man_metrics, traj_df


def calc_traj_metrics(p, 
    traj_preds:'[number of samples, target sequence length, number of output states]', 
    traj_labels,
    man_preds,
    man_labels,
    traj_min, 
    traj_max):
    #traj_preds [number of samples, target sequence length, number of output states]
    #TODO:1. manouvre specific fde and rmse table, 2.  save sample output traj imags 3. man pred error
    #man_preds = traj_preds[:,:,2:]*2
    #man_preds = np.rint(man_preds)
    #man_labels = traj_labels[:,:,2:]*2
    #man_labels = np.rint(man_labels)
    
    traj_preds = traj_preds[:,:,:2]
    in_man_labels = man_labels[:,:p.IN_SEQ_LEN]
    man_labels = man_labels[:,p.IN_SEQ_LEN:]
    in_traj_labels = traj_labels[:,:p.IN_SEQ_LEN,:2]
    traj_labels = traj_labels[:,p.IN_SEQ_LEN:,:2]
    arg_man_labels = np.argmax(man_labels, axis = -1)
    lc_frames = (arg_man_labels>0)
    lk_frames = (arg_man_labels ==0)
    total_lc_frames = np.count_nonzero(lc_frames)
    total_lk_frames = np.count_nonzero(lk_frames)
    total_frames = np.prod(traj_labels[:,:,0].shape)
    total_sequences = traj_labels.shape[0]
    #print_shape('traj_labels', traj_labels)
    print_value('total_frames', total_frames)
    print_value('total_lk_frames', total_lk_frames)
    print_value('total_lc_frames', total_lc_frames)
    #assert(total_frames ==  total_lk_frames + total_lc_frames)

    #denormalise
    traj_preds = traj_preds*(traj_max-traj_min) + traj_min
    traj_labels = traj_labels*(traj_max-traj_min) + traj_min
    #from diff to actual
    traj_preds = np.cumsum(traj_preds, axis = 1)
    traj_labels = np.cumsum(traj_labels, axis = 1)

    # fde
    fde = np.sum(np.absolute(traj_preds[:,-1,:]-traj_labels[:,-1,:]))/total_sequences
    # rmse
    mse = np.sum((traj_preds-traj_labels)**2)/total_frames
    #print_shape('traj_preds', traj_preds)
    mse_lc = 0 #np.sum((lc_frames*(traj_preds-traj_labels))**2)/total_lc_frames
    mse_lk = 0 #np.sum((lk_frames*(traj_preds-traj_labels))**2)/total_lk_frames
    rmse = np.sqrt(mse) 
    rmse_lc = 0 #np.sqrt(mse_lc) 
    rmse_lk = 0 #np.sqrt(mse_lk) 

    # man metrics TODO: update man metrics
    #TP = np.sum(np.logical_and((man_preds == man_labels), (man_labels>0))) #TODO 1: argmax man preds
    #TPnFP = np.sum(man_preds>0)
    #TPnFN = np.sum(man_labels>0)
    #print_value('TP',TP)
    recall = 0#TP/TPnFN
    precision = 0#TP/TPnFP
    accuracy =  0#np.sum(man_preds == man_labels)/total_frames

    # fde, rmse table
    prediction_ts = int(p.TGT_SEQ_LEN/p.FPS)
    if (p.TGT_SEQ_LEN/p.FPS) % 1 != 0:
        raise(ValueError('Target sequence length not dividable by FPS'))
    columns = ['<{} sec'.format(ts+1) for ts in range(prediction_ts)]
    index = ['FDE_lat', 'FDE_long', 'FDE', 'RMSE_lat', 'RMSE_long', 'RMSE']
    data = np.zeros((6, prediction_ts))
    for ts in range(prediction_ts):
        
        ts_index = (ts+1)*p.FPS
        current_total_samples = total_sequences * ts_index
        #fde
        data[0,ts] = np.sum(np.absolute(traj_preds[:,ts_index-1,0]-traj_labels[:,ts_index-1,0]))/total_sequences # 0 is laterel, 1 is longitudinal
        data[1,ts] = np.sum(np.absolute(traj_preds[:,ts_index-1,1]-traj_labels[:,ts_index-1,1]))/total_sequences # 0 is laterel, 1 is longitudinal
        data[2,ts] = np.sum(np.absolute(traj_preds[:,ts_index-1,:]-traj_labels[:,ts_index-1,:]))/total_sequences # 0 is laterel, 1 is longitudinal
        #rmse
        data[3,ts] = np.sqrt(
            np.sum((traj_preds[:,:ts_index,0]-traj_labels[:,:ts_index,0])**2)/current_total_samples
            )
        data[4,ts] = np.sqrt(
            np.sum((traj_preds[:,:ts_index,1]-traj_labels[:,:ts_index,1])**2)/current_total_samples
            )
        data[5,ts] = np.sqrt(
            np.sum((traj_preds[:,:ts_index,:]-traj_labels[:,:ts_index,:])**2)/current_total_samples
            )
    
    '''
    if p.PLOT_TRAJS:
        p.PLOT_TRAJS_DIR 
        for i in range(p.PLOT_TRAJS_NUM):
            fig = plt.figure()
            ax_traj = fig.add_subplot(2, 1, 1)
            gt_man = np.argmax(man_labels[i], axis = -1)
            pr_man = np.argmax(man_preds[i], axis = -1)

            gt_traj = traj_labels[i]
            pr_traj = traj_preds[i]
            print(gt_traj[:,1], gt_traj[:,0])
            
            ax_traj.plot(gt_traj[0,1], gt_traj[0,0], '*')
            ax_traj.plot(gt_traj[:,1], gt_traj[:,0], label = 'gt traj')
            ax_traj.plot(pr_traj[0,1], pr_traj[0,0], '*')
            ax_traj.plot(pr_traj[:,1], pr_traj[:,0], label = 'predicted traj')
            # And a corresponding grid
            ax_traj.grid(True)
            #plt.xlim(ttlc_seq[0], ttlc_seq[-1])
            #plt.ylim(0,100)
            plt.xlabel('X Coordinate (m)')
            plt.ylabel('Y Coordinate (m)')
            plt.tight_layout()
            ax_traj.legend(loc = 'lower right')
            
            ax_man = fig.add_subplot(2,1,2)
            ax_man.grid(True)
            ax_man.plot(gt_man, label = 'gt man')
            ax_man.plot(pr_man, label = 'predicted man')

            
            
            plt.show()
            exit()
    '''
    FDE_table = ''.join(['{}:{:.4f}'.format(columns[ts], data[2,ts]) for ts in range(prediction_ts)])
    RMSE_table = ''.join(['{}:{:.4f}'.format(columns[ts], data[5,ts]) for ts in range(prediction_ts)])
    result_df = pd.DataFrame(data= data, columns = columns, index = index)
    traj_metrics = (rmse, rmse_lc, rmse_lk, fde)
    man_metrics = (accuracy, precision, recall)

    return traj_metrics, man_metrics, result_df, RMSE_table, FDE_table


    

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
    w_ind = divide_prediction_window(tgt_seq_len, man_per_mode)
    man_gt, time_gt = man_vector2man_n_timing(man_gt, man_per_mode, w_ind)
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
        winning_mode = find_winning_mode(man_losses, time_losses)
    mode_loss = loss_func(mode_pr, winning_mode)
    man_loss = torch.mean(man_losses[winning_mode, np.arange(batch_size)])
    time_loss = torch.mean(man_losses[winning_mode, np.arange(batch_size)])
    lossVal = mode_loss + alpha*man_loss + beta*time_loss 
    return lossVal

def sel_high_prob_man(man_pred, n_mode, man_per_mode, tgt_seq_len, device):
    batch_size = man_pred.shape[0]
    #mode probabilities
    mode_pr = man_pred[:, 0:n_mode]
    man_pr = man_pred[:,n_mode:n_mode+ n_mode*3*man_per_mode]
    time_pr = man_pred[:,n_mode+ n_mode*3*man_per_mode:]
    man_pr = man_pr.reshape(batch_size, n_mode, man_per_mode, 3)
    time_pr = time_pr.reshape(batch_size, n_mode, tgt_seq_len)
    man_pr = torch.argmax(man_pr, dim = -1)
    high_prob_mode = torch.argmax(mode_pr, dim=1)

    time_pr = time_pr[np.arange(batch_size),high_prob_mode]
    w_ind = divide_prediction_window(tgt_seq_len, man_per_mode)
    time_pr_list = []
    for i in range(len(w_ind)):
        time_pr_list.append(torch.argmax(time_pr[:,w_ind[i,0]:w_ind[i,1]], dim=-1))

    man_vector = man_n_timing2man_vector(man_pr[np.arange(batch_size),high_prob_mode], time_pr_list, tgt_seq_len, w_ind)

    man_vector = man_vector.to(device).type(torch.long)

    return man_vector

def find_winning_mode(man_losses, time_losses, thr=0):
    # [n_mode, batch_size, ]
    ml_values, ml_index = torch.sort(man_losses+time_losses, dim=0)
    #ml_values-ml_values[0]<thr
    #tl_values, tl_index = torch.sort(time_losses, dim=0)
    return ml_index[0,:]

def divide_prediction_window(seq_len, man_per_mode):
    num_window = man_per_mode-1
    window_length = int(seq_len/num_window)
    w_ind = np.zeros((num_window, 2), dtype= np.int32)
    for i in range(num_window-1):
        w_ind[i,0] = i*window_length
        w_ind[i,1] = (i+1)*window_length
    w_ind[num_window-1,0] = (num_window-1)*window_length
    w_ind[num_window-1,1] = seq_len
    return w_ind

def man_vector2man_n_timing(man_vector, man_per_mode, w_ind):
    batch_size = man_vector.shape[0]
    man_v_list = []
    for i in range(len(w_ind)):
        man_v_list.append(man_vector[:, w_ind[i,0]:w_ind[i,1]])
    mans = torch.zeros((batch_size, man_per_mode))
    times = torch.zeros((batch_size, man_per_mode-1))
    for i, man_v in enumerate(man_v_list):
        mans[:,i] = man_v[:,0]
        
        _, times[:,i] = torch.max(man_v!=man_v[:,0:1], dim =1)
        
    times[times==0] = -1 #no manouvre change
    mans[:,-1] = man_v_list[-1][:,-1]

    return mans, times

def man_n_timing2man_vector(mans, times, tgt_seq_len, w_ind):
    batch_size = mans.shape[0]
    man_per_mode = mans.shape[1]
    man_vector = torch.zeros((batch_size,tgt_seq_len))
    for i in range(man_per_mode-1):
        for batch_itr in range(batch_size):
            man_vector[batch_itr,w_ind[i,0]:times[i][batch_itr]] = mans[batch_itr,i]
            man_vector[batch_itr,times[i][batch_itr]:w_ind[i,1]] = mans[batch_itr,i+1]
            

    return man_vector