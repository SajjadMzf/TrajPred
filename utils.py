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
import params
from sklearn import metrics
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
font = {'size'   : 22}
matplotlib.rcParams['figure.figsize'] = (18, 12)
matplotlib.rc('font', **font)
def eval_top_func(p, model, lc_loss_func, ttlc_loss_func, task, te_dataset, device, model_tag = ''):
    model = model.to(device)
    
    te_loader = utils_data.DataLoader(dataset = te_dataset, shuffle = True, batch_size = p.BATCH_SIZE, drop_last= True, pin_memory= True, num_workers= 12)

    vis_data_path = p.VIS_DIR + p.SELECTED_DATASET + '_' + model_tag + '.pickle'
    best_model_path = p.MODELS_DIR + p.SELECTED_DATASET + '_' + model_tag + '.pt'
    figure_name =  p.SELECTED_DATASET + '_' + model_tag
    
    model.load_state_dict(torch.load(best_model_path))
    
    start = time()
    
    robust_test_pred_time, test_pred_time, test_acc, test_loss, test_lc_loss, test_ttlc_loss, auc, max_j, precision, recall, f1 = eval_model(p,model, lc_loss_func, ttlc_loss_func, task, te_loader, ' N/A', device, eval_type = 'Test', vis_data_path = vis_data_path, figure_name = figure_name)
    end = time()
    total_time = end-start
    #print("Test finished in:", total_time, "sec.")
    #print("Final Test accuracy:",te_acc)
    result_dic = {
        'Test Acc': test_acc,
        'Test Robust Pred Time': robust_test_pred_time,
        'Test Pred Time': test_pred_time,
        'Test Total Loss': test_loss,
        'Test Classification Loss': test_lc_loss,
        'Test Regression Loss': test_ttlc_loss,
        'AUC': auc,
        'Max Youden Index': max_j,
        'Precision': precision,
        'Recall': recall,
        'F1':f1
    }
    return result_dic


def train_top_func(p, model, optimizer, lc_loss_func, ttlc_loss_func, task, curriculum, tr_dataset, val_dataset, device, model_tag = ''):
    
    model = model.to(device)
    
    tr_loader = utils_data.DataLoader(dataset = tr_dataset, shuffle = True, batch_size = p.BATCH_SIZE, drop_last= True, pin_memory= True, num_workers= 12)
    val_loader = utils_data.DataLoader(dataset = val_dataset, shuffle = True, batch_size = p.BATCH_SIZE, drop_last= True, pin_memory= True, num_workers= 12)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, p.LR_DECAY_EPOCH, p.LR_DECAY)
    
    best_model_path = p.MODELS_DIR + p.SELECTED_DATASET + '_' + model_tag + '.pt'

    best_val_acc = 0
    best_val_loss = float("inf")
    patience = p.PATIENCE
    best_val_pred_time = 0
    best_epoch = 0
    total_time = 0
    curriculum_flag = curriculum['loss'] or curriculum['seq'] or curriculum['virtual']
    for epoch in range(p.NUM_EPOCHS):
        #print("Epoch: {} Started!".format(epoch+1))
        start = time()
        train_model(p, model, optimizer, scheduler, tr_loader, lc_loss_func, ttlc_loss_func, task, curriculum,  epoch+1, device, calc_train_acc= False)
        val_start = time()
        val_avg_pred_time,_,val_acc,val_loss, val_lc_loss, val_ttlc_loss, auc, max_j, precision, recall, f1= eval_model(p, model, lc_loss_func, ttlc_loss_func, task, val_loader, epoch+1, device, eval_type = 'Validation')
        val_end = time()
        print('val_time:', val_end-val_start)
        #print("Validation Accuracy:",val_acc,' Avg Pred Time: ', val_avg_pred_time, " Avg Loss: ", val_loss," at Epoch", epoch+1)
        if epoch<p.CL_EPOCH and curriculum_flag:
            print("No Early Stopping in CL Epochs.")
            continue
        if val_loss<best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_pred_time = val_avg_pred_time
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            patience = p.PATIENCE
        else:
            patience -= 1
        end = time()
        total_time += end-start
        print("Validation Accuracy in best epoch:",best_val_acc,' Robust Pred Time: ', best_val_pred_time, " Avg Loss: ", best_val_loss," at Epoch", best_epoch+1)
        print("Epoch: {} finished in {} sec\n".format(epoch+1, end-start))
        
        if patience == 0:
            print(' No performance improvement in Validation data after:', epoch+1, 'Epochs!')
            break
        
        

    result_dic = {
        'EarlyStopping Epoch': best_epoch + 1,
        'Best Validaction Acc': best_val_acc,
        'Best Validation Pred Time': best_val_pred_time,
        'Best Validation Loss': best_val_loss,
    }
    return result_dic


def train_model(p, model, optimizer, scheduler, train_loader, lc_loss_func, ttlc_loss_func, task, curriculum, epoch, device, vis_step = 20, calc_train_acc = True):
    # Number of samples with correct classification
    # total size of train data
    total = len(train_loader.dataset)
    # number of batch
    num_batch = int(np.floor(total/model.batch_size))
    model_time = 0
    avg_loss = 0
    all_start = time()
    if curriculum['loss']:
        loss_ratio = p.LOSS_RATIO_CL[epoch-1]
    else:
        loss_ratio = p.LOSS_RATIO_NoCL
    
    if curriculum['seq']:
        start_seq = int(p.START_SEQ_CL[epoch-1])
        end_seq = int(p.END_SEQ_CL[epoch-1])
    else:
        start_seq = 0
        end_seq = p.SEQ_LEN-p.IN_SEQ_LEN+1
    
    # Training loop over batches of data on train dataset
    for batch_idx, (data_tuple, labels,_, ttlc_status) in enumerate(train_loader):
        #print('Batch: ', batch_idx)
        start = time()
        
            
        data_tuple = [data.to(device) for data in data_tuple]
        labels = labels.to(device)
        ttlc_status = ttlc_status.to(device)
    
        #start_point = random.randint(0,p.TR_JUMP_STEP)
        for seq_itr  in range(start_seq,end_seq, p.TR_JUMP_STEP): 
            current_data = [data[:, seq_itr:(seq_itr+p.IN_SEQ_LEN)] for data in data_tuple]
            # 1. Clearing previous gradient values.
            optimizer.zero_grad()
            if model.__class__.__name__ == 'VanillaLSTM':
                model.init_hidden()
            # 2. feeding data to model (forward method will be computed)
            output_dict = model(current_data)
            lc_pred = output_dict['lc_pred']
            ttlc_pred = output_dict['ttlc_pred']

            # 3. Calculating the loss value
            if task == params.CLASSIFICATION or task == params.DUAL:
                lc_loss = lc_loss_func(lc_pred, labels)
            else:
                lc_loss = 0
            if task == params.REGRESSION or task == params.DUAL:
                ttlc_label = torch.FloatTensor([(p.SEQ_LEN-seq_itr-p.IN_SEQ_LEN+1)/p.FPS]).unsqueeze(0).expand(*ttlc_pred.size()).requires_grad_().to(device)
                ttlc_notavailable = (ttlc_status == 0).to(torch.float).unsqueeze(-1) #prev verion =>label ==0
                ttlc_available = (ttlc_status == 1).to(torch.float).unsqueeze(-1)
                ttlc_pred = ttlc_pred * ttlc_available + ttlc_label * ttlc_notavailable
                ttlc_loss = ttlc_loss_func(ttlc_pred, ttlc_label)
            else:
                ttlc_loss = 0
            
            if task == params.DUAL:
                loss = lc_loss + loss_ratio*ttlc_loss
            else:
                loss = lc_loss + ttlc_loss
            # 4. Calculating new grdients given the loss value
            loss.backward()
            # 5. Updating the weights
            optimizer.step()
        
            
            avg_loss += loss.data/(len(train_loader))
        #if (batch_idx+1) % 100 == 0:
        #    print('Epoch: ',epoch, ' Batch: ', batch_idx+1, ' Training Loss: ', avg_loss.cpu().numpy())
        #    avg_loss = 0
        end = time()
        model_time += end-start
    all_end = time()
    all_time = all_end - all_start
    print('model time: ', model_time, 'all training time: ', all_time, 'average training loss', avg_loss)
    scheduler.step()
    all_preds = np.zeros(((num_batch*model.batch_size), p.SEQ_LEN-p.IN_SEQ_LEN+1, 3))
    all_labels = np.zeros((num_batch*model.batch_size))
    # Validation Phase on train dataset
    if calc_train_acc == True:
        raise('Depricated')
        

def eval_model(p, model, lc_loss_func, ttlc_loss_func, task, test_loader, epoch, device, eval_type = 'Validation', vis_data_path = None, figure_name = None):
    total = len(test_loader.dataset)
    # number of batch
    num_batch = int(np.floor(total/model.batch_size))
    avg_loss = 0
    avg_lc_loss = 0
    avg_ttlc_loss = 0
    all_lc_preds = np.zeros(((num_batch*model.batch_size), p.SEQ_LEN-p.IN_SEQ_LEN+1,3))
    all_att_coef = np.zeros(((num_batch*model.batch_size), p.SEQ_LEN-p.IN_SEQ_LEN+1,4))
    all_ttlc_preds = np.zeros(((num_batch*model.batch_size), p.SEQ_LEN-p.IN_SEQ_LEN+1,1))
    all_labels = np.zeros(((num_batch*model.batch_size)))
    plot_dicts = []
    
    time_counter = 0
    average_time = 0
    gf_time = 0
    nn_time = 0
    loss_ratio = 1
    
    for batch_idx, (data_tuple, labels, plot_info, ttlc_status) in enumerate(test_loader):
        #print(batch_idx, total)
        all_labels[(batch_idx*model.batch_size):((batch_idx+1)*model.batch_size)] = labels.data
        if eval_type == 'Test':
            plot_dict = {
                'tv': plot_info[0].numpy(),
                'frames': plot_info[1].numpy(),
                'preds':np.zeros((plot_info[1].shape[0], plot_info[1].shape[1], 3)),
                'ttlc_preds': np.zeros((plot_info[1].shape[0], plot_info[1].shape[1])),
                'att_coef': np.zeros((plot_info[1].shape[0], plot_info[1].shape[1], 4)),
                'att_mask': np.zeros((plot_info[1].shape[0], plot_info[1].shape[1], 11, 26)),
                'labels':labels.numpy(),
                'data_file': plot_info[2]
            }
        
        data_tuple = [data.to(device) for data in data_tuple]
        labels = labels.to(device)
        #ttlc_status = ttlc_status.to(device)
        for seq_itr  in range(0,p.SEQ_LEN-p.IN_SEQ_LEN+1):
            if model.__class__.__name__ == 'VanillaLSTM':
                    model.init_hidden()
            current_data = [data[:, seq_itr:(seq_itr+p.IN_SEQ_LEN)] for data in data_tuple]
            st_time = time()
            output_dict = model(current_data)
            end_time = time()-st_time
            lc_pred = output_dict['lc_pred']
            ttlc_pred = output_dict['ttlc_pred']
            
            if task == params.CLASSIFICATION or task == params.DUAL:
                lc_loss = lc_loss_func(lc_pred, labels)
            else:
                lc_loss = 0
            
            #_ , pred_labels = output.data.max(dim=1)
            #pred_labels = pred_labels.cpu()
        
            if eval_type == 'Test':
                if task == params.CLASSIFICATION or task == params.DUAL:
                    plot_dict['preds'][:,p.IN_SEQ_LEN-1+seq_itr,:] = F.softmax(lc_pred, dim = -1).cpu().data
                if task == params.REGRESSION or task == params.DUAL:
                    plot_dict['ttlc_preds'][:,p.IN_SEQ_LEN-1+seq_itr] = np.squeeze(ttlc_pred.cpu().detach().numpy(), -1)
                if 'REGIONATT' in p.SELECTED_MODEL:
                    plot_dict['att_coef'][:,p.IN_SEQ_LEN-1+seq_itr,:] = output_dict['attention'].cpu().data
            
            if task == params.REGRESSION or task == params.DUAL:
                all_ttlc_preds[(batch_idx*model.batch_size):((batch_idx+1)*model.batch_size), seq_itr] = ttlc_pred.cpu().data
            if task == params.CLASSIFICATION or task == params.DUAL:
                all_lc_preds[(batch_idx*model.batch_size):((batch_idx+1)*model.batch_size), seq_itr] = F.softmax(lc_pred, dim = -1).cpu().data 
                avg_lc_loss = avg_lc_loss + lc_loss.cpu().data / (len(test_loader)*(p.SEQ_LEN-p.IN_SEQ_LEN))
            if p.SELECTED_MODEL == 'REGIONATTCNN3':
                all_att_coef[(batch_idx*model.batch_size):((batch_idx+1)*model.batch_size), seq_itr] = output_dict['attention'].cpu().data
        time_counter += 1
        average_time +=end_time
        if eval_type == 'Test':
            plot_dicts.append(plot_dict)
    
    #print('Average Time per whole sequence perbatch: {}'.format(average_time/time_counter))
    #print('gf time: {}, nn time: {}'.format(gf_time, nn_time))
    
    avg_ttlc_loss, robust_pred_time, pred_time, accuracy, precision, recall, f1, FPR, auc, max_j= calc_metric(p, task, all_lc_preds, all_ttlc_preds, all_att_coef, all_labels, epoch, eval_type = eval_type, figure_name = figure_name)
    avg_loss = avg_ttlc_loss + avg_lc_loss
    print("{}: Epoch: {}, Accuracy: {:.2f}%, Robust Prediction Time: {:.2f}, Prediction Time: {:.2f}, Total LOSS: {:.2f},LC LOSS: {:.2f},TTLC LOSS: {:.2f}, PRECISION:{}, RECALL:{}, F1:{}, FPR:{}, AUC:{}, Max J:{}".format(
        eval_type, epoch, 100. * accuracy, robust_pred_time, pred_time, avg_loss, avg_lc_loss, avg_ttlc_loss, precision, recall, f1, FPR, auc, max_j))
    
    if eval_type == 'Test':
        with open(vis_data_path, "wb") as fp:
            pickle.dump(plot_dicts, fp)
        
    return robust_pred_time, pred_time, accuracy, avg_loss, avg_lc_loss, avg_ttlc_loss, auc, max_j, precision, recall, f1



def calc_metric(p, task, all_lc_preds, all_ttlc_preds, all_att_coef, all_labels, epoch=None, eval_type = 'Test', figure_name= None):
   
    num_samples = all_labels.shape[0]
    prediction_seq = p.SEQ_LEN-p.IN_SEQ_LEN+1
    all_preds = np.argmax(all_lc_preds, axis =-1)
    
    if eval_type == 'Test':
        plot_att_graphs(p, all_att_coef, prediction_seq, all_labels, all_preds, figure_name)
    if task == params.CLASSIFICATION or task == params.DUAL:
        auc, max_j = calc_roc_n_prc(p, all_lc_preds, all_labels, all_ttlc_preds, prediction_seq, num_samples, figure_name, thr_type = 'thr', eval_type = eval_type)
        accuracy, precision, recall, f1, FPR, all_TPs = calc_classification_metrics(p, all_preds, all_labels, all_ttlc_preds, prediction_seq, num_samples, eval_type, figure_name)
        robust_pred_time, pred_time = calc_avg_pred_time(p, all_TPs, all_labels, prediction_seq, num_samples)
    else:
        (accuracy, precision, recall, f1, FPR, all_TPs, auc, max_j, robust_pred_time, pred_time) = (0,0,0,0,0,0,0,0,0,0)
        avg_pred_time = 0
    if task == params.REGRESSION or task == params.DUAL:
        avg_ttlc_loss = calc_regression_metrics(p, all_ttlc_preds, all_labels, all_preds, prediction_seq, num_samples, eval_type, figure_name)
    else:
        avg_ttlc_loss = 0

    return avg_ttlc_loss, robust_pred_time, pred_time, accuracy, precision, recall, f1, FPR, auc, max_j


def plot_att_graphs(p, all_att_coef, prediction_seq, all_labels, all_preds, figure_name):
    sum_att_coef = np.zeros((3,prediction_seq, 4))
    count_att_coef = np.zeros((3,prediction_seq))
    num_samples = all_att_coef.shape[0]
    for i in range(num_samples):
        for seq_itr in range(prediction_seq):
            sum_att_coef[all_preds[i,seq_itr],seq_itr,:] += all_att_coef[i, seq_itr, :]
            count_att_coef[all_preds[i,seq_itr],seq_itr] += 1
    
    for i in range(3):
        for seq_itr in range(prediction_seq):
            sum_att_coef[i,seq_itr] = sum_att_coef[i,seq_itr]/(count_att_coef[i,seq_itr]+ 0.000000001)
    
    ''''''
    names = ['LK', 'RLC', 'LLC']
    
    # Creating dirs
    figure_dir = os.path.join(p.FIGS_DIR, 'FR_att')
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)
    plot_dir = os.path.join(figure_dir, figure_name +p.unblanaced_ext+ '.png')
    fig_obs_dir = os.path.join(figure_dir, figure_name +p.unblanaced_ext+ '.pickle')

    att_ax = plt.figure()
    ttlc_seq = (prediction_seq-np.arange(prediction_seq))/p.FPS
    # Creating Figs
    plt.plot(ttlc_seq, sum_att_coef[0,:,0], label = names[0],  linewidth=5)
    plt.plot(ttlc_seq, sum_att_coef[1,:,0], label = names[1],  linewidth=5)
    plt.plot(ttlc_seq, sum_att_coef[2,:,0], label = names[2],  linewidth=5)
    plt.xlim(ttlc_seq[0], ttlc_seq[-1])
    att_ax.legend()
    plt.xlabel('TTLC (s)')
    plt.ylabel('Average Attention Coefficient')
    plt.grid()
    att_ax.savefig(plot_dir)
    with open(fig_obs_dir, 'wb') as fid:
        pickle.dump(att_ax, fid)
    

    # Creating dirs
    figure_dir = os.path.join(p.FIGS_DIR, 'BR_att')
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)
    plot_dir = os.path.join(figure_dir, figure_name +p.unblanaced_ext+ '.png')
    fig_obs_dir = os.path.join(figure_dir, figure_name +p.unblanaced_ext+ '.pickle')

    att_ax = plt.figure()
    ttlc_seq = (prediction_seq-np.arange(prediction_seq))/p.FPS
    
    # Creating Figs
    plt.plot(ttlc_seq, sum_att_coef[0,:,1], label = names[0],  linewidth=5)
    plt.plot(ttlc_seq, sum_att_coef[1,:,1], label = names[1],  linewidth=5)
    plt.plot(ttlc_seq, sum_att_coef[2,:,1], label = names[2],  linewidth=5)
    plt.xlim(ttlc_seq[0], ttlc_seq[-1])
    att_ax.legend()
    plt.xlabel('TTLC (s)')
    plt.ylabel('Average Attention Coefficient')
    plt.grid()
    att_ax.savefig(plot_dir)
    with open(fig_obs_dir, 'wb') as fid:
        pickle.dump(att_ax, fid)
    
    # Creating dirs
    figure_dir = os.path.join(p.FIGS_DIR, 'FL_att')
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)
    plot_dir = os.path.join(figure_dir, figure_name +p.unblanaced_ext+ '.png')
    fig_obs_dir = os.path.join(figure_dir, figure_name +p.unblanaced_ext+ '.pickle')

    att_ax = plt.figure()
    ttlc_seq = (prediction_seq-np.arange(prediction_seq))/p.FPS
    
    # Creating Figs
    plt.plot(ttlc_seq, sum_att_coef[0,:,2], label = names[0],  linewidth=5)
    plt.plot(ttlc_seq, sum_att_coef[1,:,2], label = names[1],  linewidth=5)
    plt.plot(ttlc_seq, sum_att_coef[2,:,2], label = names[2],  linewidth=5)
    plt.xlim(ttlc_seq[0], ttlc_seq[-1])
    att_ax.legend()
    plt.xlabel('TTLC (s)')
    plt.ylabel('Average Attention Coefficient')
    plt.grid()
    att_ax.savefig(plot_dir)
    with open(fig_obs_dir, 'wb') as fid:
        pickle.dump(att_ax, fid)

    # Creating dirs
    figure_dir = os.path.join(p.FIGS_DIR, 'BL_att')
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)
    plot_dir = os.path.join(figure_dir, figure_name +p.unblanaced_ext+ '.png')
    fig_obs_dir = os.path.join(figure_dir, figure_name +p.unblanaced_ext+ '.pickle')

    att_ax = plt.figure()
    ttlc_seq = (prediction_seq-np.arange(prediction_seq))/p.FPS
    
    # Creating Figs
    plt.plot(ttlc_seq, sum_att_coef[0,:,3], label = names[0],  linewidth=5)
    plt.plot(ttlc_seq, sum_att_coef[1,:,3], label = names[1],  linewidth=5)
    plt.plot(ttlc_seq, sum_att_coef[2,:,3], label = names[2],  linewidth=5)
    plt.xlim(ttlc_seq[0], ttlc_seq[-1])
    att_ax.legend()
    plt.xlabel('TTLC (s)')
    plt.ylabel('Average Attention Coefficient')
    plt.grid()
    att_ax.savefig(plot_dir)
    with open(fig_obs_dir, 'wb') as fid:
        pickle.dump(att_ax, fid)
    

def calc_roc_n_prc(p, all_lc_preds, all_labels, all_ttlc_preds, prediction_seq, num_samples, figure_name, thr_type, eval_type):
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
def calc_classification_metrics(p, all_preds, all_labels, all_ttlc_preds, prediction_seq, num_samples, eval_type, figure_name):
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
        FP_TTLC[prev_FP_index:cur_FP_index] = np.squeeze(all_ttlc_preds[FP_index[:,t],t], -1)

        recall_vs_TTLC[t] = TP_vs_TTLC[t]/(TP_vs_TTLC[t] + FN_vs_TTLC[t])
        precision_vs_SEQ[t] = TP_vs_TTLC[t]/(TP_vs_TTLC[t] + FP_vs_SEQ[t])
        FPR_vs_SEQ[t] = FP_vs_SEQ[t]/(FP_vs_SEQ[t] + TN_vs_SEQ[t])
        
        acc_vs_SEQ[t] = sum(all_hits[:,t])/num_samples
    
    FP_TTLC = FP_TTLC[:cur_FP_index]
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
        
        # Creating dirs
        figure_dir = os.path.join(p.FIGS_DIR, 'FP_TTLC')
        if not os.path.exists(figure_dir):
            os.mkdir(figure_dir)
        plot_dir = os.path.join(figure_dir, figure_name +p.unblanaced_ext+ '.png')
        fig_obs_dir = os.path.join(figure_dir, figure_name +p.unblanaced_ext+  '.pickle')
        fp_ax = plt.figure()
        '''
        ax = fp_ax.add_subplot(1, 1, 1)
        major_ticks = np.arange(0, 1, 0.1)
        minor_ticks = np.arange(0, 1, 0.02)

        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)

        # And a corresponding grid
        ax.grid(which='both')

        # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        '''
        ttlc_seq = (prediction_seq-np.arange(prediction_seq))/p.FPS

        # Creating Figs
        plt.plot(np.sort(FP_TTLC), np.linspace(0, 1, len(FP_TTLC), endpoint=False), linewidth=5)
        
        plt.xlabel('Predicted TTLC(s)')
        plt.ylabel('Cumulative False Positive')
        
        plt.grid()
        fp_ax.savefig(plot_dir)
        with open(fig_obs_dir, 'wb') as fid:
            pickle.dump(fp_ax, fid)

    

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

def calc_regression_metrics(p, all_ttlc_preds_orig, all_labels, all_preds, prediction_seq, num_samples, eval_type, figure_name):
    all_ttlc_preds = np.squeeze(all_ttlc_preds_orig[all_labels!=0], -1)
    
    mse = np.zeros((prediction_seq))
    for i in range(prediction_seq):
        gt_ttlc = (prediction_seq-i)/p.FPS
        mse[i] = ((all_ttlc_preds[:,i]- gt_ttlc)**2).mean()
    avg_ttlc_loss = np.mean(mse)
    rmse = np.sqrt(mse)
    
    
    

    if eval_type == 'Test':

        '''RMSE PLOT'''
        # Creating dirs
        figure_dir = os.path.join(p.FIGS_DIR, 'rmse_vs_TTLC')
        if not os.path.exists(figure_dir):
            os.mkdir(figure_dir)
        plot_dir = os.path.join(figure_dir, figure_name +p.unblanaced_ext+ '.png')
        fig_obs_dir = os.path.join(figure_dir, figure_name +p.unblanaced_ext+ '.pickle')

        recall_ax = plt.figure()
        ttlc_seq = (prediction_seq-np.arange(prediction_seq))/p.FPS

        # Creating Figs
        plt.plot(ttlc_seq, rmse)
        plt.xlim(ttlc_seq[0], ttlc_seq[-1])
        plt.xlabel('TTLC (s)')
        plt.ylabel('RMSE(s)')
        plt.grid()
        recall_ax.savefig(plot_dir)
        with open(fig_obs_dir, 'wb') as fid:
            pickle.dump(recall_ax, fid)
        '''BOX PLOT'''
        
        box_plot_data = np.transpose(all_ttlc_preds, (1, 0))
        ttlc_seq = (prediction_seq-np.arange(prediction_seq))/p.FPS
        ttlc_seq_mat = np.tile(ttlc_seq, ( box_plot_data.shape[1], 1))
        ttlc_seq_mat = np.transpose(ttlc_seq_mat, (1,0))
        box_plot_data = box_plot_data - ttlc_seq_mat
        print('box plot size :{}'.format(box_plot_data.shape))
        box_plot_data = list(box_plot_data)
        # Creating dirs
        figure_dir = os.path.join(p.FIGS_DIR, 'box_plot_regression')
        if not os.path.exists(figure_dir):
            os.mkdir(figure_dir)
        plot_dir = os.path.join(figure_dir, figure_name +p.unblanaced_ext+  '.pdf')
        fig_obs_dir = os.path.join(figure_dir, figure_name +p.unblanaced_ext+  '.pickle')
        with PdfPages(plot_dir) as export_pdf:
            recall_ax = plt.figure()
            ttlc_seq = (prediction_seq-np.arange(prediction_seq))/p.FPS

            # Creating Figs
            plt.boxplot(box_plot_data, labels = ttlc_seq, showfliers=False, widths = 0.3, patch_artist=True)
            #plt.plot(ttlc_seq,ttlc_seq)
            plt.xlabel('Actual TTLC (s)')
            plt.ylabel('TTLC Error(s)')
            plt.xticks(rotation=90)
            plt.grid()
            plt.tight_layout()
            #plt.show()
            export_pdf.savefig()
            with open(fig_obs_dir, 'wb') as fid:
                pickle.dump(recall_ax, fid)
        

    return avg_ttlc_loss

def update_tag(model_dict):
    hyperparam_str = ''
    for hyperparam in model_dict['hyperparams']:
        abbr = hyperparam.split()
        abbr = ''.join([word[0] for word in abbr])
        
        hyperparam_str += abbr + str(model_dict['hyperparams'][hyperparam])
    hyperparam_str += model_dict['state type']
    
    return model_dict['name'] + hyperparam_str