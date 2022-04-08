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
import pandas as pd
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
font = {'size'   : 22}
matplotlib.rcParams['figure.figsize'] = (18, 12)
matplotlib.rc('font', **font)
def eval_top_func(p, model, lc_loss_func, task, te_dataset, device, model_tag = ''):
    model = model.to(device)
    
    te_loader = utils_data.DataLoader(dataset = te_dataset, shuffle = True, batch_size = p.BATCH_SIZE, drop_last= True, pin_memory= True, num_workers= 12)

    vis_data_path = p.VIS_DIR + p.SELECTED_DATASET + '_' + model_tag + '.pickle'
    best_model_path = p.MODELS_DIR + p.SELECTED_DATASET + '_' + model_tag + '.pt'
    figure_name =  p.SELECTED_DATASET + '_' + model_tag
    
    if p.SELECTED_MODEL != 'CONSTANT_PARAMETER':
        model.load_state_dict(torch.load(best_model_path))
    
    start = time()
    
    robust_test_pred_time, test_pred_time, test_acc, test_loss, test_lc_loss, auc, max_j, precision, recall, f1, rmse, fde, traj_df = eval_model(p,model, lc_loss_func, task, te_loader, te_dataset, ' N/A', device, eval_type = 'Test', vis_data_path = vis_data_path, figure_name = figure_name)
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
        'AUC': auc,
        'Max Youden Index': max_j,
        'Precision': precision,
        'Recall': recall,
        'F1':f1,
        'rmse':rmse,
        'fde': fde,
    }
    return result_dic, traj_df


def train_top_func(p, model, optimizer, lc_loss_func, task, tr_dataset, val_dataset, device, model_tag = ''):
    
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
    for epoch in range(p.NUM_EPOCHS):
        #print("Epoch: {} Started!".format(epoch+1))
        start = time()
        train_model(p, model, optimizer, scheduler, tr_loader, lc_loss_func, task,  epoch+1, device, calc_train_acc= False)
        val_start = time()
        val_avg_pred_time,_,val_acc,val_loss, val_lc_loss, auc, max_j, precision, recall, f1, rmse, fde, traj_df = eval_model(p, model, lc_loss_func, task, val_loader, val_dataset, epoch+1, device, eval_type = 'Validation')
        val_end = time()
        print('val_time:', val_end-val_start)
        #print("Validation Accuracy:",val_acc,' Avg Pred Time: ', val_avg_pred_time, " Avg Loss: ", val_loss," at Epoch", epoch+1)
        if epoch<p.CL_EPOCH :
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


def train_model(p, model, optimizer, scheduler, train_loader, lc_loss_func, task, epoch, device, vis_step = 20, calc_train_acc = True):
    # Number of samples with correct classification
    # total size of train data
    total = len(train_loader.dataset)
    # number of batch
    num_batch = int(np.floor(total/model.batch_size))
    model_time = 0
    avg_loss = 0
    avg_lc_loss = 0
    avg_traj_loss = 0
    all_start = time()
    
    loss_ratio = p.LOSS_RATIO_NoCL
    
    
    start_seq = 0
    end_seq = p.SEQ_LEN-p.IN_SEQ_LEN+1
    
    # Training loop over batches of data on train dataset
    for batch_idx, (data_tuple, labels,_, _) in enumerate(train_loader):
        #print('Batch: ', batch_idx)
        start = time()
        
            
        data_tuple = [data.to(device) for data in data_tuple]
        labels = labels.to(device)
    
        #start_point = random.randint(0,p.TR_JUMP_STEP)
        for seq_itr  in range(start_seq,end_seq, p.TR_JUMP_STEP): 
            
            if task == params.TRAJECTORYPRED:
                target_data = data_tuple[-1]
                target_data_in = target_data[:,(seq_itr+p.IN_SEQ_LEN-1):(seq_itr+p.IN_SEQ_LEN-1+p.TGT_SEQ_LEN)]
                target_data_out = target_data[:,(seq_itr+p.IN_SEQ_LEN):(seq_itr+p.IN_SEQ_LEN+p.TGT_SEQ_LEN)]
                in_data_tuple = data_tuple[:-1]
            else:
                in_data_tuple = data_tuple
            current_data = [data[:, seq_itr:(seq_itr+p.IN_SEQ_LEN)] for data in in_data_tuple]
            #print(current_data)
            # 1. Clearing previous gradient values.
            optimizer.zero_grad()
            if model.__class__.__name__ == 'VanillaLSTM':
                model.init_hidden()
            # 2. feeding data to model (forward method will be computed)
            if task == params.TRAJECTORYPRED and p.SELECTED_MODEL== 'TRANSFORMER_TRAJ':
                output_dict = model(x = current_data, y =target_data_in, y_mask = model.get_y_mask(p.TGT_SEQ_LEN).to(device))
                traj_pred = output_dict['traj_pred']
            else:
                output_dict = model(current_data)
            lc_pred = output_dict['lc_pred']
            #print(lc_pred)
            #print(traj_pred)
            #exit()

            # 3. Calculating the loss value
            if task == params.CLASSIFICATION or task == params.DUAL or task == params.TRAJECTORYPRED:
                lc_loss = lc_loss_func(lc_pred, labels)

            else:
                lc_loss = 0
            
            if task == params.TRAJECTORYPRED:
                traj_loss_func = nn.MSELoss()
                traj_loss = traj_loss_func(traj_pred, target_data_out)
            else:
                traj_loss = 0
            loss = lc_loss + traj_loss 
            # 4. Calculating new grdients given the loss value
            loss.backward()
            # 5. Updating the weights
            optimizer.step()
        

            avg_traj_loss += traj_loss.data/(len(train_loader))
            avg_lc_loss += lc_loss.data/(len(train_loader))
            avg_loss += loss.data/(len(train_loader))
        #if (batch_idx+1) % 100 == 0:
        #    print('Epoch: ',epoch, ' Batch: ', batch_idx+1, ' Training Loss: ', avg_loss.cpu().numpy())
        #    avg_loss = 0
        end = time()
        model_time += end-start
    all_end = time()
    all_time = all_end - all_start
    #print('model time: ', model_time, 'all training time: ', all_time, 'average training lc loss', avg_loss, 'average training')
    print('Total Training loss: {}, Training LC Loss: {}, Training Traj Loss: {}'.format(avg_loss, avg_lc_loss, avg_traj_loss))
    scheduler.step()
    all_preds = np.zeros(((num_batch*model.batch_size), p.SEQ_LEN-p.IN_SEQ_LEN+1, 3))
    all_labels = np.zeros((num_batch*model.batch_size))
    # Validation Phase on train dataset
    if calc_train_acc == True:
        raise('Depricated')
        

def eval_model(p, model, lc_loss_func, task, test_loader, test_dataset, epoch, device, eval_type = 'Validation', vis_data_path = None, figure_name = None):
    total = len(test_loader.dataset)
    # number of batch
    num_batch = int(np.floor(total/model.batch_size))
    avg_loss = 0
    avg_lc_loss = 0
    avg_traj_loss = 0
    all_lc_preds = np.zeros(((num_batch*model.batch_size), p.SEQ_LEN-p.IN_SEQ_LEN+1,3))
    all_traj_preds = np.zeros(((num_batch*model.batch_size), p.SEQ_LEN-p.IN_SEQ_LEN+1,p.TGT_SEQ_LEN,2))
    all_traj_labels = np.zeros(((num_batch*model.batch_size), p.SEQ_LEN-p.IN_SEQ_LEN+1,p.TGT_SEQ_LEN,2))
    all_att_coef = np.zeros(((num_batch*model.batch_size), p.SEQ_LEN-p.IN_SEQ_LEN+1,4))
    all_labels = np.zeros(((num_batch*model.batch_size)))
    plot_dicts = []
    
    time_counter = 0
    average_time = 0
    gf_time = 0
    nn_time = 0
    loss_ratio = 1
    
    for batch_idx, (data_tuple, labels, plot_info, _) in enumerate(test_loader):
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
                'traj_labels': np.zeros((plot_info[1].shape[0], plot_info[1].shape[1], p.TGT_SEQ_LEN, 2)),
                'traj_preds': np.zeros((plot_info[1].shape[0], plot_info[1].shape[1], p.TGT_SEQ_LEN, 2)),
                'labels':labels.numpy(),
                'data_file': plot_info[2]
            }
        
        data_tuple = [data.to(device) for data in data_tuple]
        labels = labels.to(device)
        for seq_itr  in range(0,p.SEQ_LEN-p.IN_SEQ_LEN+1):
            if model.__class__.__name__ == 'VanillaLSTM':
                    model.init_hidden()
            
            if task == params.TRAJECTORYPRED:
                target_data = data_tuple[-1]
                target_data_in = target_data[:,(seq_itr+p.IN_SEQ_LEN-1):(seq_itr+p.IN_SEQ_LEN-1+p.TGT_SEQ_LEN)]
                target_data_out = target_data[:,(seq_itr+p.IN_SEQ_LEN):(seq_itr+p.IN_SEQ_LEN+p.TGT_SEQ_LEN)]
                in_data_tuple = data_tuple[:-1]
            else:
                in_data_tuple = data_tuple
            current_data = [data[:, seq_itr:(seq_itr+p.IN_SEQ_LEN)] for data in in_data_tuple]
            st_time = time()
            
            if task == params.TRAJECTORYPRED and p.SELECTED_MODEL== 'TRANSFORMER_TRAJ':
                target_data_in = target_data[:,(seq_itr+p.IN_SEQ_LEN-1):(seq_itr+p.IN_SEQ_LEN)]   
                for out_seq_itr in range(p.TGT_SEQ_LEN):
                    output_dict = model(x = current_data, y =target_data_in, y_mask = model.get_y_mask(target_data_in.size(1)).to(device))
                    traj_pred = output_dict['traj_pred']
                    traj_pred = traj_pred[:,out_seq_itr:(out_seq_itr+1)]
                    target_data_in = torch.cat((target_data_in, traj_pred), dim = 1)
                traj_pred = target_data_in[:,1:]    
            elif task == params.TRAJECTORYPRED and p.SELECTED_MODEL== 'CONSTANT_PARAMETER':
                output_dict = model(current_data, test_dataset.states_min, test_dataset.states_max, test_dataset.output_states_min, test_dataset.output_states_max, target_data_in)
                traj_pred = output_dict['traj_pred']
            else:
                output_dict = model(current_data)
            
            end_time = time()-st_time
            lc_pred = output_dict['lc_pred']
            #print(lc_pred)
            #exit()
            if task == params.CLASSIFICATION or task == params.DUAL or task == params.TRAJECTORYPRED:
                lc_loss = lc_loss_func(lc_pred, labels)
            else:
                lc_loss = 0

            if task == params.TRAJECTORYPRED:
                traj_loss_func = nn.MSELoss()
                traj_loss = traj_loss_func(traj_pred, target_data_out) #todo make it selectable in models dict
            else:
                traj_loss = 0
            loss = lc_loss + traj_loss 

            #_ , pred_labels = output.data.max(dim=1)
            #pred_labels = pred_labels.cpu()
        
            if eval_type == 'Test':
                if task == params.CLASSIFICATION or task == params.DUAL or task == params.TRAJECTORYPRED:
                    plot_dict['preds'][:,p.IN_SEQ_LEN-1+seq_itr,:] = F.softmax(lc_pred, dim = -1).cpu().data
                if 'REGIONATT' in p.SELECTED_MODEL:
                    plot_dict['att_coef'][:,p.IN_SEQ_LEN-1+seq_itr,:] = output_dict['attention'].cpu().data
                if task == params.TRAJECTORYPRED:
                    traj_min = test_dataset.output_states_min
                    traj_max = test_dataset.output_states_max
                    unnormalised_traj_pred = traj_pred.cpu().data*(traj_max-traj_min) + traj_min
                    unnormalised_traj_label = target_data_out.cpu().data*(traj_max-traj_min) + traj_min
                    plot_dict['traj_labels'][:,p.IN_SEQ_LEN-1+seq_itr,:,:] = unnormalised_traj_label
                    plot_dict['traj_preds'][:,p.IN_SEQ_LEN-1+seq_itr,:,:] = unnormalised_traj_pred
                    
            
            if task == params.CLASSIFICATION or task == params.DUAL or task == params.TRAJECTORYPRED:
                all_lc_preds[(batch_idx*model.batch_size):((batch_idx+1)*model.batch_size), seq_itr] = F.softmax(lc_pred, dim = -1).cpu().data 
                avg_lc_loss = avg_lc_loss + lc_loss.cpu().data / (len(test_loader)*(p.SEQ_LEN-p.IN_SEQ_LEN))
            if task == params.TRAJECTORYPRED:
                all_traj_preds[(batch_idx*model.batch_size):((batch_idx+1)*model.batch_size), seq_itr,:,:] = traj_pred.cpu().data
                all_traj_labels[(batch_idx*model.batch_size):((batch_idx+1)*model.batch_size), seq_itr,:,:] = target_data_out.cpu().data
            if p.SELECTED_MODEL == 'REGIONATTCNN3':
                all_att_coef[(batch_idx*model.batch_size):((batch_idx+1)*model.batch_size), seq_itr] = output_dict['attention'].cpu().data
            avg_traj_loss += traj_loss.cpu().data / (len(test_loader)*(p.SEQ_LEN-p.IN_SEQ_LEN))
            avg_loss =  avg_loss + loss.cpu().data / (len(test_loader)*(p.SEQ_LEN-p.IN_SEQ_LEN))
        time_counter += 1
        average_time +=end_time
        if eval_type == 'Test':
            plot_dicts.append(plot_dict)
    
    #print('Average Time per whole sequence perbatch: {}'.format(average_time/time_counter))
    #print('gf time: {}, nn time: {}'.format(gf_time, nn_time))
    
    robust_pred_time, pred_time, accuracy, precision, recall, f1, FPR, auc, max_j, rmse, fde, traj_df = calc_metric(p, task, all_lc_preds, all_att_coef, all_labels, all_traj_preds, all_traj_labels, test_dataset.output_states_min, test_dataset.output_states_max, epoch, eval_type = eval_type, figure_name = figure_name)
    avg_loss = avg_lc_loss
    print('                                   ')
    print("{}: Epoch: {}, Accuracy: {:.2f}%, Robust Prediction Time: {:.2f}, Prediction Time: {:.2f}, Total LOSS: {:.2f},LC LOSS: {:.2f}, PRECISION:{}, RECALL:{}, F1:{}, FPR:{}, AUC:{}, Max J:{}, RMSE:{}, FDE:{}".format(
        eval_type, epoch, 100. * accuracy, robust_pred_time, pred_time, avg_loss, avg_lc_loss, precision, recall, f1, FPR, auc, max_j, rmse, fde))
    print('                                   ')
    print('-----------------------------------')
    print('                                   ')
    print(traj_df)
    
    if eval_type == 'Test':
        with open(vis_data_path, "wb") as fp:
            pickle.dump(plot_dicts, fp)
        
    return robust_pred_time, pred_time, accuracy, avg_loss, avg_lc_loss, auc, max_j, precision, recall, f1, rmse, fde, traj_df



def calc_metric(p, task, all_lc_preds, all_att_coef, all_labels, all_traj_preds, all_traj_labels, traj_label_min, traj_label_max, epoch=None, eval_type = 'Test', figure_name= None):
   
    num_samples = all_labels.shape[0]
    prediction_seq = p.SEQ_LEN-p.IN_SEQ_LEN+1
    all_preds = np.argmax(all_lc_preds, axis =-1)
    
    if eval_type == 'Test':
        plot_att_graphs(p, all_att_coef, prediction_seq, all_labels, all_preds, figure_name)
    if task == params.CLASSIFICATION or task == params.DUAL or task == params.TRAJECTORYPRED:
        auc, max_j = calc_roc_n_prc(p, all_lc_preds, all_labels,  prediction_seq, num_samples, figure_name, thr_type = 'thr', eval_type = eval_type)
        accuracy, precision, recall, f1, FPR, all_TPs = calc_classification_metrics(p, all_preds, all_labels, prediction_seq, num_samples, eval_type, figure_name)
        robust_pred_time, pred_time = calc_avg_pred_time(p, all_TPs, all_labels, prediction_seq, num_samples)
    else:
        (accuracy, precision, recall, f1, FPR, all_TPs, auc, max_j, robust_pred_time, pred_time) = (0,0,0,0,0,0,0,0,0,0)
        avg_pred_time = 0
    
    if task == params.TRAJECTORYPRED:
        rmse, fde, traj_df = calc_traj_metrics(p, all_traj_preds, all_traj_labels, traj_label_min, traj_label_max)
    else:
        rmse = 0
        fde = 0

    return robust_pred_time, pred_time, accuracy, precision, recall, f1, FPR, auc, max_j, rmse, fde, traj_df


def calc_traj_metrics(p, traj_preds, traj_labels, traj_min, traj_max):
    #traj_preds [number of samples, sequence of prediction, target sequence length, number of output states]
    #1. denormalise
    traj_preds = traj_preds*(traj_max-traj_min) + traj_min
    traj_labels = traj_labels*(traj_max-traj_min) + traj_min
    #2. from diff to actual
    traj_preds = np.cumsum(traj_preds, axis = 2)
    traj_labels = np.cumsum(traj_labels, axis = 2)
    # 3. fde
    fde = np.mean(np.absolute(traj_preds[:,:,-1,:]-traj_labels[:,:,-1,:]))
    # 4. rmse
    rmse = np.sqrt(((traj_preds-traj_labels)**2).mean())
    # 5. fde, rmse table
    prediction_ts = int(p.TGT_SEQ_LEN/p.FPS)
    if (p.TGT_SEQ_LEN/p.FPS) % 1 != 0:
        raise(ValueError('Target sequence length not dividable by FPS'))
    columns = ['<{} sec'.format(ts+1) for ts in range(prediction_ts)]
    index = ['FDE_lat', 'FDE_long', 'FDE', 'RMSE_lat', 'RMSE_long', 'RMSE']
    data = np.zeros((6, prediction_ts))
    for ts in range(prediction_ts):
        ts_index = (ts+1)*p.FPS
        #fde
        data[0,ts] = np.mean(np.absolute(traj_preds[:,:,ts_index-1,0]-traj_labels[:,:,ts_index-1,0])) # 0 is laterel, 1 is longitudinal
        data[1,ts] = np.mean(np.absolute(traj_preds[:,:,ts_index-1,1]-traj_labels[:,:,ts_index-1,1])) # 0 is laterel, 1 is longitudinal
        data[2,ts] = np.mean(np.absolute(traj_preds[:,:,ts_index-1,:]-traj_labels[:,:,ts_index-1,:])) # 0 is laterel, 1 is longitudinal
        #rmse
        data[3,ts] = np.sqrt(((traj_preds[:,:,:ts_index,0]-traj_labels[:,:,:ts_index,0])**2).mean())
        data[4,ts] = np.sqrt(((traj_preds[:,:,:ts_index,1]-traj_labels[:,:,:ts_index,1])**2).mean())
        data[5,ts] = np.sqrt(((traj_preds[:,:,:ts_index,:]-traj_labels[:,:,:ts_index,:])**2).mean())

    result_df = pd.DataFrame(data=  data, columns = columns, index= index)
    
    return rmse, fde, result_df

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
    

def calc_roc_n_prc(p, all_lc_preds, all_labels, prediction_seq, num_samples, figure_name, thr_type, eval_type):
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

def update_tag(model_dict):
    hyperparam_str = ''
    for hyperparam in model_dict['hyperparams']:
        abbr = hyperparam.split()
        abbr = ''.join([word[0] for word in abbr])
        
        hyperparam_str += abbr + str(model_dict['hyperparams'][hyperparam])
    hyperparam_str += model_dict['state type']
    
    return model_dict['name'] + hyperparam_str