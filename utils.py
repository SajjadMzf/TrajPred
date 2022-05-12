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
from debugging_utils import *
font = {'size'   : 22}
matplotlib.rcParams['figure.figsize'] = (18, 12)
matplotlib.rc('font', **font)
def eval_top_func(p, model, lc_loss_func, traj_loss_func, task, te_dataset, device, model_tag = '', tensorboard = None):
    model = model.to(device)
    
    te_loader = utils_data.DataLoader(dataset = te_dataset, shuffle = True, batch_size = p.BATCH_SIZE, drop_last= True, pin_memory= True, num_workers= 12)

    vis_data_path = p.VIS_DIR + p.SELECTED_DATASET + '_' + model_tag + '.pickle'
    best_model_path = p.MODELS_DIR + p.SELECTED_DATASET + '_' + model_tag + '.pt'
    figure_name =  p.SELECTED_DATASET + '_' + model_tag
    
    if p.SELECTED_MODEL != 'CONSTANT_PARAMETER':
        model.load_state_dict(torch.load(best_model_path))
    
    start = time()
    
    test_acc, test_loss, test_lc_loss, test_traj_loss, auc, max_j, precision, recall, f1, rmse, fde, traj_df = eval_model(p,model, lc_loss_func, traj_loss_func, task, te_loader, te_dataset, ' N/A', device, eval_type = 'Test', vis_data_path = vis_data_path, figure_name = figure_name)
    end = time()
    total_time = end-start
    #print("Test finished in:", total_time, "sec.")
    #print("Final Test accuracy:",te_acc)
    result_dic = {
        'Test Acc': test_acc,
        'Test Total Loss': test_loss,
        'Test Classification Loss': test_lc_loss,
        'AUC': auc,
        'Max Youden Index': max_j,
        'Precision': precision,
        'Recall': recall,
        'F1':f1,
        'rmse':rmse[0],
        'fde': fde,
    }
    return result_dic, traj_df


def train_top_func(p, model, optimizer, lc_loss_func, traj_loss_func, task, tr_dataset, val_dataset, device, model_tag = '', tensorboard = None):
    
    model = model.to(device)
    
    tr_loader = utils_data.DataLoader(dataset = tr_dataset, shuffle = True, batch_size = p.BATCH_SIZE, drop_last= True, pin_memory= True, num_workers= 12)
    val_loader = utils_data.DataLoader(dataset = val_dataset, shuffle = True, batch_size = p.BATCH_SIZE, drop_last= True, pin_memory= True, num_workers= 12)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, p.LR_DECAY_EPOCH, p.LR_DECAY)
    
    best_model_path = p.MODELS_DIR + p.SELECTED_DATASET + '_' + model_tag + '.pt'

    best_val_acc = 0
    best_val_loss = float("inf")
    patience = p.PATIENCE
    best_epoch = 0
    total_time = 0
    for epoch in range(p.NUM_EPOCHS):
        #print("Epoch: {} Started!".format(epoch+1))
        start = time()
        tr_loss, tr_lc_loss, tr_traj_loss = train_model(p, model, optimizer, scheduler, tr_loader, lc_loss_func, traj_loss_func, task,  epoch+1, device, calc_train_acc= False)
        val_start = time()

        val_acc,val_loss, val_lc_loss, val_traj_loss, auc, max_j, precision, recall, f1, rmse, fde, traj_df = eval_model(p, model, lc_loss_func, traj_loss_func, task, val_loader, val_dataset, epoch+1, device, eval_type = 'Validation')
        val_end = time()
        print('val_time:', val_end-val_start)
        #print("Validation Accuracy:",val_acc,' Avg Pred Time: ', val_avg_pred_time, " Avg Loss: ", val_loss," at Epoch", epoch+1)
        if tensorboard != None:
            
            tensorboard.add_scalar('tr_total_loss', tr_loss, epoch+1)
            tensorboard.add_scalar('tr_lc_loss', tr_lc_loss, epoch+1)
            tensorboard.add_scalar('tr_traj_loss', tr_traj_loss, epoch+1)

            tensorboard.add_scalar('val_total_loss', val_loss, epoch+1)
            tensorboard.add_scalar('val_lc_loss', val_lc_loss, epoch+1)
            tensorboard.add_scalar('val_traj_loss', val_traj_loss, epoch+1)
            tensorboard.add_scalar('val_rmse', rmse[0], epoch+1)
            tensorboard.add_scalar('val_fde', fde, epoch+1)
        if val_loss<best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            patience = p.PATIENCE
        else:
            patience -= 1
        end = time()
        total_time += end-start
        print("Validation Accuracy in best epoch:",best_val_acc, " Avg Loss: ", best_val_loss," at Epoch", best_epoch+1)
        print("Epoch: {} finished in {} sec\n".format(epoch+1, end-start))
        if p.DEBUG_MODE == True:
            print('debugging mode')
            break

        if patience == 0:
            print(' No performance improvement in Validation data after:', epoch+1, 'Epochs!')
            break
        
        

    result_dic = {
        'EarlyStopping Epoch': best_epoch + 1,
        'Best Validaction Acc': best_val_acc,
        'Best Validation Loss': best_val_loss,
    }
    return result_dic


def train_model(p, model, optimizer, scheduler, train_loader, lc_loss_func, traj_loss_func, task, epoch, device, vis_step = 20, calc_train_acc = True):
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
    
    start_seq = 0
    end_seq = p.SEQ_LEN-p.IN_SEQ_LEN+1
    
    # Training loop over batches of data on train dataset
    for batch_idx, (data_tuple, labels,_, _) in enumerate(train_loader):
        #print('Batch: ', batch_idx)
        
        if p.DEBUG_MODE == True:
            if batch_idx >2: ##Uncoment for debuggering
                break
        
        start = time()
        
        data_tuple = [data.to(device) for data in data_tuple]
        labels = labels.to(device)

        if task == params.TRAJECTORYPRED: # seperate traj input data from other
            target_data = data_tuple[-1]
            target_data_in = target_data[:,:p.TGT_SEQ_LEN ]
            target_data_out = target_data[:,1:(p.TGT_SEQ_LEN+1)]
            in_data_tuple = data_tuple[:-1]
        else:
            in_data_tuple = data_tuple
        current_data = in_data_tuple
        
        # 1. Clearing previous gradient values.
        optimizer.zero_grad()
        if model.__class__.__name__ == 'VanillaLSTM':
            model.init_hidden()
        # 2. feeding data to model (forward method will be computed)
        if task == params.TRAJECTORYPRED and 'TRANSFORMER_TRAJ' in p.SELECTED_MODEL:
            
            target_data_in = torch.stack([target_data_in,target_data_in,target_data_in], dim = 1)
            output_dict = model(x = current_data, y =target_data_in, y_mask = model.get_y_mask(p.TGT_SEQ_LEN).to(device))
            traj_pred = output_dict['traj_pred']
            multi_modal = output_dict['multi_modal']
        else:
            output_dict = model(current_data)
        
        #print(lc_pred)
        #print(traj_pred)
        #exit()

        # 3. Calculating the loss value
        
        lc_loss = 0
        
        if task == params.TRAJECTORYPRED:
            #traj_loss_func = nn.MSELoss()
            #print(traj_pred.size())
            if multi_modal == True:
                manouvre_index = labels#F.one_hot(labels)
                #print(manouvre_index.size())
                #print(manouvre_index)
                traj_pred = traj_pred[np.arange(traj_pred.shape[0]), manouvre_index]
            else:
                traj_pred = torch.squeeze(traj_pred, dim =1)
            #print(traj_pred.size())
            #exit()
            traj_loss = traj_loss_func(traj_pred, target_data_out) #TODO: seperate loss function for traj and label
        else:
            traj_loss = 0
        loss = lc_loss + p.TRAJ2CLASS_LOSS_RATIO*traj_loss 
        # 4. Calculating new grdients given the loss value
        loss.backward()
        # 5. Updating the weights
        optimizer.step()
    

        avg_traj_loss += traj_loss.data/(len(train_loader))
        avg_lc_loss += 0#lc_loss.data/(len(train_loader))
        avg_loss += loss.data/(len(train_loader))
        if (batch_idx+1) % 500 == 0:
            print('Epoch: ',epoch, ' Batch: ', batch_idx+1, ' Training Loss: ', avg_loss.cpu().numpy())
            avg_loss = 0
        end = time()
        model_time += end-start
        
        
    all_end = time()
    all_time = all_end - all_start
    #print('model time: ', model_time, 'all training time: ', all_time, 'average training lc loss', avg_loss, 'average training')
    print('Total Training loss: {}, Training LC Loss: {}, Training Traj Loss: {}'.format(avg_loss, avg_lc_loss, avg_traj_loss))
    scheduler.step()
    
    # Validation Phase on train dataset
    if calc_train_acc == True:
        raise('Depricated')
    return avg_loss, avg_lc_loss, avg_traj_loss
        

def eval_model(p, model, lc_loss_func, traj_loss_func, task, test_loader, test_dataset, epoch, device, eval_type = 'Validation', vis_data_path = None, figure_name = None):
    total = len(test_loader.dataset)
    # number of batch
    num_batch = int(np.floor(total/model.batch_size))
    avg_loss = 0
    avg_lc_loss = 0
    avg_traj_loss = 0
    all_lc_preds = np.zeros(((num_batch*model.batch_size), 3))
    all_traj_preds = np.zeros(((num_batch*model.batch_size), p.TGT_SEQ_LEN))
    all_traj_labels = np.zeros(((num_batch*model.batch_size), p.TGT_SEQ_LEN))
    all_att_coef = np.zeros(((num_batch*model.batch_size), 4))
    all_labels = np.zeros(((num_batch*model.batch_size)))
    plot_dicts = []
    
    time_counter = 0
    average_time = 0
    gf_time = 0
    nn_time = 0
    loss_ratio = 1
    multi_modal = False
    for batch_idx, (data_tuple, labels, plot_info, _) in enumerate(test_loader):
        
        if p.DEBUG_MODE == True:
            if batch_idx >2: ##Uncoment for debuggering
                break
        
        if eval_type == 'Test':
            (tv_id, frames, data_file) = plot_info
            batch_size = frames.shape[0]
            plot_dict = {
                'tv': tv_id.numpy(),
                'frames': frames.numpy(),
                'preds':np.zeros((batch_size, 3)),
                'ttlc_preds': np.zeros((batch_size)),
                'att_coef': np.zeros((batch_size, 4)),
                'att_mask': np.zeros((batch_size, 11, 26)),
                'traj_labels': np.zeros((batch_size, p.TGT_SEQ_LEN, 2)),
                'traj_preds': np.zeros((batch_size, p.TGT_SEQ_LEN, 2)),
                'labels':labels.numpy(),
                'data_file': data_file
            }
        
        data_tuple = [data.to(device) for data in data_tuple]
        labels = labels.to(device)
        
        if model.__class__.__name__ == 'VanillaLSTM':
                model.init_hidden()
        
        if task == params.TRAJECTORYPRED:
            target_data = data_tuple[-1]
            #print(target_data.shape)
            target_data_out = target_data[:,1:(p.TGT_SEQ_LEN+1)] #trajectory labels are target data except the first element.
            in_data_tuple = data_tuple[:-1]
        else:
            in_data_tuple = data_tuple
        current_data =  in_data_tuple
        st_time = time()
        
        if task == params.TRAJECTORYPRED and 'TRANSFORMER_TRAJ' in p.SELECTED_MODEL:
            target_data_in = target_data[:,:1] #initalise target data input to transformer decoder with the first element of target data
            target_data_in = torch.stack([target_data_in,target_data_in,target_data_in], dim = 1)
            #print(target_data_in.size())
            for out_seq_itr in range(p.TGT_SEQ_LEN):
                output_dict = model(x = current_data, y =target_data_in, y_mask = model.get_y_mask(target_data_in.size(2)).to(device))
                traj_pred = output_dict['traj_pred']
                multi_modal = output_dict['multi_modal']
                #print('model output: {}'.format(traj_pred.shape))
                traj_pred = traj_pred[:,:,out_seq_itr:(out_seq_itr+1)]
                if not multi_modal:
                    traj_pred = torch.cat([traj_pred, traj_pred, traj_pred], dim = 1)
                #print('target data in:')
                #print(target_data_in.shape)
                #print(traj_pred.shape)
                #if target_data_in.shape[1] == 0 or traj_pred.shape[1] == 0:
                #    exit()
                target_data_in = torch.cat((target_data_in, traj_pred), dim = 2)
            
            traj_pred = target_data_in[:,:,1:]    
        elif task == params.TRAJECTORYPRED and p.SELECTED_MODEL== 'CONSTANT_PARAMETER':
            target_data_in = target_data
            output_dict = model(current_data, test_dataset.states_min, test_dataset.states_max, test_dataset.output_states_min, test_dataset.output_states_max, target_data_in)
            traj_pred = output_dict['traj_pred']
            #print(traj_pred.size())
            traj_pred = traj_pred.unsqueeze(1)
            #print(traj_pred.size())
        else:
            output_dict = model(current_data)
        
        end_time = time()-st_time
        #lc_pred = output_dict['lc_pred']
        #print(lc_pred)
        #exit()
        
        lc_loss = 0

        if task == params.TRAJECTORYPRED:
            #traj_loss_func = nn.MSELoss()
            if multi_modal == True:
                #predicted_labels = F.softmax(lc_pred, dim = -1).argmax(dim = -1)
                manouvre_index = predicted_labels #in eval we use predicted label instead of ground truth.
                traj_pred = traj_pred[np.arange(traj_pred.shape[0]), manouvre_index]
            else:
                traj_pred = traj_pred[:,0]
            #print(traj_pred.size())
            #exit() 
            #print('traj pred: {}'.format(traj_pred.shape))
            #print('target data out: {}'.format(target_data_out.shape))
            traj_loss = traj_loss_func(traj_pred, target_data_out) 
        else:
            traj_loss = 0
        loss = lc_loss + p.TRAJ2CLASS_LOSS_RATIO*traj_loss 

        #_ , pred_labels = output.data.max(dim=1)
        #pred_labels = pred_labels.cpu()
    
        if eval_type == 'Test':
            #if task == params.CLASSIFICATION or task == params.DUAL or task == params.TRAJECTORYPRED:
            #    plot_dict['preds'] = F.softmax(lc_pred, dim = -1).cpu().data
            if 'REGIONATT' in p.SELECTED_MODEL:
                plot_dict['att_coef'] = output_dict['attention'].cpu().data
            if task == params.TRAJECTORYPRED:
                traj_min = test_dataset.output_states_min
                traj_max = test_dataset.output_states_max
                #print_shape('traj_pred', traj_pred)
                x_y_pred = traj_pred[:,:,:2]
                x_y_label = target_data_out[:,:,:2]
                unnormalised_traj_pred = x_y_pred.cpu().data*(traj_max-traj_min) + traj_min
                unnormalised_traj_label = x_y_label.cpu().data*(traj_max-traj_min) + traj_min
                plot_dict['traj_labels'] = unnormalised_traj_label
                plot_dict['traj_preds']= unnormalised_traj_pred
                
        
        if task == params.CLASSIFICATION or task == params.DUAL or task == params.TRAJECTORYPRED:
            #all_lc_preds[(batch_idx*model.batch_size):((batch_idx+1)*model.batch_size)] = F.softmax(lc_pred, dim = -1).cpu().data 
            avg_lc_loss = 0# avg_lc_loss + lc_loss.cpu().data / len(test_loader)
        if task == params.TRAJECTORYPRED:
            all_traj_preds[(batch_idx*model.batch_size):((batch_idx+1)*model.batch_size)] = traj_pred.cpu().data
            all_traj_labels[(batch_idx*model.batch_size):((batch_idx+1)*model.batch_size)] = target_data_out.cpu().data
        if p.SELECTED_MODEL == 'REGIONATTCNN3':
            all_att_coef[(batch_idx*model.batch_size):((batch_idx+1)*model.batch_size)] = output_dict['attention'].cpu().data
        avg_traj_loss += traj_loss.cpu().data / len(test_loader)
        avg_loss =  avg_loss + loss.cpu().data / len(test_loader)
        time_counter += 1
        average_time +=end_time
        if (batch_idx+1) % 500 == 0:
            print('Epoch: ',epoch, ' Batch: ', batch_idx+1)
            
        if eval_type == 'Test':
            plot_dicts.append(plot_dict)
    
    #print('Average Time per whole sequence perbatch: {}'.format(average_time/time_counter))
    #print('gf time: {}, nn time: {}'.format(gf_time, nn_time))
    traj_metrics, man_metrics, traj_df = calc_metric(p, task, all_lc_preds, all_att_coef, all_labels, all_traj_preds, all_traj_labels, test_dataset.output_states_min, test_dataset.output_states_max, epoch, eval_type = eval_type, figure_name = figure_name)
    accuracy, precision, recall = man_metrics
    f1=0
    FPR=0
    auc=0
    max_j=0 
    rmse = traj_metrics[0:3]
    fde = traj_metrics [3]
    print('                                   ')
    print("{}: Epoch: {}, Accuracy: {:.2f}%, Total LOSS: {:.2f},LC LOSS: {:.2f},TRAJ LOSS: {:.5f}, PRECISION:{}, RECALL:{}, F1:{}, FPR:{}, AUC:{}, Max J:{}, RMSE:{}, FDE:{}".format(
        eval_type, epoch, 100. * accuracy, avg_loss, avg_lc_loss, avg_traj_loss, precision, recall, f1, FPR, auc, max_j, rmse, fde))
    print('                                   ')
    print('-----------------------------------')
    print('                                   ')
    print(traj_df)
    
    if eval_type == 'Test':
        with open(vis_data_path, "wb") as fp:
            pickle.dump(plot_dicts, fp)
        
    return accuracy, avg_loss, avg_lc_loss, avg_traj_loss, auc, max_j, precision, recall, f1, rmse, fde, traj_df



def calc_metric(p, task, all_lc_preds, all_att_coef, all_labels, all_traj_preds, all_traj_labels, traj_label_min, traj_label_max, epoch=None, eval_type = 'Test', figure_name= None):
   
    num_samples = all_labels.shape[0]
    all_preds = np.argmax(all_lc_preds, axis =-1)
    
    
    if task == params.TRAJECTORYPRED:
        traj_metrics, man_metrics, traj_df = calc_traj_metrics(p, all_traj_preds, all_traj_labels, traj_label_min, traj_label_max)
    else:
        raise(ValueError('unsupported value for task'))

    return traj_metrics, man_metrics, traj_df


def calc_traj_metrics(p, 
    traj_preds:'[number of samples, target sequence length, number of output states]', 
    traj_labels,
    traj_min, 
    traj_max):
    #traj_preds [number of samples, target sequence length, number of output states]
    #TODO:1. manouvre specific fde and rmse table, 2.  save sample output traj imags 3. man pred error
    #man_preds = traj_preds[:,:,2:]*2
    #man_preds = np.rint(man_preds)
    man_labels = traj_labels[:,:,2:]*2
    man_labels = np.rint(man_labels)
    traj_preds = traj_preds[:,:,:2]
    traj_labels = traj_labels[:,:,:2]
    lc_frames = (man_labels>0)
    lk_frames = (man_labels ==0)
    total_lc_frames = np.count_nonzero(lc_frames)
    total_lk_frames = np.count_nonzero(lk_frames)
    total_frames = np.prod(man_labels.shape)
    total_sequences = man_labels.shape[0]
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
    mse_lc = np.sum((lc_frames*(traj_preds-traj_labels))**2)/total_lc_frames
    mse_lk = np.sum((lk_frames*(traj_preds-traj_labels))**2)/total_lk_frames
    rmse = np.sqrt(mse) 
    rmse_lc = np.sqrt(mse_lc) 
    rmse_lk = np.sqrt(mse_lk) 

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

    result_df = pd.DataFrame(data= data, columns = columns, index = index)
    traj_metrics = (rmse, rmse_lc, rmse_lk, fde)
    man_metrics = (accuracy, precision, recall)
    return traj_metrics, man_metrics, result_df


    

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

def update_tag(model_dict):
    hyperparam_str = ''
    for hyperparam in model_dict['hyperparams']:
        abbr = hyperparam.split()
        abbr = ''.join([word[0] for word in abbr])
        
        hyperparam_str += abbr + str(model_dict['hyperparams'][hyperparam])
    hyperparam_str += model_dict['state type']
    
    return model_dict['name'] + hyperparam_str