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
from kpis import *
font = {'size'   : 22}
matplotlib.rcParams['figure.figsize'] = (18, 12)
matplotlib.rc('font', **font)


def eval_top_func(p, model, lc_loss_func, traj_loss_func, te_dataset, device, tensorboard = None):
    model = model.to(device)
    
    te_loader = utils_data.DataLoader(dataset = te_dataset, shuffle = True, batch_size = p.BATCH_SIZE, drop_last= True, pin_memory= True, num_workers= 12)

    vis_data_path = p.VIS_DIR + p.experiment_tag + '.pickle'
    best_model_path = p.MODELS_DIR + p.experiment_tag + '.pt'
    figure_name =  p.experiment_tag
    
    if p.SELECTED_MODEL != 'CONSTANT_PARAMETER':
        model.load_state_dict(torch.load(best_model_path))
    
    start = time()
    
    test_acc, test_loss, test_lc_loss, test_traj_loss, auc, max_j, precision, recall, f1, rmse, fde, traj_df = eval_model(p,model, lc_loss_func, traj_loss_func, te_loader, te_dataset, ' N/A', device, eval_type = 'Test', vis_data_path = vis_data_path, figure_name = figure_name)
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


def train_top_func(p, model, optimizer, lc_loss_func, traj_loss_func, tr_dataset, val_dataset, device, tensorboard = None):
    
    model = model.to(device)
    
    tr_loader = utils_data.DataLoader(dataset = tr_dataset, shuffle = True, batch_size = p.BATCH_SIZE, drop_last= True, pin_memory= True, num_workers= 12)
    val_loader = utils_data.DataLoader(dataset = val_dataset, shuffle = True, batch_size = p.BATCH_SIZE, drop_last= True, pin_memory= True, num_workers= 12)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, p.LR_DECAY_EPOCH, p.LR_DECAY)
    
    best_model_path = p.MODELS_DIR + p.experiment_tag + '.pt'

    best_val_acc = 0
    best_val_loss = float("inf")
    patience = p.PATIENCE
    best_epoch = 0
    total_time = 0
    p.LR_WU_CURRENT_BATCH = 0
    for epoch in range(p.NUM_EPOCHS):
        #print("Epoch: {} Started!".format(epoch+1))
        start = time()
        tr_loss, tr_lc_loss, tr_traj_loss = train_model(p, model, optimizer, scheduler, tr_loader, lc_loss_func, traj_loss_func,  epoch+1, device, calc_train_acc= False)
        val_start = time()

        val_acc,val_loss, val_lc_loss, val_traj_loss, auc, max_j, precision, recall, f1, rmse, fde, traj_df = eval_model(p, model, lc_loss_func, traj_loss_func, val_loader, val_dataset, epoch+1, device, eval_type = 'Validation')
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


def train_model(p, model, optimizer, scheduler, train_loader, lc_loss_func, traj_loss_func, epoch, device, vis_step = 20, calc_train_acc = True):
    # Number of samples with correct classification
    # total size of train data
    total = len(train_loader.dataset)
    # number of batch
    num_batch = int(np.floor(total/model.batch_size))
    model_time = 0
    avg_loss = 0
    avg_lc_loss = 0
    avg_enc_lc_loss = 0
    avg_traj_loss = 0
    all_start = time()
    
    
    # Training loop over batches of data on train dataset
    for batch_idx, (data_tuple, labels,_, _) in enumerate(train_loader):
        #print('Batch: ', batch_idx)
        
        if p.DEBUG_MODE == True:
            if batch_idx >2: ##Uncoment for debuggering
                break
        
        start = time()
        
        data_tuple = [data.to(device) for data in data_tuple]
        labels = labels.to(device)
        dec_man_gt = labels[:, p.IN_SEQ_LEN:]
        enc_man_gt = labels[:,p.IN_SEQ_LEN-1:p.IN_SEQ_LEN]
           
        
        man_gt_onehot = F.one_hot(labels, num_classes= 3)
        label_enc_in = man_gt_onehot[:,:p.IN_SEQ_LEN]
        label_dec_in = man_gt_onehot[:,(p.IN_SEQ_LEN-1):(p.IN_SEQ_LEN-1+p.TGT_SEQ_LEN)]
        target_data = data_tuple[-1] # selecting target data from data tuple
        tv_only_data = target_data[:,:p.IN_SEQ_LEN] # if tv's only, we use same output states for input data
        target_data = target_data[:,(p.IN_SEQ_LEN-1):]
        target_data_in = target_data[:,:p.TGT_SEQ_LEN ] # target data into decoder
        traj_gt = target_data[:,1:(p.TGT_SEQ_LEN+1)] #ground truth data
        in_data_tuple = data_tuple[:-1]
        encoder_input = in_data_tuple[0]

        
        if p.TV_ONLY:
            encoder_input = tv_only_data
        
        
        target_data_in = torch.cat((target_data_in, label_dec_in), dim = -1) 
        # 1. Clearing previous gradient values.
        optimizer.zero_grad()
        
        # 2. feeding data to model (forward method will be computed)
        if 'TRANSFORMER_TRAJ' in p.SELECTED_MODEL:
            target_data_in = torch.stack([target_data_in,target_data_in,target_data_in], dim = 1)
            output_dict = model(x = encoder_input, y = target_data_in, y_mask = model.get_y_mask(p.TGT_SEQ_LEN).to(device))
            traj_pred = output_dict['traj_pred']
            man_pred = output_dict['man_pred']
            enc_man_pred = output_dict['enc_man_pred']

        elif 'LSTM_ED' in p.SELECTED_MODEL:
            target_data_in = torch.stack([target_data_in,target_data_in,target_data_in], dim = 1)
            output_dict = model(x = encoder_input, y = target_data_in, teacher_force = True)
            traj_pred = output_dict['traj_pred']
            man_pred = output_dict['man_pred'] #TODO: update enc man pred for lstm

        else:
            output_dict = model(encoder_input)
        
        #print(lc_pred)
        #print(traj_pred)
        #exit()

        # 3. Calculating the loss value
        
        #lc_loss = 0
        
        
        if p.MULTI_MODAL == True:
            #print_shape('traj_pred', traj_pred)
            manouvre_index = dec_man_gt
            for i in range(p.TGT_SEQ_LEN):
                current_selected_traj = traj_pred[np.arange(traj_pred.shape[0]), manouvre_index[:,i],i,:]
                current_selected_traj = torch.unsqueeze(current_selected_traj, dim = 1)
                #print_shape('current_selected_traj', current_selected_traj)
                if i == 0:
                    selected_traj_pred = current_selected_traj
                else:
                    selected_traj_pred =  torch.cat((selected_traj_pred, current_selected_traj), dim=1)
            traj_pred = selected_traj_pred
            #print_shape('traj_pred', traj_pred)
        else:
            traj_pred = traj_pred[:,0]

        #print_shape('traj_pred',traj_pred)
        #print_shape('traj_gt',traj_gt)
        traj_loss = traj_loss_func(traj_pred, traj_gt)
        
        if p.MAN_DEC_OUT:
            lc_loss = lc_loss_func(man_pred.reshape(-1,3), dec_man_gt.reshape(-1))
            enc_lc_loss = lc_loss_func(enc_man_pred.reshape(-1,3), enc_man_gt.reshape(-1))
        else:
            lc_loss = 0
            enc_lc_loss = 0
        
        loss = lc_loss + enc_lc_loss + p.TRAJ2CLASS_LOSS_RATIO*traj_loss 
        # 4. Calculating new grdients given the loss value
        loss.backward()
        # 5. Updating the weights
        if p.LR_WU and p.LR_WU_CURRENT_BATCH<=p.LR_WU_BATCHES:
            p.LR_WU_CURRENT_BATCH +=1
            lr = p.LR*p.LR_WU_CURRENT_BATCH/p.LR_WU_BATCHES
            for g in optimizer.param_groups:
                g['lr'] = lr

        optimizer.step()
    

        avg_traj_loss += traj_loss.data/(len(train_loader))
        if p.MAN_DEC_OUT:
            avg_lc_loss += lc_loss.data/(len(train_loader))
            avg_enc_lc_loss += enc_lc_loss.data/(len(train_loader))
        else:
            avg_lc_loss = 0
            avg_enc_lc_loss = 0
        avg_loss += loss.data/(len(train_loader))
        if (batch_idx+1) % 500 == 0:
            print('Epoch: ',epoch, ' Batch: ', batch_idx+1, ' Training Loss: ', avg_loss.cpu().numpy())
            avg_loss = 0
        end = time()
        model_time += end-start
        
        
    all_end = time()
    all_time = all_end - all_start
    print('Total Training loss: {}, Training ENC LC Loss: {}, LC Loss: {}, Training Traj Loss: {}'.format(avg_loss, avg_enc_lc_loss, avg_lc_loss, avg_traj_loss))
    scheduler.step()
    
    # Validation Phase on train dataset
    if calc_train_acc == True:
        raise('Depricated')
    return avg_loss, avg_lc_loss, avg_traj_loss
        

def eval_model(p, model, lc_loss_func, traj_loss_func, test_loader, test_dataset, epoch, device, eval_type = 'Validation', vis_data_path = None, figure_name = None):
    total = len(test_loader.dataset)
    num_batch = int(np.floor(total/model.batch_size))
    # Initialise Variables
    avg_loss = 0
    avg_lc_loss = 0
    avg_enc_lc_loss = 0
    avg_traj_loss = 0
    time_counter = 0
    average_time = 0
    gf_time = 0
    nn_time = 0
    loss_ratio = 1
    plot_dicts = []
    
    all_traj_preds = np.zeros(((num_batch*model.batch_size), p.TGT_SEQ_LEN, 2))
    all_traj_labels = np.zeros(((num_batch*model.batch_size), p.TGT_SEQ_LEN + p.IN_SEQ_LEN,2))
    all_man_preds = np.zeros(((num_batch*model.batch_size), p.TGT_SEQ_LEN,3))
    all_man_labels = np.zeros(((num_batch*model.batch_size), p.TGT_SEQ_LEN + p.IN_SEQ_LEN,3))
    
    
    
    for batch_idx, (data_tuple, labels, plot_info, _) in enumerate(test_loader):
        
        if p.DEBUG_MODE == True:
            if batch_idx >2: 
                break
        
        # Plot data initialisation
        if eval_type == 'Test':
            (tv_id, frames, data_file) = plot_info
            batch_size = frames.shape[0]
            plot_dict = {
                'data_file': data_file,
                'tv': tv_id.numpy(),
                'frames': frames.numpy(),
                'traj_min': test_dataset.output_states_min,
                'traj_max': test_dataset.output_states_max,
                'input_features': np.zeros((batch_size, p.IN_SEQ_LEN, 18)),
                'traj_labels': np.zeros((batch_size, p.TGT_SEQ_LEN+p.IN_SEQ_LEN, 2)),
                'traj_preds': np.zeros((batch_size, p.TGT_SEQ_LEN, 2)),
                'traj_dist_preds': np.zeros((batch_size, p.TGT_SEQ_LEN, 5)),
                'man_labels':np.zeros((batch_size, p.TGT_SEQ_LEN+p.IN_SEQ_LEN)),
                'man_preds':np.zeros((batch_size, p.TGT_SEQ_LEN,3)),
                'enc_man_pred': np.zeros((batch_size, 3)),     
            }
        data_tuple = [data.to(device) for data in data_tuple]
        
        tv_traj_data = data_tuple[-1]
        initial_traj = tv_traj_data[:,(p.IN_SEQ_LEN-1):p.IN_SEQ_LEN]
        traj_gt = tv_traj_data[:,p.IN_SEQ_LEN:(p.IN_SEQ_LEN+p.TGT_SEQ_LEN)] 
        
        in_data_tuple = data_tuple[:-1]
        encoder_input = in_data_tuple[0]
        if p.TV_ONLY:
            encoder_input = tv_traj_data[:,:p.IN_SEQ_LEN]
        
        labels = labels.to(device) # [0:IN_SEQ_LEN+p.TGT_SEQ_LEN]
        dec_man_gt = labels[:, p.IN_SEQ_LEN:]
        enc_man_gt = labels[:,p.IN_SEQ_LEN-1:p.IN_SEQ_LEN]
        man_gt_onehot = F.one_hot(labels, num_classes= 3)

        st_time = time()
        
        if 'TRANSFORMER_TRAJ' in p.SELECTED_MODEL:
            encoder_out, enc_man_pred = model.encoder_forward(x = encoder_input)
            initial_man = F.one_hot(torch.argmax(enc_man_pred, dim = -1), num_classes = 3)
            initial_man = torch.unsqueeze(initial_man, dim = 1)
            #print_shape('enc_man_pred', enc_man_pred)
            #print_shape('initial_man', initial_man)
            #print_shape('initial_traj', initial_traj)
            decoder_input = torch.cat((initial_traj, initial_man), dim =-1) 
            decoder_input = torch.stack([decoder_input,decoder_input,decoder_input], dim = 1) #multi-modal
            
            for out_seq_itr in range(p.TGT_SEQ_LEN):
                #output_dict = model(x = encoder_input, y =decoder_input, y_mask = model.get_y_mask(decoder_input.size(2)).to(device))
                traj_pred, man_pred = model.decoder_forward(y = decoder_input, 
                                                            y_mask = model.get_y_mask(decoder_input.size(2)).to(device), 
                                                            encoder_out = encoder_out)
                
                current_traj_pred = traj_pred[:,:,out_seq_itr:(out_seq_itr+1)] #traj output at current timestep
                current_man_pred = torch.argmax(man_pred[:,out_seq_itr:(out_seq_itr+1)], dim = -1)        
                
                if p.MULTI_MODAL:
                    manouvre_index = current_man_pred[:,0] 
                else:
                    manouvre_index = torch.zeros_like(current_man_pred[:,0] )
                #print_shape('current_traj_pred', current_traj_pred)
                #print_shape('manouvre_index', manouvre_index)
                current_traj_pred = current_traj_pred[np.arange(current_traj_pred.shape[0]),manouvre_index,:,:2] #only the muX and muY [batch, modal, sequence, feature]
                current_man_pred = F.one_hot(current_man_pred, num_classes = 3) 
                
                #print_shape('current_traj_pred', current_traj_pred)
                #print_shape('current_man_pred', current_man_pred)
                
                current_decoder_input = torch.cat((current_traj_pred, current_man_pred), dim = -1)
                current_decoder_input = torch.unsqueeze(current_decoder_input, dim = 1)
                current_decoder_input = torch.cat([current_decoder_input, current_decoder_input, current_decoder_input], dim = 1)
                
                decoder_input = torch.cat((decoder_input, current_decoder_input), dim = 2)
                
                if out_seq_itr ==0:
                    predicted_data_dist = traj_pred[np.arange(traj_pred.shape[0]),manouvre_index, out_seq_itr:(out_seq_itr+1)]
                else:
                    predicted_data_dist = torch.cat((predicted_data_dist, traj_pred[np.arange(traj_pred.shape[0]),manouvre_index, out_seq_itr:(out_seq_itr+1)]), dim=1)
                #print_shape('predicted_data_dist', predicted_data_dist)
            traj_pred = decoder_input[:,:,1:,:2]
            man_pred = torch.unsqueeze(man_pred, dim = 1)
            
        elif p.SELECTED_MODEL== 'CONSTANT_PARAMETER':
            output_dict = model(encoder_input, test_dataset.states_min, test_dataset.states_max, test_dataset.output_states_min, test_dataset.output_states_max, traj_labels = None)
            traj_pred = output_dict['traj_pred']
            man_pred = output_dict['man_pred'] # All zero vector for this model
            enc_man_pred = output_dict['enc_man_pred'] # All zero vector for this model
            man_pred = torch.unsqueeze(man_pred, dim = 1)
            #print(traj_pred.size())
            traj_pred = traj_pred.unsqueeze(1)
            #print(traj_pred.size())
            predicted_data_dist = traj_pred[:,0]
        elif 'LSTM_ED' in p.SELECTED_MODEL:
            
            decoder_input = torch.cat((initial_traj, initial_man), dim =-1)
            decoder_input = torch.stack([decoder_input,decoder_input,decoder_input], dim = 1)
            output_dict = model(x = encoder_input, y = decoder_input, teacher_force = False)
            predicted_data_dist = output_dict['traj_pred']
            traj_pred = predicted_data_dist[:,:,:,:2] 
            predicted_data_dist = predicted_data_dist[:,0]
            man_pred = output_dict['man_pred']
            #TODO add enc_man_pred
            man_pred = torch.unsqueeze(man_pred, dim = 1)
            
        else:
            output_dict = model(encoder_input)
        
        end_time = time()-st_time
       
        man_pred = man_pred[:,0]
        traj_pred = traj_pred[:,0]
        #print_shape('traj_gt', traj_gt)
        #print_shape('predicted_data_dist', predicted_data_dist)
        traj_loss = traj_loss_func(predicted_data_dist, traj_gt)/len(test_loader) 
        if p.MAN_DEC_OUT:
            lc_loss = lc_loss_func(man_pred.reshape(-1,3), dec_man_gt.reshape(-1))
            enc_lc_loss = lc_loss_func(enc_man_pred.reshape(-1,3), enc_man_gt.reshape(-1)) 
        else:
            lc_loss = 0
            enc_lc_loss = 0
        loss = enc_lc_loss + lc_loss + p.TRAJ2CLASS_LOSS_RATIO*traj_loss 


        if eval_type == 'Test':
            traj_min = test_dataset.output_states_min
            traj_max = test_dataset.output_states_max
            x_y_dist_pred = predicted_data_dist
            #print_shape('predicted_data_dist', predicted_data_dist)
            #exit()
            x_y_pred = traj_pred[:,:,:2]
            x_y_label = data_tuple[-1]
            unnormalised_traj_pred = x_y_pred.cpu().data*(traj_max-traj_min) + traj_min
            unnormalised_traj_label = x_y_label.cpu().data*(traj_max-traj_min) + traj_min
            plot_dict['input_features'] = encoder_input
            plot_dict['traj_labels'] = unnormalised_traj_label.numpy()
            plot_dict['traj_preds']= unnormalised_traj_pred.numpy() #TODO: remove traj pred and use traj_dist_preds instead
            plot_dict['traj_dist_preds'] = x_y_dist_pred.cpu().data.numpy()# TODO: export unnormalised dist
            plot_dict['man_labels'] = labels.cpu().data.numpy()
            plot_dict['man_preds'] = man_pred.cpu().data.numpy()
            plot_dict['enc_man_preds'] = enc_man_pred.cpu().data.numpy()
                
        all_traj_preds[(batch_idx*model.batch_size):((batch_idx+1)*model.batch_size)] = traj_pred.cpu().data
        all_traj_labels[(batch_idx*model.batch_size):((batch_idx+1)*model.batch_size)] = data_tuple[-1].cpu().data
        all_man_preds[(batch_idx*model.batch_size):((batch_idx+1)*model.batch_size)] = man_pred.cpu().data
        all_man_labels[(batch_idx*model.batch_size):((batch_idx+1)*model.batch_size)] = man_gt_onehot.cpu().data

        avg_traj_loss += traj_loss.cpu().data / len(test_loader)
        avg_loss =  avg_loss + loss.cpu().data / len(test_loader)
        
        if p.MAN_DEC_OUT:
            avg_lc_loss += lc_loss.data/(len(test_loader))
            avg_enc_lc_loss += enc_lc_loss.data/(len(test_loader))
        else:
            avg_lc_loss = 0
            avg_enc_lc_loss = 0
        
        

        time_counter += 1
        average_time +=end_time
        if (batch_idx+1) % 500 == 0:
            print('Epoch: ',epoch, ' Batch: ', batch_idx+1)
            
        if eval_type == 'Test':
            plot_dicts.append(plot_dict)
    
  
    traj_metrics, man_metrics, traj_df = calc_metric(p, all_traj_preds, all_traj_labels, all_man_preds, all_man_labels, test_dataset.output_states_min, test_dataset.output_states_max, epoch, eval_type = eval_type, figure_name = figure_name)
    accuracy, precision, recall = man_metrics
    f1=0
    FPR=0
    auc=0
    max_j=0 
    rmse = traj_metrics[0:3]
    fde = traj_metrics [3]
    print('                                   ')
    print("{}: Epoch: {}, Accuracy: {:.2f}%, Total LOSS: {},ENC_LC_LOSS: {}, LC LOSS: {},TRAJ LOSS: {}, PRECISION:{}, RECALL:{}, F1:{}, FPR:{}, AUC:{}, Max J:{}, RMSE:{}, FDE:{}".format(
        eval_type, epoch, 100. * accuracy, avg_loss, avg_enc_lc_loss, avg_lc_loss, avg_traj_loss, precision, recall, f1, FPR, auc, max_j, rmse, fde))
    print('                                   ')
    print('-----------------------------------')
    print('                                   ')
    print(traj_df)
    
    if eval_type == 'Test':
        
        with open(vis_data_path, "wb") as fp:
            pickle.dump(plot_dicts, fp)
        
    return accuracy, avg_loss, avg_lc_loss, avg_traj_loss, auc, max_j, precision, recall, f1, rmse, fde, traj_df


