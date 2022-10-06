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
import kpis

font = {'size'   : 22}
matplotlib.rcParams['figure.figsize'] = (18, 12)
matplotlib.rc('font', **font)


def eval_top_func(p, model, man_loss_func, traj_loss_func, te_dataset, device, tensorboard = None):
    model = model.to(device)
    
    te_loader = utils_data.DataLoader(dataset = te_dataset, shuffle = True, batch_size = p.BATCH_SIZE, drop_last= True, pin_memory= True, num_workers= 12)

    vis_data_path = p.VIS_DIR + p.experiment_tag + '.pickle'
    best_model_path = p.MODELS_DIR + p.experiment_tag + '.pt'
    figure_name =  p.experiment_tag
    
    if p.SELECTED_MODEL != 'CONSTANT_PARAMETER':
        model.load_state_dict(torch.load(best_model_path))
    
    start = time()
    
    test_acc,test_losses, auc, max_j, precision, recall, f1, rmse, fde, traj_df = eval_model(p, tensorboard, model, man_loss_func, traj_loss_func, te_loader, te_dataset, ' N/A', device, eval_type = 'Test', vis_data_path = vis_data_path, figure_name = figure_name)
    (test_loss, test_man_loss, test_traj_loss, test_mode_ploss, test_man_ploss, test_time_ploss) = test_losses
    end = time()
    total_time = end-start
    #print("Test finished in:", total_time, "sec.")
    #print("Final Test accuracy:",te_acc)
    result_dic = {
        'Test Acc': test_acc,
        'Test Total Loss': test_loss,
        'Test Classification Loss': test_man_loss,
        'AUC': auc,
        'Max Youden Index': max_j,
        'Precision': precision,
        'Recall': recall,
        'F1':f1,
        'rmse':rmse[0],
        'fde': fde,
    }
    return result_dic, traj_df


def train_top_func(p, model, optimizer, man_loss_func, traj_loss_func, tr_dataset, val_dataset, device, tensorboard = None):
    
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
        tr_loss, tr_man_loss, tr_traj_loss, tr_mode_ploss, tr_man_ploss, tr_time_ploss = train_model(p, tensorboard, model, optimizer, scheduler, tr_loader, man_loss_func, traj_loss_func,  epoch+1, device, calc_train_acc= False)
        val_start = time()

        val_acc, val_losses, auc, max_j, precision, recall, f1, rmse, fde, traj_df = eval_model(p, tensorboard, model, man_loss_func, traj_loss_func, val_loader, val_dataset, epoch+1, device, eval_type = 'Validation')
        
        (val_loss, val_man_loss, val_traj_loss, val_mode_ploss, val_man_ploss, val_time_ploss) = val_losses
        val_end = time()
        print('val_time:', val_end-val_start)
        #print("Validation Accuracy:",val_acc,' Avg Pred Time: ', val_avg_pred_time, " Avg Loss: ", val_loss," at Epoch", epoch+1)
        if tensorboard != None:
            
            tensorboard.add_scalar('tr_total_loss', tr_loss, epoch+1)
            tensorboard.add_scalar('tr_man_loss', tr_man_loss, epoch+1)
            tensorboard.add_scalar('tr_traj_loss', tr_traj_loss, epoch+1)
            tensorboard.add_scalar('tr_mode_ploss', tr_mode_ploss, epoch+1)
            tensorboard.add_scalar('tr_man_ploss', tr_man_ploss, epoch+1)
            tensorboard.add_scalar('tr_time_ploss', tr_time_ploss, epoch+1)

            tensorboard.add_scalar('val_total_loss', val_loss, epoch+1)
            tensorboard.add_scalar('val_man_loss', val_man_loss, epoch+1)
            tensorboard.add_scalar('val_traj_loss', val_traj_loss, epoch+1)
            tensorboard.add_scalar('val_mode_ploss', val_mode_ploss, epoch+1)
            tensorboard.add_scalar('val_man_ploss', val_man_ploss, epoch+1)
            tensorboard.add_scalar('val_time_ploss', val_time_ploss, epoch+1)
            tensorboard.add_scalar('val_rmse', rmse[0], epoch+1)
            tensorboard.add_scalar('val_fde', fde, epoch+1)
        if rmse[0]<best_val_loss:
            best_val_loss = rmse[0]
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


def train_model(p, tb, model, optimizer, scheduler, train_loader, man_loss_func, traj_loss_func, epoch, device, vis_step = 20, calc_train_acc = True):
    # Number of samples with correct classification
    # total size of train data
    total = len(train_loader.dataset)
    # number of batch
    num_batch = int(np.floor(total/model.batch_size))
    model_time = 0
    avg_loss = 0
    avg_man_loss = 0
    avg_mode_ploss = 0
    avg_man_ploss = 0
    avg_time_ploss = 0
    avg_traj_loss = 0
    
    batch_loss = 0
    batch_man_loss = 0
    batch_mode_ploss = 0
    batch_man_ploss = 0
    batch_time_ploss = 0
    batch_traj_loss = 0

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
           
        
        man_gt_onehot = F.one_hot(labels, num_classes= 3)
        label_dec_in = man_gt_onehot[:,(p.IN_SEQ_LEN-1):(p.IN_SEQ_LEN-1+p.TGT_SEQ_LEN)]
        target_data = data_tuple[-1] # selecting target data from data tuple
        target_data = target_data[:,(p.IN_SEQ_LEN-1):]
        target_data_in = target_data[:,:p.TGT_SEQ_LEN ] # target data into decoder
        traj_gt = target_data[:,1:(p.TGT_SEQ_LEN+1)] #ground truth data
        in_data_tuple = data_tuple[:-1]
        encoder_input = in_data_tuple[0]

        
        
        target_data_in = torch.cat((target_data_in, label_dec_in), dim = -1) 
        # 1. Clearing previous gradient values.
        optimizer.zero_grad()
        
        # 2. feeding data to model (forward method will be computed)
        target_data_in = torch.stack([target_data_in,target_data_in,target_data_in], dim = 1)
        output_dict = model(x = encoder_input, y = target_data_in, y_mask = model.get_y_mask(p.TGT_SEQ_LEN).to(device))
        traj_pred = output_dict['traj_pred']
        man_pred = output_dict['man_pred']
        
        if p.MULTI_MODAL == True:
            
            manouvre_index = dec_man_gt
            for i in range(p.TGT_SEQ_LEN):
                current_selected_traj = traj_pred[np.arange(traj_pred.shape[0]), manouvre_index[:,i],i,:]
                current_selected_traj = torch.unsqueeze(current_selected_traj, dim = 1)
                
                if i == 0:
                    selected_traj_pred = current_selected_traj
                else:
                    selected_traj_pred =  torch.cat((selected_traj_pred, current_selected_traj), dim=1)
            traj_pred = selected_traj_pred
            
        else:
            traj_pred = traj_pred[:,0]

        traj_loss = traj_loss_func(traj_pred, traj_gt)
        
        if p.MAN_DEC_OUT:
            man_loss, mode_ploss, man_ploss, time_ploss = man_loss_func(man_pred, dec_man_gt, n_mode = model.n_mode, man_per_mode = model.man_per_mode, device = device)
        else:
            man_loss = 0
        
        loss = man_loss + p.TRAJ2CLASS_LOSS_RATIO*traj_loss 
        # 4. Calculating new grdients given the loss value
        loss.backward()
        # 5. Updating the weights
        if p.LR_WU and p.LR_WU_CURRENT_BATCH<=p.LR_WU_BATCHES:
            p.LR_WU_CURRENT_BATCH +=1
            lr = p.LR*p.LR_WU_CURRENT_BATCH/p.LR_WU_BATCHES
            for g in optimizer.param_groups:
                g['lr'] = lr

        optimizer.step()
    

        batch_traj_loss += traj_loss.data/(len(train_loader))
        if p.MAN_DEC_OUT:
            batch_man_loss += man_loss.data/(len(train_loader))
            batch_mode_ploss += mode_ploss.data/(len(train_loader))
            batch_man_ploss += man_ploss.data/(len(train_loader))
            batch_time_ploss += time_ploss.data/(len(train_loader))

            avg_man_loss += man_loss.data/(len(train_loader))
            avg_mode_ploss += mode_ploss.data/(len(train_loader))
            avg_man_ploss += man_ploss.data/(len(train_loader))
            avg_time_ploss += time_ploss.data/(len(train_loader))


        batch_loss += loss.data/(len(train_loader))
        avg_loss += loss.data/(len(train_loader))
        
        if (batch_idx+1) % 500 == 0:
            print('Training Epoch: {}, Batch: {}, Avg Loss: {}, Man Loss:{}, Traj Loss: {}'.format(epoch, batch_idx+1, batch_loss, batch_man_loss , batch_traj_loss) )
            print('Mode Partial loss: {}, Man Partial loss: {}, Time Partial Loss: {}'.format(batch_mode_ploss, batch_man_ploss, batch_time_ploss))
            
            batch_mode_ploss = 0
            batch_man_ploss = 0
            batch_time_ploss = 0
            batch_loss = 0
            batch_man_loss = 0
            batch_traj_loss = 0
        end = time()
        model_time += end-start
        
        
    all_end = time()
    all_time = all_end - all_start
    
    scheduler.step()
    
    # Validation Phase on train dataset
    if calc_train_acc == True:
        raise('Depricated')
    return avg_loss, avg_man_loss, avg_traj_loss, avg_mode_ploss, avg_man_ploss, avg_time_ploss
        

def eval_model(p, tb, model, man_loss_func, traj_loss_func, test_loader, test_dataset, epoch, device, eval_type = 'Validation', vis_data_path = None, figure_name = None):
    total = len(test_loader.dataset)
    num_batch = int(np.floor(total/model.batch_size))
    # Initialise Variables
    avg_loss = 0
    avg_man_loss = 0
    avg_mode_ploss = 0
    avg_man_ploss = 0
    avg_time_ploss = 0
    avg_traj_loss = 0
    time_counter = 0
    average_time = 0
    plot_dicts = []
    
    all_traj_preds = np.zeros(((num_batch*model.batch_size), p.TGT_SEQ_LEN, 2))
    all_traj_labels = np.zeros(((num_batch*model.batch_size), p.TGT_SEQ_LEN + p.IN_SEQ_LEN,2))
    all_man_preds = np.zeros(((num_batch*model.batch_size), p.TGT_SEQ_LEN))
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
                'traj_preds': np.zeros((batch_size, model.n_mode, p.TGT_SEQ_LEN, 2)),
                'traj_dist_preds': np.zeros((batch_size, model.n_mode, p.TGT_SEQ_LEN, 5)),
                'man_labels':np.zeros((batch_size, p.TGT_SEQ_LEN+p.IN_SEQ_LEN)),
                'man_preds':np.zeros((batch_size, model.n_mode, p.TGT_SEQ_LEN)),
                'mode_prob': np.zeros((batch_size, model.n_mode))     
            }
        data_tuple = [data.to(device) for data in data_tuple]
        
        tv_traj_data = data_tuple[-1]
        initial_traj = tv_traj_data[:,(p.IN_SEQ_LEN-1):p.IN_SEQ_LEN]
        traj_gt = tv_traj_data[:,p.IN_SEQ_LEN:(p.IN_SEQ_LEN+p.TGT_SEQ_LEN)] 
        
        in_data_tuple = data_tuple[:-1]
        encoder_input = in_data_tuple[0]
        
        
        labels = labels.to(device) # [0:IN_SEQ_LEN+p.TGT_SEQ_LEN]
        dec_man_gt = labels[:, p.IN_SEQ_LEN:]
        man_gt_onehot = F.one_hot(labels, num_classes= 3)

        st_time = time()
        
            
            
        if p.SELECTED_MODEL== 'CONSTANT_PARAMETER':
            output_dict = model(encoder_input, test_dataset.states_min, test_dataset.states_max, test_dataset.output_states_min, test_dataset.output_states_max, traj_labels = None)
            traj_pred = output_dict['traj_pred']
            man_pred = output_dict['man_pred'] # All zero vector for this model
            man_pred = torch.unsqueeze(man_pred, dim = 1)
            #print(traj_pred.size())
            traj_pred = traj_pred.unsqueeze(1)
            #print(traj_pred.size())
            BM_predicted_data_dist = traj_pred[:,0]
            BM_traj_pred = traj_pred[:,0]
        else:
            encoder_out = model.encoder_forward(x = encoder_input)
            man_pred = model.man_decoder_forward(encoder_out)
            mode_prob, man_vectors = kpis.calc_man_vectors(man_pred, model.n_mode, model.man_per_mode, p.TGT_SEQ_LEN, device)
            '''
            traj_preds = []
            data_dist_preds = []
            for mode_itr in range(model.n_mode):
                man_pred_vector = man_vectors[:,mode_itr]
                decoder_input = initial_traj
                decoder_input = torch.stack([decoder_input,decoder_input,decoder_input], dim = 1) #multi-modal
                predicted_data_dist, traj_pred = trajectory_inference(p, model, device, decoder_input, encoder_out, man_pred_vector)
                traj_preds.append(traj_pred)
                data_dist_preds.append(predicted_data_dist)
            traj_preds = torch.stack(traj_preds, dim = 1)
            data_dist_preds = torch.stack(data_dist_preds, dim =1)
            '''
            BM_man_vector = kpis.sel_high_prob_man( man_pred, model.n_mode, model.man_per_mode, p.TGT_SEQ_LEN, device)
            decoder_input = initial_traj
            decoder_input = torch.stack([decoder_input,decoder_input,decoder_input], dim = 1) #multi-modal
            BM_predicted_data_dist, BM_traj_pred = trajectory_inference(p, model, device, decoder_input, encoder_out, BM_man_vector)
                
                
        
        end_time = time()-st_time
        #print_shape('traj_gt', traj_gt)
        #print_shape('predicted_data_dist', predicted_data_dist)
        traj_loss = traj_loss_func(BM_predicted_data_dist, traj_gt)/len(test_loader) 
        if p.MAN_DEC_OUT:
            man_loss, mode_ploss, man_ploss,time_ploss = man_loss_func(man_pred, dec_man_gt, n_mode = model.n_mode, man_per_mode = model.man_per_mode, device = device, test_phase = True)
        else:
            man_loss = 0
            mode_ploss = 0
            man_ploss = 0
            time_ploss = 0
        loss =  man_loss + p.TRAJ2CLASS_LOSS_RATIO*traj_loss 


        if eval_type == 'Test':
            traj_min = test_dataset.output_states_min
            traj_max = test_dataset.output_states_max
            x_y_dist_pred = BM_predicted_data_dist
            #print_shape('predicted_data_dist', predicted_data_dist)
            #exit()
            #x_y_pred = traj_preds[:,:,:,:2]
            x_y_label = data_tuple[-1]
            #unnormalised_traj_pred = x_y_pred.cpu().data*(traj_max-traj_min) + traj_min
            unnormalised_traj_label = x_y_label.cpu().data*(traj_max-traj_min) + traj_min
            plot_dict['input_features'] = encoder_input
            plot_dict['traj_labels'] = unnormalised_traj_label.numpy()
            #plot_dict['traj_preds']= unnormalised_traj_pred.numpy() #TODO: remove traj pred and use traj_dist_preds instead
            plot_dict['traj_dist_preds'] = x_y_dist_pred.cpu().data.numpy()# TODO: export unnormalised dist
            if p.SELECTED_MODEL != 'CONSTANT_PARAMETER':
                plot_dict['man_labels'] = labels.cpu().data.numpy()
                plot_dict['man_preds'] = man_vectors.cpu().numpy()
                plot_dict['mode_prob'] = mode_prob.detach().cpu().numpy()
            
                
        all_traj_preds[(batch_idx*model.batch_size):((batch_idx+1)*model.batch_size)] = BM_traj_pred.cpu().data
        all_traj_labels[(batch_idx*model.batch_size):((batch_idx+1)*model.batch_size)] = data_tuple[-1].cpu().data
        
        all_man_labels[(batch_idx*model.batch_size):((batch_idx+1)*model.batch_size)] = man_gt_onehot.cpu().data

        avg_traj_loss += traj_loss.cpu().data / len(test_loader)
        avg_loss =  avg_loss + loss.cpu().data / len(test_loader)
        
        if p.MAN_DEC_OUT:
            avg_man_loss += man_loss.data/(len(test_loader))
            avg_mode_ploss += mode_ploss.data/((len(test_loader)))
            avg_man_ploss += man_ploss.data/((len(test_loader)))
            avg_time_ploss += time_ploss.data/((len(test_loader)))
        else:
            avg_man_loss = 0
            avg_mode_ploss = 0
            avg_man_ploss = 0
            avg_time_ploss = 0
        
        

        time_counter += 1
        average_time +=end_time
        if eval_type == 'Test':
            plot_dicts.append(plot_dict)
        if (batch_idx+1) % 500 == 0:
            print('Epoch: ',epoch, ' Batch: ', batch_idx+1)
            
            
        
    
  
    traj_metrics, man_metrics, traj_df = kpis.calc_metric(p, all_traj_preds, all_traj_labels, all_man_preds, all_man_labels, test_dataset.output_states_min, test_dataset.output_states_max, epoch, eval_type = eval_type, figure_name = figure_name)
    accuracy, precision, recall = man_metrics
    f1=0
    FPR=0
    auc=0
    max_j=0 
    rmse = traj_metrics[0:3]
    fde = traj_metrics [3]
    print('                                   ')
    print("{}: Epoch: {}, Accuracy: {:.2f}%, Total LOSS: {}, MAN LOSS: {},MODE PLOSS: {},MAN PLOSS: {},TIME LOSS: {},TRAJ LOSS: {}, PRECISION:{}, RECALL:{}, F1:{}, FPR:{}, AUC:{}, Max J:{}, RMSE:{}, FDE:{}".format(
        eval_type, epoch, 100. * accuracy, avg_loss, avg_man_loss, avg_mode_ploss, avg_man_ploss, avg_time_ploss, avg_traj_loss, precision, recall, f1, FPR, auc, max_j, rmse, fde))
    print('                                   ')
    print('-----------------------------------')
    print('                                   ')
    print(traj_df)
    
    if eval_type == 'Test':
        
        with open(vis_data_path, "wb") as fp:
            pickle.dump(plot_dicts, fp)
    avg_losses = avg_loss, avg_man_loss, avg_traj_loss, avg_mode_ploss, avg_man_ploss, avg_time_ploss
    return accuracy, avg_losses, auc, max_j, precision, recall, f1, rmse, fde, traj_df

def trajectory_inference(p, model, device, decoder_input, encoder_out, man_pred_vector):
    for out_seq_itr in range(p.TGT_SEQ_LEN):
        #output_dict = model(x = encoder_input, y =decoder_input, y_mask = model.get_y_mask(decoder_input.size(2)).to(device))
        traj_pred = model.traj_decoder_forward(y = decoder_input, 
                                                    y_mask = model.get_y_mask(decoder_input.size(2)).to(device), 
                                                    encoder_out = encoder_out)
        
        current_traj_pred = traj_pred[:,:,out_seq_itr:(out_seq_itr+1)] #traj output at current timestep
        current_man_pred = man_pred_vector[:,out_seq_itr:(out_seq_itr+1)]      
        
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
        
        current_decoder_input = current_traj_pred 
        current_decoder_input = torch.unsqueeze(current_decoder_input, dim = 1)
        current_decoder_input = torch.cat([current_decoder_input, current_decoder_input, current_decoder_input], dim = 1)
        
        decoder_input = torch.cat((decoder_input, current_decoder_input), dim = 2)
        
        if out_seq_itr ==0:
            predicted_data_dist = traj_pred[np.arange(traj_pred.shape[0]),manouvre_index, out_seq_itr:(out_seq_itr+1)]
        else:
            predicted_data_dist = torch.cat((predicted_data_dist, traj_pred[np.arange(traj_pred.shape[0]),manouvre_index, out_seq_itr:(out_seq_itr+1)]), dim=1)
        #print_shape('predicted_data_dist', predicted_data_dist)
    traj_pred = decoder_input[:,:,1:,:2]
    #print(traj_pred)
    #print(predicted_data_dist[:,:,:,:2])
    #exit()
    traj_pred = traj_pred[:,0]
    return predicted_data_dist, traj_pred

def multi_modal_inference(p, model, test_loader, test_dataset, epoch, device, eval_type = 'Validation', vis_data_path = None, figure_name = None):
    return 0

def extract_modes(p, man_prob, prob_thr = 0.33, derivative_thr =0.5):
    # man_prob dim = [batch size, tgt_seq_len, 3]
    batch_size = man_prob.shape[0]
    man_prob_deriv = np.zeros_like(man_prob) 
    
    xat = lambda t: man_prob[:,2+t:p.TGT_SEQ_LEN+t-2]
    man_prob_deriv[:,2:tgt_seq_len-2] = (p.FPS/10)*(-2*xat(-2) 
                                                -1*xat(-1)
                                                +1*xat(1)
                                                +2*xat(2))
    
    high_prob_mans = np.zeros((batch_size,3))
    high_prob_mans = np.any(man_prob>= prob_thr, axis = 1)
    for man_itr in range(3):
        high_prob_mans[:,man_itr] = np.any(man_prob>= prob_thr, axis = 1)
    



