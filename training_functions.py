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
import models_functions as mf
font = {'size'   : 22}
matplotlib.rcParams['figure.figsize'] = (18, 12)
matplotlib.rc('font', **font)


def eval_top_func(p, model_eval_func, model_kpi_func, model, loss_func_tuple, te_dataset, device, tensorboard = None):
    model = model.to(device)
    
    te_loader = utils_data.DataLoader(dataset = te_dataset, shuffle = True, batch_size = p.BATCH_SIZE, drop_last= True, pin_memory= True, num_workers= 12)

    vis_data_path = p.VIS_DIR + p.experiment_tag + '.pickle'
    best_model_path = p.MODELS_DIR + p.experiment_tag + '.pt'
    figure_name =  p.experiment_tag
    
    if p.SELECTED_MODEL != 'CONSTANT_PARAMETER':
        model.load_state_dict(torch.load(best_model_path))
    
    print_dict, kpi_dict = eval_model(p, tensorboard, model_eval_func, model_kpi_func, model, loss_func_tuple, te_loader, te_dataset, ' N/A', device, eval_type = 'Test', vis_data_path = vis_data_path, figure_name=figure_name)
    
    print('Test Losses:\n'+ ''.join(['{}:{}\n'.format(k, print_dict[k]) for k in print_dict]))

    print('Test KPIs:\n'+ ''.join(['{}:{}\n'.format(k, kpi_dict[k]) for k in kpi_dict]))
    
    return kpi_dict


def train_top_func(p, model_train_func, model_eval_func, model_kpi_func, model,loss_func_tuple, optimizer , tr_dataset, val_dataset, device, tensorboard = None):
    
    
    tr_loader = utils_data.DataLoader(dataset = tr_dataset, shuffle = True, batch_size = p.BATCH_SIZE, drop_last= True, pin_memory= True, num_workers= 12)
    val_loader = utils_data.DataLoader(dataset = val_dataset, shuffle = True, batch_size = p.BATCH_SIZE, drop_last= True, pin_memory= True, num_workers= 12)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, p.LR_DECAY_EPOCH, p.LR_DECAY)
    
    best_model_path = p.MODELS_DIR + p.experiment_tag + '.pt'

    if p.LOWER_BETTER_VAL_SCORE:
        best_val_score = float("inf")
    else:
        best_val_score = 0

    patience = p.PATIENCE
    best_epoch = 0
    total_time = 0
    p.LR_WU_CURRENT_BATCH = 0
    for epoch in range(p.NUM_EPOCHS):
        #print("Epoch: {} Started!".format(epoch+1))
        start = time()
        tr_print_dict = train_model(p, tensorboard, model_train_func, model, loss_func_tuple, optimizer, scheduler, tr_loader,  epoch+1, device)
        
        val_start = time()

        val_print_dict, val_kpi_dict = eval_model(p, tensorboard, model_eval_func, model_kpi_func, model, loss_func_tuple, val_loader, val_dataset, epoch+1, device, eval_type = 'Validation')
        val_score = val_print_dict[p.VAL_SCORE]
        val_end = time()
        #print("Validation Accuracy:",val_acc,' Avg Pred Time: ', val_avg_pred_time, " Avg Loss: ", val_loss," at Epoch", epoch+1)
        #if tensorboard != None:   
            #tensorboard.add_scalar('tr_total_loss', tr_loss, epoch+1)   
        if (p.LOWER_BETTER_VAL_SCORE and val_score<best_val_score) or (not p.LOWER_BETTER_VAL_SCORE and val_score>best_val_score):
            best_val_score = val_score
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            patience = p.PATIENCE
        else:
            patience -= 1
        end = time()
        total_time += end-start
        print('Epoch: {}, Total Time: {}, Validation time: {}'.format(epoch+1, end-start, val_end-val_start))     
        print('Best Epoch so far: {}, Validation Score: {}={}'.format(best_epoch+1, p.VAL_SCORE, best_val_score))
        print('Training Metrics:\n'+ ''.join(['{}:{}\n'.format(k,tr_print_dict[k]) for k in tr_print_dict]))
        print('Validation Metrics:\n'+ ''.join(['{}:{}\n'.format(k,val_print_dict[k]) for k  in val_print_dict]))

        if p.DEBUG_MODE == True:
            print('Debugging Mode Active.')
            break

        if patience == 0:
            print(' No performance improvement in Validation data after:', epoch+1, 'Epochs!')
            break
        
        

    result_dic = {
        'EarlyStopping Epoch': best_epoch + 1,
        'Best Validation Loss': best_val_score,
    }
    return result_dic


def train_model(p, tb, model_train_func, model, loss_func_tuple, optimizer, scheduler, train_loader, epoch, device, vis_step = 20):
    # Number of samples with correct classification
    # total size of train data
    # number of batch
    model_time = 0

    all_start = time()
    
    vis_print_dict = []
    print_dict = []
    # Training loop over batches of data on train dataset
    for batch_idx, (data_tuple, labels,_, _) in enumerate(train_loader):
        #print('Batch: ', batch_idx)
        
        if p.DEBUG_MODE == True:
            if batch_idx >3: ##Uncoment for debuggering
                break
        
        data_tuple = [data.to(device) for data in data_tuple]
        label_tuple = (labels.to(device),)
        
        # 1. Clearing previous gradient values.
        optimizer.zero_grad()
        
        # 2. Run the Model 
        #print(model_train_func)
        loss, batch_print_info_dict = model_train_func(p, data_tuple, label_tuple, model, loss_func_tuple, device)

        # 3. Calculating new grdients given the loss value
        loss.backward()

        # 4. Updating the weights
        if p.LR_WU and p.LR_WU_CURRENT_BATCH<=p.LR_WU_BATCHES:
            p.LR_WU_CURRENT_BATCH +=1
            lr = p.LR*p.LR_WU_CURRENT_BATCH/p.LR_WU_BATCHES
            for g in optimizer.param_groups:
                g['lr'] = lr

        optimizer.step()
        # For epoch level printing
        if batch_idx == 0:
            print_dict = batch_print_info_dict
        else:
            for k in print_dict:
                print_dict[k] += batch_print_info_dict[k]/len(train_loader)
        
        # Every X batch print vis_print_dict
        if batch_idx % 500 == 0:
            if batch_idx !=0:
                print('Training Epoch: {}, Batch: {}/{}\n'.format(epoch, int(batch_idx/500), int(len(train_loader)/500))+ ''.join(['{}:{}\n'.format(k,vis_print_dict[k]) for k in vis_print_dict]))
            vis_print_dict = batch_print_info_dict
        else:
            for k in vis_print_dict:
                vis_print_dict[k] += batch_print_info_dict[k]/500
           
    all_end = time()
    all_time = all_end - all_start
    
    scheduler.step()
    
    return print_dict
        
def eval_model(p, tb, model_eval_func, model_kpi_func, model, loss_func_tuple, test_loader, test_dataset, epoch, device, eval_type = 'Validation', vis_data_path = None, figure_name = None):
    total = len(test_loader.dataset)
    num_batch = int(np.floor(total/model.batch_size))
    # Initialise Variables
    
    plot_dicts = []
    print_dict = {}
    kpi_input_dict = {}
   
    for batch_idx, (data_tuple, labels, plot_info, _) in enumerate(test_loader):
        
        if p.DEBUG_MODE == True:
            if batch_idx >100: 
                break
        
        
        data_tuple = [data.to(device) for data in data_tuple]
        label_tuple = (labels.to(device),)
        
        batch_print_info_dict, batch_kpi_input_dict = model_eval_func(p, data_tuple, plot_info, test_dataset, label_tuple, model, loss_func_tuple, device, eval_type)

            
        if batch_idx == 0:
            for k in batch_kpi_input_dict:
                kpi_input_dict[k] = [batch_kpi_input_dict[k]]
            for k in batch_print_info_dict:
                print_dict[k] = batch_print_info_dict[k]
        else:
            for k in batch_kpi_input_dict:
                kpi_input_dict[k].append(batch_kpi_input_dict[k])
            
            for k in batch_print_info_dict:
                print_dict[k] += batch_print_info_dict[k]/len(test_loader)
        
        if (batch_idx+1) % 500 == 0:
            print('Epoch: ',epoch, ' Batch: ', batch_idx+1, '/{}'.format(len(test_loader)))
               
    kpi_dict = model_kpi_func(p, kpi_input_dict, test_dataset.output_states_min, test_dataset.output_states_max, figure_name)
   
    if eval_type == 'Test':
        with open(vis_data_path, "wb") as fp:
            pickle.dump(kpi_input_dict, fp)
    
    return print_dict, kpi_dict


    



