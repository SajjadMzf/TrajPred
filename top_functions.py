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

import kpis
import export
font = {'size'   : 22}
matplotlib.rcParams['figure.figsize'] = (18, 12)
matplotlib.rc('font', **font)

def deploy_top_func(p, model_deploy_func, model, de_dataset, device):
    model = model.to(device)
    de_loader = utils_data.DataLoader(dataset = de_dataset, shuffle = False,
                                       batch_size = p.BATCH_SIZE, 
                                       drop_last= False, 
                                       num_workers= 0)
    vis_data_path = p.VIS_DIR + p.experiment_tag + '.pickle'
    best_model_path = p.WEIGHTS_DIR + p.experiment_tag + '.pt'
    figure_name =  p.experiment_tag
    if p.SELECTED_MODEL != 'CONSTANT_PARAMETER':
        model.load_state_dict(torch.load(best_model_path))
    export_dict = deploy_model(p, model, model_deploy_func, de_loader, de_dataset,
                                device, vis_data_path = vis_data_path, figure_name=figure_name)
    return export_dict

def eval_top_func(p, model_eval_func, model_kpi_func, model, loss_func_tuple,
                   te_dataset, device, tensorboard = None):
    model = model.to(device)
    
    te_loader = utils_data.DataLoader(dataset = te_dataset, shuffle = True, 
                                      batch_size = p.BATCH_SIZE, drop_last= True,
                                        pin_memory= True, num_workers= 0)

    vis_data_path = p.VIS_DIR + p.experiment_tag + '.pickle'
    best_model_path = p.WEIGHTS_DIR + p.experiment_tag + '.pt'
    figure_name =  p.experiment_tag
    
    if p.SELECTED_MODEL != 'CONSTANT_PARAMETER':
        model.load_state_dict(torch.load(best_model_path))
    
    print_dict, kpi_dict = eval_model(p, tensorboard, model_eval_func, 
                                      model_kpi_func, model, loss_func_tuple, te_loader, te_dataset,
                                        ' N/A', device, eval_type = 'Test', 
                                        vis_data_path = vis_data_path, figure_name=figure_name)
    
    print(' ************ TEST KPIs  ************ :\n'+ ''.join(['{}:{}\n'.format(k, print_dict[k]) for k in print_dict]))
    for k in kpi_dict:
        if 'histogram' not in k:
            print(''.join('{}:{}'.format(k,kpi_dict[k])))
    return kpi_dict


def train_top_func(p, model_train_func, model_eval_func, model_kpi_func, 
                   model,loss_func_tuple, optimizer , tr_dataset, val_dataset,
                     device,prev_best_model = None, prev_itr = 1, tensorboard = None):
    
    tr_loader = utils_data.DataLoader(dataset = tr_dataset, 
                                      shuffle = True, 
                                      batch_size = p.BATCH_SIZE,
                                      drop_last= True,
                                      num_workers= 0)
    val_loader = utils_data.DataLoader(dataset = val_dataset, 
                                       shuffle = True, 
                                       batch_size = p.BATCH_SIZE, 
                                       drop_last= True, 
                                       num_workers= 0)
    
    
    
    if prev_best_model is not None:
        model.load_state_dict(torch.load(prev_best_model))
    
    best_model_path = p.WEIGHTS_DIR + p.experiment_tag + '.pt'

    if p.LOWER_BETTER_VAL_SCORE:
        best_val_score = float("inf")
    else:
        best_val_score = 0

    
    best_itr = 0
    p.LR_WU_CURRENT_BATCH = 0
    training_start = time()
    
    
    for itr in range(prev_itr, p.NUM_ITRS):
        print('ITR:{}/{}'.format(itr, p.NUM_ITRS))
        start = time()
        tr_print_dict, lr = train_step(p, model_train_func, model, 
                                    loss_func_tuple, optimizer, 
                                    tr_dataset, tr_loader,itr, device)
        
        print('Training Metrics:\n'+ ''.join(['{}:{}\n'.format(k,tr_print_dict[k])\
                                               for k in tr_print_dict if 'histogram' not in k]))
            
        print('LR:{}'.format(lr))
        
        if itr%p.VAL_FREQ==0 or p.DEBUG_MODE == True:
            print("Validating at itr {}/{} ...".format(itr, p.NUM_ITRS))
            val_start = time()
            val_print_dict, val_kpi_dict = eval_model(p, tensorboard, 
                                                      model_eval_func, 
                                                      model_kpi_func, 
                                                      model, 
                                                      loss_func_tuple, 
                                                      val_loader,
                                                      val_dataset,
                                                      itr,
                                                      device, 
                                                      eval_type = 'Validation')
            if p.VAL_SCORE in val_print_dict:
                val_score = val_print_dict[p.VAL_SCORE]
            else:
                val_score = val_kpi_dict[p.VAL_SCORE]
            val_end = time()
            
            
            if (p.LOWER_BETTER_VAL_SCORE and val_score<best_val_score) \
                or (not p.LOWER_BETTER_VAL_SCORE and val_score>best_val_score):
                best_val_score = val_score
                best_itr = itr
                torch.save(model.state_dict(), best_model_path)
                

            val_print_dict['Validation Time'] = val_end-val_start
            val_print_dict['Best ITR'] = best_itr
            val_print_dict['Best Val Score ({})'.format(p.VAL_SCORE)] = best_val_score
              
            print('Validation Metrics:\n'+ ''.join(['{}:{}\n'\
                                                    .format(k,val_print_dict[k]) for k  in val_print_dict]))
            
           
            for k in val_print_dict:
                tensorboard.add_scalar('Validation_itr_' + k, val_print_dict[k], itr)
            
            print(' ************ Validation KPIs ************:\n')
            for k in val_kpi_dict:
                print(k)
                if 'histogram' not in k:
                    print(''.join('{}:{}'.format(k,val_kpi_dict[k])))
            
                if 'histogram' in k:
                    tensorboard.add_histogram('Validation_' + k, val_kpi_dict[k], itr)
                elif 'rmse_table' in k:
                    for i in range(p.MAX_IN_SEQ_LEN):
                        tensorboard.add_scalar('Validation_' + k + str(i), val_kpi_dict[k][0,i], itr)
                elif ('group' not in k) and ('min' not in k) and('mnll' not in k) and ('list' not in k) :
                    tensorboard.add_scalar('Validation_' + k, val_kpi_dict[k], itr)
            
            print('************ Timings ************:\n')
            print('Total Time: {}'.format(time()-training_start, ))

            if p.DEBUG_MODE == True:
                print('Debugging Mode Active.')
                break

        end = time()
        
        tr_print_dict['Total Time'] = end-start
        tr_print_dict['Itr'] = itr  
        tensorboard.add_scalar('LR',lr, itr)
        for k in tr_print_dict:
            if 'histogram' in k:
                tensorboard.add_histogram('Train_itr_' + k, tr_print_dict[k], itr)
            else:
                tensorboard.add_scalar('Train_itr_' + k, tr_print_dict[k], itr)      

    result_dic = {
        'Best Itr': best_itr,
        'Best Validation Loss': best_val_score,
    }
    return result_dic


def train_step(p, model_train_func, model, loss_func_tuple, 
                optimizer, train_dataset, train_loader, 
                itr, device):
    
    model.train()
    # Training loop over batches of data on train dataset
    try:
        (data_tuple, man,_) = next(train_loader)
    except:
        data_itr = iter(train_loader)
        (data_tuple, man,_) = next(data_itr)
    
    
    data_tuple = [data.to(device) for data in data_tuple]
    man = man.to(device)
    # 1. Clearing previous gradient values.
    optimizer.zero_grad()
    
    # 2. Run the Model 
    #print(model_train_func)
    loss, batch_print_info_dict = model_train_func(p, data_tuple, man, model, 
                                                   train_dataset, loss_func_tuple, device)

    # 3. Calculating new grdients given the loss value
    loss.backward()

    # 4. Updating the weights
    
    if p.LR_WU and itr<=p.LR_WU_BATCHES:
        lr = (p.LR*itr)/p.LR_WU_BATCHES
        lr = lr/math.sqrt(p.LR_WU_BATCHES)
        for g in optimizer.param_groups:
            g['lr'] = lr
    else:
        lr = p.LR/math.sqrt(itr)
        for g in optimizer.param_groups:
            g['lr'] = lr
        
    optimizer.step()
    
    return batch_print_info_dict, lr

def deploy_model(p, model, model_deploy_func, de_loader, de_dataset, device, vis_data_path = None, figure_name=None):
    total = len(de_loader.dataset)
    #print('Total test data',total)
    #exit()
    num_batch = int(np.floor(total/model.batch_size))
    # Initialise Variables
    export_dict = {}
    #model.eval()
    for batch_idx, (data_tuple, man, plot_info) in enumerate(de_loader):
        if p.DEBUG_MODE == True:
            if batch_idx >2: 
                break

        data_tuple = [data.to(device) for data in data_tuple]  
        with torch.no_grad():
            batch_export_dict = model_deploy_func(p, data_tuple, plot_info, de_dataset, model, device)
    
        if batch_idx % 500 == 0 and batch_idx !=0:
            print('Deploy Batch: {}/{}'.format(batch_idx, len(de_loader)))
        
        if batch_idx == 0:
            for k in batch_export_dict:
                export_dict[k] = [batch_export_dict[k]]
        else:
            for k in batch_export_dict:
                export_dict[k].append(batch_export_dict[k])    
    return export_dict        
def eval_model(p, tb, model_eval_func, model_kpi_func, model, loss_func_tuple, test_loader, test_dataset, epoch, device, eval_type = 'Validation', vis_data_path = None, figure_name = None):
    total = len(test_loader.dataset)
    #print('Total test data',total)
    #exit()
    num_batch = int(np.floor(total/model.batch_size))
    # Initialise Variables
    
    plot_dicts = []
    print_dict = {}
    kpi_input_dict = {}
    #model.eval()
    
    n_batchs = len(test_loader) if eval_type != 'Validation' else p.MAX_VAL_ITR
    for batch_idx, (data_tuple, man, plot_info) in enumerate(test_loader):
        if eval_type== 'Validation' and batch_idx>=p.MAX_VAL_ITR:
            break
        if p.DEBUG_MODE == True:
            if batch_idx >2: 
                break
        
        data_tuple = [data.to(device) for data in data_tuple]
        man = man.to(device)
        with torch.no_grad():
            batch_print_info_dict, batch_kpi_input_dict = model_eval_func(p, data_tuple, man, plot_info, test_dataset, model, loss_func_tuple, device, eval_type)
    
        if batch_idx == 0:
            for k in batch_kpi_input_dict:
                kpi_input_dict[k] = [batch_kpi_input_dict[k]]
            for k in batch_print_info_dict:
                print_dict[k] = batch_print_info_dict[k]
        else:
            for k in batch_kpi_input_dict:
                kpi_input_dict[k].append(batch_kpi_input_dict[k])
            
            for k in batch_print_info_dict:
                print_dict[k] += batch_print_info_dict[k]/n_batchs
        
        if (batch_idx+1) % 20 == 0:
            print('Epoch: ',epoch, ' Batch: ', batch_idx+1, '/{}'.format(n_batchs))
               
    kpi_dict = model_kpi_func(p, kpi_input_dict, test_dataset.output_states_min, test_dataset.output_states_max, figure_name)
    if eval_type == 'Test':
        with open(vis_data_path, "wb") as fp:
            pickle.dump(kpi_input_dict, fp)
    
    return print_dict, kpi_dict