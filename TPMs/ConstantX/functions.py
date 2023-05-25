import torch
import torch.nn.functional as F
import numpy as np
from time import time
import pdb

def CONSTANTX_training(p, data_tuple, man, model, dataset, loss_func_tuple, device):
    print('ConstantX does not require training!')
    exit()


def CONSTANTX_deploy(p, data_tuple, plot_info, dataset, model, device):
    (tv_id, frames, data_file) = plot_info
    
    traj_data = data_tuple[-1]
    
    traj_gt = traj_data[:,p.MAX_IN_SEQ_LEN:]
    feature_data = data_tuple[0]
    input_padding_mask = data_tuple[1]
    with torch.no_grad():
        traj_pred = model.forward(x = feature_data, 
                                    input_padding_mask = input_padding_mask,
                                    states_min = dataset.states_min,
                                    states_max = dataset.states_max,
                                    output_states_min = dataset.output_states_min,
                                    output_states_max = dataset.output_states_max),
    
    
    traj_pred = traj_pred[0]
    # Trajectory inference for all modes!
    unnormalised_traj_pred = traj_pred.cpu().data.numpy()
    traj_gt = traj_gt.cpu().data.numpy()
    traj_max = dataset.output_states_max
    traj_min = dataset.output_states_min
    unnormalised_traj_pred = unnormalised_traj_pred*(traj_max-traj_min) + traj_min 
    unnormalised_traj_pred = np.cumsum(unnormalised_traj_pred, axis = 1)
    unnormalised_traj_gt = traj_gt*(traj_max-traj_min) + traj_min
    unnormalised_traj_gt = np.cumsum(unnormalised_traj_gt, axis = 1)
    batch_export_dict = {    
        'data_file': data_file,
        'tv': tv_id.numpy(),
        'frames': frames.numpy(),
        'traj_pred': unnormalised_traj_pred,
        'traj_gt': unnormalised_traj_gt,
    }

    return batch_export_dict

def CONSTANTX_evaluation(p, data_tuple, man, plot_info, dataset, 
                       model, loss_func_tuple, device, eval_type):
    (tv_id, frames, data_file) = plot_info
    
    traj_loss_func = loss_func_tuple[0]
    
    
    traj_data = data_tuple[-1]
    traj_initial_input = traj_data[:,(p.MAX_IN_SEQ_LEN-1):p.MAX_IN_SEQ_LEN] 
    traj_gt = traj_data[:,p.MAX_IN_SEQ_LEN:(p.MAX_IN_SEQ_LEN+p.TGT_SEQ_LEN)]
    
    traj_track_gt = traj_data[:,:(p.MAX_IN_SEQ_LEN+p.TGT_SEQ_LEN)]
    feature_data = data_tuple[0]
    input_padding_mask = data_tuple[1]
    with torch.no_grad():
        traj_pred = model.forward(x = feature_data, 
                                    input_padding_mask = input_padding_mask,
                                    states_min = dataset.states_min,
                                    states_max = dataset.states_max,
                                    output_states_min = dataset.output_states_min,
                                    output_states_max = dataset.output_states_max),
    
    
        
    traj_pred = traj_pred[0]
    
    batch_print_info_dict = {
        'Total Loss': 0,
    }
    #print(traj_gt.shape)
    batch_kpi_input_dict = {    
        'data_file': data_file,
        'tv': tv_id.numpy(),
        'frames': frames.numpy(),
        'traj_min': dataset.output_states_min,
        'traj_max': dataset.output_states_max,  
        'input_features': feature_data.cpu().data.numpy(),
        'input padding mask': input_padding_mask.cpu().data.numpy().astype(int),
        'traj_gt': traj_gt.cpu().data.numpy(),
        'traj_track_gt': traj_track_gt.cpu().data.numpy(),
        'traj_dist_preds': traj_pred.data.numpy(),
    }
    return batch_print_info_dict, batch_kpi_input_dict
