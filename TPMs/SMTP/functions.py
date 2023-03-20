import torch
import torch.nn.functional as F
import numpy as np
from time import time

from ..MMnTP import utils
from ..DMTP.functions import DMT_trajectory_inference

def SMTP_training(p, data_tuple, label_tuple, model, loss_func_tuple, device):
    traj_loss_func = loss_func_tuple[0]
    mode_loss_func = loss_func_tuple[1]
    man_data = label_tuple[0]
    #man_data_onehot = F.one_hot(man_data, num_classes= 3)
    #man_input = man_data_onehot[:,(p.IN_SEQ_LEN-1):(p.IN_SEQ_LEN-1+p.TGT_SEQ_LEN)]
    man_gt = man_data[:, p.IN_SEQ_LEN:(p.IN_SEQ_LEN+p.TGT_SEQ_LEN)]
    mode_gt = utils.static_mode_from_man(man_gt.cpu().detach().numpy())
    mode_gt = torch.from_numpy(mode_gt).to(device)
    traj_data = data_tuple[-1]
    traj_input = traj_data[:,(p.IN_SEQ_LEN-1):(p.IN_SEQ_LEN-1+p.TGT_SEQ_LEN) ] 
    traj_gt = traj_data[:,p.IN_SEQ_LEN:(p.IN_SEQ_LEN+p.TGT_SEQ_LEN)]
    
    feature_data = data_tuple[0]
    
    output_dict = model(x = feature_data, y = traj_input, y_mask = utils.get_y_mask(p.TGT_SEQ_LEN).to(device))
    traj_pred = output_dict['traj_pred']
    mode_prob = output_dict['mode_prob_pred']
    
    
    traj_pred = traj_pred[np.arange(traj_pred.shape[0]), mode_gt]
    
    traj_loss = traj_loss_func(traj_pred, traj_gt)
    
   
    mode_loss_func = mode_loss_func()
    mode_loss = mode_loss_func(mode_prob, mode_gt)
    training_loss = mode_loss + p.TRAJ2CLASS_LOSS_RATIO*traj_loss 
    batch_print_info_dict = {
        'Total Loss': training_loss.cpu().data.numpy()/model.batch_size,
        'Traj Loss': traj_loss.cpu().data.numpy()/model.batch_size,
        'Mode Loss': mode_loss.cpu().data.numpy()/model.batch_size,
    }
    return training_loss, batch_print_info_dict

def SMTP_evaluation(p, data_tuple, plot_info, dataset, label_tuple, model, loss_func_tuple, device, eval_type):
    (tv_id, frames, data_file) = plot_info
    
    traj_loss_func = loss_func_tuple[0]
    mode_loss_func = loss_func_tuple[1]

    man_data = label_tuple[0]
    man_gt = man_data[:, p.IN_SEQ_LEN:(p.IN_SEQ_LEN+p.TGT_SEQ_LEN)]
    mode_gt = utils.static_mode_from_man(man_gt.cpu().detach().numpy())
    mode_gt = torch.from_numpy(mode_gt).to(device)

    traj_data = data_tuple[-1]
    traj_initial_input = traj_data[:,(p.IN_SEQ_LEN-1):p.IN_SEQ_LEN] 
    traj_gt = traj_data[:,p.IN_SEQ_LEN:(p.IN_SEQ_LEN+p.TGT_SEQ_LEN)]
    traj_track_gt = traj_data[:,:(p.IN_SEQ_LEN+p.TGT_SEQ_LEN)]
    feature_data = data_tuple[0]
    with torch.no_grad():
        encoder_out = model.encoder_forward(x = feature_data)
        mode_prob = model.mode_decoder_forward(encoder_out)
    mode_loss_func = mode_loss_func()
    mode_loss = mode_loss_func(mode_prob, mode_gt)
    
    mode_prob = F.softmax(mode_prob,dim = 1) #softmax should be after applying loss function
    hp_mode = np.argmax(mode_prob.cpu().detach().numpy(), axis = 1)
    
    BM_predicted_data_dist, BM_traj_pred = DMT_trajectory_inference(p, model, device, traj_initial_input, encoder_out, hp_mode)

    traj_loss = traj_loss_func(BM_predicted_data_dist, traj_gt)
    
    evaluation_loss =  mode_loss + p.TRAJ2CLASS_LOSS_RATIO*traj_loss 
    
    # Trajectory inference for all modes!
    if eval_type == 'Test' and p.MULTI_MODAL_EVAL == True:
        traj_preds = []
        data_dist_preds = []
        for mode_itr in range(model.n_mode):
            selected_mode = np.ones_like(hp_mode)*mode_itr
            predicted_data_dist, traj_pred = DMT_trajectory_inference(p, model, device, traj_initial_input, encoder_out, selected_mode)
            traj_preds.append(traj_pred)
            data_dist_preds.append(predicted_data_dist)
        traj_preds = torch.stack(traj_preds, dim = 1)
        data_dist_preds = torch.stack(data_dist_preds, dim =1)
    else:
        data_dist_preds = []
        for mode_itr in range(model.n_mode):
            data_dist_preds.append(BM_predicted_data_dist)
        data_dist_preds = torch.stack(data_dist_preds, dim =1)

        
    batch_print_info_dict = {
        'Total Loss': evaluation_loss.cpu().data.numpy()/model.batch_size,
        'Traj Loss': traj_loss.cpu().data.numpy()/model.batch_size,
        'Mode Loss': mode_loss.cpu().data.numpy()/model.batch_size,
    }

    batch_kpi_input_dict = {    
        'data_file': data_file,
        'tv': tv_id.numpy(),
        'frames': frames.numpy(),
        'traj_min': dataset.output_states_min,
        'traj_max': dataset.output_states_max,  
        'input_features': feature_data.cpu().data.numpy(),
        'traj_gt': traj_gt.cpu().data.numpy(),
        'traj_track_gt': traj_track_gt.cpu().data.numpy(),
        'traj_dist_preds': data_dist_preds.cpu().data.numpy(),
        'mode_gt': mode_gt.cpu().data.numpy(),
        'mode_prob': mode_prob.detach().cpu().data.numpy(),
    }
    return batch_print_info_dict, batch_kpi_input_dict

