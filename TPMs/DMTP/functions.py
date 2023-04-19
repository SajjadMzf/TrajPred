import torch
import torch.nn.functional as F
import numpy as np
from time import time
from ..MMnTP import utils

def DMTP_training(p, data_tuple, label_tuple, model, loss_func_tuple, device):
    traj_loss_func = loss_func_tuple[0]
    mode_loss_func = loss_func_tuple[1]
    #man_data = label_tuple[0]
    #man_data_onehot = F.one_hot(man_data, num_classes= 3)
    #man_input = man_data_onehot[:,(p.IN_SEQ_LEN-1):(p.IN_SEQ_LEN-1+p.TGT_SEQ_LEN)]
    #man_gt = man_data[:, p.IN_SEQ_LEN:(p.IN_SEQ_LEN+p.TGT_SEQ_LEN)]

    traj_data = data_tuple[-1]
    traj_input = traj_data[:,(p.IN_SEQ_LEN-1):(p.IN_SEQ_LEN-1+p.TGT_SEQ_LEN) ] 
    traj_gt = traj_data[:,p.IN_SEQ_LEN:(p.IN_SEQ_LEN+p.TGT_SEQ_LEN)]
    
    decoder_input = traj_input
    feature_data = data_tuple[0]
    
    output_dict = model(x = feature_data, y = decoder_input, 
                        y_mask = utils.get_y_mask(p.TGT_SEQ_LEN).to(device))
    traj_pred = output_dict['traj_pred']
    mode_prob_pred = output_dict['mode_prob_pred']
    
    total_samples = traj_pred.shape[0]
    n_mode = traj_pred.shape[1]
    fde = np.zeros((total_samples, n_mode))
    for i in range(n_mode):
        fde[:,i] = \
            np.sum(np.absolute(traj_pred[:,i,-1,:2].cpu().detach().numpy()-\
                               traj_gt[:,-1,:].cpu().detach().numpy()), axis=-1)
    
    best_mode = np.argmin(fde, axis = 1)
   
    
    traj_pred = traj_pred[np.arange(traj_pred.shape[0]),best_mode]
    
    traj_loss = traj_loss_func(traj_pred, traj_gt)
    mode_loss_func = mode_loss_func()
    mode_loss = mode_loss_func(mode_prob_pred, torch.from_numpy(best_mode).to(device))
    
    
    training_loss = mode_loss + p.TRAJ2CLASS_LOSS_RATIO*traj_loss 
    batch_print_info_dict = {
        'Total Loss': training_loss.cpu().data.numpy()/model.batch_size,
        'Traj Loss': traj_loss.cpu().data.numpy()/model.batch_size,
        'Mode Loss': mode_loss.cpu().data.numpy()/model.batch_size,
        'Best Mode_histogram': best_mode
    }
    return training_loss, batch_print_info_dict

def DMTP_evaluation(p, data_tuple, plot_info, dataset, \
                    label_tuple, model, loss_func_tuple, device, eval_type):
    (tv_id, frames, data_file) = plot_info
    
    traj_loss_func = loss_func_tuple[0]
    mode_loss_func = loss_func_tuple[1]

    #man_data = label_tuple[0]
    #man_gt = man_data[:, p.IN_SEQ_LEN:(p.IN_SEQ_LEN+p.TGT_SEQ_LEN)]
    
    traj_data = data_tuple[-1]
    traj_initial_input = traj_data[:,(p.IN_SEQ_LEN-1):p.IN_SEQ_LEN] 
    traj_gt = traj_data[:,p.IN_SEQ_LEN:(p.IN_SEQ_LEN+p.TGT_SEQ_LEN)]
    traj_track_gt = traj_data[:,:(p.IN_SEQ_LEN+p.TGT_SEQ_LEN)]
    feature_data = data_tuple[0]
    with torch.no_grad():
        encoder_out = model.encoder_forward(x = feature_data)
        mode_prob_pred = model.mode_decoder_forward(encoder_out)
    mode_prob_pred = F.softmax(mode_prob_pred, dim = 1)
    decoder_input = traj_initial_input
    hp_mode = np.argmax(mode_prob_pred.cpu().detach().numpy(), axis = 1)

    BM_predicted_data_dist, BM_traj_pred = \
        DMT_trajectory_inference(p, model, device, decoder_input, encoder_out, hp_mode)
    
   
    

    traj_loss = traj_loss_func(BM_predicted_data_dist, traj_gt)
    
    #TODO: calculating mode loss requires multi modal evaluation
    evaluation_loss =  p.TRAJ2CLASS_LOSS_RATIO*traj_loss 
    
    # Trajectory inference for all modes!
    if eval_type == 'Test' and p.MULTI_MODAL_EVAL == True:
        traj_preds = []
        data_dist_preds = []
        for mode_itr in range(model.n_mode):
            selected_mode = np.ones_like(hp_mode)*mode_itr
            decoder_input = traj_initial_input
            predicted_data_dist, traj_pred = \
                DMT_trajectory_inference(p, model, device, decoder_input,\
                                          encoder_out, selected_mode)
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
    }

    batch_kpi_input_dict = {    
        'data_file': data_file,
        'tv': tv_id.numpy(),
        'frames': frames.numpy(),
        'traj_min': dataset.output_states_min,
        'traj_max': dataset.output_states_max,  
        'input_features': feature_data.cpu().data.numpy(),
        'traj_track_gt': traj_track_gt.cpu().data.numpy(),
        'traj_gt': traj_gt.cpu().data.numpy(),
        'traj_dist_preds': data_dist_preds.cpu().data.numpy(),
        'mode_prob': mode_prob_pred.detach().cpu().data.numpy(),
    }
    return batch_print_info_dict, batch_kpi_input_dict



def DMT_trajectory_inference(p, model, device, decoder_input, encoder_out, selected_mode):
    #decoder input [batch size, seq_len, feature size=2]
    for out_seq_itr in range(p.TGT_SEQ_LEN):
        #output_dict = model(x = encoder_input,\
        #  y =decoder_input, y_mask = utils.get_y_mask(decoder_input.size(2)).to(device))
        
        traj_pred = \
            model.traj_decoder_forward(y = decoder_input, 
                                        y_mask = utils.get_y_mask(decoder_input.size(1)).to(device), 
                                        encoder_out = encoder_out)
        #traj_pred = [batch_size, n_mode, seq_len, feature size = 5]
        current_decoder_input = \
            traj_pred[np.arange(traj_pred.shape[0]),\
                      selected_mode,out_seq_itr:(out_seq_itr+1),:2] #traj output at current timestep
        
        decoder_input = torch.cat((decoder_input, current_decoder_input), dim = 1)
        
        if out_seq_itr ==0:
            predicted_data_dist = \
                traj_pred[np.arange(traj_pred.shape[0]),selected_mode, out_seq_itr:(out_seq_itr+1)]
        else:
            predicted_data_dist = \
                torch.cat((predicted_data_dist, 
                           traj_pred[np.arange(traj_pred.shape[0]),
                                     selected_mode, out_seq_itr:(out_seq_itr+1)]), dim=1)
        #print_shape('predicted_data_dist', predicted_data_dist)
    traj_pred = decoder_input[:,1:,:2]
    
    return predicted_data_dist, traj_pred