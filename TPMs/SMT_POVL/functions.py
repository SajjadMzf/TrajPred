import torch
import torch.nn.functional as F
import numpy as np
from time import time
from . import utils

def SMTPOVL_training(p, data_tuple, label_tuple, model,
                     dataset, loss_func_tuple, device):
    traj_loss_func = loss_func_tuple[0]
    mode_loss_func = loss_func_tuple[1]
    man_data = label_tuple
    #man_data_onehot = F.one_hot(man_data, num_classes= 3)
    #man_input = man_data_onehot[:,(p.MAX_IN_SEQ_LEN-1):(p.MAX_IN_SEQ_LEN-1+p.TGT_SEQ_LEN)]
    man_gt = man_data[:, p.MAX_IN_SEQ_LEN:(p.MAX_IN_SEQ_LEN+p.TGT_SEQ_LEN)]
    mode_gt = utils.static_mode_from_man(man_gt.cpu().detach().numpy())
    mode_gt = torch.from_numpy(mode_gt).to(device)
    traj_data = data_tuple[-1]
    traj_input = traj_data[:,(p.MAX_IN_SEQ_LEN-1):(p.MAX_IN_SEQ_LEN-1+p.TGT_SEQ_LEN) ] 
    traj_gt = traj_data[:,p.MAX_IN_SEQ_LEN:(p.MAX_IN_SEQ_LEN+p.TGT_SEQ_LEN)]
    
    feature_data = data_tuple[0]
    input_padding_mask = data_tuple[1]

    output_dict = model(x = feature_data, 
                        y = traj_input, 
                        y_mask = utils.get_y_mask(p.TGT_SEQ_LEN).to(device),
                        input_padding_mask = input_padding_mask)
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

def SMTPOVL_evaluation(p, data_tuple, man_data, plot_info, dataset,
                        model, loss_func_tuple, device, eval_type):
    (tv_id, frames, data_file) = plot_info
    
    traj_loss_func = loss_func_tuple[0]
    mode_loss_func = loss_func_tuple[1]

    man_gt = man_data[:, p.MAX_IN_SEQ_LEN:(p.MAX_IN_SEQ_LEN+p.TGT_SEQ_LEN)]
    mode_gt = utils.static_mode_from_man(man_gt.cpu().detach().numpy())
    mode_gt = torch.from_numpy(mode_gt).to(device)

    traj_data = data_tuple[-1]
    traj_initial_input = traj_data[:,(p.MAX_IN_SEQ_LEN-1):p.MAX_IN_SEQ_LEN] 
    traj_gt = traj_data[:,p.MAX_IN_SEQ_LEN:(p.MAX_IN_SEQ_LEN+p.TGT_SEQ_LEN)]
    traj_track_gt = traj_data[:,:(p.MAX_IN_SEQ_LEN+p.TGT_SEQ_LEN)]
    feature_data = data_tuple[0]
    input_padding_mask = data_tuple[1]
    with torch.no_grad():
        encoder_out = model.encoder_forward(x = feature_data,
                                            input_padding_mask = input_padding_mask)
        mode_prob = model.mode_decoder_forward(encoder_out,
                                               input_padding_mask = input_padding_mask)
    mode_loss_func = mode_loss_func()
    mode_loss = mode_loss_func(mode_prob, mode_gt)
    
    mode_prob = F.softmax(mode_prob,dim = 1) #softmax should be after applying loss function
    hp_mode = np.argmax(mode_prob.cpu().detach().numpy(), axis = 1)
    
    BM_predicted_data_dist, BM_traj_pred = \
        DMTPOVL_trajectory_inference(p, model, device, traj_initial_input,
                                     input_padding_mask, encoder_out, hp_mode)

    traj_loss = traj_loss_func(BM_predicted_data_dist, traj_gt)
    
    evaluation_loss =  mode_loss + p.TRAJ2CLASS_LOSS_RATIO*traj_loss 
    
    # Trajectory inference for all modes!
    if eval_type == 'Test' and p.MULTI_MODAL_EVAL == True:
        traj_preds = []
        data_dist_preds = []
        for mode_itr in range(model.n_mode):
            selected_mode = np.ones_like(hp_mode)*mode_itr
            predicted_data_dist, traj_pred = \
                DMTPOVL_trajectory_inference(p, model, device, traj_initial_input,
                                             input_padding_mask, encoder_out, selected_mode)
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
        'input padding mask': input_padding_mask.cpu().data.numpy().astype(int),
        'traj_gt': traj_gt.cpu().data.numpy(),
        'traj_track_gt': traj_track_gt.cpu().data.numpy(),
        'traj_dist_preds': data_dist_preds.cpu().data.numpy(),
        'mode_gt': mode_gt.cpu().data.numpy(),
        'mode_prob': mode_prob.detach().cpu().data.numpy(),
    }
    return batch_print_info_dict, batch_kpi_input_dict




def SMTPOVL_deploy(p, data_tuple, plot_info, dataset, model, device):
    
    (tv_id, frames, data_file) = plot_info
    
    traj_data = data_tuple[-1]
    
    traj_initial_input = traj_data[:,(p.MAX_IN_SEQ_LEN-1):p.MAX_IN_SEQ_LEN] 
    traj_gt = traj_data[:,p.MAX_IN_SEQ_LEN:]
    feature_data = data_tuple[0]
    input_padding_mask = data_tuple[1]
    
    with torch.no_grad():
        encoder_out = model.encoder_forward(x = feature_data,
                                            input_padding_mask = input_padding_mask)
        mode_prob_pred = model.mode_decoder_forward(encoder_out,
                                                    input_padding_mask = input_padding_mask)
    mode_prob_pred = F.softmax(mode_prob_pred, dim = 1)
    decoder_input = traj_initial_input
    hp_mode = np.argmax(mode_prob_pred.cpu().detach().numpy(), axis = 1)

    
   
    

    # Trajectory inference for all modes!
    if p.MULTI_MODAL_EVAL == True:
        traj_preds = []
        data_dist_preds = []
        for mode_itr in range(model.n_mode):
            selected_mode = np.ones_like(hp_mode)*mode_itr
            decoder_input = traj_initial_input
            predicted_data_dist, traj_pred = \
                DMTPOVL_trajectory_inference(p, model, device, decoder_input,\
                                          input_padding_mask, encoder_out, selected_mode)
            traj_preds.append(traj_pred)
            data_dist_preds.append(predicted_data_dist)
        traj_preds = torch.stack(traj_preds, dim = 1)
        data_dist_preds = torch.stack(data_dist_preds, dim =1)
    else:
        data_dist_preds = []
        BM_predicted_data_dist, BM_traj_pred = \
            DMTPOVL_trajectory_inference(p, model, device, decoder_input, 
                                        input_padding_mask, encoder_out, hp_mode)
        for mode_itr in range(model.n_mode):
            data_dist_preds.append(BM_predicted_data_dist)
        data_dist_preds = torch.stack(data_dist_preds, dim =1)
    
    
    traj_gt = traj_gt.cpu().data.numpy()
    unnormalised_traj_pred = data_dist_preds.cpu().data.numpy()
    unnormalised_traj_pred = unnormalised_traj_pred[:,:,:,:2]
    traj_max = dataset.output_states_max
    traj_min = dataset.output_states_min
    unnormalised_traj_pred = unnormalised_traj_pred*(traj_max-traj_min) + traj_min 
    unnormalised_traj_pred = np.cumsum(unnormalised_traj_pred, axis = 2)
    unnormalised_traj_gt = traj_gt*(traj_max-traj_min) + traj_min
    unnormalised_traj_gt = np.cumsum(unnormalised_traj_gt, axis = 1)
    batch_export_dict = {    
        'data_file': data_file,
        'tv': tv_id.numpy(),
        'frames': frames.numpy(),
        'traj_pred': unnormalised_traj_pred,
        'traj_gt': unnormalised_traj_gt,
        'mode_prob': mode_prob_pred.detach().cpu().data.numpy(),
    }


    return batch_export_dict





def DMTPOVL_trajectory_inference(p, model, device, decoder_input, 
                                 input_padding_mask, encoder_out, selected_mode):
    for out_seq_itr in range(p.TGT_SEQ_LEN):
        traj_pred = \
            model.traj_decoder_forward(y = decoder_input, 
                                       input_padding_mask = input_padding_mask,
                                       y_mask = utils.get_y_mask(decoder_input.size(1)).to(device), 
                                       encoder_out = encoder_out)
        
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
    traj_pred = decoder_input[:,1:,:2]
    
    return predicted_data_dist, traj_pred

