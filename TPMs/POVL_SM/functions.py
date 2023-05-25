import torch
import torch.nn.functional as F
import gc
import numpy as np
from time import time
from functools import reduce
import operator as op
from . import utils

def POVL_SM_training(p, data_tuple, man_data, model, dataset, loss_func_tuple, device):
    '''
    start_all = torch.cuda.Event(enable_timing=True)
    end_all = torch.cuda.Event(enable_timing=True)

    start_model = torch.cuda.Event(enable_timing=True)
    end_model = torch.cuda.Event(enable_timing=True)

    start_loss = torch.cuda.Event(enable_timing=True)
    end_loss = torch.cuda.Event(enable_timing=True)


    start_all.record()
    '''
    traj_loss_func = loss_func_tuple[0]
    #print(man_data.shape)
    traj_data = data_tuple[-1]
    traj_input = traj_data[:,(p.MAX_IN_SEQ_LEN-1):(p.MAX_IN_SEQ_LEN-1+p.TGT_SEQ_LEN) ] 
    traj_gt = traj_data[:,p.MAX_IN_SEQ_LEN:(p.MAX_IN_SEQ_LEN+p.TGT_SEQ_LEN)]
    
    decoder_input = traj_input
    feature_data = data_tuple[0]
    input_padding_mask = data_tuple[1]

    #start_model.record()
    output_dict = model(x = feature_data, 
                        y = decoder_input, 
                        input_padding_mask = input_padding_mask, 
                        y_mask = utils.get_y_mask(p.TGT_SEQ_LEN).to(device))
    traj_pred = output_dict['traj_pred']
    '''
    obj_sizes = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                obj_size = reduce(op.mul, obj.size()) if len(obj.size()) > 0 else 0
                obj_sizes +=obj_size
                print(obj_size, type(obj), obj.size())
        except: pass
    print(obj_sizes)
    exit()
    '''
    #end_model.record()
    #torch.cuda.synchronize()
    #model_time = start_model.elapsed_time(end_model)

    
    

    #start_loss.record()
    traj_loss = traj_loss_func(traj_pred, traj_gt)
    
    training_loss = traj_loss 
    #end_all.record()
    #end_loss.record()
    #torch.cuda.synchronize()
    #total_time = start_all.elapsed_time(end_all)
    #loss_time = start_loss.elapsed_time(end_loss)
    batch_print_info_dict = {
        'Total Loss': training_loss.cpu().data.numpy()/model.batch_size,
    }
    return training_loss, batch_print_info_dict

def POVL_SM_deploy(p, data_tuple, plot_info, dataset, model, device):
    (tv_id, frames, data_file) = plot_info
    
    traj_data = data_tuple[-1]
    
    traj_initial_input = traj_data[:,(p.MAX_IN_SEQ_LEN-1):p.MAX_IN_SEQ_LEN] 
    traj_gt = traj_data[:,p.MAX_IN_SEQ_LEN:]
    feature_data = data_tuple[0]
    input_padding_mask = data_tuple[1]
    with torch.no_grad():
        encoder_out = model.encoder_forward(x = feature_data, 
                                                     input_padding_mask = input_padding_mask)
    
    decoder_input = traj_initial_input
    BM_predicted_data_dist, BM_traj_pred = POVL_SM_trajectory_inference(p, 
                                                                        model, 
                                                                        device, 
                                                                        decoder_input, 
                                                                        input_padding_mask, 
                                                                        encoder_out)

    
    # Trajectory inference for all modes!
    traj_gt = traj_gt.cpu().data.numpy()
    unnormalised_traj_pred = BM_traj_pred.cpu().data.numpy()
    unnormalised_traj_pred = unnormalised_traj_pred[:,:,:2]
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

def POVL_SM_evaluation(p, data_tuple, man_data,plot_info, dataset, 
                       model, loss_func_tuple, device, eval_type):
    (tv_id, frames, data_file) = plot_info
    
    traj_loss_func = loss_func_tuple[0]
    
    
    traj_data = data_tuple[-1]
    traj_initial_input = traj_data[:,(p.MAX_IN_SEQ_LEN-1):p.MAX_IN_SEQ_LEN] 
    traj_gt = traj_data[:,p.MAX_IN_SEQ_LEN:(p.MAX_IN_SEQ_LEN+p.TGT_SEQ_LEN)]
    
    traj_gt_export = traj_data
    traj_track_gt = traj_data[:,:(p.MAX_IN_SEQ_LEN+p.TGT_SEQ_LEN)]
    feature_data = data_tuple[0]
    input_padding_mask = data_tuple[1]
    with torch.no_grad():
        encoder_out = model.encoder_forward(x = feature_data, 
                                                     input_padding_mask = input_padding_mask)
    
    decoder_input = traj_initial_input
    predicted_data_dist, traj_pred = POVL_SM_trajectory_inference(p, 
                                                                  model, 
                                                                  device, 
                                                                  decoder_input,
                                                                  input_padding_mask, 
                                                                  encoder_out)

    traj_loss = traj_loss_func(predicted_data_dist, traj_gt)
    evaluation_loss =  traj_loss 
    
    
        
    batch_print_info_dict = {
        'Total Loss': evaluation_loss.cpu().data.numpy()/model.batch_size,
    }
    #print(traj_gt.shape)
    #print(traj_pred.shape)
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
        'traj_dist_preds': predicted_data_dist.cpu().data.numpy(),
    }
    return batch_print_info_dict, batch_kpi_input_dict

def POVL_SM_trajectory_inference(p, model, device, decoder_input, 
                                 input_padding_mask, encoder_out):
    for out_seq_itr in range(p.TGT_SEQ_LEN):
        #output_dict = model(x = encoder_input, y =decoder_input, y_mask = utils.get_y_mask(decoder_input.size(2)).to(device))
        with torch.no_grad():
            traj_pred = model.traj_decoder_forward(y = decoder_input, 
                                                        input_padding_mask = input_padding_mask,
                                                        y_mask = utils.get_y_mask(decoder_input.size(1)).to(device), 
                                                        encoder_out = encoder_out)
        
        current_traj_pred = traj_pred[:,out_seq_itr:(out_seq_itr+1)] #traj output at current timestep
        
        #print(decoder_input.shape)
        #print(current_traj_pred.shape)
        decoder_input = torch.cat((decoder_input, current_traj_pred[:,:,:2]), dim = 1)
        if out_seq_itr==0:
            traj_pred_dist = current_traj_pred
        else:
            traj_pred_dist = torch.cat((traj_pred_dist, current_traj_pred), dim = 1)
        
    traj_pred = decoder_input[:,1:,:2]
    return traj_pred_dist, traj_pred


