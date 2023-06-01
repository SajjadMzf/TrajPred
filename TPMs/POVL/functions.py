import torch
import torch.nn.functional as F
import numpy as np
import pdb
from time import time

from . import utils

def POVL_training(p, data_tuple, man_data, model, dataset, loss_func_tuple, device):
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
    man_loss_func = loss_func_tuple[1]
    man_input = man_data[:,(p.MAX_IN_SEQ_LEN-1):(p.MAX_IN_SEQ_LEN-1+p.TGT_SEQ_LEN)]
    man_input = F.one_hot(man_input, num_classes= 3)
    
    man_gt = man_data[:, p.MAX_IN_SEQ_LEN:(p.MAX_IN_SEQ_LEN+p.TGT_SEQ_LEN)]
    #print(man_data.shape)
    traj_data = data_tuple[-1]
    traj_input = traj_data[:,(p.MAX_IN_SEQ_LEN-1):(p.MAX_IN_SEQ_LEN-1+p.TGT_SEQ_LEN) ] 
    traj_gt = traj_data[:,p.MAX_IN_SEQ_LEN:(p.MAX_IN_SEQ_LEN+p.TGT_SEQ_LEN)]
    
    decoder_input = torch.cat((traj_input, man_input), dim = -1)
    decoder_input = torch.stack([decoder_input, decoder_input, decoder_input], dim =1 )
    feature_data = data_tuple[0]
    input_padding_mask = data_tuple[1]
    map_data = data_tuple[2]

    #start_model.record()
    output_dict = model(x = feature_data, y = decoder_input, map = map_data, 
                        input_padding_mask = input_padding_mask, 
                        y_mask = utils.get_y_mask(p.TGT_SEQ_LEN).to(device))
    traj_pred = output_dict['traj_pred']
    man_pred = output_dict['man_pred']
    #end_model.record()
    #torch.cuda.synchronize()
    #model_time = start_model.elapsed_time(end_model)

    if p.MULTI_MODAL == True:
        manouvre_index = man_gt # [batch_size, tgt_seq_len]
        batch_sweep = np.arange(model.batch_size).reshape(-1,1)
        batch_sweep = np.tile(batch_sweep, (1, p.TGT_SEQ_LEN))
        seq_sweep = np.arange(p.TGT_SEQ_LEN).reshape(1,-1)
        seq_sweep = np.tile(seq_sweep, (model.batch_size, 1))
        traj_pred = traj_pred[batch_sweep, manouvre_index, seq_sweep]     
    else:
        traj_pred = traj_pred[:,0]
    
    

    #start_loss.record()
    traj_loss = traj_loss_func(traj_pred, traj_gt)
    '''
    traj_max = dataset.output_states_min
    traj_min = dataset.output_states_max
    traj_pred_np = traj_pred[:,:,:2].cpu().data.numpy()
    traj_gt_np = traj_gt[:,:,:2].cpu().data.numpy()
    unnorm_traj_pred = traj_pred_np*(traj_max-traj_min) + traj_min
    unnorm_traj_pred = np.cumsum(unnorm_traj_pred, axis = -1)
    unnorm_traj_gt = traj_gt_np*(traj_max-traj_min) + traj_min
    unnorm_traj_gt = np.cumsum(unnorm_traj_gt, axis = -1)
    
    mse_traj_loss_np = np.sum(np.sum((unnorm_traj_pred-unnorm_traj_gt)**2, axis = -1), axis = -1)/(p.TGT_SEQ_LEN*2)
    
    traj_loss_ol = np.zeros((p.MAX_IN_SEQ_LEN))
    n_samples_ol = np.zeros((p.MAX_IN_SEQ_LEN))
    input_padding_mask_np = p.MAX_IN_SEQ_LEN - np.argmin(input_padding_mask.cpu().data.numpy(), axis = -1)
    for o_len in range(1, p.MAX_IN_SEQ_LEN+1):
        ovl_index = input_padding_mask_np==o_len
        n_samples_ol[o_len-1] = np.sum(ovl_index)
        if n_samples_ol[o_len-1]>0:
            traj_loss_ol[o_len-1] = np.sum(mse_traj_loss_np[ovl_index])/(n_samples_ol[o_len-1])
    '''
    if p.MAN_DEC_OUT:
        man_loss, mode_ploss, man_ploss, time_ploss, time_bar_pred =\
              man_loss_func(p, man_pred, man_gt, n_mode = model.n_mode,
                             man_per_mode = model.man_per_mode, device = device)
    else:
        man_loss = 0
    
    training_loss = man_loss + p.TRAJ2CLASS_LOSS_RATIO*traj_loss 
    #end_all.record()
    #end_loss.record()
    #torch.cuda.synchronize()
    #total_time = start_all.elapsed_time(end_all)
    #loss_time = start_loss.elapsed_time(end_loss)
    batch_print_info_dict = {
        #'Total time':total_time,
        #'Model time':model_time,
        #'Loss Time': loss_time,
        'Total Loss': training_loss.cpu().data.numpy()/model.batch_size,
        'Traj Loss': traj_loss.cpu().data.numpy()/model.batch_size,
        'Man Loss': man_loss.cpu().data.numpy()/model.batch_size if p.MAN_DEC_OUT else 0,
        'Mode Partial Loss': mode_ploss.cpu().data.numpy()/model.batch_size if p.MAN_DEC_OUT else 0,
        'Man Partial Loss': man_ploss.cpu().data.numpy()/model.batch_size if p.MAN_DEC_OUT else 0,
        'Time Partial Loss': time_ploss.cpu().data.numpy()/model.batch_size if p.MAN_DEC_OUT else 0,
    }
    '''
    for i in range(p.MAX_IN_SEQ_LEN):
        batch_print_info_dict['TRAJ_LOSS OL{}'.format(i+1)] = traj_loss_ol[i]
    for i in range(p.MAX_IN_SEQ_LEN):
        batch_print_info_dict['N_SAMPLES OL{}'.format(i+1)] = n_samples_ol[i]*500
    '''
    return training_loss, batch_print_info_dict

def POVL_deploy(p, data_tuple, plot_info, dataset, model, device):
    (tv_id, frames, data_file) = plot_info
    
    traj_data = data_tuple[-1]
    traj_initial_input = traj_data[:,(p.MAX_IN_SEQ_LEN-1):p.MAX_IN_SEQ_LEN] 
    traj_gt = traj_data[:,p.MAX_IN_SEQ_LEN:]
    feature_data = data_tuple[0]
    input_padding_mask = data_tuple[1]
    map_data = data_tuple[2]
    with torch.no_grad():
        encoder_out, map_out = \
            model.encoder_forward(x = feature_data, map = map_data, input_padding_mask = input_padding_mask)
        man_pred = model.man_decoder_forward(encoder_out, map_out, input_padding_mask)
    mode_prob, man_vectors = utils.calc_man_vectors(man_pred, model.n_mode, model.man_per_mode, p.TGT_SEQ_LEN, device)
    mode_prob = F.softmax(mode_prob,dim = 1)

    BM_man_vector = utils.sel_high_prob_man(man_pred, model.n_mode, model.man_per_mode, p.TGT_SEQ_LEN, device)
    decoder_input = traj_initial_input
    decoder_input = torch.stack([decoder_input,decoder_input,decoder_input], dim = 1) #multi-modal
    BM_predicted_data_dist, BM_traj_pred =\
          POVL_trajectory_inference(p, model, device, decoder_input, input_padding_mask, encoder_out, BM_man_vector)

    
    # Trajectory inference for all modes!
    if p.MULTI_MODAL_EVAL == True and p.MULTI_MODAL==True:
        traj_preds = []
        data_dist_preds = []
        for mode_itr in range(model.n_mode):
            man_pred_vector = man_vectors[:,mode_itr]
            decoder_input = traj_initial_input
            decoder_input = torch.stack([decoder_input,decoder_input,decoder_input], dim = 1) #multi-modal
            predicted_data_dist, traj_pred =\
                  POVL_trajectory_inference(p, model, device, decoder_input, input_padding_mask, encoder_out, man_pred_vector)
            traj_preds.append(traj_pred)
            data_dist_preds.append(predicted_data_dist)
        traj_preds = torch.stack(traj_preds, dim = 1)
        data_dist_preds = torch.stack(data_dist_preds, dim =1)
    else:
        data_dist_preds = []
        for mode_itr in range(model.n_mode):
            data_dist_preds.append(BM_predicted_data_dist)
        data_dist_preds = torch.stack(data_dist_preds, dim =1)
    unnormalised_traj_pred = data_dist_preds.cpu().data.numpy()
    unnormalised_traj_pred = unnormalised_traj_pred[:,:,:,:2]
    traj_min = dataset.output_states_min
    traj_max = dataset.output_states_max
    unnormalised_traj_pred = unnormalised_traj_pred*(traj_max-traj_min) + traj_min 
    unnormalised_traj_pred = np.cumsum(unnormalised_traj_pred, axis = 2)
    traj_gt = traj_gt.cpu().data.numpy()
    unnormalised_traj_gt = traj_gt*(traj_max-traj_min) + traj_min
    unnormalised_traj_gt = np.cumsum(unnormalised_traj_gt, axis = 1)
    
    batch_export_dict = {    
        'data_file': data_file,
        'tv': tv_id.numpy(),
        'frames': frames.numpy(),
        'traj_pred': unnormalised_traj_pred,
        'traj_gt': unnormalised_traj_gt,
        'mode_prob': F.softmax(mode_prob, dim = -1).detach().cpu().data.numpy(),
    }

    return batch_export_dict




def POVL_evaluation(p, data_tuple,man_data, plot_info, dataset, model, loss_func_tuple, device, eval_type):
    (tv_id, frames, data_file) = plot_info
    
    traj_loss_func = loss_func_tuple[0]
    man_loss_func = loss_func_tuple[1]

    man_gt = man_data[:, p.MAX_IN_SEQ_LEN:(p.MAX_IN_SEQ_LEN+p.TGT_SEQ_LEN)]
    
    w_ind = utils.divide_prediction_window(p.TGT_SEQ_LEN, model.man_per_mode)

    _, time_bar_gt = utils.man_vector2man_n_timing(man_gt, model.man_per_mode, w_ind)
    #man_bar_gt = man_bar_gt.to(device).type(torch.long)

    traj_data = data_tuple[-1]
    traj_initial_input = traj_data[:,(p.MAX_IN_SEQ_LEN-1):p.MAX_IN_SEQ_LEN] 
    traj_gt = traj_data[:,p.MAX_IN_SEQ_LEN:(p.MAX_IN_SEQ_LEN+p.TGT_SEQ_LEN)]
    traj_gt_export = traj_data
    traj_track_gt = traj_data[:,:(p.MAX_IN_SEQ_LEN+p.TGT_SEQ_LEN)]
    feature_data = data_tuple[0]
    input_padding_mask = data_tuple[1]
    map_data = data_tuple[2]
    with torch.no_grad():
        encoder_out, map_out =\
              model.encoder_forward(x = feature_data, map = map_data, input_padding_mask = input_padding_mask)
        man_pred = model.man_decoder_forward(encoder_out, map_out, input_padding_mask)
    mode_prob, man_vectors = \
        utils.calc_man_vectors(man_pred, model.n_mode, model.man_per_mode, p.TGT_SEQ_LEN, device)
    mode_prob = F.softmax(mode_prob,dim = 1)

    BM_man_vector = utils.sel_high_prob_man(man_pred, model.n_mode, model.man_per_mode, p.TGT_SEQ_LEN, device)
    decoder_input = traj_initial_input
    decoder_input = torch.stack([decoder_input,decoder_input,decoder_input], dim = 1) #multi-modal
    BM_predicted_data_dist, BM_traj_pred = \
        POVL_trajectory_inference(p, model, device, decoder_input,input_padding_mask, encoder_out, BM_man_vector)

    traj_loss = traj_loss_func(BM_predicted_data_dist, traj_gt)
    if p.MAN_DEC_OUT:
        man_loss, mode_ploss, man_ploss,time_ploss, time_bar_pred = \
            man_loss_func(p, man_pred, man_gt, n_mode = model.n_mode,
                           man_per_mode = model.man_per_mode, device = device, test_phase = True)
    else:
        man_loss = 0
        mode_ploss = 0
        man_ploss = 0
        time_ploss = 0
        time_bar_pred = torch.zeros((model.batch_size, model.n_mode, model.man_per_mode-1))
    evaluation_loss =  man_loss + p.TRAJ2CLASS_LOSS_RATIO*traj_loss 
    
    # Trajectory inference for all modes!
    if p.MULTI_MODAL_EVAL == True and p.MULTI_MODAL==True:
        traj_preds = []
        data_dist_preds = []
        for mode_itr in range(model.n_mode):
            man_pred_vector = man_vectors[:,mode_itr]
            decoder_input = traj_initial_input
            decoder_input = torch.stack([decoder_input,decoder_input,decoder_input], dim = 1) #multi-modal
            predicted_data_dist, traj_pred = \
                POVL_trajectory_inference(p, model, device, 
                                          decoder_input,input_padding_mask, encoder_out, man_pred_vector)
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
        'Man Loss': man_loss.cpu().data.numpy()/model.batch_size if p.MAN_DEC_OUT else 0,
        'Mode Partial Loss': mode_ploss.cpu().data.numpy()/model.batch_size if p.MAN_DEC_OUT else 0,
        'Man Partial Loss': man_ploss.cpu().data.numpy()/model.batch_size if p.MAN_DEC_OUT else 0,
        'Time Partial Loss': time_ploss.cpu().data.numpy()/model.batch_size if p.MAN_DEC_OUT else 0,
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
        'man_gt': man_gt.cpu().data.numpy(),
        'man_preds': man_vectors.cpu().data.numpy(),
        'time_bar_gt': time_bar_gt.cpu().data.numpy(),
        'time_bar_preds': time_bar_pred.cpu().data.numpy(),
        'mode_prob': mode_prob.detach().cpu().data.numpy(),
    }
    return batch_print_info_dict, batch_kpi_input_dict

def POVL_trajectory_inference(p, model, device, decoder_input, input_padding_mask, encoder_out, man_pred_vector):
    for out_seq_itr in range(p.TGT_SEQ_LEN):
        #output_dict = model(x = encoder_input, y =decoder_input, y_mask = utils.get_y_mask(decoder_input.size(2)).to(device))
        with torch.no_grad():
            traj_pred = model.traj_decoder_forward(y = decoder_input, 
                                                        input_padding_mask = input_padding_mask,
                                                        y_mask = utils.get_y_mask(decoder_input.size(2)).to(device), 
                                                        encoder_out = encoder_out)
        
        current_traj_pred = traj_pred[:,:,out_seq_itr:(out_seq_itr+1)] #traj output at current timestep
        current_man_pred = man_pred_vector[:,out_seq_itr:(out_seq_itr+1)]      
        
        if p.MULTI_MODAL:
            manouvre_index = current_man_pred[:,0] 
        else:
            manouvre_index = torch.zeros_like(current_man_pred[:,0] )
        #print_shape('current_traj_pred', current_traj_pred)
        #print_shape('manouvre_index', manouvre_index)
        current_traj_pred = \
            current_traj_pred[np.arange(current_traj_pred.shape[0]),manouvre_index,:,:2] #only the muX and muY [batch, modal, sequence, feature]
        current_man_pred = F.one_hot(current_man_pred, num_classes = 3) 
        
        #print_shape('current_traj_pred', current_traj_pred)
        #print_shape('current_man_pred', current_man_pred)
        
        current_decoder_input = current_traj_pred 
        current_decoder_input = torch.unsqueeze(current_decoder_input, dim = 1)
        current_decoder_input = \
            torch.cat([current_decoder_input, current_decoder_input, current_decoder_input], dim = 1)
        
        decoder_input = torch.cat((decoder_input, current_decoder_input), dim = 2)
        
        if out_seq_itr ==0:
            predicted_data_dist =\
                  traj_pred[np.arange(traj_pred.shape[0]),manouvre_index, out_seq_itr:(out_seq_itr+1)]
        else:
            predicted_data_dist = \
                torch.cat((predicted_data_dist, traj_pred[np.arange(traj_pred.shape[0]),
                                                          manouvre_index, out_seq_itr:(out_seq_itr+1)]), dim=1)
        #print_shape('predicted_data_dist', predicted_data_dist)
    traj_pred = decoder_input[:,:,1:,:2]
    #print(traj_pred)
    #print(predicted_data_dist[:,:,:,:2])
    #exit()
    traj_pred = traj_pred[:,0]
    return predicted_data_dist, traj_pred




def MTPM_loss(p, man_pred, man_vec_gt, n_mode, man_per_mode, device, test_phase = False, time_reg = True):
    # man pred: [batch_size, (1+3*man_per_mode + tgt_seq_len)*modes]
    # man_gt: [batch_size, tgt_seq_len]
    
    tgt_seq_len = man_vec_gt.shape[1]
    w_ind = utils.divide_prediction_window(tgt_seq_len, man_per_mode)
    man_gt, time_gt = utils.man_vector2man_n_timing(man_vec_gt, man_per_mode, w_ind)
    man_gt = man_gt.to(device).type(torch.long)
    time_gt = time_gt.to(device).type(torch.long)
    man_vec_gt = man_vec_gt.to(device).type(torch.long)
    batch_size = man_pred.shape[0]
    #mode probabilities
    mode_pr = man_pred[:, 0:n_mode] # mode prediction: probability of modes
    man_pr = man_pred[:,n_mode:n_mode+ n_mode*3*man_per_mode] # manouvre prediction: order of manouvres 
    time_pr = man_pred[:,n_mode+ n_mode*3*man_per_mode:] # timing of the manouvre
    
    

    man_pr = man_pr.reshape(batch_size, n_mode, man_per_mode, 3)
    man_pr_class = torch.argmax(man_pr, dim = -1)
    man_pr = torch.permute(man_pr,(0,1,3,2))
    if time_reg:
        time_pr = time_pr.reshape(batch_size, n_mode, man_per_mode-1).clone()
        for i in range(man_per_mode-1):
            time_pr[:,:,i] = (time_pr[:,:,i]/2 + 0.5)*(w_ind[i,1]-w_ind[i,0]) 
        time_bar_pred = time_pr
        
    else:
        time_pr = time_pr.reshape(batch_size, n_mode, tgt_seq_len)
    
        time_pr_list = []
        for i in range(len(w_ind)):
            time_pr_list.append(time_pr[:,:,w_ind[i,0]:w_ind[i,1]])
        
    
    loss_func = torch.nn.CrossEntropyLoss(ignore_index = -1)
    loss_func_no_r = torch.nn.CrossEntropyLoss(ignore_index = -1, reduction = 'none')
    reg_loss_func_no_r = torch.nn.MSELoss(reduction = 'none')
    man_loss_list = []
    time_loss_list = []
    for mode_itr in range(n_mode):
        
        man_loss_list.append(torch.sum(loss_func_no_r(man_pr[:,mode_itr], man_gt), dim = 1)) 
        if time_reg:
            mode_time_loss = \
                torch.sum(torch.mul(reg_loss_func_no_r(time_pr[:,mode_itr], time_gt.float()), 
                                    (time_gt!=-1).float()), dim = -1)
        else:
            mode_time_loss = 0
            for i, mode_time_pr in enumerate(time_pr_list):
                mode_time_loss += loss_func_no_r(mode_time_pr[:,mode_itr], time_gt[:,i])
        time_loss_list.append(mode_time_loss)
    man_losses = torch.stack(man_loss_list, dim = 1)
    time_losses = torch.stack(time_loss_list, dim = 1)
    
    man_pr = torch.permute(man_pr, (0,1,3,2))
    man_pr = man_pr.reshape(batch_size*n_mode, man_per_mode,3)
    man_pr_argmax = torch.argmax(man_pr, dim = -1)
    #time_pr = time_pr.reshape(batch_size*n_mode, tgt_seq_len)
    # Un comment for man acc loss calculation
    '''
    time_pr_arg_list = []
    for i in range(len(w_ind)):
        time_pr_arg_list.append(torch.argmax(time_pr[:,w_ind[i,0]:w_ind[i,1]], dim = -1))
    
    man_vec_pr = utils.man_n_timing2man_vector(man_pr_argmax, time_pr_arg_list, tgt_seq_len, w_ind, device)
    man_vec_pr = man_vec_pr.reshape(batch_size, n_mode, tgt_seq_len)
    #print(man_vec_pr[:,0].shape)
    man_acc = calc_man_acc_torch( man_vec_pr, man_vec_gt, device)
    '''
    if test_phase:
        winning_mode = torch.argmax(mode_pr, dim=1)
    else:
        #winning_mode = torch.argmin(man_vec_loss, dim = 1)
        winning_mode = torch.argmin(man_losses, dim = 1)
        #winning_mode = torch.argmin(man_acc, dim = 1)

    mode_loss = loss_func(mode_pr, winning_mode)
    man_loss = torch.mean(man_losses[np.arange(batch_size), winning_mode])
    time_loss = torch.mean(time_losses[np.arange(batch_size), winning_mode])
    lossVal = mode_loss + man_loss + time_loss 
    return lossVal, mode_loss, man_loss, time_loss, time_bar_pred