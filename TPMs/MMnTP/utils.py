import numpy as np
import torch
from scipy import stats

def calc_man_vectors(man_pred, n_mode, man_per_mode, tgt_seq_len, device, time_reg = True):
    batch_size = man_pred.shape[0]
    #mode probabilities
    mode_pr = man_pred[:, 0:n_mode]
    man_pr = man_pred[:,n_mode:n_mode+ n_mode*3*man_per_mode]
    time_pr = man_pred[:,n_mode+ n_mode*3*man_per_mode:]
    man_pr = man_pr.reshape(batch_size, n_mode, man_per_mode, 3)
    w_ind = divide_prediction_window(tgt_seq_len, man_per_mode)
    
    if time_reg:
        time_pr = time_pr.reshape(batch_size, n_mode, man_per_mode-1).clone()
        for i in range(man_per_mode-1):
            time_pr[:,:,i] = (time_pr[:,:,i]/2 + 0.5)*(w_ind[i,1]-w_ind[i,0]) 
        time_bar_pr = torch.round(time_pr).int()
       
    else:
        time_pr = time_pr.reshape(batch_size, n_mode, tgt_seq_len)
        time_bar_pr = torch.zeros((batch_size, n_mode, man_per_mode-1), device = device).int()
        
        for mode_itr in range(n_mode):
            for i in range(len(w_ind)):
                time_bar_pr[:,mode_itr,i] = torch.argmax(time_pr[:,mode_itr,w_ind[i,0]:w_ind[i,1]], dim=-1)
    
    man_pr = torch.argmax(man_pr, dim = -1)
    '''
    time_pr_list = []
    for j in range(n_mode):
        time_pr_list.append([])
        for i in range(len(w_ind)):
            time_pr_list[j].append(torch.argmax(time_pr[:,j,w_ind[i,0]:w_ind[i,1]], dim=-1))
    '''
    man_vectors = []
    for i in range(n_mode):
        man_vectors.append(man_n_timing2man_vector(man_pr[:,i], time_bar_pr[:,i], tgt_seq_len, w_ind))
    man_vectors = torch.stack(man_vectors, dim=1)
    man_vectors = man_vectors.to(device).type(torch.long)

    return mode_pr,man_vectors 

def sel_high_prob_man( man_pred, n_mode, man_per_mode, tgt_seq_len, device, time_reg = True):
    batch_size = man_pred.shape[0]
    #mode probabilities
    mode_pr = man_pred[:, 0:n_mode]
    man_pr = man_pred[:,n_mode:n_mode+ n_mode*3*man_per_mode]
    time_pr = man_pred[:,n_mode+ n_mode*3*man_per_mode:]
    man_pr = man_pr.reshape(batch_size, n_mode, man_per_mode, 3)
    w_ind = divide_prediction_window(tgt_seq_len, man_per_mode)
    if time_reg:
        time_pr = time_pr.reshape(batch_size, n_mode, man_per_mode-1).clone()
        for i in range(man_per_mode-1):
            time_pr[:,:,i] = (time_pr[:,:,i]/2 + 0.5)*(w_ind[i,1]-w_ind[i,0])
      
    else:
        time_pr = time_pr.reshape(batch_size, n_mode, tgt_seq_len)
    high_prob_mode = torch.argmax(mode_pr, dim=1)
    time_pr = time_pr[np.arange(batch_size),high_prob_mode]
    
    
    if time_reg:
        #for i in range(man_per_mode-1):
        #    time_pr[:,:,i] = time_pr[:,:,i]*(w_ind[i,1]-w_ind[i,0])
        time_bar_pr = torch.round(time_pr).int()
    else:
        time_bar_pr = torch.zeros((batch_size, man_per_mode-1), device = device).int()
        for i in range(len(w_ind)):
            time_bar_pr[:,i] = torch.argmax(time_pr[:,w_ind[i,0]:w_ind[i,1]], dim=-1)
        
    man_pr = torch.argmax(man_pr, dim = -1)
    
    man_vector = man_n_timing2man_vector(man_pr[np.arange(batch_size),high_prob_mode], time_bar_pr, tgt_seq_len, w_ind)
    #print(man_vector)
    #exit()
    man_vector = man_vector.to(device).type(torch.long)

    return man_vector

def find_winning_mode(loss_value, thr=0):
    # 
    ml_values, ml_index = torch.sort(loss_value , dim=1)
    #ml_values-ml_values[0]<thr
    #tl_values, tl_index = torch.sort(time_losses, dim=0)
    return ml_index[0,:]

def divide_prediction_window(seq_len, man_per_mode):
    
    num_window = man_per_mode-1
    window_length = int(seq_len/num_window)
    w_ind = np.zeros((num_window, 2), dtype= np.int32)
    for i in range(num_window-1):
        w_ind[i,0] = i*window_length
        w_ind[i,1] = (i+1)*window_length
    w_ind[num_window-1,0] = (num_window-1)*window_length
    w_ind[num_window-1,1] = seq_len
    return w_ind

def man_vector2man_n_timing(man_vector, man_per_mode, w_ind):
    batch_size = man_vector.shape[0]
    man_v_list = []
    for i in range(len(w_ind)):
        man_v_list.append(man_vector[:, w_ind[i,0]:w_ind[i,1]])
    mans = torch.zeros((batch_size, man_per_mode))
    times = torch.zeros((batch_size, man_per_mode-1))
    for i, man_v in enumerate(man_v_list):
        mans[:,i] = man_v[:,0]
        
        _, times[:,i] = torch.max(man_v!=man_v[:,0:1], dim =1)
        
    times[times==0] = -1 #no manouvre change
    mans[:,-1] = man_v_list[-1][:,-1]
    #print('man_vector:', man_vector)
    #print(mans)
    #print(times)
    #exit()
    return mans, times

def man_n_timing2man_vector(mans, times, tgt_seq_len, w_ind, device = torch.device("cpu")):
    batch_size = mans.shape[0]
    man_per_mode = mans.shape[1]
    #print(batch_size)
    #print(times.shape)
    #print(w_ind)

    man_vector = torch.zeros((batch_size,tgt_seq_len), device = device)
    for i in range(man_per_mode-1):
        times[:,i] = torch.clamp(times[:,i], min = 0, max = (w_ind[i,1]-w_ind[i,0]))
        for batch_itr in range(batch_size):
            man_vector[batch_itr,w_ind[i,0]:w_ind[i,0]+times[batch_itr, i]] = mans[batch_itr,i]
            man_vector[batch_itr,w_ind[i,0]+times[batch_itr, i]:w_ind[i,1]] = mans[batch_itr,i+1]     
    return man_vector

def prob_activation_func(x):
    muY = x[:,:,0:1]
    muX = x[:,:,1:2]
    sigY = x[:,:,2:3]
    sigX = x[:,:,3:4]
    rho = x[:,:,4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    x = torch.cat([muX, muY, sigX, sigY, rho],dim=2)
    return x

def get_y_mask(size) -> torch.tensor:
    # Generates a squeare matrix where the each row allows one word more to be seen
    mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
    mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
    
    # EX for size=5:
    # [[0., -inf, -inf, -inf, -inf],
    #  [0.,   0., -inf, -inf, -inf],
    #  [0.,   0.,   0., -inf, -inf],
    #  [0.,   0.,   0.,   0., -inf],
    #  [0.,   0.,   0.,   0.,   0.]]
    
    return mask

def static_mode_from_man(man):
    m = stats.mode(man, axis = 1)
    return m[0][:,0]