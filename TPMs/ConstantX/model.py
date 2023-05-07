import random
import numpy as np 
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import torch.nn.functional as F
import logging
from time import time
import math
import pdb

class ConstantX(nn.Module):
    def __init__(self, batch_size, device, hyperparams_dict, parameters,
                  drop_prob = 0.1):
        super(ConstantX, self).__init__()

        self.batch_size = batch_size
        self.device = device
        ## Parameters set for campatibility with framework
        self.multi_modal = parameters.MULTI_MODAL
        self.man_dec_out = parameters.MAN_DEC_OUT
        if self.multi_modal or self.man_dec_out:
            raise(ValueError('multi modality or manouvre based outputs are\
                              not supported in ConstantX model'))
            exit()
        self.in_seq_len = parameters.MAX_IN_SEQ_LEN
        self.out_seq_len = parameters.TGT_SEQ_LEN
        self.fps = parameters.FPS
        self.input_dim = 18
        self.output_dim = 2
        #print('Constant Model should only be run with ours_
        # states that includes velocity, acceleration features')
        self.unused_layer = nn.Linear(1,1)   

        # Dimension of transformer model ()
        self.constant_parameter = hyperparams_dict['parameter']
        
        
    def forward(self, 
                x, 
                input_padding_mask,
                states_min, 
                states_max,
                output_states_min, 
                output_states_max):
        traj_pred = torch.ones((self.batch_size, self.out_seq_len, 2),
                                requires_grad = False )
        #print(len(x))
        
        x = x.cpu()
        input_padding_mask = input_padding_mask.cpu()
        #classification => always lane keeping

        #trajectory prediction
        lat_vel = x[:,:,0]
        long_vel = x[:,:,1]
        lat_acc = x[:,:,2]
        long_acc = x[:,:,3]
        lat_vel_min = states_min[0]
        lat_vel_max = states_max[0]
        long_vel_min = states_min[1]
        long_vel_max = states_max[1]
        lat_acc_min = states_min[2]
        lat_acc_max = states_max[2]
        long_acc_min = states_min[3]
        long_acc_max = states_max[3]
        
        if self.constant_parameter == 'Final Velocity':
            lat_vel_unnormalised = (lat_vel_max-lat_vel_min)*lat_vel[:,-1] +\
                  lat_vel_min
            long_vel_unnormalised = (long_vel_max-long_vel_min)*long_vel[:,-1] +\
                  long_vel_min
            #TODO: remove "*5" after getting new data by running the data preprocess.
            lat_dist_unnormalised = lat_vel_unnormalised/self.fps
            long_dist_unnormalised = long_vel_unnormalised/self.fps
            #print(self.traj_pred.shape)
            #exit()

            traj_pred[:,:,0] *= lat_dist_unnormalised.unsqueeze(-1)
            traj_pred[:,:,1] *= long_dist_unnormalised.unsqueeze(-1)
            
        elif self.constant_parameter == 'Mean Velocity':
            in_seq_len = torch.sum(input_padding_mask==False, dim = 1)
            lat_vel = torch.mul(lat_vel, input_padding_mask==False)
            long_vel = torch.mul(long_vel, input_padding_mask==False)
            lat_vel_mean = torch.div(torch.sum(lat_vel, dim = 1),in_seq_len)
            long_vel_mean = torch.div(torch.sum(long_vel, dim = 1),in_seq_len)
            lat_vel_unnormalised = (lat_vel_max-lat_vel_min)*lat_vel_mean +\
                  lat_vel_min
            long_vel_unnormalised = (long_vel_max-long_vel_min)*long_vel_mean +\
                  long_vel_min
            lat_dist_unnormalised = lat_vel_unnormalised/self.fps
            long_dist_unnormalised = long_vel_unnormalised/self.fps
            #pdb.set_trace()
            #print(self.traj_pred.shape)
            #exit()

            traj_pred[:,:,0] *= lat_dist_unnormalised.unsqueeze(-1)
            traj_pred[:,:,1] *= long_dist_unnormalised.unsqueeze(-1)
        elif self.constant_parameter == 'Final Acceleration':
            raise ValueError('TBD')
        elif self.constant_parameter == 'Mean Acceleration':
            raise ValueError('TBD')
        traj_pred = (traj_pred-output_states_min)/\
            (output_states_max-output_states_min)
        return traj_pred



