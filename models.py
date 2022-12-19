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
from debugging_utils import *
import models_functions as mf

class MMnTP(nn.Module): 
    def __init__(self, batch_size, device, hyperparams_dict, parameters, drop_prob = 0.1):
        super(MMnTP, self).__init__()

        self.batch_size = batch_size
        self.device = device
        
        self.model_dim = hyperparams_dict['model dim']# Dimension of transformer model ()
        self.ff_dim = hyperparams_dict['feedforward dim']
        self.classifier_dim = hyperparams_dict['classifier dim']
        self.layers_num = hyperparams_dict['layer number']
        self.head_num = hyperparams_dict['head number']
        self.n_mode = hyperparams_dict['number of modes']
        self.man_per_mode = hyperparams_dict['manouvre per mode']
        self.multi_modal = parameters.MULTI_MODAL
        self.time_prediction = hyperparams_dict['time prediction']
        self.prob_output = hyperparams_dict['probabilistic output']
        self.man_dec_in = parameters.MAN_DEC_IN
        self.in_seq_len = parameters.IN_SEQ_LEN
        self.tgt_seq_len = parameters.TGT_SEQ_LEN
        self.decoder_in_dim = 2
        #(mode_prob(1) + number of manoeuvre classes(3)* (number of change periods+1(self.man_per_mode)) +  timing classification tgt_seq_len )*n_mode
        if self.time_prediction == 'regression':
            self.man_output_dim = (1+3*self.man_per_mode + self.man_per_mode-1)*self.n_mode 
        elif self.time_prediction == 'classification':
            self.man_output_dim = (1+3*self.man_per_mode + self.tgt_seq_len)*self.n_mode 
        else:
            raise(ValueError('Undefined time prediction method'))
        if self.man_dec_in:
            self.decoder_in_dim += 3
        
        self.input_dim = 18
        
        if self.prob_output:
            self.output_dim = 5 # muY, muX, sigY, sigX, rho 
        else:
            self.output_dim = 2
        
        self.dropout = nn.Dropout(drop_prob)
        
        ''' 1. Positional encoder: '''
        self.positional_encoder = PositionalEncoding(dim_model=self.model_dim, dropout_p=drop_prob, max_len=100)
        
        ''' 2. Transformer Encoder: '''
        self.encoder_embedding = nn.Linear(self.input_dim, self.model_dim)
        encoder_layers = nn.TransformerEncoderLayer(self.model_dim, self.head_num, self.ff_dim, drop_prob, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.layers_num)
        
        ''' 3. Transformer Decoder: '''
        self.decoder_embedding = nn.Linear(self.decoder_in_dim, self.model_dim)
        self.man_decoder_embedding = nn.Linear(2, self.model_dim)
        decoder_layers = nn.TransformerDecoderLayer(self.model_dim, self.head_num, self.ff_dim, drop_prob, batch_first = True)
        #rlc_decoder_layers = nn.TransformerDecoderLayer(self.model_dim, self.head_num, self.ff_dim, drop_prob, batch_first = True)
        #llc_decoder_layers = nn.TransformerDecoderLayer(self.model_dim, self.head_num, self.ff_dim, drop_prob, batch_first = True)
        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, self.layers_num)
        #self.rlc_transformer_decoder = nn.TransformerDecoder(rlc_decoder_layers, self.layers_num)
        #self.llc_transformer_decoder = nn.TransformerDecoder(llc_decoder_layers, self.layers_num)
        
        ''' 5. Trajectory Output '''
        
        self.lk_trajectory_fc_1 = nn.Linear(self.model_dim, self.classifier_dim)
        self.rlc_trajectory_fc_1 = nn.Linear(self.model_dim, self.classifier_dim)
        self.llc_trajectory_fc_1 = nn.Linear(self.model_dim, self.classifier_dim)
        
        self.lk_trajectory_fc_2 = nn.Linear(self.classifier_dim, self.output_dim)       
        self.rlc_trajectory_fc_2 = nn.Linear(self.classifier_dim, self.output_dim)
        self.llc_trajectory_fc_2 = nn.Linear(self.classifier_dim, self.output_dim)

        self.lk_trajectory_fc = nn.Linear(self.model_dim, self.output_dim)       
        self.rlc_trajectory_fc = nn.Linear(self.model_dim, self.output_dim)
        self.llc_trajectory_fc = nn.Linear(self.model_dim, self.output_dim)
        
        ''' 6. Manouvre Output '''
        
        self.dec_man_fc1 = nn.Linear(self.in_seq_len*self.model_dim, self.classifier_dim)
        self.dec_man_fc2 = nn.Linear(self.classifier_dim, self.man_output_dim)
        
    def forward(self, x, y, y_mask):      
        encoder_out = self.encoder_forward(x)
        man_pred = self.man_decoder_forward(encoder_out)
        traj_pred = self.traj_decoder_forward(y, y_mask, encoder_out)
        
        return {'traj_pred':traj_pred, 'man_pred': man_pred }
    
    def encoder_forward(self, x):
        #encoder
        x = self.encoder_embedding(x)
        x = self.positional_encoder(x)
        encoder_out = self.transformer_encoder(x)
        
        return encoder_out
    
    def man_decoder_forward(self, encoder_out):
        encoder_out_flattened = encoder_out.reshape(self.batch_size, self.in_seq_len*self.model_dim)
        man_pred = self.dec_man_fc2(F.relu(self.dec_man_fc1(encoder_out_flattened)))
        return man_pred

    def traj_decoder_forward(self, y, y_mask, encoder_out):
        
        encoder_out_flattened = encoder_out.reshape(self.batch_size, self.in_seq_len*self.model_dim)
        
        #traj decoder
        y = self.decoder_embedding(y[:,0,:,:self.decoder_in_dim])
        y = self.positional_encoder(y)
        decoder_out = self.transformer_decoder(y, encoder_out, tgt_mask = y_mask)
        ''' 
        if self.multi_modal: #if single modal lk represents the single modal not the lane keeping man anymore
            rlc_y = self.decoder_embedding(y[:,1,:,:self.decoder_in_dim])
            rlc_y = self.positional_encoder(rlc_y)
            rlc_decoder_out = self.lk_transformer_decoder(rlc_y, encoder_out, tgt_mask = y_mask) #TODO: LK transformer is being used for all three modes
            
            llc_y = self.decoder_embedding(y[:,2,:,:self.decoder_in_dim])
            llc_y = self.positional_encoder(llc_y)
            llc_decoder_out = self.lk_transformer_decoder(llc_y, encoder_out, tgt_mask = y_mask) #TODO: LK transformer is being used for all three modes
        '''
        #traj decoder linear layer
        
        lk_traj_pred = self.lk_trajectory_fc(decoder_out)
        #lk_traj_pred = self.lk_trajectory_fc_2(F.relu(self.lk_trajectory_fc_1(decoder_out)))
        if self.multi_modal:
            #rlc_traj_pred = self.rlc_trajectory_fc_2(F.relu(self.rlc_trajectory_fc_1(decoder_out)))
            #llc_traj_pred = self.llc_trajectory_fc_2(F.relu(self.llc_trajectory_fc_1(decoder_out)))
            rlc_traj_pred = self.rlc_trajectory_fc(decoder_out)
            llc_traj_pred = self.llc_trajectory_fc(decoder_out)
        if self.prob_output:
            lk_traj_pred = mf.prob_activation_func(lk_traj_pred)
            if self.multi_modal:
                rlc_traj_pred = mf.prob_activation_func(rlc_traj_pred)
                llc_traj_pred = mf.prob_activation_func(llc_traj_pred) 
        if self.multi_modal:
            traj_pred = torch.stack([lk_traj_pred, rlc_traj_pred, llc_traj_pred], dim=1) # lk =0, rlc=1, llc=2
        else:
            traj_pred = torch.stack([lk_traj_pred, lk_traj_pred, lk_traj_pred], dim=1) 
        #print_shape('decoder_out', decoder_out)
        
        return traj_pred 


    

class ConstantX(nn.Module):
    def __init__(self, batch_size, device, hyperparams_dict, parameters, drop_prob = 0.1):
        super(ConstantX, self).__init__()

        self.batch_size = batch_size
        self.device = device
        self.multi_modal = parameters.MULTI_MODAL
        self.man_dec_out = parameters.MAN_DEC_OUT
        if self.multi_modal or self.man_dec_out:
            raise(ValueError('multi modality or manouvre based outputs are not supported in ConstantX model'))
            exit()
        self.constant_parameter = hyperparams_dict['parameter']# Dimension of transformer model ()
        
        self.in_seq_len = parameters.IN_SEQ_LEN
        self.out_seq_len = parameters.TGT_SEQ_LEN
        self.fps = parameters.FPS
        self.input_dim = 18
        self.output_dim = 2
        #print('Constant Model should only be run with ours_states that includes velocity, acceleration features')
        

        self.unused_layer = nn.Linear(1,1)   
    def forward(self, x, states_min, states_max, output_states_min, output_states_max, traj_labels):
        traj_pred = torch.ones((self.batch_size, self.out_seq_len, 2), requires_grad = False )
        #print(len(x))
        
        x = x.cpu()
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
        
        if self.constant_parameter == 'Last Velocity':
            lat_vel_unnormalised = (lat_vel_max-lat_vel_min)*lat_vel[:,-1] + lat_vel_min
            long_vel_unnormalised = (long_vel_max-long_vel_min)*long_vel[:,-1] + long_vel_min
            lat_dist_unnormalised = lat_vel_unnormalised/self.fps
            long_dist_unnormalised = long_vel_unnormalised/self.fps
            #print(self.traj_pred.shape)
            #exit()

            traj_pred[:,:,0] *= lat_dist_unnormalised.unsqueeze(-1)
            traj_pred[:,:,1] *= long_dist_unnormalised.unsqueeze(-1)
            
        elif self.constant_parameter == 'Mean Velocity':
            lat_vel_unnormalised = (lat_vel_max-lat_vel_min)*lat_vel.mean(-1) + lat_vel_min
            long_vel_unnormalised = (long_vel_max-long_vel_min)*long_vel.mean(-1) + long_vel_min
            lat_dist_unnormalised = lat_vel_unnormalised/self.fps
            long_dist_unnormalised = long_vel_unnormalised/self.fps
            #print(self.traj_pred.shape)
            #exit()

            traj_pred[:,:,0] *= lat_dist_unnormalised.unsqueeze(-1)
            traj_pred[:,:,1] *= long_dist_unnormalised.unsqueeze(-1)
        elif self.constant_parameter == 'Last Acceleration':
            raise ValueError('TBD')
        elif self.constant_parameter == 'Mean Acceleration':
            raise ValueError('TBD')
        
        traj_pred = (traj_pred-output_states_min)/(output_states_max-output_states_min)
        
        labels = torch.zeros_like(traj_pred)
        return {'traj_pred':traj_pred.to(self.device),
        'man_pred':torch.zeros((traj_pred.shape[0],traj_pred.shape[1],3), device=self.device),
        'enc_man_pred': torch.zeros((traj_pred.shape[0],3), device = self.device)}



class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified (batch first) version from: https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        #print(token_embedding.size())
        #exit()
        return self.dropout(token_embedding + self.pos_encoding[:, :token_embedding.size(1), :])



class DMTP(nn.Module): 
    def __init__(self, batch_size, device, hyperparams_dict, parameters, drop_prob = 0.1):
        super(DMTP, self).__init__()

        self.batch_size = batch_size
        self.device = device
        
        self.model_dim = hyperparams_dict['model dim']# Dimension of transformer model ()
        self.ff_dim = hyperparams_dict['feedforward dim']
        self.layers_num = hyperparams_dict['layer number']
        self.classifier_dim = hyperparams_dict['classifier dim']
        self.head_num = hyperparams_dict['head number']
        self.n_mode = hyperparams_dict['number of modes']
        self.multi_modal = parameters.MULTI_MODAL
        
        self.prob_output = hyperparams_dict['probabilistic output']
        self.in_seq_len = parameters.IN_SEQ_LEN
        self.tgt_seq_len = parameters.TGT_SEQ_LEN
        self.decoder_in_dim = 2
        self.input_dim = 18
        self.mode_output_dim = self.n_mode
        if self.prob_output:
            self.output_dim = 5*self.n_mode # muY, muX, sigY, sigX, rho 
        else:
            self.output_dim = 2*self.n_mode
        
        self.dropout = nn.Dropout(drop_prob)
        
        ''' 1. Positional encoder: '''
        self.positional_encoder = PositionalEncoding(dim_model=self.model_dim, dropout_p=drop_prob, max_len=100)
        
        ''' 2. Transformer Encoder: '''
        self.encoder_embedding = nn.Linear(self.input_dim, self.model_dim)
        encoder_layers = nn.TransformerEncoderLayer(self.model_dim, self.head_num, self.ff_dim, drop_prob, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.layers_num)
        
        ''' 3. Transformer Decoder: '''
        self.decoder_embedding = nn.Linear(self.decoder_in_dim, self.model_dim)
        self.man_decoder_embedding = nn.Linear(2, self.model_dim)
        decoder_layers = nn.TransformerDecoderLayer(self.model_dim, self.head_num, self.ff_dim, drop_prob, batch_first = True)
        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, self.layers_num)
        
        ''' 5. Trajectory Output '''
        self.trajectory_fc = nn.Linear(self.model_dim, self.output_dim)
        self.mode_prob_fc = nn.Linear(self.model_dim, self.output_dim)
        
        ''' 6. Manouvre Output '''
        self.dec_mode_fc1 = nn.Linear(self.in_seq_len*self.model_dim, self.classifier_dim)
        self.dec_mode_fc2 = nn.Linear(self.classifier_dim, self.mode_output_dim)

    def forward(self, x, y, y_mask):      
        encoder_out = self.encoder_forward(x)
        mode_prob_pred = self.mode_decoder_forward(encoder_out)
        traj_pred = self.traj_decoder_forward(y, y_mask, encoder_out)
        
        return {'traj_pred':traj_pred, 'mode_prob_pred': mode_prob_pred}
    
    def encoder_forward(self, x):
        #encoder
        x = self.encoder_embedding(x)
        x = self.positional_encoder(x)
        encoder_out = self.transformer_encoder(x)
        
        return encoder_out
    
    def mode_decoder_forward(self, encoder_out):
        encoder_out_flattened = encoder_out.reshape(self.batch_size, self.in_seq_len*self.model_dim)
        mode_prob_pred = self.dec_mode_fc2(F.relu(self.dec_mode_fc1(encoder_out_flattened)))
        return mode_prob_pred

    def traj_decoder_forward(self, y, y_mask, encoder_out):
        encoder_out_flattened = encoder_out.reshape(self.batch_size, self.in_seq_len*self.model_dim)    
        #traj decoder
        y = self.decoder_embedding(y[:,:,:self.decoder_in_dim])
        y = self.positional_encoder(y)
        decoder_out = self.transformer_decoder(y, encoder_out, tgt_mask = y_mask)
        
        if self.multi_modal== False:
            raise(ValueError('Single Modal not supported'))
        
        #traj decoder linear layer
        
        traj_pred = self.trajectory_fc(decoder_out)
        
        if self.prob_output:
            seq_len = traj_pred.shape[1]
            traj_pred = traj_pred.reshape(-1, seq_len*self.n_mode,5)
            
            traj_pred = mf.prob_activation_func(traj_pred)
            traj_pred = traj_pred.reshape(-1, seq_len,self.n_mode,5)
            
            traj_pred = torch.permute(traj_pred, (0,2,1,3))
        else:
            raise(ValueError('not supported yet!'))
        
        return traj_pred 

class SMTP(nn.Module): 
    def __init__(self, batch_size, device, hyperparams_dict, parameters, drop_prob = 0.1):
        super(SMTP, self).__init__()

        self.batch_size = batch_size
        self.device = device
        
        self.model_dim = hyperparams_dict['model dim']# Dimension of transformer model ()
        self.ff_dim = hyperparams_dict['feedforward dim']
        self.layers_num = hyperparams_dict['layer number']
        self.classifier_dim = hyperparams_dict['classifier dim']
        self.head_num = hyperparams_dict['head number']
        self.n_mode = hyperparams_dict['number of modes']
        self.multi_modal = parameters.MULTI_MODAL
        
        self.prob_output = hyperparams_dict['probabilistic output']
        self.in_seq_len = parameters.IN_SEQ_LEN
        self.tgt_seq_len = parameters.TGT_SEQ_LEN
        self.decoder_in_dim = 2
        self.input_dim = 18
        self.mode_output_dim = self.n_mode
        if self.multi_modal == False:
            raise(ValueError('Single Modality not supported'))
        if self.n_mode != 3:
            raise(ValueError('Mode counts should be equal to manouvre counts (3).'))
        
        if self.prob_output:
            self.output_dim = 5 # muY, muX, sigY, sigX, rho 
        else:
            self.output_dim = 2
        
        self.dropout = nn.Dropout(drop_prob)
        
        ''' 1. Positional encoder: '''
        self.positional_encoder = PositionalEncoding(dim_model=self.model_dim, dropout_p=drop_prob, max_len=100)
        
        ''' 2. Transformer Encoder: '''
        self.encoder_embedding = nn.Linear(self.input_dim, self.model_dim)
        encoder_layers = nn.TransformerEncoderLayer(self.model_dim, self.head_num, self.ff_dim, drop_prob, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.layers_num)
        
        ''' 3. Transformer Decoder: '''
        self.decoder_embedding = nn.Linear(self.decoder_in_dim, self.model_dim)
        self.man_decoder_embedding = nn.Linear(2, self.model_dim)
        
        lk_decoder_layers = nn.TransformerDecoderLayer(self.model_dim, self.head_num, self.ff_dim, drop_prob, batch_first = True)
        rlc_decoder_layers = nn.TransformerDecoderLayer(self.model_dim, self.head_num, self.ff_dim, drop_prob, batch_first = True)
        llc_decoder_layers = nn.TransformerDecoderLayer(self.model_dim, self.head_num, self.ff_dim, drop_prob, batch_first = True)
        
        self.lk_transformer_decoder = nn.TransformerDecoder(lk_decoder_layers, self.layers_num)
        self.rlc_transformer_decoder = nn.TransformerDecoder(rlc_decoder_layers, self.layers_num)
        self.llc_transformer_decoder = nn.TransformerDecoder(llc_decoder_layers, self.layers_num)
        ''' 5. Trajectory Output '''
        self.lk_trajectory_fc = nn.Linear(self.model_dim, self.output_dim)
        self.rlc_trajectory_fc = nn.Linear(self.model_dim, self.output_dim)
        self.llc_trajectory_fc = nn.Linear(self.model_dim, self.output_dim)
        
        
        ''' 6. Manouvre Output '''
        self.dec_mode_fc1 = nn.Linear(self.in_seq_len*self.model_dim, self.classifier_dim)
        self.dec_mode_fc2 = nn.Linear(self.classifier_dim, self.mode_output_dim)

    def forward(self, x, y, y_mask):      
        encoder_out = self.encoder_forward(x)
        mode_prob_pred = self.mode_decoder_forward(encoder_out)
        traj_pred = self.traj_decoder_forward(y, y_mask, encoder_out)
        
        return {'traj_pred':traj_pred, 'mode_prob_pred': mode_prob_pred}
    
    def encoder_forward(self, x):
        #encoder
        x = self.encoder_embedding(x)
        x = self.positional_encoder(x)
        encoder_out = self.transformer_encoder(x)
        
        return encoder_out
    
    def mode_decoder_forward(self, encoder_out):
        encoder_out_flattened = encoder_out.reshape(self.batch_size, self.in_seq_len*self.model_dim)
        mode_prob_pred = self.dec_mode_fc2(F.relu(self.dec_mode_fc1(encoder_out_flattened)))
        return mode_prob_pred

    def traj_decoder_forward(self, y, y_mask, encoder_out):
        encoder_out_flattened = encoder_out.reshape(self.batch_size, self.in_seq_len*self.model_dim)    
        
        #traj decoder
        lk_y = self.decoder_embedding(y[:,:,:self.decoder_in_dim])
        lk_y = self.positional_encoder(lk_y)
        lk_decoder_out = self.lk_transformer_decoder(lk_y, encoder_out, tgt_mask = y_mask)
        
        rlc_y = self.decoder_embedding(y[:,:,:self.decoder_in_dim])
        rlc_y = self.positional_encoder(rlc_y)
        rlc_decoder_out = self.rlc_transformer_decoder(rlc_y, encoder_out, tgt_mask = y_mask)
        
        llc_y = self.decoder_embedding(y[:,:,:self.decoder_in_dim])
        llc_y = self.positional_encoder(llc_y)
        llc_decoder_out = self.llc_transformer_decoder(llc_y, encoder_out, tgt_mask = y_mask)

        #traj decoder linear layer
        
        lk_traj_pred = self.lk_trajectory_fc(lk_decoder_out)
        rlc_traj_pred = self.rlc_trajectory_fc(rlc_decoder_out)
        llc_traj_pred = self.llc_trajectory_fc(llc_decoder_out)

        if self.prob_output:
            lk_traj_pred = mf.prob_activation_func(lk_traj_pred)
            rlc_traj_pred = mf.prob_activation_func(rlc_traj_pred)
            llc_traj_pred = mf.prob_activation_func(llc_traj_pred) 
        
        traj_pred = torch.stack([lk_traj_pred, rlc_traj_pred, llc_traj_pred], dim=1) # lk =0, rlc=1, llc=2
            
        return traj_pred 