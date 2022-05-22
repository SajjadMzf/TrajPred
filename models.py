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
import constants
import math
from debugging_utils import *


class LSTM_EncDec(nn.Module):

    def __init__(self, batch_size, device, hyperparams_dict, parameters, drop_prob = 0.5):
        super(LSTM_EncDec, self).__init__()
        
        self.batch_size = batch_size
        self.device = device
        
        self.hidden_dim = hyperparams_dict['hidden dim']
        self.num_layers = hyperparams_dict['layer number']
        self.task = hyperparams_dict['task']
        self.multi_modal = hyperparams_dict['multi modal']
        self.prob_output = hyperparams_dict['probabilistic output']
        self.in_seq_len = parameters.IN_SEQ_LEN
        self.output_dim = 2
        self.dropout = nn.Dropout(drop_prob)
        
        if parameters.TV_ONLY:
            self.input_dim = 2
        else:
            self.input_dim = 18
        
        if self.man_based:
            self.input_dim += 3
            self.decoder_in_dim += 3
        
        if self.prob_output:
            self.output_dim = 5 # muY, muX, sigY, sigX, rho 
        else:
            self.output_dim = 2


        # encoder
        self.lstm_enc = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first = True, dropout= drop_prob)
        
        # decoder
        self.lstm_dec = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first = True, dropout= drop_prob)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        
        # Define the output layer
        
        

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def lstm_forward(self, x_in):
        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, seq_len, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        
        
        x = x_in.view(self.batch_size, self.in_seq_len, self.input_dim)
        lstm_out, self.hidden = self.lstm(x)
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        lstm_out = lstm_out.transpose(0,1)
        out = self.dropout(lstm_out[-1].view(self.batch_size, -1))
        return out
    
    def forward(self,x_in):
        lstm_out = self.lstm_forward(x_in)
        
        
        return {'lc_pred':lc_pred, 'ttlc_pred':ttlc_pred, 'features': lstm_out}
    
    def prob_activation_func(self,x):
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
        

class NovelTransformerTraj(nn.Module): 
    def __init__(self, batch_size, device, hyperparams_dict, parameters, drop_prob = 0.1):
        super(NovelTransformerTraj, self).__init__()

        self.batch_size = batch_size
        self.device = device
        
        self.model_dim = hyperparams_dict['model dim']# Dimension of transformer model ()
        self.ff_dim = hyperparams_dict['feedforward dim']
        self.classifier_dim = hyperparams_dict['classifier dim']
        self.layers_num = hyperparams_dict['layer number']
        self.head_num = hyperparams_dict['head number']
        self.task = hyperparams_dict['task']
        self.multi_modal = hyperparams_dict['multi modal']
        self.prob_output = hyperparams_dict['probabilistic output']
        self.in_seq_len = parameters.IN_SEQ_LEN
        self.input_dim = 18
        if self.prob_output:
            self.output_dim = 5 # muY, muX, sigY, sigX, rho 
        else:
            self.output_dim = 2
        self.decoder_in_dim = 2
        self.dropout = nn.Dropout(drop_prob)
        
        ''' 1. Positional encoder: '''
        self.positional_encoder = PositionalEncoding(dim_model=self.model_dim, dropout_p=drop_prob, max_len=100)
        
        ''' 2.a Temporal Transformer Encoder: '''
        self.temp_encoder_embedding = nn.Linear(self.input_dim, self.model_dim)
        temp_encoder_layers = nn.TransformerEncoderLayer(self.model_dim, self.head_num, self.ff_dim, drop_prob, batch_first = True)
        self.temp_transformer_encoder = nn.TransformerEncoder(temp_encoder_layers, self.layers_num)
        
        ''' 2.b Spatial Transformer Encoder: '''
        self.spat_encoder_embedding = nn.Linear(self.in_seq_len, self.model_dim)
        spat_encoder_layers = nn.TransformerEncoderLayer(self.model_dim, self.head_num, self.ff_dim, drop_prob, batch_first = True)
        self.spat_transformer_encoder = nn.TransformerEncoder(spat_encoder_layers, self.layers_num)
        

        self.decoder_embedding = nn.Linear(self.decoder_in_dim, self.model_dim)
        ''' 3.a Temporal Transformer Decoder: '''
        if self.multi_modal == False:
            temp_decoder_layers = nn.TransformerDecoderLayer(self.model_dim, self.head_num, self.ff_dim, drop_prob, batch_first = True)
            self.temp_transformer_decoder = nn.TransformerDecoder(temp_decoder_layers, self.layers_num)
        else:
            temp_lk_decoder_layers = nn.TransformerDecoderLayer(self.model_dim, self.head_num, self.ff_dim, drop_prob, batch_first = True)
            temp_rlc_decoder_layers = nn.TransformerDecoderLayer(self.model_dim, self.head_num, self.ff_dim, drop_prob, batch_first = True)
            temp_llc_decoder_layers = nn.TransformerDecoderLayer(self.model_dim, self.head_num, self.ff_dim, drop_prob, batch_first = True)
            
            self.temp_lk_transformer_decoder = nn.TransformerDecoder(temp_lk_decoder_layers, self.layers_num)
            self.temp_rlc_transformer_decoder = nn.TransformerDecoder(temp_rlc_decoder_layers, self.layers_num)
            self.temp_llc_transformer_decoder = nn.TransformerDecoder(temp_llc_decoder_layers, self.layers_num)
        
        ''' 3.b Spatial Transformer Decoder: '''
        if self.multi_modal == False:
            spat_decoder_layers = nn.TransformerDecoderLayer(self.model_dim, self.head_num, self.ff_dim, drop_prob, batch_first = True)
            self.spat_transformer_decoder = nn.TransformerDecoder(spat_decoder_layers, self.layers_num)
        else:
            spat_lk_decoder_layers = nn.TransformerDecoderLayer(self.model_dim, self.head_num, self.ff_dim, drop_prob, batch_first = True)
            spat_rlc_decoder_layers = nn.TransformerDecoderLayer(self.model_dim, self.head_num, self.ff_dim, drop_prob, batch_first = True)
            spat_llc_decoder_layers = nn.TransformerDecoderLayer(self.model_dim, self.head_num, self.ff_dim, drop_prob, batch_first = True)
            
            self.spat_lk_transformer_decoder = nn.TransformerDecoder(spat_lk_decoder_layers, self.layers_num)
            self.spat_rlc_transformer_decoder = nn.TransformerDecoder(spat_rlc_decoder_layers, self.layers_num)
            self.spat_llc_transformer_decoder = nn.TransformerDecoder(spat_llc_decoder_layers, self.layers_num)
        

        ''' 4. Trajectory Output '''
        if self.multi_modal == False:
            self.trajectory_fc = nn.Linear(self.model_dim, self.output_dim)
        else:
            self.lk_trajectory_fc = nn.Linear(self.model_dim, self.output_dim)
            self.rlc_trajectory_fc = nn.Linear(self.model_dim, self.output_dim)
            self.llc_trajectory_fc = nn.Linear(self.model_dim, self.output_dim)
        
        ''' 5. Manouvre Output '''
        self.man_fc = nn.Linear(self.model_dim,3)
    
    def forward(self, x, y, y_mask):
        
        
        #temporal encoder
        #print_shape('x',x)
        temp_x = self.temp_encoder_embedding(x)
        temp_x = self.positional_encoder(temp_x)
        temp_encoder_out = self.temp_transformer_encoder(temp_x)
        #print_shape('temporal encoder output', temp_encoder_out)
        
        spatial_x = torch.permute(x, (0, 2, 1))
        #print_shape('spatial x', spatial_x)
        spatial_x = self.spat_encoder_embedding(spatial_x)
        spatial_x = self.positional_encoder(spatial_x)
        spat_encoder_out = self.spat_transformer_encoder(spatial_x)
        #print_shape('spatial encoder output', spat_encoder_out)
        #exit()
        #temp decoder
        #print_shape('y',y)
        if self.multi_modal == False:
            y = self.decoder_embedding(y[:,0])
            y = self.positional_encoder(y)
            temp_decoder_out = self.temp_transformer_decoder(y, temp_encoder_out, tgt_mask = y_mask)
            #print_shape('temporal decoder output', temp_decoder_out)
        else:
            lk_y = self.decoder_embedding(y[:,0])
            lk_y = self.positional_encoder(lk_y)
            temp_lk_decoder_out = self.temp_lk_transformer_decoder(lk_y, temp_encoder_out, tgt_mask = y_mask)
            
            rlc_y = self.decoder_embedding(y[:,1])
            rlc_y = self.positional_encoder(rlc_y)
            temp_rlc_decoder_out = self.temp_rlc_transformer_decoder(rlc_y, temp_encoder_out, tgt_mask = y_mask)
            
            llc_y = self.decoder_embedding(y[:,2])
            llc_y = self.positional_encoder(llc_y)
            temp_llc_decoder_out = self.temp_llc_transformer_decoder(llc_y, temp_encoder_out, tgt_mask = y_mask)
            
        # spat decoder
        if self.multi_modal == False:
            spat_decoder_out = self.spat_transformer_decoder(temp_decoder_out, spat_encoder_out, tgt_mask = y_mask)
            #print_shape('spatial decoder output', spat_decoder_out)
        else:
            
            spat_lk_decoder_out = self.spat_lk_transformer_decoder(temp_lk_decoder_out, spat_encoder_out, tgt_mask = y_mask)
            spat_rlc_decoder_out = self.spat_rlc_transformer_decoder(temp_rlc_decoder_out, spat_encoder_out, tgt_mask = y_mask)
            spat_llc_decoder_out = self.spat_llc_transformer_decoder(temp_llc_decoder_out, spat_encoder_out, tgt_mask = y_mask)


        

        #trajectory prediction
        if self.multi_modal == False:
            traj_pred = self.trajectory_fc(spat_decoder_out)
            #print_shape('traj_pred', traj_pred)
            if self.prob_output:
                traj_pred = self.prob_activation_func(traj_pred)
            traj_pred = torch.stack([traj_pred], dim=1)
            #print_shape('traj_pred', traj_pred)
            
        else:
            lk_traj_pred = self.lk_trajectory_fc(spat_lk_decoder_out)
            rlc_traj_pred = self.rlc_trajectory_fc(spat_rlc_decoder_out)
            llc_traj_pred = self.llc_trajectory_fc(spat_llc_decoder_out)
            if self.prob_output:
                lk_traj_pred = self.prob_activation_func(lk_traj_pred)
                rlc_traj_pred = self.prob_activation_func(rlc_traj_pred)
                llc_traj_pred = self.prob_activation_func(llc_traj_pred)
            traj_pred = torch.stack([lk_traj_pred, rlc_traj_pred, llc_traj_pred], dim=1)
        man_pred = self.man_fc(spat_decoder_out)
        
        return {'traj_pred':traj_pred, 'man_pred': man_pred,'multi_modal': self.multi_modal}
    
    def get_y_mask(self, size) -> torch.tensor:
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

    def prob_activation_func(self,x):
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



class ConstantX(nn.Module):
    def __init__(self, batch_size, device, hyperparams_dict, parameters, drop_prob = 0.1):
        super(ConstantX, self).__init__()

        self.batch_size = batch_size
        self.device = device
        
        self.constant_parameter = hyperparams_dict['parameter']# Dimension of transformer model ()
        self.task = hyperparams_dict['task']
        self.in_seq_len = parameters.IN_SEQ_LEN
        self.out_seq_len = parameters.TGT_SEQ_LEN
        self.fps = parameters.FPS
        self.input_dim = 18
        self.output_dim = 2
        print('Constant Model should only be run with ours_states that includes velocity, acceleration features')
        

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
        #traj_pred = torch.cat((traj_pred, labels[:,:,0:1]), dim = -1)
        #traj_pred *= traj_labels[:,0:1,:].to(self.device)
        #print('traj label')
        #print(traj_labels[0])
        #print(self.traj_pred[0])
        #exit()
        return {'traj_pred':traj_pred.to(self.device), 'multi_modal': False}

class TransformerTraj(nn.Module): 
    def __init__(self, batch_size, device, hyperparams_dict, parameters, drop_prob = 0.1):
        super(TransformerTraj, self).__init__()

        self.batch_size = batch_size
        self.device = device
        
        self.model_dim = hyperparams_dict['model dim']# Dimension of transformer model ()
        self.ff_dim = hyperparams_dict['feedforward dim']
        self.classifier_dim = hyperparams_dict['classifier dim']
        self.layers_num = hyperparams_dict['layer number']
        self.head_num = hyperparams_dict['head number']
        self.task = hyperparams_dict['task']
        self.multi_modal = hyperparams_dict['multi modal']
        self.prob_output = hyperparams_dict['probabilistic output']
        self.man_based= parameters.MAN_BASED
        self.in_seq_len = parameters.IN_SEQ_LEN
        self.decoder_in_dim = 2
        
        if parameters.TV_ONLY:
            self.input_dim = 2
        else:
            self.input_dim = 18
        if self.man_based:
            self.input_dim += 3
            self.decoder_in_dim += 3
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
        if self.multi_modal == False:
            decoder_layers = nn.TransformerDecoderLayer(self.model_dim, self.head_num, self.ff_dim, drop_prob, batch_first = True)
            self.transformer_decoder = nn.TransformerDecoder(decoder_layers, self.layers_num)
        else:
            lk_decoder_layers = nn.TransformerDecoderLayer(self.model_dim, self.head_num, self.ff_dim, drop_prob, batch_first = True)
            rlc_decoder_layers = nn.TransformerDecoderLayer(self.model_dim, self.head_num, self.ff_dim, drop_prob, batch_first = True)
            llc_decoder_layers = nn.TransformerDecoderLayer(self.model_dim, self.head_num, self.ff_dim, drop_prob, batch_first = True)
            
            self.lk_transformer_decoder = nn.TransformerDecoder(lk_decoder_layers, self.layers_num)
            self.rlc_transformer_decoder = nn.TransformerDecoder(rlc_decoder_layers, self.layers_num)
            self.llc_transformer_decoder = nn.TransformerDecoder(llc_decoder_layers, self.layers_num)
        ''' 4. Classification Output '''
        self.classifier_fc1 = nn.Linear(self.model_dim, self.classifier_dim)
        self.classifier_fc2 = nn.Linear(self.classifier_dim,3)

        ''' 5. Trajectory Output '''
        if self.multi_modal == False:
            self.trajectory_fc = nn.Linear(self.model_dim, self.output_dim)
        else:
            self.lk_trajectory_fc = nn.Linear(self.model_dim, self.output_dim)
            self.rlc_trajectory_fc = nn.Linear(self.model_dim, self.output_dim)
            self.llc_trajectory_fc = nn.Linear(self.model_dim, self.output_dim)
        ''' 6. Manouvre Output '''
        self.man_fc = nn.Linear(self.model_dim, 3)
    
    def forward(self, x, y, y_mask):
        #print(len(x))
        #print_shape('x',x)
        #encoder
        x = self.encoder_embedding(x)
        x = self.positional_encoder(x)
        encoder_out = self.transformer_encoder(x)
        
        #decoder
        if self.multi_modal == False:
            y = self.decoder_embedding(y[:,0])
            y = self.positional_encoder(y)
            decoder_out = self.transformer_decoder(y, encoder_out, tgt_mask = y_mask)
        else:
            lk_y = self.decoder_embedding(y[:,0])
            lk_y = self.positional_encoder(lk_y)
            lk_decoder_out = self.lk_transformer_decoder(lk_y, encoder_out, tgt_mask = y_mask)
            
            rlc_y = self.decoder_embedding(y[:,1])
            rlc_y = self.positional_encoder(rlc_y)
            rlc_decoder_out = self.rlc_transformer_decoder(rlc_y, encoder_out, tgt_mask = y_mask)
            
            llc_y = self.decoder_embedding(y[:,2])
            llc_y = self.positional_encoder(llc_y)
            llc_decoder_out = self.llc_transformer_decoder(llc_y, encoder_out, tgt_mask = y_mask)

        #trajectory prediction
        if self.multi_modal == False:
            traj_pred = self.trajectory_fc(decoder_out)
            if self.prob_output:
                traj_pred = self.prob_activation_func(traj_pred)
            traj_pred = torch.stack([traj_pred], dim=1)
        else:
            lk_traj_pred = self.lk_trajectory_fc(lk_decoder_out)
            rlc_traj_pred = self.rlc_trajectory_fc(rlc_decoder_out)
            llc_traj_pred = self.llc_trajectory_fc(llc_decoder_out)
            if self.prob_output:
                lk_traj_pred = self.prob_activation_func(lk_traj_pred)
                rlc_traj_pred = self.prob_activation_func(rlc_traj_pred)
                llc_traj_pred = self.prob_activation_func(llc_traj_pred)
                
            traj_pred = torch.stack([lk_traj_pred, rlc_traj_pred, llc_traj_pred], dim=1) # lk =0, rlc=1, llc=2
        #print_shape('decoder_out', decoder_out)
        man_pred = self.man_fc(decoder_out)
        #print_shape('man_pred', man_pred)
        return {'traj_pred':traj_pred, 'man_pred': man_pred, 'multi_modal': self.multi_modal}
    
    def prob_activation_func(self,x):
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

    

    def get_y_mask(self, size) -> torch.tensor:
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




class VanillaLSTM(nn.Module):

    def __init__(self, batch_size, device, hyperparams_dict, parameters, drop_prob = 0.5):
        super(VanillaLSTM, self).__init__()
        
        self.batch_size = batch_size
        self.device = device
        
        self.hidden_dim = hyperparams_dict['hidden dim']
        self.num_layers = hyperparams_dict['layer number']
        self.only_tv = hyperparams_dict['tv only']
        self.task = hyperparams_dict['task']

        self.in_seq_len = parameters.IN_SEQ_LEN
        # Initial Convs
        if self.only_tv:
            self.input_dim = 2
        else:
            self.input_dim = 18

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first = True, dropout= drop_prob)
        self.dropout = nn.Dropout(drop_prob)
        # Define the output layer
        
        if self.task == constants.CLASSIFICATION or self.task == constants.DUAL:
            self.fc1 = nn.Linear(self.hidden_dim, 128)
            self.fc2 = nn.Linear(128,3)
        
        if self.task == constants.REGRESSION or self.task == constants.DUAL:
            self.fc1_ttlc = nn.Linear(self.hidden_dim, 512)
            self.fc2_ttlc = nn.Linear(512,1)
        
        

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def lstm_forward(self, x_in):
        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, seq_len, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        x_in = x_in[0]
        if self.only_tv:
            x_in = x_in[:,:,:2]
        x = x_in.view(self.batch_size, self.in_seq_len, self.input_dim)
        lstm_out, self.hidden = self.lstm(x)
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        lstm_out = lstm_out.transpose(0,1)
        out = self.dropout(lstm_out[-1].view(self.batch_size, -1))
        return out
    
    def ttlc_forward(self, x):
        x = self.dropout(x)
        out = F.relu(self.fc1_ttlc(x))
        out = self.dropout(out)
        out = F.relu(self.fc2_ttlc(out))
        return out
    
    def lc_forward(self, x):
        x = self.dropout(x)
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

    def forward(self,x_in):
        lstm_out = self.lstm_forward(x_in)
        
        if self.task == constants.CLASSIFICATION or self.task == constants.DUAL:
            lc_pred = self.lc_forward(lstm_out)
        else:
            lc_pred = 0
        
        if self.task == constants.REGRESSION or self.task == constants.DUAL:
            ttlc_pred = self.ttlc_forward(lstm_out)
        else:
            ttlc_pred = 0
        
        return {'lc_pred':lc_pred, 'ttlc_pred':ttlc_pred, 'features': lstm_out}


# Previous methods
class MLP(nn.Module):

    def __init__(self, batch_size, device, hyperparams_dict, parameters, drop_prob = 0.5):
        super(MLP, self).__init__()
        
        self.batch_size = batch_size
        self.device = device
        
        self.hidden_dim = hyperparams_dict['hidden dim']
        self.only_tv = hyperparams_dict['tv only']
        self.task = hyperparams_dict['task']

        self.in_seq_len = parameters.IN_SEQ_LEN
        # Initial Convs
        if self.only_tv:
            self.input_dim = 2
        else:
            self.input_dim = 18
        
        self.dropout = nn.Dropout(drop_prob)

        if self.task == constants.CLASSIFICATION or self.task == constants.DUAL:
                self.fc1 = nn.Linear(self.input_dim*self.in_seq_len, self.hidden_dim)
                self.fc2 = nn.Linear(self.hidden_dim,3)
            
        if self.task == constants.REGRESSION or self.task == constants.DUAL:
            self.fc1_ttlc = nn.Linear(self.input_dim*self.in_seq_len, 512)
            self.fc2_ttlc = nn.Linear(512,1)

    def lc_forward(self,x_in):
        
        x_in = x_in[0]
        if self.only_tv:
            x_in = x_in[:,:,:2]
        x = x_in.view(self.batch_size, self.in_seq_len * self.input_dim)
        h1 = F.relu(self.fc1(x))
        out = self.dropout(h1)
        out = self.fc2(out)
        return out

    def ttlc_forward(self,x_in):
        
        x_in = x_in[0]
        if self.only_tv:
            x_in = x_in[:,:,:2]
        x = x_in.view(self.batch_size, self.in_seq_len * self.input_dim)
        h1 = F.relu(self.fc1_ttlc(x))
        out = self.dropout(h1)
        out = self.fc2_ttlc(out)

        return out
    
    def forward(self, x_in):
        features = 0
        if self.task == constants.CLASSIFICATION or self.task == constants.DUAL:
            lc_pred = self.lc_forward(x_in)
        else:
            lc_pred = 0
        
        if self.task == constants.REGRESSION or self.task == constants.DUAL:
            ttlc_pred = self.ttlc_forward(x_in)
        else:
            ttlc_pred = 0
        
        return {'lc_pred':lc_pred, 'ttlc_pred':ttlc_pred, 'features': features}

        


class VanillaLSTM(nn.Module):

    def __init__(self, batch_size, device, hyperparams_dict, parameters, drop_prob = 0.5):
        super(VanillaLSTM, self).__init__()
        
        self.batch_size = batch_size
        self.device = device
        
        self.hidden_dim = hyperparams_dict['hidden dim']
        self.num_layers = hyperparams_dict['layer number']
        self.only_tv = hyperparams_dict['tv only']
        self.task = hyperparams_dict['task']

        self.in_seq_len = parameters.IN_SEQ_LEN
        # Initial Convs
        if self.only_tv:
            self.input_dim = 2
        else:
            self.input_dim = 18

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first = True, dropout= drop_prob)
        self.dropout = nn.Dropout(drop_prob)
        # Define the output layer
        
        if self.task == constants.CLASSIFICATION or self.task == constants.DUAL:
            self.fc1 = nn.Linear(self.hidden_dim, 128)
            self.fc2 = nn.Linear(128,3)
        
        if self.task == constants.REGRESSION or self.task == constants.DUAL:
            self.fc1_ttlc = nn.Linear(self.hidden_dim, 512)
            self.fc2_ttlc = nn.Linear(512,1)
        
        

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def lstm_forward(self, x_in):
        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, seq_len, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        x_in = x_in[0]
        if self.only_tv:
            x_in = x_in[:,:,:2]
        x = x_in.view(self.batch_size, self.in_seq_len, self.input_dim)
        lstm_out, self.hidden = self.lstm(x)
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        lstm_out = lstm_out.transpose(0,1)
        out = self.dropout(lstm_out[-1].view(self.batch_size, -1))
        return out
    
    def ttlc_forward(self, x):
        x = self.dropout(x)
        out = F.relu(self.fc1_ttlc(x))
        out = self.dropout(out)
        out = F.relu(self.fc2_ttlc(out))
        return out
    
    def lc_forward(self, x):
        x = self.dropout(x)
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

    def forward(self,x_in):
        lstm_out = self.lstm_forward(x_in)
        
        if self.task == constants.CLASSIFICATION or self.task == constants.DUAL:
            lc_pred = self.lc_forward(lstm_out)
        else:
            lc_pred = 0
        
        if self.task == constants.REGRESSION or self.task == constants.DUAL:
            ttlc_pred = self.ttlc_forward(lstm_out)
        else:
            ttlc_pred = 0
        
        return {'lc_pred':lc_pred, 'ttlc_pred':ttlc_pred, 'features': lstm_out}
        
class VanillaGRU(nn.Module):

    def __init__(self, batch_size, device, hyperparams_dict, parameters, drop_prob = 0.5):
        super(VanillaGRU, self).__init__()
        
        self.batch_size = batch_size
        self.device = device
        
        self.hidden_dim = hyperparams_dict['hidden dim']
        self.num_layers = hyperparams_dict['layer number']
        self.only_tv = hyperparams_dict['tv only']
        self.task = hyperparams_dict['task']

        self.in_seq_len = parameters.IN_SEQ_LEN
        # Initial Convs
        if self.only_tv:
            self.input_dim = 2
        else:
            self.input_dim = 18

        self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers, batch_first = True, dropout= drop_prob)
        self.dropout = nn.Dropout(drop_prob)
        # Define the output layer
        
        if self.task == constants.CLASSIFICATION or self.task == constants.DUAL:
            self.fc1 = nn.Linear(self.hidden_dim, 128)
            self.fc2 = nn.Linear(128,3)
        
        if self.task == constants.REGRESSION or self.task == constants.DUAL:
            self.fc1_ttlc = nn.Linear(self.hidden_dim, 512)
            self.fc2_ttlc = nn.Linear(512,1)
        
        

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def gru_forward(self, x_in):
        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, seq_len, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        x_in = x_in[0]
        if self.only_tv:
            x_in = x_in[:,:,:2]
        x = x_in.view(self.batch_size, self.in_seq_len, self.input_dim)
        gru_out, self.hidden = self.gru(x)
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        gru_out = gru_out.transpose(0,1)
        out = self.dropout(gru_out[-1].view(self.batch_size, -1))
        return out
    
    def ttlc_forward(self, x):
        x = self.dropout(x)
        out = F.relu(self.fc1_ttlc(x))
        out = self.dropout(out)
        out = F.relu(self.fc2_ttlc(out))
        return out
    
    def lc_forward(self, x):
        x = self.dropout(x)
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

    def forward(self,x_in):
        gru_out = self.gru_forward(x_in)
        
        if self.task == constants.CLASSIFICATION or self.task == constants.DUAL:
            lc_pred = self.lc_forward(gru_out)
        else:
            lc_pred = 0
        
        if self.task == constants.REGRESSION or self.task == constants.DUAL:
            ttlc_pred = self.ttlc_forward(gru_out)
        else:
            ttlc_pred = 0
        
        return {'lc_pred':lc_pred, 'ttlc_pred':ttlc_pred, 'features': gru_out}
        

class VanillaCNN(nn.Module):

    def __init__(self, batch_size, device, hyperparams_dict, parameters, drop_prob = 0.5):
        super(VanillaCNN, self).__init__()
        
        self.batch_size = batch_size
        self.device = device
        # Hyperparams:
        self.num_channels = hyperparams_dict['channel number']
        self.kernel_size = hyperparams_dict['kernel size']
        self.single_merged_ch = hyperparams_dict['merge channels']
        self.task = hyperparams_dict['task']
        self.padding = int((self.kernel_size -1)/2) 
        
        self.in_seq_len = parameters.IN_SEQ_LEN
        self.image_height = parameters.IMAGE_HEIGHT
        self.image_width = parameters.IMAGE_WIDTH 
        # Initial Convs
        if self.single_merged_ch:
            self.in_channel = self.in_seq_len
      
        self.init_conv1 = nn.Conv2d(self.in_channel,self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding)  # (100,75)
        self.init_pool2 = nn.MaxPool2d(2, padding = 1) 
        self.init_conv3 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding) # (51,38)
        self.init_pool4 = nn.MaxPool2d(2)
        self.init_conv5 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding) # (25, 19)
        self.init_pool6 = nn.MaxPool2d(2, padding = 1)
        self.dropout = nn.Dropout(drop_prob)
        
        if self.task == constants.CLASSIFICATION or self.task == constants.DUAL:
            self.fc1 = nn.Linear(2*11*13*self.num_channels, 128)
            self.fc2 = nn.Linear(128,3)
        
        if self.task == constants.REGRESSION or self.task == constants.DUAL:
            self.fc1_ttlc = nn.Linear(2*11*13*self.num_channels, 512)
            self.fc2_ttlc = nn.Linear(512,1)
        

    def conv_forward(self, x_in):
        
        x_in = x_in[0]
        if self.single_merged_ch:
            x_in = torch.mean(x_in, 2, True)
        x = x_in.view(-1, self.in_channel, self.image_height, self.image_width)
        conv1_out = self.init_conv1(x)
        conv2_out = F.relu(self.init_pool2(conv1_out))
        conv3_out = self.init_conv3(conv2_out)
        conv4_out = F.relu(self.init_pool4(conv3_out))     
        conv5_out = self.init_conv5(conv4_out) 
        conv6_out = F.relu(self.init_pool6(conv5_out)) 
        return (conv1_out, conv2_out, conv3_out, conv4_out, conv5_out, conv6_out)
    
    def lc_forward(self, conv_out):
        x = conv_out.view(-1, 2*11*13*self.num_channels)
        x = self.dropout(x)
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return out
            
    def ttlc_forward(self, conv_out):
        x = conv_out.view(-1, 2*11*13*self.num_channels)
        x = self.dropout(x)
        out = F.relu(self.fc1_ttlc(x))
        out = self.dropout(out)
        out = F.relu(self.fc2_ttlc(out))
        return out
    
    def forward(self,x_in):
        (conv1_out, conv2_out, conv3_out, conv4_out, conv5_out, conv6_out) = self.conv_forward(x_in)
        if self.task == constants.CLASSIFICATION or self.task == constants.DUAL:
            lc_pred = self.lc_forward(conv6_out)
        else:
            lc_pred = 0
        
        if self.task == constants.REGRESSION or self.task == constants.DUAL:
            ttlc_pred = self.ttlc_forward(conv6_out)
        else:
            ttlc_pred = 0
        
        return {'lc_pred':lc_pred, 'ttlc_pred':ttlc_pred, 'features': conv6_out}

        

class ATTCNN3(nn.Module):

    def __init__(self, batch_size, device, hyperparams_dict, parameters, drop_prob = 0.5):
        super(ATTCNN3, self).__init__()
        
        self.batch_size = batch_size
        self.device = device
        # Hyperparams:
        self.num_channels = hyperparams_dict['channel number']
        self.kernel_size = hyperparams_dict['kernel size']
        self.single_merged_ch = hyperparams_dict['merge channels']
        self.task = hyperparams_dict['task']
        self.padding = int((self.kernel_size -1)/2) 

        self.in_seq_len = parameters.IN_SEQ_LEN
        self.image_height = parameters.IMAGE_HEIGHT
        self.image_width = parameters.IMAGE_WIDTH 
        # Initial Convs
        if self.single_merged_ch:
            self.in_channel = self.in_seq_len


        self.init_conv1 = nn.Conv2d(self.in_channel,self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding)  # (100,75)
        self.init_pool2 = nn.MaxPool2d(2, padding = 1) 
        self.init_conv3 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding) # (51,38)
        self.init_pool4 = nn.MaxPool2d(2)
        self.init_conv5 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=1, padding = self.padding) # (25, 19)
        self.init_pool6 = nn.MaxPool2d(2, padding = 1)
        self.dropout = nn.Dropout(drop_prob)
        self.fc1 = nn.Linear(2*11*13*self.num_channels, 4)

        if self.task == constants.CLASSIFICATION or self.task == constants.DUAL:
            self.fc3 = nn.Linear(12*26*self.num_channels,128)# 6*13
            self.fc4 = nn.Linear(128, 3)

        if self.task == constants.REGRESSION or self.task == constants.DUAL:
            self.fc1_ttlc = nn.Linear(12*26*self.num_channels, 512)
            self.fc2_ttlc = nn.Linear(512,1)

    def conv_forward(self, x_in):
        
        x_in = x_in[0]
        if self.single_merged_ch:
            x_in = torch.mean(x_in, 2, True)
        x = x_in.view(-1, self.in_channel, self.image_height, self.image_width)
        conv1_out = self.init_conv1(x)
        conv2_out = F.relu(self.init_pool2(conv1_out))
        conv3_out = self.init_conv3(conv2_out)
        conv4_out = F.relu(self.init_pool4(conv3_out))     
        conv5_out = self.init_conv5(conv4_out) 
        conv6_out = F.relu(self.init_pool6(conv5_out)) 
        
        return conv6_out
    
    def attention_coef_forward(self, conv_out):
        x = conv_out.view(-1, 2*11*13*self.num_channels)
        out = F.softmax(self.fc1(x), dim = -1)
        return out

    def lc_forward(self, attended_features):        
        x = attended_features.view(-1, 26*12*self.num_channels)
        x = self.dropout(x)
        out = F.relu(self.fc3(x))
        out = self.dropout(out)
        out = self.fc4(out)
        return out

    def ttlc_forward(self, attended_features):
        
        x = attended_features.view(-1, 26*12*self.num_channels)
        x = self.dropout(x)
        out = F.relu(self.fc1_ttlc(x))
        out = self.dropout(out)
        out = F.relu(self.fc2_ttlc(out))
        return out

    
    def forward(self,x_in, seq_itr = 0):
        conv6_out= self.conv_forward(x_in)
        
        attention_coef = self.attention_coef_forward(conv6_out)
        front_right = conv6_out[:,:,:6, :13]
        front_left = conv6_out[:,:,5:,:13]
        back_right = conv6_out[:,:,:6,13:]
        back_left = conv6_out[:,:,5:,13:]
        
        front_right_coef = attention_coef[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(*front_right.size())
        front_left_coef = attention_coef[:,1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(*front_right.size())
        back_right_coef = attention_coef[:,2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(*front_right.size())
        back_left_coef = attention_coef[:,3].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(*front_right.size())
        if seq_itr<10:
            front_right =front_right * front_right_coef
            front_left =front_left * front_left_coef
            back_right =back_right * back_right_coef
            back_left =back_left * back_left_coef

        front = torch.cat((front_right, front_left), dim = 2)
        back = torch.cat((back_right, back_left), dim = 2)
        attended_features = torch.cat((front, back), dim = 3)
        
        if self.task == constants.CLASSIFICATION or self.task == constants.DUAL:
            lc_pred = self.lc_forward(attended_features)
        else:
            lc_pred = 0
        
        if self.task == constants.REGRESSION or self.task == constants.DUAL:
            ttlc_pred = self.ttlc_forward(attended_features)
        else:
            ttlc_pred = 0
        return {'lc_pred':lc_pred, 'ttlc_pred':ttlc_pred, 'features': conv6_out, 'attention': attention_coef}
        

