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

from . import utils

class POVL(nn.Module): 
    def __init__(self, batch_size, device, hyperparams_dict, parameters, drop_prob = 0.1):
        super(POVL, self).__init__()

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
        self.max_in_seq_len = parameters.MAX_IN_SEQ_LEN
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
        
        self.input_dim = parameters.FEATURE_SIZE
        self.map_dim = parameters.MAP_FEATURES
        #+ parameters.MAP_FEATURES if parameters.USE_MAP_FEATURES else parameters.FEATURE_SIZE
        self.MAP_ENCODER = parameters.USE_MAP_FEATURES

        if self.prob_output:
            self.output_dim = 5 # muY, muX, sigY, sigX, rho 
        else:
            self.output_dim = 2
        
        self.dropout = nn.Dropout(drop_prob)
        
        ''' 0. map encoder '''
        self.map_ff = nn.Linear(self.map_dim*15, 128)

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
        if self.MAP_ENCODER:
            dec_man_fc1_in_size = 128 + self.max_in_seq_len*self.model_dim
        else:
            dec_man_fc1_in_size =  self.max_in_seq_len*self.model_dim
        self.dec_man_fc1 = nn.Linear(dec_man_fc1_in_size, self.classifier_dim)
        self.dec_man_fc2 = nn.Linear(self.classifier_dim, self.man_output_dim)
        
    def forward(self, x, y,map, input_padding_mask, y_mask):      
        self.batch_size = x.shape[0]
        encoder_out, map_encoder_out = self.encoder_forward(x, map, input_padding_mask)
        man_pred = self.man_decoder_forward(encoder_out, map_encoder_out, input_padding_mask)
        traj_pred = self.traj_decoder_forward(y, input_padding_mask,y_mask, encoder_out)
        
        return {'traj_pred':traj_pred, 'man_pred': man_pred }
    
    def encoder_forward(self, x, map, input_padding_mask):
        #encoder
        self.batch_size = x.shape[0]
        
        x = self.encoder_embedding(x)
        x = self.positional_encoder(x)
        encoder_out = self.transformer_encoder(x, src_key_padding_mask = input_padding_mask)
        if self.MAP_ENCODER:
            map_encoder_out = F.relu(self.map_ff(map.reshape(self.batch_size, -1)))
        else:
            map_encoder_out = 0
        return encoder_out, map_encoder_out
    
    def man_decoder_forward(self, encoder_out, map_encoder_out, input_padding_mask):
        #print(encoder_out.shape)
        #print(input_padding_mask.shape)
        input_padding_mask = torch.unsqueeze(input_padding_mask, dim = -1)
        encoder_out = torch.mul(encoder_out, torch.logical_not(input_padding_mask))
        man_dec_in = encoder_out.reshape(self.batch_size, self.max_in_seq_len*self.model_dim)
        if self.MAP_ENCODER:
            man_dec_in = torch.cat((man_dec_in, map_encoder_out), dim=1)
        man_pred = self.dec_man_fc2(F.relu(self.dec_man_fc1(man_dec_in)))
        return man_pred

    def traj_decoder_forward(self, y, input_padding_mask, y_mask, encoder_out):
        
        
        
        #traj decoder
        y = self.decoder_embedding(y[:,0,:,:self.decoder_in_dim])
        y = self.positional_encoder(y)
        decoder_out = self.transformer_decoder(y, encoder_out, memory_key_padding_mask = input_padding_mask,tgt_mask = y_mask)
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
            lk_traj_pred = utils.prob_activation_func(lk_traj_pred)
            if self.multi_modal:
                rlc_traj_pred = utils.prob_activation_func(rlc_traj_pred)
                llc_traj_pred = utils.prob_activation_func(llc_traj_pred) 
        if self.multi_modal:
            traj_pred = torch.stack([lk_traj_pred, rlc_traj_pred, llc_traj_pred], dim=1) # lk =0, rlc=1, llc=2
        else:
            traj_pred = torch.stack([lk_traj_pred, lk_traj_pred, lk_traj_pred], dim=1) 
        #print_shape('decoder_out', decoder_out)
        
        return traj_pred 


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
  