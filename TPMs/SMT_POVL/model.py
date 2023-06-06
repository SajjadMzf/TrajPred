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

from ..MMnTP import utils

class SMT_POVL(nn.Module): 
    def __init__(self, batch_size, device, hyperparams_dict, parameters, drop_prob = 0.1):
        super(SMT_POVL, self).__init__()

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
        self.max_in_seq_len = parameters.MAX_IN_SEQ_LEN
        self.tgt_seq_len = parameters.TGT_SEQ_LEN
        self.decoder_in_dim = 2
        
        self.input_dim = parameters.FEATURE_SIZE+ parameters.MAP_FEATURES \
            if parameters.USE_MAP_FEATURES else parameters.FEATURE_SIZE
        
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
        
        decoder_layers = nn.TransformerDecoderLayer(self.model_dim, self.head_num, self.ff_dim, drop_prob, batch_first = True)
        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, self.layers_num)
        ''' 5. Trajectory Output '''
        self.lk_traj_fc = nn.Linear(self.model_dim, self.output_dim)
        self.rlc_traj_fc = nn.Linear(self.model_dim, self.output_dim)
        self.llc_traj_fc = nn.Linear(self.model_dim, self.output_dim)
        
        
        ''' 6. Manouvre Output '''
        self.dec_mode_fc1 = nn.Linear(self.max_in_seq_len*self.model_dim, self.classifier_dim)
        self.dec_mode_fc2 = nn.Linear(self.classifier_dim, self.mode_output_dim)

    def forward(self, x, y, input_padding_mask, y_mask):      
        encoder_out = self.encoder_forward(x, input_padding_mask)
        mode_prob_pred = self.mode_decoder_forward(encoder_out, input_padding_mask)
        traj_pred = self.traj_decoder_forward(y, y_mask, encoder_out, input_padding_mask)
        
        return {'traj_pred':traj_pred, 'mode_prob_pred': mode_prob_pred}
    
    def encoder_forward(self, x, input_padding_mask):
        #encoder
        x = self.encoder_embedding(x)
        x = self.positional_encoder(x)
        encoder_out = self.transformer_encoder(x, src_key_padding_mask = input_padding_mask)
        
        return encoder_out
    
    def mode_decoder_forward(self, encoder_out, input_padding_mask):
        input_padding_mask = torch.unsqueeze(input_padding_mask, dim = -1)
        encoder_out = torch.mul(encoder_out, torch.logical_not(input_padding_mask))
        encoder_out_flattened = encoder_out.reshape(self.batch_size, self.max_in_seq_len*self.model_dim)
        mode_prob_pred = self.dec_mode_fc2(F.relu(self.dec_mode_fc1(encoder_out_flattened)))
        return mode_prob_pred

    def traj_decoder_forward(self, y, y_mask, encoder_out, input_padding_mask):
        encoder_out_flattened = encoder_out.reshape(self.batch_size, self.max_in_seq_len*self.model_dim)    
        
        #traj decoder
        y = self.decoder_embedding(y[:,:,:self.decoder_in_dim])
        y = self.positional_encoder(y)
        decoder_out = self.transformer_decoder(y, encoder_out, tgt_mask = y_mask,
                                                memory_key_padding_mask = input_padding_mask)
        
        #traj decoder linear layer
        
        lk_traj_pred = self.lk_traj_fc(decoder_out)
        rlc_traj_pred = self.rlc_traj_fc(decoder_out)
        llc_traj_pred = self.llc_traj_fc(decoder_out)

        if self.prob_output:
            lk_traj_pred = utils.prob_activation_func(lk_traj_pred)
            rlc_traj_pred = utils.prob_activation_func(rlc_traj_pred)
            llc_traj_pred = utils.prob_activation_func(llc_traj_pred) 
        if self.multi_modal:
            traj_pred = torch.stack([lk_traj_pred, rlc_traj_pred, llc_traj_pred], dim=1) # lk =0, rlc=1, llc=2
        else:
            traj_pred = torch.stack([lk_traj_pred, lk_traj_pred, lk_traj_pred], dim=1)    
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
  