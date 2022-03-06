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
import params
import utils
import math



class TransformerClassifier(nn.Module): 
    def __init__(self, batch_size, device, hyperparams_dict, parameters, drop_prob = 0.1):
        super(TransformerClassifier, self).__init__()

        self.batch_size = batch_size
        self.device = device
        
        self.model_dim = hyperparams_dict['model dim']
        self.ff_dim = hyperparams_dict['feedforward dim']
        self.mlp_dim = hyperparams_dict['mlp dim']
        self.layers_num = hyperparams_dict['layer number']
        self.head_num = hyperparams_dict['head number']
        self.task = hyperparams_dict['task']
        self.in_seq_len = parameters.IN_SEQ_LEN
        self.input_dim = 18

        ##### Positional encoder:
        pos = torch.arange(0.0, self.in_seq_len).unsqueeze(1)
        pos_encoding = torch.zeros((self.in_seq_len, self.model_dim))

        sin_den = 10000 ** (torch.arange(0.0, self.model_dim, 2)/self.model_dim) # sin for even item of position's dimension
        cos_den = 10000 ** (torch.arange(1.0, self.model_dim, 2)/self.model_dim) # cos for odd 

        pos_encoding[:, 0::2] = torch.sin(pos / sin_den) 
        pos_encoding[:, 1::2] = torch.cos(pos / cos_den)

        # Shape (pos_embedding) --> [max len, d_model]
        # Adding one more dimension in-between
        pos_encoding = pos_encoding.unsqueeze(0)
        # Shape (pos_embedding) --> [1, max len, d_model]

        self.dropout = nn.Dropout(drop_prob)
        self.register_buffer('pos_encoding', pos_encoding)
        ######

        # Only using Encoder of Transformer model
        encoder_layers = nn.TransformerEncoderLayer(self.model_dim, self.head_num, self.ff_dim, drop_prob, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.layers_num)
        self.embedding_layer = nn.Linear(self.input_dim, self.model_dim)
        if self.task == params.CLASSIFICATION or self.task == params.DUAL:
            self.fc1 = nn.Linear(self.model_dim, self.mlp_dim)
            self.fc2 = nn.Linear(self.mlp_dim,3)
        
        if self.task == params.REGRESSION or self.task == params.DUAL:
            self.fc1_ttlc = nn.Linear(self.model_dim, 512)
            self.fc2_ttlc = nn.Linear(512,1)

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
    
    def forward(self, x):#TODO make embedding works for sequence data
        x = x[0]
        #x = x.reshape(self.batch_size*self.in_seq_len, self.input_dim)
        x_list = []
        for i in range(self.in_seq_len):
            #print(i)
            x_temp = self.embedding_layer(x[:,i,:])
            x_temp = torch.unsqueeze(x_temp, 1)
            x_list.append(x_temp)
        x = torch.cat(x_list, dim = 1)
        #print(x.shape)
        #exit()
        #x = self.embedding_layer(x)
        #x = x.reshape(self.batch_size, self.in_seq_len, self.model_dim)
        
        x = self.dropout(x + self.pos_encoding)
        
        out = self.transformer_encoder(x)
        transformer_out = out.mean(dim = 1)

        if self.task == params.CLASSIFICATION or self.task == params.DUAL:
            lc_pred = self.lc_forward(transformer_out)
        else:
            lc_pred = 0
        
        if self.task == params.REGRESSION or self.task == params.DUAL:
            ttlc_pred = self.ttlc_forward(transformer_out)
        else:
            ttlc_pred = 0
        
        return {'lc_pred':lc_pred, 'ttlc_pred':ttlc_pred, 'features': transformer_out}


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
        
        if self.task == params.CLASSIFICATION or self.task == params.DUAL:
            self.fc1 = nn.Linear(self.hidden_dim, 128)
            self.fc2 = nn.Linear(128,3)
        
        if self.task == params.REGRESSION or self.task == params.DUAL:
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
        
        if self.task == params.CLASSIFICATION or self.task == params.DUAL:
            lc_pred = self.lc_forward(lstm_out)
        else:
            lc_pred = 0
        
        if self.task == params.REGRESSION or self.task == params.DUAL:
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

        if self.task == params.CLASSIFICATION or self.task == params.DUAL:
                self.fc1 = nn.Linear(self.input_dim*self.in_seq_len, self.hidden_dim)
                self.fc2 = nn.Linear(self.hidden_dim,3)
            
        if self.task == params.REGRESSION or self.task == params.DUAL:
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
        if self.task == params.CLASSIFICATION or self.task == params.DUAL:
            lc_pred = self.lc_forward(x_in)
        else:
            lc_pred = 0
        
        if self.task == params.REGRESSION or self.task == params.DUAL:
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
        
        if self.task == params.CLASSIFICATION or self.task == params.DUAL:
            self.fc1 = nn.Linear(self.hidden_dim, 128)
            self.fc2 = nn.Linear(128,3)
        
        if self.task == params.REGRESSION or self.task == params.DUAL:
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
        
        if self.task == params.CLASSIFICATION or self.task == params.DUAL:
            lc_pred = self.lc_forward(lstm_out)
        else:
            lc_pred = 0
        
        if self.task == params.REGRESSION or self.task == params.DUAL:
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
        
        if self.task == params.CLASSIFICATION or self.task == params.DUAL:
            self.fc1 = nn.Linear(self.hidden_dim, 128)
            self.fc2 = nn.Linear(128,3)
        
        if self.task == params.REGRESSION or self.task == params.DUAL:
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
        
        if self.task == params.CLASSIFICATION or self.task == params.DUAL:
            lc_pred = self.lc_forward(gru_out)
        else:
            lc_pred = 0
        
        if self.task == params.REGRESSION or self.task == params.DUAL:
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
        
        if self.task == params.CLASSIFICATION or self.task == params.DUAL:
            self.fc1 = nn.Linear(2*11*13*self.num_channels, 128)
            self.fc2 = nn.Linear(128,3)
        
        if self.task == params.REGRESSION or self.task == params.DUAL:
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
        if self.task == params.CLASSIFICATION or self.task == params.DUAL:
            lc_pred = self.lc_forward(conv6_out)
        else:
            lc_pred = 0
        
        if self.task == params.REGRESSION or self.task == params.DUAL:
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

        if self.task == params.CLASSIFICATION or self.task == params.DUAL:
            self.fc3 = nn.Linear(12*26*self.num_channels,128)# 6*13
            self.fc4 = nn.Linear(128, 3)

        if self.task == params.REGRESSION or self.task == params.DUAL:
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
        
        if self.task == params.CLASSIFICATION or self.task == params.DUAL:
            lc_pred = self.lc_forward(attended_features)
        else:
            lc_pred = 0
        
        if self.task == params.REGRESSION or self.task == params.DUAL:
            ttlc_pred = self.ttlc_forward(attended_features)
        else:
            ttlc_pred = 0
        return {'lc_pred':lc_pred, 'ttlc_pred':ttlc_pred, 'features': conv6_out, 'attention': attention_coef}
        

