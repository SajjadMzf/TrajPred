import torch
from torch.utils.data import Dataset
import os
import numpy as np 
import h5py
import matplotlib.pyplot as plt
class LCDataset(Dataset):
    def __init__(self, 
    dataset_dir, 
    data_files, 
    data_type, 
    state_type = '', 
    keep_plot_info = True, 
    traj_output = False, 
    states_min = 0, 
    states_max = 0, 
    output_states_min = 0, 
    output_states_max = 0):

        super(LCDataset, self).__init__()
        self.data_files = data_files
        self.dataset_dirs = [os.path.join(dataset_dir, data_file) for data_file in data_files]
        self.file_size = []
        self.dataset_size = 0
        self.state_data_name = 'state_'+ state_type + '_data'
        self.data_type= data_type
        self.traj_output = traj_output
        if data_type == 'image':
            self.image_only = True
            self.state_only = False
        elif data_type == 'state':
            self.image_only = False
            self.state_only = True
        
        self.keep_plot_info = keep_plot_info
        for data_file in self.dataset_dirs:
            npy_file = data_file.replace('.h5','.npy')
            dataset_in_file_size = np.load(npy_file)
            self.dataset_size += dataset_in_file_size #Total number of frames
            self.file_size.append(self.dataset_size) #Number of frames in each file
        
        self.file_size = np.array(self.file_size)
        if data_type == 'state':
            if np.all(states_min) == 0 or np.all(states_max) == 0:
                self.states_min, self.states_max = self.get_features_range(self.state_data_name)
            else:
                self.states_min = states_min
                self.states_max = states_max
        else:
            self.states_min = 0
            self.states_max = 1
        
        if traj_output:
            if np.all(output_states_min) == 0 or np.all(output_states_max) == 0:
                self.output_states_min, self.output_states_max = self.get_features_range('output_states_data')
            else:
                self.output_states_min = output_states_min
                self.output_states_max = output_states_max
        else:
            self.output_states_min = 0
            self.output_states_max = 1

    def __len__(self):
        return self.dataset_size

    def get_features_range(self , feature_name):
        for dataset_dir in self.dataset_dirs:
            states_min = []
            states_max = []
            with h5py.File(dataset_dir, 'r') as f:
                
                state_data = f[feature_name]
                #state_data = state_data.reshape((state_data.shape[0]*state_data.shape[1],state_data.shape[2]))
                states_min.append(np.min(np.min(state_data, axis = 0), axis = 0))
                states_max.append(np.max(np.max(state_data, axis = 0), axis = 0))
        states_min = np.stack(states_min, axis = 0)
        states_max = np.stack(states_max, axis = 0)
        states_min = states_min.min(axis = 0)
        states_max = states_max.max(axis = 0)
        return states_min, states_max
                


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()        
        file_itr = np.argmax(self.file_size>idx)
        if file_itr>0:
            sample_itr = idx - self.file_size[file_itr-1]-1
        else:
            sample_itr = idx-1
        
        with h5py.File(self.dataset_dirs[file_itr], 'r') as f:
            image_data = f['image_data']
            if self.data_type == 'state':
                state_data = f[self.state_data_name]
            
            labels_data = f['labels']
            ttlc_available = f['ttlc_available']

            if self.keep_plot_info:
                frame_data = f['frame_data']
                tv_data = f['tv_data']
                tv_id = tv_data[sample_itr]
                frames = frame_data[sample_itr]
                plot_output = [tv_id, frames, self.data_files[file_itr]]
            else:
                plot_output = ()
            
            if self.image_only:
                images = torch.from_numpy(image_data[sample_itr].astype(np.float32))
                data_output = [images]
            elif self.state_only:
                states = state_data[sample_itr]
                states = (states-self.states_min)/(self.states_max-self.states_min)
                states = torch.from_numpy(states.astype(np.float32))
                data_output = [states]
            else:
                states = state_data[sample_itr]
                states = (states-self.states_min)/(self.states_max-self.states_min)
                images = torch.from_numpy(image_data[sample_itr].astype(np.float32))
                states = torch.from_numpy(states.astype(np.float32))
                data_output = [images, states]
            if self.traj_output:
                output_state_data = f['output_states_data']
                output_states = output_state_data[sample_itr]
                output_states = (output_states-self.output_states_min)/(self.output_states_max-self.output_states_min)
                output_states = torch.from_numpy(output_states.astype(np.float32))
                data_output.append(output_states)

            label = labels_data[sample_itr].astype(np.long)
            ttlc_status = ttlc_available[sample_itr].astype(np.long)          
        return data_output, label, plot_output, ttlc_status

