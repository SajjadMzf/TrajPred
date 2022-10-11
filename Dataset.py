import torch
from torch.utils.data import Dataset
import os
import numpy as np 
import h5py
import matplotlib.pyplot as plt
from debugging_utils import *
class LCDataset(Dataset):
    def __init__(self, 
    dataset_dir, 
    data_files, 
    data_type, 
    in_seq_len,
    out_seq_len,
    end_of_seq_skip_len,
    state_type = '', 
    keep_plot_info = True, 
    states_min = 0, 
    states_max = 0, 
    unbalanced = False,
    force_recalc_start_indexes = False,
    import_states = False, # import min-max of states
    output_states_min = 0, 
    output_states_max = 0,
    deploy_data = False):

        super(LCDataset, self).__init__()
        self.data_files = data_files
        #print(data_files)
        #exit()
        self.dataset_dirs = [os.path.join(dataset_dir, data_file) for data_file in data_files]
        self.file_size = []
        self.dataset_size = 0
        self.state_data_name = 'state_'+ state_type + '_data'
        self.data_type= data_type
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.end_of_seq_skip_len = end_of_seq_skip_len
        self.unbalanced = unbalanced
        self.deploy_data = deploy_data
        if self.deploy_data:   
            self.total_seq_len = self.in_seq_len 
        else:
            self.total_seq_len = self.in_seq_len + self.out_seq_len
        if data_type == 'image':
            self.image_only = True
            self.state_only = False
        elif data_type == 'state':
            self.image_only = False
            self.state_only = True
        
        self.keep_plot_info = keep_plot_info
        
        
        self.start_indexes = []
        for data_file in self.dataset_dirs:
            start_indexes = self.get_samples_start_index(in_seq_len, out_seq_len, end_of_seq_skip_len, data_file, force_recalc = force_recalc_start_indexes)
            self.start_indexes.append(start_indexes)
        
        if unbalanced == False:
            self.balance_dataset(force_recalc_start_indexes)
        
        for start_index in self.start_indexes:
            self.dataset_size += len(start_index)
            self.file_size.append(self.dataset_size) #Cumulative Number of samples
        self.file_size = np.array(self.file_size)
        print_value('dataset_size', self.dataset_size)
        if data_type == 'state':
            if import_states == True:
                self.states_min = states_min
                self.states_max = states_max
            else:
                self.states_min, self.states_max = self.get_features_range(self.state_data_name)

            for i in range(len(self.states_min)):
                if self.states_min[i] == self.states_max[i]:
                    self.states_max[i] += np.finfo('float').eps
            


        else:
            self.states_min = 0
            self.states_max = 1
        
        
        
        if import_states == True:
            self.output_states_min = output_states_min
            self.output_states_max = output_states_max
        else:
            self.output_states_min, self.output_states_max = self.get_features_range('output_states_data')
        
        for i in range(len(self.output_states_min)):
            if self.output_states_min[i] == self.output_states_max[i]:
                self.output_states_max[i] += np.finfo('float').eps
        
        #print('State min: {},\n State max: {}'.format(self.states_min, self.states_max))

    def __len__(self):
        return self.dataset_size

    def get_features_range(self , feature_name):
        for dataset_dir in self.dataset_dirs:
            states_min = []
            states_max = []
            with h5py.File(dataset_dir, 'r') as f:
                
                state_data = f[feature_name]
                #state_data = state_data.reshape((state_data.shape[0]*state_data.shape[1],state_data.shape[2]))
                states_min.append(np.min(state_data, axis = 0))
                states_max.append(np.max(state_data, axis = 0))
        states_min = np.stack(states_min, axis = 0)
        states_max = np.stack(states_max, axis = 0)
        states_min = states_min.min(axis = 0)
        states_max = states_max.max(axis = 0)
        #print('diff')
        #print(states_min)
        #print(states_max-states_min)
        #assert(np.any(states_max-states_min) != 0)
        
        #print('states_min:{}'.format(states_min))
        #print('states_max:{}'.format(states_max))
        #print('states_diff:{}'.format(states_max-states_min))
        #print('states_min_all:{}'.format(np.all(states_min)))
        #print('states_max_all:{}'.format(np.all(states_max)))
        return states_min, states_max
                
    def get_samples_start_index(self, in_seq_len, out_seq_len, end_of_seq_skip_len, data_file, force_recalc = False):
        if self.deploy_data:
            deploy_flag = 'D'
            sample_length = in_seq_len
        else:
            deploy_flag = ''
            sample_length = in_seq_len+out_seq_len+end_of_seq_skip_len
        sample_start_indx_file = data_file.replace('.h5', '_start_indx_{}_{}_{}_{}.npy'.format(in_seq_len, out_seq_len, end_of_seq_skip_len, deploy_flag))
        if force_recalc or (not os.path.exists(sample_start_indx_file)):
            samples_start_index = []
            with h5py.File(data_file, 'r') as f:
                tv_ids = f['tv_data']
                len_scenario = tv_ids.shape[0]
                for itr, tv_id in enumerate(tv_ids):
                    if (itr+sample_length) <= len_scenario:
                        if np.all(tv_ids[itr:(itr+sample_length)] == tv_id):
                            samples_start_index.append(itr)         
            samples_start_index = np.array(samples_start_index)
            np.save(sample_start_indx_file, samples_start_index)
            print('Saving file: {}, total data: {}'.format(sample_start_indx_file, len(samples_start_index)))
        else:
            #print('loading {}'.format(sample_start_indx_file))
            samples_start_index = np.load(sample_start_indx_file)

        return samples_start_index
    
    def balance_dataset(self, force_recalc = False):
        #force_recalc = True
        unbalanced_indexes = self.start_indexes
        self.start_indexes = []
        for itr, data_file in enumerate(self.dataset_dirs):
            sample_start_indx_file = data_file.replace('.h5', '_start_indx_{}_{}_{}_B.npy'.format(self.in_seq_len, self.out_seq_len, self.end_of_seq_skip_len))
            if force_recalc or (not os.path.exists(sample_start_indx_file)):
                print('Balancing dataset file: {}'.format(sample_start_indx_file))
                with h5py.File(data_file, 'r') as f:
                    labels_data = f['labels']
                    balanced_scenarios = np.zeros_like(unbalanced_indexes[itr])
                    
                    lc_count_in_lc_scenarios = 0
                    lk_count_in_lc_scenarios = 0
                    
                    for start_index_itr, start_index in enumerate(unbalanced_indexes[itr]):
                        label = abs(labels_data[(start_index+self.in_seq_len):(start_index+self.total_seq_len)])
                        #lc_count += np.count_nonzero(label>0)
                        #lk_count += np.count_nonzero(label==0)
                        
                        balanced_scenarios[start_index_itr] = np.any(label)*2 # 2 is lc scenario
                        if np.any(label):
                            lc_count_in_lc_scenarios += np.count_nonzero(label>0)
                            lk_count_in_lc_scenarios += np.count_nonzero(label==0)
                     
                    #print_value('lc_count_in_lc_scenarios',lc_count_in_lc_scenarios)
                    #print_value('lk_count_in_lc_scenarios',lk_count_in_lc_scenarios)
                    if lc_count_in_lc_scenarios> lk_count_in_lc_scenarios + self.out_seq_len:
                        lk_balanced_count = int((lc_count_in_lc_scenarios-lk_count_in_lc_scenarios)/self.out_seq_len)
                        lk_args = np.argwhere(balanced_scenarios == 0)
                        #print(lk_args.shape)
                        lk_balanced_args = np.random.permutation(lk_args[:,0])[:lk_balanced_count]
                        #print(balanced_scenarios.shape)
                        #exit()
                        print_value('lk_balanced_count', lk_balanced_count)
                        balanced_scenarios[lk_balanced_args] = 1 # 1 is lk scenario

                balanced_start_indexes = unbalanced_indexes[itr][balanced_scenarios>0]
                np.save(sample_start_indx_file, balanced_start_indexes)
            else:
                #print('loading {}'.format(sample_start_indx_file))
                balanced_start_indexes = np.load(sample_start_indx_file)
            self.start_indexes.append(balanced_start_indexes)
        self.start_indexes = np.array(self.start_indexes)

    def __getitem__(self, idx):
        assert(idx != self.dataset_size)
        if torch.is_tensor(idx):
            idx = idx.tolist()       
        file_itr = np.argmax(self.file_size>idx)
        if file_itr>0:
            sample_itr = idx - self.file_size[file_itr-1]
        else:
            sample_itr = idx
        
    
        start_index = self.start_indexes[file_itr][sample_itr]
        
        with h5py.File(self.dataset_dirs[file_itr], 'r') as f:
            
            if self.data_type == 'state':
                state_data = f[self.state_data_name]
            
            labels_data = f['labels']
            ttlc_available = f['ttlc_available']

            if self.keep_plot_info:
                frame_data = f['frame_data']
                tv_data = f['tv_data']
                tv_id = tv_data[start_index] # constant number for all frames of same scenario  
                frames = frame_data[start_index:(start_index+self.total_seq_len)]
                plot_output = [tv_id, frames, self.data_files[file_itr]]
            else:
                plot_output = ()
            
            if self.image_only:
                image_data = f['image_data']
                images = torch.from_numpy(image_data[start_index:(start_index+self.in_seq_len)].astype(np.float32))
                data_output = [images]
            elif self.state_only:
                states = state_data[start_index:(start_index+self.in_seq_len)]
                
                states = (states-self.states_min)/(self.states_max-self.states_min)
                states = torch.from_numpy(states.astype(np.float32))
                data_output = [states]
            else:
                states = state_data[start_index:(start_index+self.in_seq_len)]
                states = (states-self.states_min)/(self.states_max-self.states_min)
                images = torch.from_numpy(image_data[start_index:(start_index+self.in_seq_len)].astype(np.float32))
                states = torch.from_numpy(states.astype(np.float32))
                data_output = [images, states]
            
            output_state_data = f['output_states_data']
            output_states = output_state_data[(start_index):(start_index+self.total_seq_len)]
            output_states = (output_states-self.output_states_min)/(self.output_states_max-self.output_states_min)
            
            output_states = torch.from_numpy(output_states.astype(np.float32))
            #label = torch.from_numpy(label)
            data_output.append(output_states)
                

            label = np.absolute(labels_data[(start_index):(start_index+self.total_seq_len)].astype(np.long))
            ttlc_status = ttlc_available[start_index].astype(np.long)  # constant number for all frames of same scenario        
        return data_output, label, plot_output, ttlc_status