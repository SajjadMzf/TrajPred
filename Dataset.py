import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import numpy as np 
import h5py
import matplotlib.pyplot as plt
import sys
import pickle
from random import shuffle
import math
import pdb
import time

class LCDataset(Dataset):
    def __init__(self, 
    dataset_dir, 
    data_files,
    data_type, 
    index_file,
        #'max_in_seq_len, out_seq_len, skip_seq_len,
        # _B for balanced(_U for unbalanced),tr_ratio,
        # abb_val_ratio,val_ratio,test_ratio,  
    state_type = '',
    use_map_features = False,
    keep_plot_info = True, 
    states_min = 0, 
    states_max = 0, 
    force_recalc_start_indexes = False,
    import_states = False, # import min-max of states
    output_states_min = 0, 
    output_states_max = 0,
    deploy_data = False):
        '''
        Index File Format: IndexGroup_InSeqLen_OutSeqLen_SkipSeqLen_BalancedIndex_TrRatio_AbbValRatio_ValRatio_TeRatio.npy
        IndexGroup Option: Tr, Val, Te, AbbTr, AbbVal, AbbTe
        BalancedIndex Option: B for balanced, U for unbalanced.
        '''
        super(LCDataset, self).__init__()
        
        self.data_files = data_files
        #print(data_files)
        #exit()
        self.index_file = index_file
        print(index_file)
        self.main_dir = dataset_dir
        self.dataset_dirs = \
            [os.path.join(dataset_dir, data_file) for data_file in data_files]
        self.index_file_dirs = os.path.join(dataset_dir, self.index_file)
        self.file_size = []
        self.dataset_size = 0
        self.state_data_name = 'state_'+ state_type #+ '_data'
        self.data_type= data_type
        self.deploy_data = deploy_data
        self.parse_index_file()
        self.use_map_features = use_map_features
        
        
        self.keep_plot_info = keep_plot_info
        
        self.start_indexes = \
            self.get_samples_start_index(force_recalc_start_indexes)
        
        self.dataset_size = len(self.start_indexes)
        print('{}: {}'.format(self.index_group, self.dataset_size))
       
        if data_type == 'state':
            if import_states == True:
                self.states_min = states_min
                self.states_max = states_max
            else:
                self.states_min, self.states_max = \
                    self.get_features_range(self.state_data_name)

            for i in range(len(self.states_min)):
                if self.states_min[i] == self.states_max[i]:
                    print('Warning! Feature {} min and max values are equal!'.format(i))
                    self.states_max[i] += 1
        else:
            self.states_min = 0
            self.states_max = 1  
        
        if import_states == True:
            self.output_states_min = output_states_min
            self.output_states_max = output_states_max
        else:
            self.output_states_min, self.output_states_max = \
                self.get_features_range('output_states_data')
        
        for i in range(len(self.output_states_min)):
            if self.output_states_min[i] == self.output_states_max[i]:
                self.output_states_max[i] += np.finfo('float').eps
    
        self.load_data()

    def __len__(self):
        return self.dataset_size

    def parse_index_file(self):
        index_data = self.index_file.split('_')
        try:
            self.index_group = index_data[0]
            self.min_in_seq_len = int(index_data[1])
            self.max_in_seq_len = int(index_data[2])
            self.out_seq_len = int(index_data[3]) if self.index_group != 'De' else 0
            #self.end_of_seq_skip_len = int(index_data[3])
            self.unbalanced_status = index_data[4]
            if index_data[4] == 'B':
                self.unbalanced = False
            elif index_data[4] == 'U':
                self.unbalanced = True
            else: 
                raise(ValueError('wrong dataset format'))
            self.tr_ratio = float(index_data[5])
            self.abb_val_ratio = float(index_data[6])
            self.val_ratio = float(index_data[7]) 
            self.te_ratio = float(index_data[8]) 
            self.de_ratio = float(index_data[9])
        except:
            print('Wrong index file format.')

    def get_features_range(self , feature_name):
        states_min = []
        states_max = []
        for dataset_dir in self.dataset_dirs:    
            with h5py.File(dataset_dir, 'r') as f:
                state_data = f[feature_name]
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
                
    def get_samples_start_index(self,force_recalc = False):
        
        if force_recalc or (not os.path.exists(self.index_file_dirs)):
            samples_start_index = []
                                
            print('Extracing start indexes for all index groups')
            valid_tv = -1
            for file_itr, data_file in enumerate(self.dataset_dirs):
                #if file_itr>1:
                #    break
                print('File: {}'.format(data_file))
                with h5py.File(data_file, 'r') as f:
                    tv_ids = f['tv_data']
                    len_data_serie = tv_ids.shape[0]
                    current_tv = -1
                    for itr, tv_id in enumerate(tv_ids):
                        
                        if self.index_group == 'Te':
                            if tv_id != current_tv:
                                sample_seq_len =  self.min_in_seq_len+self.out_seq_len
                                if (itr+sample_seq_len) <= len_data_serie:
                                    if np.all(tv_ids[itr:(itr+sample_seq_len)] == tv_id):
                                        samples_start_index.append([])
                                        valid_tv+=1
                                        current_tv = tv_id
                                        for te_itr in range(itr+self.min_in_seq_len, len_data_serie-self.out_seq_len):
                                            in_seq_len = min(te_itr-itr, self.max_in_seq_len)
                                            if np.all(tv_ids[te_itr:(te_itr+self.out_seq_len)] == tv_id):
                                                print('{}/{}/{}'.\
                                                      format(in_seq_len, te_itr-in_seq_len, len_data_serie), end = '\r')
                                                samples_start_index[valid_tv]\
                                                    .append([file_itr, te_itr-in_seq_len, in_seq_len])
                                            else:
                                                break

                        elif self.index_group == 'De':
                            if tv_id != current_tv and (itr+self.min_in_seq_len) <= len_data_serie:
                                samples_start_index.append([])
                                valid_tv+=1
                                current_tv = tv_id
                                for te_itr in range(itr+self.min_in_seq_len, len_data_serie):
                                    
                                    if tv_ids[te_itr-1] == tv_id and tv_ids[te_itr-self.min_in_seq_len] == tv_id:
                                        in_seq_len = min(te_itr-itr, self.max_in_seq_len)
                                        print('{}/{}/{}'.format(in_seq_len, te_itr-in_seq_len, len_data_serie), end = '\r')
                                        samples_start_index[valid_tv].append([file_itr, te_itr-in_seq_len, in_seq_len])
                                    else:
                                        break
                        else:
                            for in_seq_len in range(self.min_in_seq_len, self.max_in_seq_len+1):
                                print('{}/{}/{}'.format(in_seq_len, itr, len(tv_ids)), end = '\r')
                                sample_seq_len =  in_seq_len+self.out_seq_len  #+self.end_of_seq_skip_len 
                                if (itr+sample_seq_len) <= len_data_serie:
                                    if np.all(tv_ids[itr:(itr+sample_seq_len)] == tv_id):
                                        if tv_id != current_tv:
                                            samples_start_index.append([])
                                            valid_tv+=1
                                            current_tv = tv_id
                                        samples_start_index[valid_tv].append([file_itr, itr, in_seq_len])                       
                        
                            
    
            samples_start_index = \
                [np.array(samples_start_index[itr]) for itr in range(len(samples_start_index))]
            shuffle(samples_start_index)
            

            n_tracks = len(samples_start_index)
            tr_samples = int(n_tracks*self.tr_ratio)
            abbVal_samples = int(n_tracks*self.abb_val_ratio)
            val_samples = int(n_tracks*self.val_ratio)
            te_samples = int(n_tracks*self.te_ratio)
            de_samples = int(n_tracks*self.de_ratio)
            index_groups = ['Tr', 'Val', 'AbbTe', 'Te', 'AbbTr', 'AbbVal', 'De']
            unbalanced_inds = ['U', 'B']
            
            start_indexes = {}
            start_indexes['B'] = {}
            start_indexes['U'] = {}
            
            start_indexes['U']['Tr'] = samples_start_index[:tr_samples]
            start_indexes['U']['Val'] = \
                samples_start_index[tr_samples:(tr_samples + val_samples)]
            start_indexes['U']['AbbTe'] = \
                samples_start_index[tr_samples:(tr_samples + val_samples)]
            start_indexes['U']['Te'] = \
                samples_start_index[(tr_samples + val_samples):\
                                    (tr_samples + val_samples + te_samples)]
            start_indexes['U']['AbbTr'] = \
                samples_start_index[abbVal_samples:tr_samples]
            start_indexes['U']['AbbVal'] = samples_start_index[:abbVal_samples]
            start_indexes['U']['De'] = \
                samples_start_index[(tr_samples+val_samples+ te_samples):\
                                    (tr_samples+val_samples+ te_samples+de_samples)]
            for index_group in index_groups: 
                print('Balancing {} dataset...'.format(index_group))
                #print(len(start_indexes['U'][index_group]))
                if len(start_indexes['U'][index_group]) == 0:
                    start_indexes['U'][index_group] = np.array([])
                    start_indexes['B'][index_group] = np.array([])
                else:
                    #print(index_group)
                    #print(start_indexes['U'][index_group])
                    start_indexes['U'][index_group] = \
                        np.concatenate(start_indexes['U'][index_group] , axis = 0)
                    start_indexes['B'][index_group] = \
                        self.balance_dataset(start_indexes['U'][index_group]) \
                            if index_group != 'De' else np.array([])
            
            for ub_ind in unbalanced_inds:
                index_file = modify_index_file(self.index_file, unbalanced_ind = ub_ind)
                for index_group in index_groups:
                    index_file = modify_index_file(index_file, index_group = index_group)
                    
                    index_file_dir = os.path.join(self.main_dir, index_file)
                    #samples_start_index = np.concatenate(start_indexes['B'][index_group] , axis = 0)
                    random_itrs = np.random.permutation(len(start_indexes[ub_ind][index_group]))
                    start_indexes[ub_ind][index_group] = start_indexes[ub_ind][index_group][random_itrs]
                    print('{}-{}: {}'.format(index_group, ub_ind, len(start_indexes[ub_ind][index_group])))
                    np.save(index_file_dir, start_indexes[ub_ind][index_group])
            
            samples_start_index = start_indexes[self.unbalanced_status][self.index_group]
                
        else:
            samples_start_index = np.load(self.index_file_dirs)
       
        return samples_start_index
    
    def balance_dataset(self, start_index):
        #force_recalc = True

        lc_count_in_lc_scenarios = 0
        lk_count_in_lc_scenarios = 0
        balanced_scenarios = np.zeros((len(start_index)))
        for itr in range(len(start_index)):
            print('{}/{}'.format(itr, len(start_index)), end = '\r')
            file_itr = start_index[itr, 0]
            with h5py.File(self.dataset_dirs[file_itr], 'r') as f: 
                #TODO you may save all labels in an array before this loop
                labels_data = f['labels']
                start_itr = start_index[itr,1]
                in_seq_len = start_index[itr,2]
                label = abs(labels_data[(start_itr+in_seq_len):(start_itr + in_seq_len + self.out_seq_len)])    
                balanced_scenarios[itr] = np.any(label)*2 \
                    # 2 is lc scenario, if there is a lc man at any time-step 
                    # of sample, considered it in balanced dataset
                if np.any(label):
                    lc_count_in_lc_scenarios += np.count_nonzero(label>0)
                    lk_count_in_lc_scenarios += np.count_nonzero(label==0)
                  
        if lc_count_in_lc_scenarios> lk_count_in_lc_scenarios + self.out_seq_len:
            lk_balanced_count = int((lc_count_in_lc_scenarios-lk_count_in_lc_scenarios)/self.out_seq_len)
            lk_args = np.argwhere(balanced_scenarios == 0)
            lk_balanced_args = np.random.permutation(lk_args[:,0])[:lk_balanced_count]
            balanced_scenarios[lk_balanced_args] = 1 # 1 is balanced lk scenario
        

        return start_index[balanced_scenarios>0]

    def load_data(self):
        self.state_data = []
        self.frame_data = [] 
        self.tv_data = []
        self.output_data = [] 
        self.man_data = []
        
        start_time = time.time()
        for dataset_dir in self.dataset_dirs:
            with h5py.File(dataset_dir, 'r') as f:
                state_data_i = f[self.state_data_name]
                state_data_i = state_data_i[:]
                
                state_data_i = (state_data_i-self.states_min)/(self.states_max-self.states_min)
                state_data_i = torch.from_numpy(state_data_i.\
                                                astype(np.float32))
                self.state_data.append(state_data_i)
                output_data_i = f['output_states_data']
                output_data_i = output_data_i[:]
                output_data_i = (output_data_i-self.output_states_min)/(self.output_states_max-self.output_states_min)
                output_data_i = torch.from_numpy(output_data_i.\
                                                 astype(np.float32))
                self.output_data.append(output_data_i)
                man_data_i = f['labels']
                man_data_i = man_data_i[:]
                self.man_data.append(man_data_i)
                frame_data_i = f['frame_data']
                self.frame_data.append(frame_data_i[:])
                tv_data_i = f['tv_data']
                self.tv_data.append(tv_data_i[:])
        end_time = time.time()
        print('Data Loaded in {} sec'.format(end_time-start_time))


         
    def __getitem__(self, idx):
        file_itr = self.start_indexes[idx,0]
        start_index = self.start_indexes[idx,1]
        in_seq_len = self.start_indexes[idx, 2]
        
        if self.data_type == 'state':
            state_data = self.state_data[file_itr]
            states = state_data[start_index:(start_index+in_seq_len)]
            #seq_len, feature size
            p2d = (0,0, self.max_in_seq_len-in_seq_len,0)
            states = F.pad(states, p2d, 'constant', 0)
            
            padding_mask = np.ones((self.max_in_seq_len), dtype=bool)
            padding_mask[self.max_in_seq_len-in_seq_len:] = False
            
            padding_mask = torch.from_numpy(padding_mask.astype(bool))
            data_output = [states, padding_mask]
          
        else:
            raise(ValueError('undefined data type'))

        if self.keep_plot_info:
            frame_data = self.frame_data[file_itr]
            tv_data = self.tv_data[file_itr]
            tv_id = tv_data[start_index] # constant number for all frames of same scenario  
            p1d = (self.max_in_seq_len-in_seq_len,0)
            frames =  frame_data[start_index:(start_index + in_seq_len + self.out_seq_len)]
            frames = torch.from_numpy(frames)
            frames = F.pad(frames, p1d, 'constant', -1)
            frames = frames.numpy()
            plot_output = [tv_id, frames, self.data_files[file_itr]]
        else:
            plot_output = ()
        
        output_state_data = self.output_data[file_itr]
        output_states = \
            output_state_data[(start_index):(start_index + in_seq_len + self.out_seq_len)]
        p2d = (0,0, self.max_in_seq_len-in_seq_len,0)
        output_states = F.pad(output_states, p2d, 'constant', -1)
        data_output.append(output_states)


        man_data = self.man_data[file_itr]
        man = np.absolute(\
            man_data[(start_index):(start_index + in_seq_len + self.out_seq_len)]\
                .astype(np.int64))
        man = torch.from_numpy(man)
        p1d = (self.max_in_seq_len-in_seq_len,0)
        man = F.pad(man, p1d, 'constant', -1)
        return data_output, man, plot_output

def get_index_file(p, d_class, index_group):
    '''
        Index File Format: IndexGroup_InSeqLen_OutSeqLen_SkipSeqLen_BalancedIndex_TrRatio_AbbValRatio_ValRatio_TeRatio.npy
        IndexGroup Option: Tr, Val, Te, AbbTr, AbbVal, AbbTe
        BalancedIndex Option: B for balanced, U for unbalanced.
    '''
    if p.ABLATION:
        index_group = 'Abb' + index_group
    if p.UNBALANCED:
        unbalanced_ind = 'U'
    else:
        unbalanced_ind = 'B'
    index_file = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.npy'\
        .format(index_group, p.MIN_IN_SEQ_LEN, p.MAX_IN_SEQ_LEN, p.TGT_SEQ_LEN,\
                 unbalanced_ind, d_class.TR_RATIO, d_class.ABBVAL_RATIO, \
                    d_class.VAL_RATIO, d_class.TE_RATIO, d_class.DE_RATIO, \
                        d_class.SELECTED_DATASET)
    return index_file

def modify_index_file(index_file,index_group = None, unbalanced_ind = None):
    
    index_list = index_file.split('_')
    if index_group is not None:
        index_list[0] = index_group
    if unbalanced_ind is not None:
        index_list[4] = unbalanced_ind
    index_file = '_'.join(index_list)
    return index_file
                