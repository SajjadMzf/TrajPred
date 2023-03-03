from PIL import Image
import os
import cv2
import numpy as np 
import pickle
import h5py
import matplotlib.pyplot as plt
from matplotlib import gridspec
from moviepy.video.io.bindings import mplfig_to_npimage
from mpl_toolkits.mplot3d import Axes3D

import torch
from scipy.stats import multivariate_normal
from mpl_toolkits.axes_grid1 import make_axes_locatable

import time as time_func
import random
import math
import torch.utils.data as utils_data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pdb


import sys

sys.path.insert(1,'../')
import Dataset 
import TPMs 
import params 
import top_functions 
import kpis
import utils
import read_csv as rc
import param as p
import plot_func as pf



class BEVPlotter:
    """
    This class is for plotting results.
    """
    def __init__(
        self,
        fps:'Desired fps',
        result_file:'File contains results',
        dataset_name: 'Dataset  Name',
        force_resort):
        
        self.result_file = result_file
        ''' 1. Representation Properties:'''
        
        # 1.1 Scales and distances
        
        self.traj_vis_dir = "../../SAMPLE/" + p.model_name + "/traj_vis"
        if not os.path.exists(self.traj_vis_dir):
            os.makedirs(self.traj_vis_dir)
        
        self.sorted_scenarios, self.plot_ids, self.in_seq_len, self.tgt_seq_len = utils.read_scenarios(self.result_file, force_resort= force_resort)

        
    def plot(self, file_id_pairs = None, remove_ids_list = None):
        
        plot_ids = file_id_pairs if file_id_pairs is not None else self.plot_ids
        if len(plot_ids)>p.MAX_PLOTS:
            plot_ids = [plot_ids[i] for i in range(p.MAX_PLOTS)]
        self.remove_ids_list = remove_ids_list
        for i, plot_id in enumerate(plot_ids):
            self.plot_one_scenario(plot_id)
    
    
    def plot_one_scenario(self,plot_id):
        
        if plot_id not in self.plot_ids:
            print('file tv pair {}-{} cannot be found!'.format(plot_id[0], plot_id[1]))
            return 
        else:
            scenario_itr = self.plot_ids.index(plot_id)
        

        tv_id = self.sorted_scenarios[scenario_itr]['tv']
        data_file = self.sorted_scenarios[scenario_itr]['data_file']
        traj_min = self.sorted_scenarios[scenario_itr]['traj_min']
        traj_max = self.sorted_scenarios[scenario_itr]['traj_max']
        with open(p.map_paths[data_file], 'rb') as handle:
            map_data = pickle.load(handle)
        track_path = p.track_paths[data_file]
        print(track_path)
        print(p.map_paths[data_file])
        pickle_path = p.frame_pickle_paths[data_file]
        frames_data = rc.read_track_csv(track_path, pickle_path, group_by = 'frames', reload = False, fr_div = p.fr_div)
        
        driving_dir = map_data['driving_dir']
        
        print('FILE-TV: {}-{}, List of Available Frames: {}, dd:{}'.format(plot_id[0], plot_id[1], self.sorted_scenarios[scenario_itr]['times'], driving_dir ))
        
        
        np.set_printoptions(precision=2, suppress=True)
        
        
        # for each time-step
        images = []
        
        for j,time in enumerate(self.sorted_scenarios[scenario_itr]['times']):
            
            if p.PLOT_MAN== False:
                man_preds = []
                man_labels = []
            else:
                man_preds = self.sorted_scenarios[scenario_itr]['man_preds'][j]
                man_labels = self.sorted_scenarios[scenario_itr]['man_labels'][j]
            mode_prob = self.sorted_scenarios[scenario_itr]['mode_prob'][j]
            traj_labels = self.sorted_scenarios[scenario_itr]['traj_labels'][j]
            
            traj_preds = self.sorted_scenarios[scenario_itr]['traj_dist_preds'][j][:,:, :2]
            frames = self.sorted_scenarios[scenario_itr]['frames'][j]
            if plot_id[0]==44 and plot_id[1] == 290:
                pdb.set_trace()
            scenario_tuple = (traj_min, traj_max, man_labels, man_preds, mode_prob, traj_labels, traj_preds, frames, frames_data, map_data)
            image = self.plot_one_frame(scenario_itr, tv_id, scenario_tuple, j)
            images.append(image)
        images = np.array(images)
        scenario_id = 'File{}_TV{}_SN{}_F{}'.format(data_file, tv_id, scenario_itr, frames[0])
        pf.save_image_sequence(p.model_name, images, self.traj_vis_dir, scenario_id, self.remove_ids_list is not None)              

    
    def plot_one_frame(self, scenario_itr, tv_id, scenario_tuple, time):
        summary_image = False
        (traj_min, traj_max, man_labels, man_preds, mode_prob, traj_labels, traj_preds, frames, frames_data, map_data) = scenario_tuple
        driving_dir = map_data['driving_dir']
        image_height = int(map_data['image_height']*p.Y_IMAGE_SCALE)
        image_width = int(map_data['image_width']*p.X_IMAGE_SCALE)
        lane_markings = map_data['lane_nodes_frenet']

        in_seq_len = self.in_seq_len
        tgt_seq_len = self.tgt_seq_len
        frame = frames[in_seq_len-1]
        #print(frames.shape)
        frame_list = [frame_data[rc.FRAME][0] for frame_data in frames_data]
        frame_data = frames_data[frame_list.index(frame)]
        
        
        traj_labels = traj_labels*(traj_max-traj_min)+traj_min
        traj_labels = np.cumsum(traj_labels, axis = 0)
        traj_preds =  traj_preds*(traj_max-traj_min)+traj_min
        traj_preds = np.cumsum(traj_preds, axis = 1)
        #print(traj_labels.shape)
        #pdb.set_trace()  
        
        image = pf.plot_frame(
            lane_markings,
            frame_data,
            tv_id, 
            driving_dir,
            frame,
            man_labels,
            man_preds,
            mode_prob,
            traj_labels,
            traj_preds,
            image_width,
            image_height)            
        return image
        
        

if __name__ =="__main__":
    bev_plotter = BEVPlotter( 
        fps = p.FPS,
        result_file = p.RESULT_FILE,
        dataset_name = p.DATASET,
        force_resort= True)

    bev_plotter.plot(file_id_pairs = None, remove_ids_list = None)
    
    