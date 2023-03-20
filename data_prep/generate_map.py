import os
import cv2
import numpy as np 
import pickle
import h5py
import matplotlib.pyplot as plt
import read_csv as rc
import math
import param as p
from utils import rendering_funcs as rf
import pandas
import pdb
# python -m pdb -c continue 
class GenerateMap:
    """This class is for rendering extracted scenarios from HighD dataset recording files (needs to be called seperately for each scenario).
    """
    def __init__(
        self,
        map_path,
        dataset_name):
        with open(map_path, 'rb') as handle:
            lane_markings = pickle.load(handle)
        self.lane_types = lane_markings['lane_types']
        self.frenet_lms = lane_markings['lane_nodes_frenet']
        self.map_export_dir = "../../Dataset/" + dataset_name + "/Map"
        if not os.path.exists(self.map_export_dir):
            os.makedirs(self.map_export_dir)
        self.map_file_name = map_path.split('/')[-1]

    def generate_map(self):
        print('Generating map data for: ', self.map_file_name)
        map_data = self.initialise_map()
        
        for i in range(map_data.shape[0]):
            for j in range(map_data.shape[1]):
                x = i*p.MAP_XRES+self.min_x
                y = self.max_y-j*p.MAP_YRES
                lane_itr = self.get_lane_itr(x,y, self.frenet_lms)
                map_data[i,j] = self.get_cell_feature(x, self.frenet_lms[lane_itr], self.lane_types[lane_itr])
        file_dir = os.path.join(self.map_export_dir, self.map_file_name)
        with open(file_dir, 'wb') as handle:
            pickle.dump((map_data, self.min_x, self.max_y, p.MAP_XRES, p.MAP_YRES), handle, protocol = pickle.HIGHEST_PROTOCOL)
        
    
    
    def get_lane_itr(self, x,y, lms):
        lane_itr = 0
        closest_itrs = []
        closest_ys = []
        for itr, lm in enumerate(lms):
            closest_itr = np.argmin(np.abs(lm['l'][:,0]-x))
            closest_itrs.append(closest_itr)
            closest_ys.append(lm['l'][closest_itr,1])
        closest_ys = np.array(closest_ys)
        if np.any(y>closest_ys) == False:
            lane_itr = len(lms)-1
        else:
            lane_itr = max(np.nonzero(y>closest_ys)[0][0]-1,0)
        return lane_itr
    
    def get_cell_feature(self,cur_x, lm, lt):
        x_array = np.linspace(cur_x, cur_x+p.MAP_VISION, p.MAP_CELL_LENGTH)
        r_type = np.zeros((p.MAP_CELL_LENGTH), dtype= float)
        l_type = np.zeros((p.MAP_CELL_LENGTH), dtype= float)
        lane_width = np.zeros((p.MAP_CELL_LENGTH), dtype= float)
        for i, x in enumerate(x_array):
            r_closest_itr = np.argmin(np.abs(lm['r'][:,0]-x))
            l_closest_itr = np.argmin(np.abs(lm['l'][:,0]-x))
            r_type[i] = lt['r'][r_closest_itr]
            l_type[i] = lt['l'][l_closest_itr]
            lane_width[i] = abs(lm['r'][r_closest_itr,1]-lm['l'][l_closest_itr,1])
            
        cell = np.stack((lane_width, r_type, l_type), axis= 0)
        return cell/4.0
    
    def initialise_map(self):
        
        min_x = math.inf
        min_y = math.inf
        max_x = -math.inf
        max_y = -math.inf
        for lm in self.frenet_lms:
            min_x = min(min(min(lm['r'][:,0]), min(lm['l'][:,0])), min_x) 
            min_y = min(min(min(lm['r'][:,1]), min(lm['l'][:,1])), min_y) 
            max_x = max(max(max(lm['r'][:,0]), max(lm['l'][:,0])), max_x) 
            max_y = max(max(max(lm['r'][:,1]), max(lm['l'][:,1])), max_y) 
        self.min_x = min_x
        self.max_y = max_y
        min_x = math.floor(min_x/p.MAP_XRES)
        min_y = math.floor(min_y/p.MAP_YRES)
        max_x = math.ceil((max_x-p.MAP_LENGTH_EXT)/p.MAP_XRES)
        max_y = math.ceil(max_y/p.MAP_YRES)
        empty_map = np.zeros((max_x-min_x, max_y-min_y, p.MAP_CELL_CHANNEL, p.MAP_CELL_LENGTH), dtype= float)
        print(empty_map.shape)
        return empty_map

    
if __name__ == '__main__':
    for i in range(len(p.map_paths)):
        gm = GenerateMap( p.map_paths[i], p.DATASET)
        gm.generate_map()