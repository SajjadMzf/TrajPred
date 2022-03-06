import os
import numpy as np
import pickle
import math
import read_csv as rc
import param as p
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
def get_last_idxs(
    tv_data:'Array containing track data of the TV', 
    driving_dir:'Driving direction of the TV', 
    )-> '(List of) last frame index(es) of the scenario(s)  and the labels for the scenario(s)' :
    
    tv_lane = tv_data[rc.LANE_ID]
    tv_lane_list = list(set(tv_lane))

    last_idxs = []
    labels = []
    if len(tv_lane_list) == 1:
        return last_idxs, labels, True
    
    tv_lane_mem = tv_lane
    prev_last_idx = 0
    while True:
        start_lane = tv_lane[0]
        last_idxs.append(prev_last_idx + np.nonzero(tv_lane!= start_lane)[0][0]) # the idx of the first element in tv_lane which is not equal to start_lane
        prev_last_idx = last_idxs[-1]
        tv_lane = tv_lane_mem[prev_last_idx:]
        cur_lane = tv_lane[0]
        if driving_dir == 1:
            if  cur_lane < start_lane:
                label = 1 # right lane change
            elif cur_lane > start_lane:
                label = 2 # left lane change                      
        elif driving_dir == 2:
            if cur_lane > start_lane:
                label = 1 # right lane change
            elif cur_lane < start_lane:
                label = 2 # left lane change
        labels.append(label)
        if np.all(tv_lane == tv_lane[0]):
            break
    return last_idxs, labels, True
    
def get_data_list(dataset_name, file_numbers):
    LC_states_dir = "../../Dataset/" + dataset_name + "/Scenarios" + p.dir_ext
    fr_div = 25/p.FPS
    
    feature_list = []
    left_lane_pos_list = []
    right_lane_pos_list = []
    lc_indx_list = []
    driving_dir_list = []
    lc_type_list = []
    for file_number in file_numbers:
        # read files
        data_tracks, _ = rc.read_track_csv(p.track_paths[file_number], p.track_pickle_paths[file_number], group_by = 'tracks', reload = False, fr_div = fr_div)
        frames_data, image_width = rc.read_track_csv(p.track_paths[file_number], p.frame_pickle_paths[file_number], group_by = 'frames', fr_div = fr_div)
        statics = rc.read_static_info(p.static_paths[file_number])
        metas = rc.read_meta_info(p.meta_paths[file_number])
        
            # define some useful functions:
        corner_x = lambda itr: int(frame_data[rc.X][itr])
        corner_y = lambda itr: int((frame_data[rc.Y][itr]))
        veh_width = lambda itr: int(frame_data[rc.WIDTH][itr])
        veh_height = lambda itr: int(frame_data[rc.HEIGHT][itr])
        center_x = lambda itr: int(frame_data[rc.X][itr] + veh_width(itr)/2)
        center_y = lambda itr: int(frame_data[rc.Y][itr] + veh_height(itr)/2)
        lateral_pos = lambda itr, lane_ind: abs(frame_data[rc.Y][itr] + frame_data[rc.HEIGHT][itr]/2- tv_lane_markings[lane_ind])
        tv_lateral_pos = lambda lane_ind: abs(tv_data[rc.Y] + tv_data[rc.HEIGHT][0]/2- tv_lane_markings[lane_ind])
        

        for tv_idx, tv_data in enumerate(data_tracks):
            driving_dir = statics[tv_idx+1][rc.DRIVING_DIRECTION] # statics is 1-based array
            tv_id = tv_data[rc.TRACK_ID] 
            lc_last_idxs, labels, _ = get_last_idxs(tv_data, driving_dir, )
            tv_lane_ind = None
            if len(lc_last_idxs)>1:# skip scenarios with more than 1 LC
                continue
            for scenario_idx, tv_last_idx in enumerate(lc_last_idxs): # These lane_idxes are for tv_data  
                if labels[scenario_idx]!=2:# only left lanes
                    continue
                frame = tv_data[rc.FRAME][tv_last_idx-1]
                frame_data = frames_data[int(frame/fr_div -1)],
                frame_data = frame_data[0]
                tv_itr = np.nonzero(frame_data[rc.TRACK_ID] == tv_id)[0][0]
            
                # tv lane
                tv_lane_markings = metas[rc.UPPER_LANE_MARKINGS] if driving_dir == 1 else metas[rc.LOWER_LANE_MARKINGS]
                tv_lane_markings = tv_lane_markings.astype(int)
                if tv_lane_ind is None:
                    tv_lane_ind = 0
                    for ind, value in reversed(list(enumerate(tv_lane_markings))):
                        if center_y(tv_itr)>value:
                            tv_lane_ind = ind
                            break
                if tv_lane_ind+1>=len(tv_lane_markings):# len is 1-based lane_ind is 0-based
                    raise ValueError('tv lane beyond valid lanes')
                tv_left_lane_ind = tv_lane_ind + 1 if driving_dir==1 else tv_lane_ind
                tv_right_lane_ind = tv_lane_ind if driving_dir==1 else  tv_lane_ind + 1
                left_lane_pos_list.append(tv_lane_markings[tv_left_lane_ind])
                right_lane_pos_list.append(tv_lane_markings[tv_right_lane_ind])
                feature_list.append(tv_data[rc.Y_VELOCITY])
                #feature_list.append(tv_lateral_pos(tv_left_lane_ind))
                    #tv_data[rc.Y])
                lc_indx_list.append(tv_last_idx)
                driving_dir_list.append(driving_dir)
                lc_type_list.append(labels[scenario_idx])

    return  feature_list, left_lane_pos_list, right_lane_pos_list, lc_indx_list, driving_dir_list, lc_type_list     

def plot_all(parameter, dataset_name, file_numbers):
    fr_div = 25/p.FPS
    # lateral_pos_list: [Number of scenarios, number of frames in each scenario (variable)]
    feature_list, left_lane_pos_list, right_lane_pos_list, lc_indx_list, driving_dir_list, lc_type_list = get_data_list(dataset_name, file_numbers)
    
    max_pre_lc = 0
    max_past_lc = 0
    for idx, feature in enumerate(feature_list):
        if lc_indx_list[idx]>max_pre_lc:
            max_pre_lc = lc_indx_list[idx]
        if len(feature)-lc_indx_list[idx]>max_past_lc:
            max_past_lc = len(feature)-lc_indx_list[idx]
    #print(max_pre_lc, max_past_lc)
    #print(len(feature))
    data_list = [[] for _ in range(max_past_lc+max_pre_lc)]
    data_timesteps = (np.arange(0,max_past_lc+max_pre_lc)-max_pre_lc)/p.FPS
    
    min_data = math.inf
    max_data = math.inf*-1
    for idx, data_array in enumerate(feature_list):
        offset = max_pre_lc-lc_indx_list[idx]
        if min(data_array)<min_data:
            min_data = min(data_array)
        if max(data_array)>max_data:
            max_data = max(data_array)
        for i, data in enumerate(data_array):
            data_list[i+offset].append(data)
    #print(data_list)
    
    first_slice = 0
    last_slice = max_past_lc+max_pre_lc
    # Delete data with less than 100 samples:
    for fr, data in enumerate(data_list):
        if len(data)<50:
            first_slice =fr
        else:
            break
    for fr, data in reversed(list(enumerate(data_list))):
        if len(data)<50:
            last_slice = fr
        else:
            break
    data_list = data_list[first_slice:last_slice]
    data_timesteps = data_timesteps[first_slice:last_slice]
    #plot histogram
    hist_bins = np.linspace(min_data, max_data, 100)
    hist_data = []
    for fr, data in enumerate(data_list):
        hist = np.histogram(data, hist_bins,(min_data, max_data))
        hist_data.append(hist[0])
    
    #normalise data:
    hist_data = [np.array([value/sum(data) for value in data]) for data in hist_data]

    hist_data = np.stack(hist_data, axis = 0)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    hist_bins = np.asarray(hist_bins[0:-1])
    X, Y = np.meshgrid(hist_bins,data_timesteps)
    ax.contour3D( X, Y, hist_data, 100)
    ax.set_xlabel('Value of the feature')
    ax.set_ylabel('timesteps')
    ax.set_zlabel('Distribution')
    ax.set_title('Distribution of LC feature in different timesteps')
    plt.show()



            
            
            

if __name__ == '__main__':
    file_numbers = np.arange(1,3)
    plot_all('lateral pos', p.DATASET, file_numbers)