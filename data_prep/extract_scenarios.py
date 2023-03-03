import os
import numpy as np
import pickle
import pdb

import read_csv as rc
import param as p
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
font = {'size'   : 24}
matplotlib.rcParams['figure.figsize'] = (16, 12)

matplotlib.rc('font', **font)
class ExtractScenarios:
    """This class is for extracting LC and LK scenarios from HighD dataset recording files (needs to be instantiated seperately for each recording).
    """
    def __init__(
        self, 
        file_num:'Number of recording file being rendered',
        track_path:'Path to track file', 
        track_pickle_path:'Path to track pickle file', 
        frame_pickle_path:'Path to frame pickle file',
        static_path:'Path to static file',
        meta_path:'Path to meta file',
        dataset_name:'Dataset Name' = None):

        
        
        #self.metas = rc.read_meta_info(meta_path)
        #self.fr_div = int(self.metas[rc.FRAME_RATE]/p.FPS)
        self.fr_div = int(p.IN_FPS/p.FPS)
        if self.fr_div<=0:
            raise(ValueError('non integer fr_div'))
        self.track_path = track_path

        self.file_num = file_num       
        
        self.LC_states_dir = "../../Dataset/"+ dataset_name +"/Scenarios"
        if not os.path.exists(self.LC_states_dir):
            os.makedirs(self.LC_states_dir)
        self.data_tracks = rc.read_track_csv(track_path, track_pickle_path, group_by = 'tracks', reload = True, fr_div = self.fr_div)
        self.data_frames = rc.read_track_csv(track_path, frame_pickle_path, group_by = 'frames', reload = True, fr_div = self.fr_div)
        self.track_list = [data_track[rc.TRACK_ID][0] for data_track in self.data_tracks]
        self.frame_list = [data_frame[rc.FRAME][0] for data_frame in self.data_frames]
        
        #self.statics = rc.read_static_info(static_path)

    def extract_and_save(self): 
        
        scenarios = self.get_scenarios()
        print("File Number; {}, ALL extracted scenarios: {}, ".format(self.file_num,len(scenarios)))
        file_dir = os.path.join(self.LC_states_dir, str(self.file_num).zfill(2)+ '.pickle')
        with open(file_dir, 'wb') as handle:
            pickle.dump(scenarios, handle, protocol= pickle.HIGHEST_PROTOCOL)
        return len(scenarios)

    def get_scenarios(
        self
        )-> 'List of dicts containing scenarios':
        
        scenarios = []
        for tv_idx, tv_data in enumerate(self.data_tracks):
            #print('tv idx: {}'.format(tv_idx))
            
            if tv_idx%500 == 0:
                print('Scenario {} out of: {}'.format(tv_idx, len(self.data_tracks)))
            
            driving_dir = p.driving_dir#self.statics[tv_data[rc.TRACK_ID][0]][rc.DRIVING_DIRECTION] # statics is based on id
            tv_id = tv_data[rc.TRACK_ID][0]
            #if tv_id == 284:
            #    a = 2
            total_frames = len(tv_data[rc.FRAME])
            if total_frames<= p.LINFIT_WINDOW:
                print('{} has short length({})'.format(tv_idx, total_frames))
                continue
            grid_data = np.zeros((total_frames, 3*13))
            for fr_indx in range(total_frames):
                frame = tv_data[rc.FRAME][fr_indx]
                frame_data = self.data_frames[self.frame_list.index(frame)]
                grid_data[fr_indx] = self.get_grid_data(tv_id, frame_data, driving_dir)
            #if self.file_num == 44 and tv_id ==290:
            #    pdb.set_trace()

            label_array =  self.get_label(tv_idx, tv_data, driving_dir)
            
            scenario = {
                    'file': self.file_num,
                    'tv':tv_id,
                    'x':tv_data[rc.X],
                    'y': tv_data[rc.Y],
                    'ttlc_available':True,
                    'frames':tv_data[rc.FRAME], 
                    'grid': grid_data, 
                    'label':label_array, 
                    'driving_dir':driving_dir,
                    'svs': self.get_svs(tv_data, 0, total_frames),
                    'images': None,
                    'states_wirth': None,
                    'states_shou': None,
                    'states_ours': None,
                    'output_states': None,

                    }
            scenarios.append(scenario)
        
        return scenarios
   
    
    def get_label(self, tv_idx, tv_data, driving_dir):
        tv_lane = tv_data[rc.LANE_ID]
        total_track_len = len(tv_lane)
        tv_lane_list = list(set(tv_lane))
        last_idxs = []
        labels = []
        label_array_y = np.zeros((total_track_len))
        
        if len(tv_lane_list) == 1:
            return label_array_y
        
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
        
        tv_x = tv_data[rc.X]
        tv_y = tv_data[rc.Y]
        
        indexes = np.arange(total_track_len)
        y_gradients = [np.polynomial.polynomial.polyfit(indexes[i:i+p.LINFIT_WINDOW], tv_y[i:i+p.LINFIT_WINDOW], deg=1 )[1] for i in range(0, total_track_len-p.LINFIT_WINDOW)]#tv_y - tv_y[last_idxs[0]]#[tv_y[i+p.LINFIT_WINDOW]-tv_y[i] for i in range(0, total_track_len-p.LINFIT_WINDOW)]
        
        # assuming same grad for final linfit_window timestep
        for i in range(p.LINFIT_WINDOW):
            y_gradients.append(y_gradients[-1])
        
        y_gradients = np.array(y_gradients)
        #Assumption: y axis from image bottom to top,
        non_lc_y_gradient = lambda gradient_array, label, driving_dir: gradient_array>=0 if (label == 2 and driving_dir==1) or (label == 1 and driving_dir==2)   else gradient_array<=0
        for idx, last_idx in enumerate(last_idxs):
            non_lc_points_after_crossing_y = non_lc_y_gradient(y_gradients[last_idx:], labels[idx], driving_dir)
            if np.any(non_lc_points_after_crossing_y):
                end_point = last_idx + np.nonzero(non_lc_points_after_crossing_y)[0][0]-1
            else: 
                end_point = total_track_len
                
            #end_point = last_idx
            non_lc_points_before_crossing_y = non_lc_y_gradient(y_gradients[:last_idx], labels[idx], driving_dir)
            if np.any(non_lc_points_before_crossing_y):
                start_point = np.nonzero(non_lc_points_before_crossing_y)[0][-1]
            else: 
                start_point = 0
                
            label_array_y[start_point:end_point]= np.ones((end_point-start_point))*labels[idx]
        
        for last_idx in last_idxs:
            label_array_y[last_idx] *= -1
            
        if p.PLOT_LABELS:
            
            plt.subplot(3,1,1)
            plt.plot(indexes, y_gradients, linewidth = 5)
            plt.plot(indexes, np.zeros_like(y_gradients),'--', linewidth = 3)
            plt.xlabel('Frame')
            plt.ylabel('y_Gradients')
            
            
            #plt.plot(indexes, gradients)
            for last_idx in last_idxs:
                plt.plot(last_idx, y_gradients[last_idx], marker='o', markersize = 10)
            plt.grid()
            plt.subplot(3,1,2)
            plt.plot(indexes, tv_y, linewidth = 5)
            for last_idx in last_idxs:
                plt.plot(indexes, np.ones_like(tv_y)*tv_y[last_idx],'--', linewidth = 3)
            plt.xlabel('Frame')
            plt.ylabel('Y (m)')
            
            #for i in range(0, total_track_len-p.LINFIT_WINDOW):
            #    plt.plot(tv_x[i:i+p.LINFIT_WINDOW], y_gradients[i]*tv_x[i:i+p.LINFIT_WINDOW] + biases[i])
            #if driving_dir == 1:
            #    plt.gca().invert_xaxis()
            plt.grid()
            
            plt.subplot(3,1,3)
            plt.plot(indexes, abs(label_array_y), linewidth = 5)
            plt.xlabel('Frame')
            plt.ylabel('Label_y')
            plt.grid()
            #plt.show()
            if os.path.exists('./labelling_figures/') == False:
                os.makedirs('./labelling_figures/')
            figure_name = 'labelling_figures/{}_{}.png'.format(self.file_num, tv_idx)
            plt.savefig(figure_name)
            plt.close()
            
        return label_array_y
    # TODO: to be updated
    def get_svs(
        self, 
        tv_data,
        tv_first_idx, 
        tv_last_idx
        ):
        pv_id = tv_data[rc.PRECEDING_ID][tv_first_idx:tv_last_idx]
        fv_id = tv_data[rc.FOLLOWING_ID][tv_first_idx:tv_last_idx]
        rv1_id = tv_data[rc.RIGHT_PRECEDING_ID][tv_first_idx:tv_last_idx]
        rv2_id = tv_data[rc.RIGHT_ALONGSIDE_ID][tv_first_idx:tv_last_idx]
        rv3_id = tv_data[rc.RIGHT_FOLLOWING_ID][tv_first_idx:tv_last_idx]
        lv1_id = tv_data[rc.LEFT_PRECEDING_ID][tv_first_idx:tv_last_idx]
        lv2_id = tv_data[rc.LEFT_ALONGSIDE_ID][tv_first_idx:tv_last_idx]
        lv3_id = tv_data[rc.LEFT_FOLLOWING_ID][tv_first_idx:tv_last_idx]
        return {'id':np.stack([pv_id, fv_id, rv1_id, rv2_id, rv3_id, lv1_id, lv2_id, lv3_id], axis = 0), 'vis':None}
    
    def get_grid_data(self, tv_id, frame_data, driving_dir):
        '''
        Get Grid data required for CS-LSTM model
        '''
        grid_data = np.zeros((3*13))
        tv_itr = np.nonzero(frame_data[rc.TRACK_ID] ==tv_id)[0][0]
        tv_lane = frame_data[rc.LANE_ID][tv_itr]
        tv_x = frame_data[rc.X][tv_itr]
        tv_lane_itrs = np.nonzero(frame_data[rc.LANE_ID] == tv_lane)[0] 
        plus_lane_itrs = np.nonzero(frame_data[rc.LANE_ID] == tv_lane+1)[0]
        minus_lane_itrs = np.nonzero(frame_data[rc.LANE_ID] == tv_lane-1)[0]
        left_lane_itrs = plus_lane_itrs if driving_dir==1 else minus_lane_itrs
        right_lane_itrs = minus_lane_itrs if driving_dir==1 else plus_lane_itrs
        
        for ll_itr in left_lane_itrs:
            xdist2tv = frame_data[rc.X][ll_itr]-tv_x
            if abs(xdist2tv)<p.grid_max_x:
                grid_ind = int(np.around(xdist2tv/16.667 + 6))
                assert(grid_ind>=0)
                grid_data[grid_ind] = frame_data[rc.TRACK_ID][ll_itr]
        
        for tl_itr in tv_lane_itrs:
            xdist2tv = frame_data[rc.X][tl_itr]-tv_x
            if  abs(xdist2tv)<p.grid_max_x and xdist2tv!=0:
                grid_ind = int(13 + np.around(xdist2tv/16.667 + 6))
                assert(grid_ind>=0)
                grid_data[grid_ind] = frame_data[rc.TRACK_ID][tl_itr]

        for rl_itr in right_lane_itrs:
            xdist2tv = frame_data[rc.X][rl_itr]-tv_x
            if abs(xdist2tv)<p.grid_max_x:
                grid_ind = int(26 + np.around(xdist2tv/16.667 + 6))
                assert(grid_ind>=0)
                grid_data[grid_ind] = frame_data[rc.TRACK_ID][rl_itr]
        return grid_data

def divide_prediction_window(seq_len, man_per_mode):
    num_window = man_per_mode-1
    window_length = int(seq_len/num_window)
    w_ind = np.zeros((num_window, 2))
    for i in range(num_window-1):
        w_indx[i,0] = i*window_length
        w_indx[i,1] = (i+1)*window_length
    w_ind[num_window-1,0] = (num_window-1)*window_length
    w_ind[num_window-1,1] = seq_len
    return w_ind
    