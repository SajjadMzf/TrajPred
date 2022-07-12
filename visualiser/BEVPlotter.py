from PIL import Image
import os
import cv2
import numpy as np 
import pickle
import h5py
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage
from mpl_toolkits.mplot3d import Axes3D
import read_csv as rc
import param as p
import torch
from scipy.stats import multivariate_normal

from time import time
import random

import torch
import torch.utils.data as utils_data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys
sys.path.insert(1,'../')
import Dataset 
import models 
import params 
import training_functions 



class BEVPlotter:
    """This class is for plotting results).
    """
    def __init__(
        self,
        fps:'Desired fps',
        result_file:'File contains results',
        dataset_name: 'Dataset  Name',
        num_output:'Number of samples need to be plotted -1 means all'):
        
        self.lines_width = p.LINES_WIDTH
        self.result_file = result_file
        self.num_output = num_output
        self.fps = fps
        self.dtype = np.float
        self.filled = 1
        self.dash_lines = tuple([8,8])
        ''' 1. Representation Properties:'''
        
        # 1.1 Scales and distances
        
        self.traj_vis_dir = "../../SAMPLE/" + p.model_name + "/traj_vis"
        if not os.path.exists(self.traj_vis_dir):
            os.makedirs(self.traj_vis_dir)
        #print(torch.cuda.is_available())
        with open(self.result_file, 'rb') as handle:
            self.scenarios = pickle.load(handle)

    def sort_scenarios(self):
        sorted_scenarios_dict = []
        tv_id_file_list = []
        for _, scenario in enumerate(self.scenarios):
            for batch_itr, tv_id in enumerate(scenario['tv']):
                data_file = int(scenario['data_file'][batch_itr].split('.')[0])
                if (data_file,tv_id) not in tv_id_file_list:
                    tv_id_file_list.append((data_file,tv_id))
                    sorted_scenarios_dict.append({'tv': tv_id,
                                                'data_file':data_file,
                                                'traj_min': scenario['traj_min'],# Assuming traj min and max are the same for all scenarios
                                                'traj_max':scenario['traj_max'],
                                                'input_features': [],
                                                'times':[], 
                                                'man_labels':[], 
                                                'man_preds':[], 
                                                'enc_man_preds':[], 
                                                'traj_labels':[], 
                                                'traj_preds':[],
                                                'traj_dist_preds':[], 
                                                'frames':[],
                                                })
                in_seq_len = scenario['man_labels'].shape[1]-scenario['man_preds'].shape[1]
                tgt_seq_len = scenario['man_preds'].shape[1]
                sorted_index = tv_id_file_list.index(((data_file,tv_id)))
                sorted_scenarios_dict[sorted_index]['times'].append(scenario['frames'][batch_itr,in_seq_len])
                sorted_scenarios_dict[sorted_index]['man_labels'].append(scenario['man_labels'][batch_itr]) 
                sorted_scenarios_dict[sorted_index]['man_preds'].append(scenario['man_preds'][batch_itr])
                sorted_scenarios_dict[sorted_index]['enc_man_preds'].append(scenario['enc_man_preds'][batch_itr])
                sorted_scenarios_dict[sorted_index]['traj_labels'].append(scenario['traj_labels'][batch_itr])
                sorted_scenarios_dict[sorted_index]['traj_preds'].append(scenario['traj_preds'][batch_itr])
                sorted_scenarios_dict[sorted_index]['traj_dist_preds'].append(scenario['traj_dist_preds'][batch_itr])
                sorted_scenarios_dict[sorted_index]['frames'].append(scenario['frames'][batch_itr])
                sorted_scenarios_dict[sorted_index]['input_features'].append(scenario['input_features'][batch_itr])
                
                
        
        for i in range(len(sorted_scenarios_dict)):
            times_array = np.array(sorted_scenarios_dict[i]['times'])
            sorted_indxs = np.argsort(times_array).astype(int)
            sorted_scenarios_dict[i]['times'] = [sorted_scenarios_dict[i]['times'][indx] for indx in sorted_indxs]
            sorted_scenarios_dict[i]['man_labels'] = [sorted_scenarios_dict[i]['man_labels'][indx] for indx in sorted_indxs]
            sorted_scenarios_dict[i]['man_preds'] = [sorted_scenarios_dict[i]['man_preds'][indx] for indx in sorted_indxs]
            sorted_scenarios_dict[i]['enc_man_preds'] = [sorted_scenarios_dict[i]['enc_man_preds'][indx] for indx in sorted_indxs]
            sorted_scenarios_dict[i]['traj_labels'] = [sorted_scenarios_dict[i]['traj_labels'][indx] for indx in sorted_indxs]
            sorted_scenarios_dict[i]['traj_preds'] = [sorted_scenarios_dict[i]['traj_preds'][indx] for indx in sorted_indxs]
            sorted_scenarios_dict[i]['traj_dist_preds'] = [sorted_scenarios_dict[i]['traj_dist_preds'][indx] for indx in sorted_indxs]
            sorted_scenarios_dict[i]['frames'] = [sorted_scenarios_dict[i]['frames'][indx] for indx in sorted_indxs]
            sorted_scenarios_dict[i]['input_features'] = [sorted_scenarios_dict[i]['input_features'][indx] for indx in sorted_indxs]
            
        self.in_seq_len = in_seq_len
        self.tgt_seq_len = tgt_seq_len
        return sorted_scenarios_dict
    
    def whatif_render(self, dl_params,  scenario_number, wif_man):
        if torch.cuda.is_available() and dl_params.CUDA:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            torch.cuda.manual_seed_all(0)
        else:
            self.device = torch.device("cpu")
                
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True
        np.random.seed(1)
        random.seed(1)

        # Instantiate Model:
        
        model = dl_params.model_dictionary['ref'](1, self.device, dl_params.model_dictionary['hyperparams'], dl_params)
        model = model.to(self.device)
        best_model_path = '../'+dl_params.MODELS_DIR + dl_params.experiment_tag + '.pt'
        model.load_state_dict(torch.load(best_model_path))

        sorted_dict = self.sort_scenarios()
        plotted_data_number = 0
        
        tv_id = sorted_dict[scenario_number]['tv']
        data_file = sorted_dict[scenario_number]['data_file']
        print('TV ID: {}, List of Available Frames: {}'.format(tv_id, sorted_dict[scenario_number]['times']))
        traj_min = sorted_dict[scenario_number]['traj_min']
        traj_max = sorted_dict[scenario_number]['traj_max']
        for j,time in enumerate(sorted_dict[scenario_number]['times']):
            man_labels = sorted_dict[scenario_number]['man_labels'][j]
            man_preds = sorted_dict[scenario_number]['man_preds'][j]
            enc_man_preds = sorted_dict[scenario_number]['enc_man_preds'][j]
            traj_labels = sorted_dict[scenario_number]['traj_labels'][j]
            if p.PROBABILISTIC_PLOT:
                traj_preds = sorted_dict[scenario_number]['traj_dist_preds'][j]
                traj_preds[:,:2] = traj_preds[:,:2]*(traj_max-traj_min) + traj_min
            else:
                traj_preds = sorted_dict[scenario_number]['traj_preds'][j]
            frames = sorted_dict[scenario_number]['frames'][j]
            input_features = sorted_dict[scenario_number]['input_features'][j]

            initial_traj = traj_labels[self.in_seq_len-1:self.in_seq_len]
            initial_traj = (initial_traj-traj_min)/(traj_max-traj_min)
            wif_traj_pred = self.eval_model(dl_params, model, input_features, initial_traj, wif_man)
            wif_traj_pred = wif_traj_pred*(traj_max-traj_min) + traj_min

            scenario_tuple = (traj_min, traj_max, man_labels, man_preds, enc_man_preds, traj_labels, traj_preds, frames, data_file)
            self.render_single_scenario(scenario_number, tv_id, scenario_tuple, plotted_data_number, summary_image =True, wif_man = wif_man[1:], wif_traj = wif_traj_pred)
            plotted_data_number += 1
            print("Scene Number: {}".format(plotted_data_number))
            
    def iterative_render(self):
        sorted_dict = self.sort_scenarios()
        plotted_data_number = 0
        for i in range(len(sorted_dict)):
            tv_id = sorted_dict[i]['tv']
            data_file = sorted_dict[i]['data_file']
            traj_min = sorted_dict[i]['traj_min']
            traj_max = sorted_dict[i]['traj_max']
            print('TV ID: {}, List of Available Frames: {}'.format(tv_id, sorted_dict[i]['times']))
            for j,time in enumerate(sorted_dict[i]['times']):
                man_labels = sorted_dict[i]['man_labels'][j]
                man_preds = sorted_dict[i]['man_preds'][j]
                enc_man_preds = sorted_dict[i]['enc_man_preds'][j]
                traj_labels = sorted_dict[i]['traj_labels'][j]
                if p.PROBABILISTIC_PLOT:
                    traj_preds = sorted_dict[i]['traj_dist_preds'][j]
                    traj_preds[:,:2] = traj_preds[:,:2]*(traj_max-traj_min) + traj_min
                else:
                    traj_preds = sorted_dict[i]['traj_preds'][j]
                frames = sorted_dict[i]['frames'][j]
                scenario_tuple = (traj_min, traj_max, man_labels, man_preds, enc_man_preds, traj_labels, traj_preds, frames, data_file)
                self.render_single_scenario(i, tv_id, scenario_tuple, plotted_data_number, summary_image =True)
                plotted_data_number += 1
                print("Scene Number: {}".format(plotted_data_number))
                if plotted_data_number >= self.num_output:
                    break
            if plotted_data_number >= self.num_output:
                break 
        return plotted_data_number

    def render_scenarios(self)-> "Number of rendered and saved scenarios":
        plotted_data_number = 0
        prev_data_file = -1
        for i, scenario in enumerate(self.scenarios):
            traj_min = scenario['traj_min']
            traj_max = scenario['traj_max']
            for batch_itr, tv_id in enumerate(scenario['tv']):
                man_labels = scenario['man_labels'][batch_itr]
                man_preds = scenario['man_preds'][batch_itr]
                enc_man_preds = scenario['enc_man_preds'][batch_itr]
                traj_labels = scenario['traj_labels'][batch_itr]
                if p.PROBABILISTIC_PLOT:
                    traj_preds = scenario['traj_dist_preds'][batch_itr]
                    traj_preds[:,:2] = traj_preds[:,:2]*(traj_max-traj_min) + traj_min
                else:
                    traj_preds = scenario['traj_preds'][batch_itr]
                
                frames = scenario['frames'][batch_itr]
                data_file = int(scenario['data_file'][batch_itr].split('.')[0])
                scenario_tuple = (traj_min, traj_max, man_labels, man_preds, enc_man_preds, traj_labels, traj_preds, frames, data_file)
                self.render_single_scenario(i, tv_id, scenario_tuple, plotted_data_number, summary_image = False)
                plotted_data_number += 1
                print("Scene Number: {}".format(plotted_data_number))
                if plotted_data_number >= self.num_output:
                    break
            if plotted_data_number >= self.num_output:
                break 
        return plotted_data_number

    def render_single_scenario(self, scenario_number, tv_id, scenario_tuple, plot_number, summary_image, wif_man = [], wif_traj = []):
        (traj_min, traj_max, man_labels, man_preds, enc_man_preds, traj_labels, traj_preds, frames, data_file) = scenario_tuple
        enc_man_preds = self.softmax(enc_man_preds)
        for i in range(man_preds.shape[0]):
            man_preds[i] = self.softmax(man_preds[i])
        in_seq_len = man_labels.shape[0]-man_preds.shape[0]
        tgt_seq_len = man_preds.shape[0]
        
        track_path = p.track_paths[data_file-1]
        static_path = p.static_paths[data_file-1]
        pickle_path = p.frame_pickle_paths[data_file-1]
        meta_path = p.meta_paths[data_file-1]
        self.metas = rc.read_meta_info(meta_path)
        self.fr_div = self.metas[rc.FRAME_RATE]/self.fps
        self.frames_data, image_width = rc.read_track_csv(track_path, pickle_path, group_by = 'frames', reload = False, fr_div = self.fr_div)
        self.statics = rc.read_static_info(static_path)
        prev_data_file = data_file
        self.image_width = int(image_width*p.X_IMAGE_SCALE )
        self.image_height = int(self.metas[rc.LOWER_LANE_MARKINGS][-1]*p.Y_IMAGE_SCALE + p.BORDER_PIXELS)

        driving_dir = self.statics[tv_id][rc.DRIVING_DIRECTION]
                        
        #self.plot_overview(man_labels, man_preds, traj_labels,traj_preds, plotted_data_number)
        traj_imgs = []
        tv_track = []
        tv_future_track = ([],[], [])
        tv_lane_ind = None
        for fr in range(in_seq_len+tgt_seq_len):
            frame = frames[fr]
            traj_img, tv_track, tv_future_track, tv_lane_ind = self.plot_frame(
                data_file,
                self.frames_data[int(frame/self.fr_div -1)],
                tv_id, 
                driving_dir,
                frame,
                man_labels,
                man_preds,
                enc_man_preds,
                traj_labels,
                traj_preds,
                traj_min,
                traj_max,
                fr,
                in_seq_len,
                tv_track =  tv_track,
                tv_lane_ind = tv_lane_ind,
                tv_future_track = tv_future_track,
                wif_man = wif_man,
                wif_traj = wif_traj
                )
            if not summary_image:
                traj_imgs.append(traj_img)
            elif summary_image and fr == in_seq_len-1:
                traj_imgs.append(traj_img)
                break
        
        traj_imgs = np.array(traj_imgs)
        scenario_id = 'File{}_TV{}_SN{}_PN{}'.format(data_file, tv_id, scenario_number, plot_number)
        self.save_image_sequence(p.model_name, traj_imgs, self.traj_vis_dir,scenario_id , summary_image)
        
    def plot_frame(
        self, 
        file_num:'File Number',
        frame_data:'Data array of current frame', 
        tv_id:'ID of the TV', 
        driving_dir:'TV driving direction',
        frame:'frame',
        man_labels, 
        man_preds,
        enc_man_preds,
        traj_labels,
        traj_preds,
        traj_min,
        traj_max,
        seq_fr:'frame sequence',
        in_seq_len,
        tv_lane_ind,
        tv_track = [],
        tv_future_track = [],
        wif_man = [],
        wif_traj = []
        ):
        
        assert(frame_data[rc.FRAME]==frame) 
        image = np.ones((self.image_height, self.image_width,3), dtype=np.int32)*p.COLOR_CODES['BACKGROUND']

        tv_itr = np.nonzero(frame_data[rc.TRACK_ID] == tv_id)[0][0]
        
        tv_lane_markings = self.metas[rc.UPPER_LANE_MARKINGS]*p.Y_IMAGE_SCALE if driving_dir == 1 else self.metas[rc.LOWER_LANE_MARKINGS]*p.Y_IMAGE_SCALE
        tv_lane_markings = tv_lane_markings.astype(int)

        vehicle_in_frame_number = len(frame_data[rc.TRACK_ID])
        
        corner_x = lambda itr: int(frame_data[rc.X][itr]*p.X_IMAGE_SCALE)
        corner_y = lambda itr: int(frame_data[rc.Y][itr]*p.Y_IMAGE_SCALE) 
        veh_width = lambda itr: int(frame_data[rc.WIDTH][itr]*p.X_IMAGE_SCALE)
        veh_height = lambda itr: int(frame_data[rc.HEIGHT][itr]*p.Y_IMAGE_SCALE)
        center_x = lambda itr: int(frame_data[rc.X][itr]*p.X_IMAGE_SCALE+ frame_data[rc.WIDTH][itr]*p.X_IMAGE_SCALE/2)
        center_y = lambda itr: int(frame_data[rc.Y][itr]*p.Y_IMAGE_SCALE+ frame_data[rc.HEIGHT][itr]*p.Y_IMAGE_SCALE/2)  
        center2corner_x = lambda center, itr: int(center - frame_data[rc.WIDTH][itr]*p.X_IMAGE_SCALE/2)
        center2corner_y = lambda center, itr: int(center - frame_data[rc.HEIGHT][itr]*p.Y_IMAGE_SCALE/2)
        fix_sign = lambda x: x if driving_dir ==1 else -1*x
        

        '''3. Draw Lines'''
        # Draw lines
        image = self.draw_lines(image, self.image_width, 
                                (self.metas[rc.LOWER_LANE_MARKINGS]*p.Y_IMAGE_SCALE),
                                (self.metas[rc.UPPER_LANE_MARKINGS]*p.Y_IMAGE_SCALE))
        
        '''Draw vehicles'''
        if not p.HIDE_SVS:
            for itr in range(vehicle_in_frame_number):
                if itr != tv_itr:
                    image = self.draw_vehicle(image, corner_x(itr), corner_y(itr), veh_width(itr), veh_height(itr), p.COLOR_CODES['SV'])
        
        image = image.astype(np.uint8)
        if seq_fr == 0:
            tv_track.append((center_x(tv_itr), center_y(tv_itr)))
        else:
            initial_x = tv_track[0][0]
            initial_y = tv_track[0][1]
            dx = 0
            dy = 0
            for i in range(seq_fr):
                dx += fix_sign(traj_labels[i,1]*p.X_IMAGE_SCALE)
                dy += fix_sign(traj_labels[i,0]*p.Y_IMAGE_SCALE)
            tv_track.append((int(initial_x+dx), int(initial_y+dy)))
        
        gt_track_history_len = min(len(tv_track), in_seq_len)
        for i in range(gt_track_history_len):
            if i ==0:
                continue
            image = cv2.line(image, tv_track[i], tv_track[i-1], p.COLOR_CODES['GT_TRAJ'], 2)
            top_left = tv_track[i-1] + np.array([-2,-2])
            bottom_right = tv_track[i-1] + np.array([2,2])
            image = cv2.rectangle(image, tuple(top_left), tuple(bottom_right), color = p.COLOR_CODES['GT_TRAJ'], thickness = 1)
    
        #print('Sequence: {}, dx: {}, dy: {}, x{}, y{} '.format(seq_fr, ))
        image = self.draw_vehicle(image, center2corner_x(tv_track[-1][0], tv_itr), center2corner_y(tv_track[-1][1], tv_itr), veh_width(tv_itr), veh_height(tv_itr), p.COLOR_CODES['TV'])
        image = image.astype(np.uint8)
        # calculate future trajectories
        if seq_fr == in_seq_len-1:
            tv_gt_future_track = []
            #tv_gt_future_track.append((center_x(tv_itr), center_y(tv_itr)))
            tv_gt_future_track.append((tv_track[-1][0], tv_track[-1][1]))
            tv_pr_future_track = []
            wif_future_track = []
            #tv_pr_future_track.append((center_x(tv_itr), center_y(tv_itr)))
            tv_pr_future_track.append((tv_track[-1][0], tv_track[-1][1]))
            wif_future_track.append((tv_track[-1][0], tv_track[-1][1]))
            tgt_seq_len = traj_preds.shape[0]
            initial_x = tv_gt_future_track[-1][0]
            initial_y = tv_gt_future_track[-1][1]
            dx_gt = 0
            dy_gt = 0
            for fut_fr in range(in_seq_len, tgt_seq_len+in_seq_len):
                dx_gt += fix_sign(traj_labels[fut_fr,1]*p.X_IMAGE_SCALE)
                dy_gt += fix_sign(traj_labels[fut_fr,0]*p.Y_IMAGE_SCALE)
                tv_gt_future_track.append((int(initial_x+dx_gt), int(initial_y+dy_gt)))

            dx_pr = 0
            dy_pr = 0  
            for fut_fr in range(tgt_seq_len):
                dx_pr += fix_sign(traj_preds[fut_fr,1]*p.X_IMAGE_SCALE)
                dy_pr += fix_sign(traj_preds[fut_fr,0]*p.Y_IMAGE_SCALE)
                tv_pr_future_track.append((int(initial_x+dx_pr), int(initial_y+dy_pr)))
            
            tv_future_track = (tv_gt_future_track, tv_pr_future_track)
            
            if p.WHAT_IF_RENDERING:
                dx_wif = 0
                dy_wif = 0  
                for fut_fr in range(tgt_seq_len):
                    dx_wif += fix_sign(wif_traj[fut_fr,1]*p.X_IMAGE_SCALE)
                    dy_wif += fix_sign(wif_traj[fut_fr,0]*p.Y_IMAGE_SCALE)
                    wif_future_track.append((int(initial_x+dx_wif), int(initial_y+dy_wif)))
            
            if p.WHAT_IF_RENDERING:
                tv_future_track = (tv_gt_future_track, tv_pr_future_track, wif_future_track)
            else:
                tv_future_track = (tv_gt_future_track, tv_pr_future_track)  


        tv_gt_future_track = tv_future_track[0]
        tv_pr_future_track = tv_future_track[1]
        for i in range(len(tv_gt_future_track)):
            if i ==0:
                continue
            image = cv2.line(image, tv_gt_future_track[i], tv_gt_future_track[i-1], p.COLOR_CODES['GT_TRAJ'], 2)
            top_left = tv_gt_future_track[i-1] + np.array([-3,-3])
            bottom_right = tv_gt_future_track[i-1] + np.array([3,3])
            image = cv2.rectangle(image, tuple(top_left), tuple(bottom_right), color = p.COLOR_CODES['GT_TRAJ'], thickness = -1)

        if p.PROBABILISTIC_PLOT:
            height = image.shape[0]
            width = image.shape[1]
            #y_vector = np.arange(0,y_max,1)
            #x_vector = np.arange(0,x_max,1)
            #xx, yy = np.meshgrid(x_vector, y_vector)
            #xxyy = np.c_[xx.ravel(), yy.ravel()]
            z = np.zeros((height, width))

            
            y_min = traj_min[0]
            x_min = traj_min[1]
            y_max = traj_max[0]
            x_max = traj_max[1]
            for i in range(len(tv_pr_future_track)):
                if i == 0:
                    continue
                muX = tv_pr_future_track[i][0]
                muY = tv_pr_future_track[i][1]
                sigY = traj_preds[i-1, 2]*(y_max-y_min) # sigY = standard deviation of y
                sigX = traj_preds[i-1, 3]*(x_max-x_min) # sigX = standard deviation of x
                rho = traj_preds[i-1, 4]
                z += self.plot_single_heatmap(height, width, muY, muX, sigY, sigX, rho,p.CUT_OFF_SIGMA_RATIO)

        else:
            for i in range(len(tv_pr_future_track)):
                if i ==0:
                    continue
                image = cv2.line(image, tv_pr_future_track[i], tv_pr_future_track[i-1], p.COLOR_CODES['PR_TRAJ'], 2)
                image = cv2.circle(image, tv_pr_future_track[i-1], 4, p.COLOR_CODES['PR_TRAJ'], thickness = -1)

        if p.WHAT_IF_RENDERING:
            wif_future_track = tv_future_track[2]
            for i in range(len(wif_future_track)):
                if i ==0:
                    continue
                image = cv2.line(image, wif_future_track[i], wif_future_track[i-1], p.COLOR_CODES['WIF_TRAJ'], 2)
                top = wif_future_track[i-1] + np.array([0,-2])
                left = wif_future_track[i-1] + np.array([-2,2])
                right = wif_future_track[i-1] + np.array([2,2])
                pts = np.array([top, left, right])
                pts = pts.reshape((-1, 1, 2))
                image = cv2.polylines(image, [pts], True, p.COLOR_CODES['WIF_TRAJ'], thickness = 2)


        if tv_lane_ind is None:
            tv_lane_ind = 0
            for ind, value in reversed(list(enumerate(tv_lane_markings))):
                if center_y(tv_itr)>value:
                    tv_lane_ind = ind
                    break
        
        #print(man_labels)
        #print(man_preds)
        #exit()
        if p.PLOT_MAN and seq_fr == in_seq_len -1:
            
            fig = plt.figure(figsize=(15, 3))
            ax = fig.add_subplot(111)
            if seq_fr>= in_seq_len-1:
                x = np.arange(in_seq_len+1,in_seq_len + tgt_seq_len+1)
                y = np.argmax(man_preds,axis = 1)
                ax.plot(x,y, p.MARKERS['PR_TRAJ'], color = 'red', alpha = 1)#, fillstyle = 'none', markersize=10)
                if p.WHAT_IF_RENDERING:
                    wif_y = wif_man
                    ax.plot(x,wif_y, p.MARKERS['WIF_TRAJ'], color = 'green', alpha = 1)#, fillstyle = 'none', markersize=10)
            gt_x = np.arange(1, in_seq_len + tgt_seq_len+1)
            gt_y = man_labels
            ax.plot(gt_x, gt_y,p.MARKERS['GT_TRAJ'], color = 'blue', alpha = 1)#, fillstyle = 'none', markersize=10)
            plt.xlabel('Frame')
            plt.xticks(rotation=90)
            plt.ylabel('Manouvre Label')
            ax.grid(True)
            plt.yticks([-1,0,1,2,3], ['','LK', 'RLC', 'LLC', ''])
            plt.xticks(gt_x,gt_x)
            fig.tight_layout(pad=1)
            #plt.show()
            #plt.close()
            man_image = mplfig_to_npimage(fig)
            plt.close('all')
            man_bar = np.ones((man_image.shape[0]+20, image.shape[1],3), dtype = np.int32)
            man_bar[:,:,:] = p.COLOR_CODES['BACKGROUND']
            man_bar[0:man_image.shape[0], 0:man_image.shape[1], :] = man_image
            man_bar = man_bar[:,:,[2,1,0]]
            image = np.concatenate((image, man_bar), axis = 0)
                
        if p.PLOT_TEXTS:
            text_bar = np.ones((40+3*p.LINE_BREAK, image.shape[1],3), dtype = np.int32)
            text_bar[:,:,:]= p.COLOR_CODES['BACKGROUND']
            gt_man_text = 'GT:{}'.format(p.CLASS[man_labels[seq_fr]])
            cv2.putText(text_bar,gt_man_text,
                                (20,25), 
                                p.FONT, 
                                p.FSCALE,
                                p.FCOLOR,
                                p.LINETYPE)

            if seq_fr>=in_seq_len-1:
                #print(seq_fr)
                
                enc_man_text = 'ENC PR: LK:{:.2f}, RLC:{:.2f}, LLC:{:.2f}'.format(enc_man_preds[0], enc_man_preds[1], enc_man_preds[2])
                cv2.putText(text_bar,enc_man_text,
                                    (20,50), 
                                    p.FONT, 
                                    p.FSCALE,
                                    p.FCOLOR,
                                    p.LINETYPE)
                
                pr_man_text = 'PR: LK:{:.2f}, RLC:{:.2f}, LLC:{:.2f}'.format(man_preds[seq_fr-(in_seq_len),0], man_preds[seq_fr-(in_seq_len),1], man_preds[seq_fr-(in_seq_len),2])
                cv2.putText(text_bar,pr_man_text,
                                    (200,25), 
                                    p.FONT, 
                                    p.FSCALE,
                                    p.FCOLOR,
                                    p.LINETYPE)
            image = np.concatenate((text_bar, image), axis = 0)


        
        
        
        return  image, tv_track, tv_future_track, tv_lane_ind 
        
 

    def plot_single_heatmap(self,height, width, muY, muX, sigY, sigX, rho,cut_off_sig_ratio):
        z = np.zeros((height, width))
        m = (muY, muX)
        s = np.array([[sigY^2, rho*sigX*sigY],[rho*sigX*sigY, sigX^2]])
        k = multivariate_normal(mean = m, cov = s)
        y_min = muY-cut_off_sig_ratio*sigY
        x_min = muX-cut_off_sig_ratio*sigX
        y_max = muY+cut_off_sig_ratio*sigY
        x_max = muX+cut_off_sig_ratio*sigX
        x_res = 100
        y_res = 100
        x = np.linspace(x_min, x_max, x_res)
        y = np.linspace(y_min, y_max, y_res)
        xx, yy = np.meshgrid(x,y)
        xxyy = np.c_(xx.ravel(), yy.ravel())
        heatmap = k.pdf(xxyy)
        z = 10

        


        return z

    def eval_model(self,dl_params, model, input_features, initial_traj, wif_man):
        wif_man = torch.from_numpy(wif_man)

        wif_man = torch.unsqueeze(wif_man, dim =0)
        x = input_features
        x = torch.unsqueeze(x, dim = 0) #batch dim
        y = torch.from_numpy(initial_traj) 
        y = torch.unsqueeze(y, dim = 0) #batch dim
        x = x.to(self.device)
        y = y.to(self.device)
        wif_man = wif_man.to(self.device)
        wif_man_one_hot = F.one_hot(wif_man, num_classes =3)
        y = torch.cat((y, wif_man_one_hot[:,0:1]), dim =-1) # TODO this has to be replaced by predicted man label by encoder
            
        y = torch.stack((y,y,y), dim = 1) #mm
        

        for seq_itr in range(self.tgt_seq_len):
            output_dict = model(x = x, y = y, y_mask = model.get_y_mask(y.size(2)).to(self.device))
            traj_dist_pred = output_dict['traj_pred']
            traj_dist_pred = traj_dist_pred[:,:,seq_itr:seq_itr+1]
            enc_man_pred = output_dict['enc_man_pred']
            #if seq_itr == 0:
            #    current_man_pred = torch.argmax(enc_man_pred, dim = -1)
            #else:
            #    current_man_pred = torch.argmax(man_pred[:,seq_itr-1], dim = -1)
            selected_traj_pred = traj_dist_pred[:,wif_man[0,seq_itr+1],:,:2]
            if dl_params.MAN_DEC_OUT==True:
                #man_pred_dec_in = man_pred[:,seq_itr:(seq_itr+1)]
                #man_pred_dec_in = F.one_hot(torch.argmax(man_pred_dec_in, dim = -1), num_classes = 3) 
                #man_pred_dec_in = torch.unsqueeze(man_pred_dec_in, dim = 1)
                #print_shape('traj_pred_dec_in', traj_pred_dec_in)
                #print_shape('man_pred_dec_in', man_pred_dec_in)
                selected_traj_pred = torch.cat((selected_traj_pred, wif_man_one_hot[:,(seq_itr+1):(seq_itr+2)]), dim = -1)
                selected_traj_pred = torch.stack([selected_traj_pred, selected_traj_pred, selected_traj_pred], dim =1)
            else:
                print('Not supported')
            y = torch.cat((y, selected_traj_pred), dim = 2)
            #selected_traj_pred = torch.stack([selected_traj_pred,selected_traj_pred,selected_traj_pred], dim = )
        y = y[0,0,1:,:2]
        return y.cpu().detach().numpy()
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0) # only difference

    def draw_vehicle(self, image, x, y, width, height, color_code):
        image[y:(y+height), x:(x+width)] = color_code
        return image

    def draw_lines(self,lane_channel, width, lower_lines, upper_lines):
        # Continues lines
        color_code = p.COLOR_CODES['LANE']
        lane_channel[int(upper_lines[0]):(int(upper_lines[0]) + self.lines_width), 0:width] = color_code
        lane_channel[int(upper_lines[-1]):(int(upper_lines[-1]) + self.lines_width), 0:width] = color_code
        lane_channel[int(lower_lines[0]):(int(lower_lines[0]) + self.lines_width), 0:width] = color_code
        lane_channel[int(lower_lines[-1]):(int(lower_lines[-1]) + self.lines_width), 0:width] = color_code
        
        # Broken Lines
        filled = int(self.dash_lines[0])
        total_line = int((self.dash_lines[0]+ self.dash_lines[1]))
        for line in upper_lines[1:-1]:
            for i in range(int(width/total_line)):
                lane_channel[int(line):(int(line) + self.lines_width), (i*total_line):(i*total_line+filled)] = color_code

        for line in lower_lines[1:-1]:
            for i in range(int(width/total_line)):
                lane_channel[int(line):(int(line) + self.lines_width), (i*total_line):(i*total_line+filled)] = color_code
        
        return lane_channel     
           
        
    def save_image_sequence(
        self,
        experiment_dir,
        images:'Image sequence',
        save_dir:'Save directory',
        sample_id,
        summary_image
        ):
        #folder_dir = os.path.join(os.path.join(save_dir, experiment_dir), sample_id)
        folder_dir = os.path.join(save_dir, experiment_dir)
        if summary_image:
            folder_dir = os.path.join(folder_dir, 'summary_images')
        else:
            folder_dir = os.path.join(folder_dir, 'all timesteps')
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)
        size = (images[0].shape[1], images[0].shape[0])
        if not summary_image:
            video_dir = os.path.join(folder_dir,sample_id + '.avi')
            video_out = cv2.VideoWriter(video_dir, cv2.VideoWriter_fourcc(*'DIVX'), 1, size, isColor = True)

        for fr, image in enumerate(images):
            if not summary_image:
                video_out.write(image.astype(np.uint8))
            file_dir = os.path.join(folder_dir, sample_id + '_' +str(fr+1)+'.png')
            if not cv2.imwrite(file_dir, image):
                raise Exception("Could not write image: " + file_dir)
        if not summary_image:
            video_out.release()

'''
def wif_man_encoder(sequence_length, initial_man, man_change_series):
    wif_man = np.zeros((sequence_length+1))
    for i in range(sequence_length):
        
        wif_man[i] = 
'''
if __name__ =="__main__":
    bev_plotter = BEVPlotter( 
        fps = p.FPS,
        result_file = p.RESULT_FILE,
        dataset_name = p.DATASET,
        num_output = p.NUM_OUTPUT)
    if p.WHAT_IF_RENDERING:
        dl_params = params.ParametersHandler('ManouvreTransformerTraj.yaml', 'highD.yaml', '../config')
        experiment_file = '../experiments/ManouvreTransformerTraj_highD_2022-06-14 15:14:02.050065'
        dl_params.import_experiment(experiment_file)
        #initial_man = 0
        #wif_man = np.array([[0,0],[1,10]])
        #wif_man = wif_man_encoder(35, initial_man, wif_man)
        #wif_man = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #35 + 1
        #wif_man = np.array(wif_man)
        wif_man = np.ones((31), dtype=int)*2

        bev_plotter.whatif_render(dl_params, scenario_number= 14, wif_man = wif_man)
    elif p.ITERATIVE_RENDERING:
        bev_plotter.iterative_render()
    else:
        bev_plotter.render_scenarios()