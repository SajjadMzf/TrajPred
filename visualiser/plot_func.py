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
import torch
import torch.utils.data as utils_data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pdb
import read_csv as rc
import param as p
import utils

def plot_frame(
        lane_markings,
        frame_data, 
        tv_id, 
        driving_dir,
        frame,
        man_labels, 
        man_preds,
        mode_prob,
        traj_labels, #[in_seq_len + tgt_seq_len ,2]
        traj_preds,# [n_mode, tgt_seq_len, 2]
        image_width,
        image_height
        ):
        print(int(frame),end="\r")
        # convert coordinate to image coordinate:
        lane_y_max = max([max(lane['l'][:,1]) for lane in lane_markings])
        lane_y_min = min([min(lane['r'][:,1]) for lane in lane_markings])
        lane_x_max = max([max(lane['l'][:,0]) for lane in lane_markings])
        lane_x_min = min([min(lane['l'][:,0]) for lane in lane_markings])
        x_ = lambda x: (x - lane_x_min) 
        y_ = lambda y: (lane_y_max-y) 
        
        x_l = lambda x: int((x - lane_x_min) * p.X_IMAGE_SCALE)
        y_l = lambda y: int((lane_y_max-y) *p.Y_IMAGE_SCALE)


        assert(frame_data[rc.FRAME][0]==frame) 
        image = np.ones((image_height, image_width,3), dtype=np.int32)*p.COLOR_CODES['BACKGROUND']
        
        tv_itr = np.nonzero(frame_data[rc.TRACK_ID] == tv_id)[0][0]
        n_vehicles = len(frame_data[rc.TRACK_ID])
        n_mode = traj_preds.shape[0]
        tgt_seq_len = traj_preds.shape[1]
        in_seq_len = traj_labels.shape[0] - tgt_seq_len
        #pdb.set_trace()
        #TODO: remove temp fix sign here and in render scenarios
        #traj_labels = -1*traj_labels
        #traj_preds = -1*traj_preds

        # set ref to frame=in_seq_len
        traj_labels -= traj_labels[in_seq_len-1] 
        #for i in range(n_mode):
        #    traj_preds[i] -= traj_preds[i,in_seq_len-1]
        #pdb.set_trace()
        # swap x,y 
        traj_labels = traj_labels[:,[1,0]]
        traj_preds = traj_preds[:, :, [1,0]]
        
        
        #assert(traj_labels[in_seq_len-1,0] == 0)
        #assert(traj_labels[in_seq_len-1,1] == 0)
        traj_labels[:,0] = x_(traj_labels[:,0]+frame_data[rc.X][tv_itr])*p.X_IMAGE_SCALE
        traj_labels[:,1] = y_(traj_labels[:,1]+frame_data[rc.Y][tv_itr])*p.Y_IMAGE_SCALE
        traj_preds[:,:,0] = x_(traj_preds[:,:,0]+frame_data[rc.X][tv_itr])*p.X_IMAGE_SCALE
        traj_preds[:,:,1] = y_(traj_preds[:,:,1]+frame_data[rc.Y][tv_itr])*p.Y_IMAGE_SCALE
        traj_labels = traj_labels.astype(int)
        traj_preds = traj_preds.astype(int)
        #print(traj_labels[in_seq_len-1,0])
        #print(x_(frame_data[rc.X][tv_itr])*p.X_IMAGE_SCALE)
        #pdb.set_trace()
        
        corner_x = lambda itr: int((x_(frame_data[rc.X][itr])-frame_data[rc.WIDTH][itr]/2)*p.X_IMAGE_SCALE)
        corner_y = lambda itr: int((y_(frame_data[rc.Y][itr])-frame_data[rc.HEIGHT][itr]/2)*p.Y_IMAGE_SCALE) 
        veh_width = lambda itr: int(frame_data[rc.WIDTH][itr]*p.X_IMAGE_SCALE)
        veh_height = lambda itr: int(frame_data[rc.HEIGHT][itr]*p.Y_IMAGE_SCALE)
        center2corner_x = lambda center, itr: int(center - frame_data[rc.WIDTH][itr]*p.X_IMAGE_SCALE/2)
        center2corner_y = lambda center, itr: int(center - frame_data[rc.HEIGHT][itr]*p.Y_IMAGE_SCALE/2)
        

        '''3. Draw Lines'''
        # Draw lane markings
        #image = draw_lane_markings(image, image_width, lane_markings)
        
        '''Draw vehicles'''
        if not p.HIDE_SVS:
            for itr in range(n_vehicles):
                if itr != tv_itr:
                    assert(corner_x(itr)>0)
                    assert(corner_x(itr)<image_width)
                    #assert(corner_y(itr)>0)
                    #assert(corner_y(itr)<image_height)
                    
                    image = draw_vehicle(image, corner_x(itr), corner_y(itr), veh_width(itr), veh_height(itr), p.COLOR_CODES['SV'])
                #else:
                #    image = draw_vehicle(image, corner_x(itr), corner_y(itr), veh_width(itr), veh_height(itr), p.COLOR_CODES['TV'])
        image = image.astype(np.uint8)
        
        # Draw Lane Markings:
        
        for lane in lane_markings:
            # TAKING AVG on lane y
            lane['r'][:,1] = np.min(lane['r'][:,1])
            lane['l'][:,1] = np.min(lane['l'][:,1])
            
            for itr in range(len(lane['r'])-1):
                cv2.line(image, (x_l(lane['r'][itr,0]), y_l(lane['r'][itr,1])),(x_l(lane['r'][itr+1,0]), y_l(lane['r'][itr+1,1])), p.COLOR_CODES['LANE'], thickness= 3)
            for itr in range(len(lane['l'])-1):
                cv2.line(image, (x_l(lane['l'][itr,0]), y_l(lane['l'][itr,1])),(x_l(lane['l'][itr+1,0]), y_l(lane['l'][itr+1,1])), p.COLOR_CODES['LANE'], thickness= 3)
    
        
        #Draw TV GT track
        for i in range(len(traj_labels)-1):
            #line
            image = cv2.line(image, tuple(traj_labels[i]), tuple(traj_labels[i+1]), p.COLOR_CODES['GT_TRAJ'], 2)
            # dot
            top_left = traj_labels[i] + np.array([-2,-2])
            bottom_right = traj_labels[i] + np.array([2,2])
            image = cv2.rectangle(image, tuple(top_left), tuple(bottom_right), p.COLOR_CODES['GT_TRAJ'], 1)
    
        image = draw_vehicle(image, center2corner_x(traj_labels[in_seq_len-1,0], tv_itr), center2corner_y(traj_labels[in_seq_len-1,1], tv_itr), veh_width(tv_itr), veh_height(tv_itr), p.COLOR_CODES['TV'])
        image = image.astype(np.uint8)
        
        sorted_modes = np.argsort(mode_prob)[::-1]
        n_modes = len(mode_prob)
        for mode_itr in range(min(n_modes,p.N_PLOTTED_TRAJS)):    
            
            bm = sorted_modes[mode_itr]
            for i in range(len(traj_preds[bm])):
                if i ==0:
                    continue
                image = cv2.line(image, tuple(traj_preds[bm][i]), tuple(traj_preds[bm][i-1]), p.COLOR_CODES['PR_TRAJ'][mode_itr], 2)
                image = cv2.circle(image, tuple(traj_preds[bm][i-1]), 4, p.COLOR_CODES['PR_TRAJ'][mode_itr], thickness = -1)

        if p.PLOT_MAN: 
            
            fig, ax = plt.subplots(figsize=(16, 4))
            ax.grid(axis = 'x')
            ax.set_axisbelow(True)
            mode_prob = utils.softmax(mode_prob)
            for i in range(p.N_PLOTTED_MODES):
                #print(i)
                msv = man_preds[sorted_modes[i]]
                if i == p.N_PLOTTED_MODES-1:
                    plot_xlabel = True
                else:
                    plot_xlabel = False
                hbar(ax,(p.N_PLOTTED_MODES-i-1)*0.75,'',msv2hbar(msv),p.COLOR_NAMES[i],plot_xlabel)
            hbar(ax,(p.N_PLOTTED_MODES)*0.75, 'Ground-Truth', msv2hbar(man_labels), 'black', edgecolor='dimgray')
            
            ax.yaxis.set_ticks(np.arange(0, 0.75*(p.N_PLOTTED_MODES+1), 0.75))
            ylabels = [item.get_text() for item in ax.get_yticklabels()]
            ylabels[p.N_PLOTTED_MODES] = 'Ground-Truth'
            #print(ylabels)   
            for i in range(p.N_PLOTTED_MODES):
                prob = mode_prob[sorted_modes[i]]
                ylabels[p.N_PLOTTED_MODES-i-1] = 'Mode#{} (P={}%)'.format(i+1, int(prob*100))
            
            ax.set_yticklabels(ylabels)
            #for tick in ax.get_yticklabels():
            #    tick.set_rotation(45)
        
            fig.tight_layout(pad=1)
            #plt.show()
            #exit()
            man_image = mplfig_to_npimage(fig)
            plt.close('all')
            man_bar = np.ones((man_image.shape[0]+20, image.shape[1],3), dtype = np.int32)
            man_bar[:,:,:] = p.COLOR_CODES['BACKGROUND']
            man_bar[0:man_image.shape[0], 0:man_image.shape[1], :] = man_image
            man_bar = man_bar[:,:,[2,1,0]]
            image = np.concatenate((image, man_bar), axis = 0)
                         
        return  image        

def draw_lane_markings(image, image_width, lane_markings):
    # TODO: draw lane markings
    return image

def plot_single_heatmap(z, height, width, muY, muX, sigY, sigX, rho,cut_off_sig_ratio):
    
    m = (muY, muX)
    s = np.array([[sigY**2, rho*sigX*sigY],[rho*sigX*sigY, sigX**2]])
    k = multivariate_normal(mean = m, cov = s)
    y_min = int(muY-cut_off_sig_ratio*sigY)
    x_min = int(muX-cut_off_sig_ratio*sigX)
    y_max = int(muY+cut_off_sig_ratio*sigY)
    x_max = int(muX+cut_off_sig_ratio*sigX)
    x = np.arange(x_min, x_max)
    y = np.arange(y_min, y_max)
    xx, yy = np.meshgrid(x,y)
    xxyy = np.c_[yy.ravel(), xx.ravel()]
    heatmap = k.pdf(xxyy)
    heatmap = heatmap.reshape(y_max-y_min, x_max-x_min)
    #plt.imshow(heatmap)
    #plt.show()
    z[y_min:y_max, x_min:x_max] += heatmap
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
        traj_dist_pred = output_dict['traj_dist_preds']
        traj_dist_pred = traj_dist_pred[:,:,seq_itr:seq_itr+1]
        
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



def draw_vehicle( image, x, y, width, height, color_code):
    image[y:(y+height), x:(x+width)] = color_code
    return image

def draw_lines(lane_channel, width, lower_lines, upper_lines):
    # Continues lines
    color_code = p.COLOR_CODES['LANE']
    lane_channel[int(upper_lines[0]):(int(upper_lines[0]) + p.lines_width), 0:width] = color_code
    lane_channel[int(upper_lines[-1]):(int(upper_lines[-1]) + p.lines_width), 0:width] = color_code
    lane_channel[int(lower_lines[0]):(int(lower_lines[0]) + p.lines_width), 0:width] = color_code
    lane_channel[int(lower_lines[-1]):(int(lower_lines[-1]) + p.lines_width), 0:width] = color_code
    
    # Broken Lines
    filled = int(p.dash_lines[0])
    total_line = int((p.dash_lines[0]+ p.dash_lines[1]))
    for line in upper_lines[1:-1]:
        for i in range(int(width/total_line)):
            lane_channel[int(line):(int(line) + p.lines_width), (i*total_line):(i*total_line+filled)] = color_code

    for line in lower_lines[1:-1]:
        for i in range(int(width/total_line)):
            lane_channel[int(line):(int(line) + p.lines_width), (i*total_line):(i*total_line+filled)] = color_code
    
    return lane_channel     
        
    
def save_image_sequence(
    experiment_dir,
    images:'Image sequence',
    save_dir:'Save directory',
    sample_id,
    sen_analysis,
    ):
    #folder_dir = os.path.join(os.path.join(save_dir, experiment_dir), sample_id)
    folder_dir = os.path.join(save_dir, experiment_dir)
    if sen_analysis:
        folder_dir = os.path.join(folder_dir, 'Sensitivity Analysis')
    else:
        folder_dir = os.path.join(folder_dir, 'all timesteps')
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
    size = (images[0].shape[1], images[0].shape[0])
    if not sen_analysis:
        video_dir = os.path.join(folder_dir,sample_id + '.avi')
        video_out = cv2.VideoWriter(video_dir, cv2.VideoWriter_fourcc(*'DIVX'), 1, size, isColor = True)

    for fr, image in enumerate(images):
        if not sen_analysis:
            video_out.write(image.astype(np.uint8))
        file_dir = os.path.join(folder_dir, sample_id + '_' +str(fr+1)+'.png')
        if not cv2.imwrite(file_dir, image):
            raise Exception("Could not write image: " + file_dir)
    if not sen_analysis:
        video_out.release()



def msv2hbar(msv):
    cats = []
    data = []
    while True:
        cats.append(msv[0])
        if np.all(msv ==cats[-1]):
            data.append(len(msv))
            break
        else:
            data.append(np.flatnonzero(msv!=cats[-1])[0])
        msv = msv[data[-1]:]
    hbar_data = np.stack((cats, data), axis = 0).astype(np.float)
    hbar_data[1] = hbar_data[1]/p.FPS
    assert(sum(hbar_data[1])==5)
    return hbar_data


def bgr2rgba(bgr):
    return (bgr[2]/255, bgr[1]/255, bgr[0]/255, 1)

def hbar(ax, loc, label,data ,color, x_label = False, edgecolor = 'black'):
   
    category_ids = data[0].astype(int)
    data = data[1,]
    data_cum = data.cumsum(axis=0)
    

    
    ax.invert_yaxis()
    
    ax.set_xlim(0, np.sum(data, axis=0))
    #ax.xaxis.set_visible(False)
    #ax.yaxis.set_visible(False)

    ax.set_xlabel('Prediction Time (s)')
    ax.set_ylabel('Manoeuvre Vectors')
    category_names = p.CLASS[category_ids]
    category_hatchs = [p.HATCHS[i] for i in category_ids]
    for i, (colname, hatch) in enumerate(zip(category_names, category_hatchs)):
        width = data[i]
        start = data_cum[i] - width
        ax.barh(loc,width= width, left=start, height=0.5,
                label=colname, color = color, edgecolor=edgecolor, hatch=hatch)
        #ax.barh(loc,width = width, left=start, height=0.5,
        #        label=colname, color = 'none')
        
        
        xcenter = start + width / 2
        #r, g, b, _ = color
        text_color = 'white' #if r * g * b < 0.5 else 'darkgrey'
        ax.text(xcenter, loc, category_names[i], ha='center', va='center',
                color=text_color, fontsize = 'large', fontstyle='oblique')
    

    return ax