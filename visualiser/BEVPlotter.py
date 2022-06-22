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
                                                'traj_min': scenario['traj_min'],
                                                'traj_max':scenario['traj_max'],
                                                'input_features': [],
                                                'times':[], 
                                                'man_labels':[], 
                                                'man_preds':[], 
                                                'enc_man_preds':[], 
                                                'traj_labels':[], 
                                                'traj_preds':[], 
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
                sorted_scenarios_dict[sorted_index]['frames'].append(scenario['frames'][batch_itr])
                sorted_scenarios_dict[sorted_index]['input_features'].append(scenario['input_fetures'][batch_itr])
                
                
        
        for i in range(len(sorted_scenarios_dict)):
            times_array = np.array(sorted_scenarios_dict[i]['times'])
            sorted_indxs = np.argsort(times_array).astype(int)
            sorted_scenarios_dict[i]['times'] = [sorted_scenarios_dict[i]['times'][indx] for indx in sorted_indxs]
            sorted_scenarios_dict[i]['man_labels'] = [sorted_scenarios_dict[i]['man_labels'][indx] for indx in sorted_indxs]
            sorted_scenarios_dict[i]['man_preds'] = [sorted_scenarios_dict[i]['man_preds'][indx] for indx in sorted_indxs]
            sorted_scenarios_dict[i]['enc_man_preds'] = [sorted_scenarios_dict[i]['enc_man_preds'][indx] for indx in sorted_indxs]
            sorted_scenarios_dict[i]['traj_labels'] = [sorted_scenarios_dict[i]['traj_labels'][indx] for indx in sorted_indxs]
            sorted_scenarios_dict[i]['traj_preds'] = [sorted_scenarios_dict[i]['traj_preds'][indx] for indx in sorted_indxs]
            sorted_scenarios_dict[i]['frames'] = [sorted_scenarios_dict[i]['frames'][indx] for indx in sorted_indxs]
            sorted_scenarios_dict[i]['input_features'] = [sorted_scenarios_dict[i]['input_features'][indx] for indx in sorted_indxs]
            
        return sorted_scenarios_dict
    
    def whatif_render(self, scenario_number, wif_man):
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
            traj_preds = sorted_dict[scenario_number]['traj_preds'][j]
            frames = sorted_dict[scenario_number]['frames'][j]
            input_features = sorted_dict[scenario_number]['input_features'][j]
            scenario_tuple = (man_labels, man_preds, enc_man_preds, traj_labels, traj_preds, frames, data_file)
            self.render_single_scenario(tv_id, scenario_tuple, plotted_data_number, summary_image =True)
            plotted_data_number += 1
            print("Scene Number: {}".format(plotted_data_number))
            if plotted_data_number >= self.num_output:
                break
        
    def iterative_render(self):
        sorted_dict = self.sort_scenarios()
        plotted_data_number = 0
        for i in range(len(sorted_dict)):
            tv_id = sorted_dict[i]['tv']
            data_file = sorted_dict[i]['data_file']
            print('TV ID: {}, List of Available Frames: {}'.format(tv_id, sorted_dict[i]['times']))
            for j,time in enumerate(sorted_dict[i]['times']):
                man_labels = sorted_dict[i]['man_labels'][j]
                man_preds = sorted_dict[i]['man_preds'][j]
                enc_man_preds = sorted_dict[i]['enc_man_preds'][j]
                traj_labels = sorted_dict[i]['traj_labels'][j]
                traj_preds = sorted_dict[i]['traj_preds'][j]
                frames = sorted_dict[i]['frames'][j]
                scenario_tuple = (man_labels, man_preds, enc_man_preds, traj_labels, traj_preds, frames, data_file)
                self.render_single_scenario(tv_id, scenario_tuple, plotted_data_number, summary_image =True)
                plotted_data_number += 1
                print("Scene Number: {}".format(plotted_data_number))
                if plotted_data_number >= self.num_output:
                    break
            if plotted_data_number >= self.num_output:
                break 
        return plotted_data_number


    def render_single_scenario(self, tv_id, scenario_tuple, scenario_number, summary_image):
        (man_labels, man_preds, enc_man_preds, traj_labels, traj_preds, frames, data_file) = scenario_tuple
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
        self.image_width = int(image_width*p.IMAGE_SCALE )
        self.image_height = int(self.metas[rc.LOWER_LANE_MARKINGS][-1]*p.IMAGE_SCALE + p.BORDER_PIXELS)

        driving_dir = self.statics[tv_id][rc.DRIVING_DIRECTION]
                        
        #self.plot_overview(man_labels, man_preds, traj_labels,traj_preds, plotted_data_number)
        traj_imgs = []
        tv_track = []
        tv_future_track = ([],[])
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
                fr,
                in_seq_len,
                tv_track =  tv_track,
                tv_lane_ind = tv_lane_ind,
                tv_future_track = tv_future_track
                )
            if not summary_image:
                traj_imgs.append(traj_img)
            elif summary_image and fr == in_seq_len-1:
                traj_imgs.append(traj_img)
                break
        
        traj_imgs = np.array(traj_imgs)
        scenario_id = '{}_{}_{}'.format(data_file, tv_id, scenario_number)
        self.save_image_sequence(p.model_name, traj_imgs, self.traj_vis_dir,scenario_id , summary_image)
        
    
    def render_scenarios(self)-> "Number of rendered and saved scenarios":
        plotted_data_number = 0
        prev_data_file = -1
        for _, scenario in enumerate(self.scenarios):
            for batch_itr, tv_id in enumerate(scenario['tv']):
                man_labels = scenario['man_labels'][batch_itr]
                man_preds = scenario['man_preds'][batch_itr]
                enc_man_preds = scenario['enc_man_preds'][batch_itr]
                traj_labels = scenario['traj_labels'][batch_itr]
                traj_preds = scenario['traj_preds'][batch_itr]
                frames = scenario['frames'][batch_itr]
                data_file = int(scenario['data_file'][batch_itr].split('.')[0])
                scenario_tuple = (man_labels, man_preds, enc_man_preds, traj_labels, traj_preds, frames, data_file)
                self.render_single_scenario(tv_id, scenario_tuple, plotted_data_number, summary_image = False)
                plotted_data_number += 1
                print("Scene Number: {}".format(plotted_data_number))
                if plotted_data_number >= self.num_output:
                    break
            if plotted_data_number >= self.num_output:
                break 
        return plotted_data_number


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
        seq_fr:'frame sequence',
        in_seq_len,
        tv_lane_ind,
        tv_track = [],
        tv_future_track = []
        ):
        
        assert(frame_data[rc.FRAME]==frame) 
        image = np.ones((self.image_height, self.image_width,3), dtype=np.int32)*p.COLOR_CODES['BACKGROUND']

        tv_itr = np.nonzero(frame_data[rc.TRACK_ID] == tv_id)[0][0]
        
        tv_lane_markings = self.metas[rc.UPPER_LANE_MARKINGS]*p.IMAGE_SCALE if driving_dir == 1 else self.metas[rc.LOWER_LANE_MARKINGS]*p.IMAGE_SCALE
        tv_lane_markings = tv_lane_markings.astype(int)

        vehicle_in_frame_number = len(frame_data[rc.TRACK_ID])
        
        corner_x = lambda itr: int(frame_data[rc.X][itr]*p.IMAGE_SCALE)
        corner_y = lambda itr: int(frame_data[rc.Y][itr]*p.IMAGE_SCALE) 
        veh_width = lambda itr: int(frame_data[rc.WIDTH][itr]*p.IMAGE_SCALE)
        veh_height = lambda itr: int(frame_data[rc.HEIGHT][itr]*p.IMAGE_SCALE)
        center_x = lambda itr: int(frame_data[rc.X][itr]*p.IMAGE_SCALE+ frame_data[rc.WIDTH][itr]*p.IMAGE_SCALE/2)
        center_y = lambda itr: int(frame_data[rc.Y][itr]*p.IMAGE_SCALE+ frame_data[rc.HEIGHT][itr]*p.IMAGE_SCALE/2)  
        center2corner_x = lambda center, itr: int(center - frame_data[rc.WIDTH][itr]*p.IMAGE_SCALE/2)
        center2corner_y = lambda center, itr: int(center - frame_data[rc.HEIGHT][itr]*p.IMAGE_SCALE/2)
        fix_sign = lambda x: x if driving_dir ==1 else -1*x
        

        '''3. Draw Lines'''
        # Draw lines
        image = self.draw_lines(image, self.image_width, 
                                (self.metas[rc.LOWER_LANE_MARKINGS]*p.IMAGE_SCALE),
                                (self.metas[rc.UPPER_LANE_MARKINGS]*p.IMAGE_SCALE))
        
        '''Draw vehicles'''
        if not p.HIDE_SVS:
            for itr in range(vehicle_in_frame_number):
                if itr != tv_itr:
                    image = self.draw_vehicle(image, corner_x(itr), corner_y(itr), veh_width(itr), veh_height(itr), p.COLOR_CODES['SV'])
        
        image = image.astype(np.uint8)
        if seq_fr == 0:
            tv_track.append((center_x(tv_itr), center_y(tv_itr)))
        if not p.HIDE_TVS_TRAJ_HISTORY and seq_fr>0 and seq_fr<in_seq_len:
            initial_x = tv_track[0][0]
            initial_y = tv_track[0][1]
            dx = 0
            dy = 0
            for i in range(seq_fr):
                dx += fix_sign(traj_labels[seq_fr,1]*p.IMAGE_SCALE)
                dy += fix_sign(traj_labels[seq_fr,0]*p.IMAGE_SCALE)
            tv_track.append((int(initial_x+dx), int(initial_y+dy)))
        for i in range(len(tv_track)):
            if i ==0:
                continue
            image = cv2.line(image, tv_track[i], tv_track[i-1], p.COLOR_CODES['GT_TRAJ'], 2)
            image = cv2.circle(image, tv_track[i-1], 4, p.COLOR_CODES['GT_TRAJ'], -1)
    
        image = self.draw_vehicle(image, center2corner_x(tv_track[-1][0], tv_itr), center2corner_y(tv_track[-1][1], tv_itr), veh_width(tv_itr), veh_height(tv_itr), p.COLOR_CODES['TV'])
        image = image.astype(np.uint8)

        if seq_fr == in_seq_len-1:
            tv_gt_future_track = []
            #tv_gt_future_track.append((center_x(tv_itr), center_y(tv_itr)))
            tv_gt_future_track.append((tv_track[-1][0], tv_track[-1][1]))
            tv_pr_future_track = []
            #tv_pr_future_track.append((center_x(tv_itr), center_y(tv_itr)))
            tv_pr_future_track.append((tv_track[-1][0], tv_track[-1][1]))
            tgt_seq_len = traj_preds.shape[0]
            initial_x = tv_gt_future_track[-1][0]
            initial_y = tv_gt_future_track[-1][1]
            dx_gt = 0
            dy_gt = 0
            for fut_fr in range(in_seq_len, tgt_seq_len+in_seq_len):
                dx_gt += fix_sign(traj_labels[fut_fr,1]*p.IMAGE_SCALE)
                dy_gt += fix_sign(traj_labels[fut_fr,0]*p.IMAGE_SCALE)
                tv_gt_future_track.append((int(initial_x+dx_gt), int(initial_y+dy_gt)))

            dx_pr = 0
            dy_pr = 0  
            for fut_fr in range(tgt_seq_len):
                dx_pr += fix_sign(traj_preds[fut_fr,1]*p.IMAGE_SCALE)
                dy_pr += fix_sign(traj_preds[fut_fr,0]*p.IMAGE_SCALE)
                tv_pr_future_track.append((int(initial_x+dx_pr), int(initial_y+dy_pr)))
            tv_future_track = (tv_gt_future_track, tv_pr_future_track)


        tv_gt_future_track = tv_future_track[0]
        tv_pr_future_track = tv_future_track[1]
        for i in range(len(tv_gt_future_track)):
            if i ==0:
                continue
            image = cv2.line(image, tv_gt_future_track[i], tv_gt_future_track[i-1], p.COLOR_CODES['GT_TRAJ'], 2)
            image = cv2.circle(image, tv_gt_future_track[i-1], 4, p.COLOR_CODES['GT_TRAJ'], -1)

        for i in range(len(tv_pr_future_track)):
            if i ==0:
                continue
            image = cv2.line(image, tv_pr_future_track[i], tv_pr_future_track[i-1], p.COLOR_CODES['PR_TRAJ'], 2)
            image = cv2.circle(image, tv_pr_future_track[i-1], 4, p.COLOR_CODES['PR_TRAJ'], -1)
                
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
                x = np.arange(in_seq_len,in_seq_len + tgt_seq_len)
                y = np.argmax(man_preds,axis = 1)
                ax.plot(x,y, 'o-')
            gt_x = np.arange(1, in_seq_len + tgt_seq_len+1)
            gt_y = man_labels
            ax.plot(gt_x, gt_y)
            plt.xlabel('Frame')
            plt.ylabel('Manouvre Label')
            ax.grid(True)
            plt.yticks([-1,0,1,2,3], ['','LK', 'RLC', 'LLC', ''])
            plt.xticks(gt_x,gt_x)
            fig.tight_layout(pad=3)
            #plt.show()
            #plt.close()
            man_image = mplfig_to_npimage(fig)
            plt.close('all')
            man_bar = np.ones((man_image.shape[0]+20, image.shape[1],3), dtype = np.int32)
            man_bar[:,:,:] = p.COLOR_CODES['BACKGROUND']
            man_bar[0:man_image.shape[0], 0:man_image.shape[1], :] = man_image
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

if __name__ =="__main__":
    bev_plotter = BEVPlotter( 
        fps = p.FPS,
        result_file = p.RESULT_FILE,
        dataset_name = p.DATASET,
        num_output = p.NUM_OUTPUT)
    if p.WHATIF_RENDERING:
        bev_plotter.whatif_render()
    elif p.ITERATIVE_RENDERING:
        bev_plotter.iterative_render()
    else:
        bev_plotter.render_scenarios()