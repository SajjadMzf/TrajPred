import os
import cv2
import numpy as np 
import pickle
import h5py
import matplotlib.pyplot as plt
import read_csv as rc
import param as p
from utils import rendering_funcs as rf

class RenderScenarios:
    """This class is for rendering extracted scenarios from HighD dataset recording files (needs to be called seperately for each scenario).
    """
    def __init__(
        self,
        file_num:'Number of recording file being rendered',
        track_path:'Path to track file', 
        pickle_path:'Path to pickle file', 
        static_path:'Path to static file',
        meta_path:'Path to meta file',
        dataset_name: 'Dataset  Name'):
        self.seq_len = p.SEQ_LEN
        self.metas = rc.read_meta_info(meta_path)
        self.fr_div = self.metas[rc.FRAME_RATE]/p.FPS
        self.track_path = track_path
        self.scenarios = []
        self.file_num = file_num
        
        # Default Settings
        self.save_whole_imgs = p.save_whole_imgs
        self.save_cropped_imgs = p.save_cropped_imgs
        
        ''' 1. Representation Properties:'''
        self.filled = True
        self.empty = False
        self.dtype = bool
        # 1.1 Scales and distances
        self.image_scaleW = p.image_scaleW
        self.image_scaleH = p.image_scaleH
        self.lines_width = 1
        self.dash_lines = tuple([8,8])
        self.highway_top_margin = int(5 * self.image_scaleH)
        self.highway_bottom_margin = int(5 * self.image_scaleH)
        self.cropped_height = p.cropped_height
        self.cropped_width = p.cropped_width
        # 1.3 Others
        self.mid_barrier = False
        
        self.LC_whole_imgs_rdir = "../../Dataset/" + dataset_name + "/WholeImages" + p.dir_ext
        self.LC_cropped_imgs_rdir = "../../Dataset/" + dataset_name + "/CroppedImages" + p.dir_ext
        
        self.LC_states_dir = "../../Dataset/" + dataset_name + "/Scenarios" + p.dir_ext 
        self.LC_image_dataset_rdir = "../../Dataset/" + dataset_name + "/RenderedDataset" + p.dir_ext
         
        self.frames_data, image_width = rc.read_track_csv(track_path, pickle_path, group_by = 'frames', fr_div = self.fr_div)
        self.statics = rc.read_static_info(static_path)
        
        self.image_width = int(image_width * self.image_scaleW)
        self.image_height = int((self.metas[rc.LOWER_LANE_MARKINGS][-1])*self.image_scaleH + self.highway_top_margin + self.highway_bottom_margin)
        self.update_dirs()
        
    def load_scenarios(self):
        file_dir = os.path.join(self.LC_states_dir, str(self.file_num).zfill(2) + '.pickle')
        with open(file_dir, 'rb') as handle:
            self.scenarios = pickle.load(handle)
    
    def save_dataset(self):
        file_dir = os.path.join(self.LC_image_dataset_dir, str(self.file_num).zfill(2) + '.h5')
        npy_dir = os.path.join(self.LC_image_dataset_dir, str(self.file_num).zfill(2) + '.npy')
        hf = h5py.File(file_dir, 'w')
        valid_itrs = [False if scenario['images'] is None else True for scenario in self.scenarios]
        data_num = valid_itrs.count(True)
        image_data = hf.create_dataset('image_data', shape = (data_num, self.seq_len, 3, self.cropped_height, self.cropped_width), dtype = np.bool)
        frame_data = hf.create_dataset('frame_data', shape = (data_num, self.seq_len), dtype = np.float32)       
        tv_data = hf.create_dataset('tv_data', shape = (data_num,), dtype = np.int)
        labels = hf.create_dataset('labels', shape = (data_num,), dtype = np.float32)
        state_wirth_data = hf.create_dataset('state_wirth_data', shape = (data_num, self.seq_len, 18), dtype = np.float32)
        state_shou_data = hf.create_dataset('state_shou_data', shape = (data_num, self.seq_len, 18), dtype = np.float32)
        state_ours_data = hf.create_dataset('state_ours_data', shape = (data_num, self.seq_len, 18), dtype = np.float32)
        ttlc_available = hf.create_dataset('ttlc_available', shape = (data_num,), dtype = np.bool)
        

        data_itr = 0
        for itr, validity in enumerate(valid_itrs):
            if validity == False:
                continue
            temp = self.scenarios[itr]['images']
            temp = np.transpose(temp,[0,3,1,2])# Chanel first
            image_data[data_itr, :] = temp
            state_wirth_data[data_itr, :] = self.scenarios[itr]['states_wirth']
            state_shou_data[data_itr, :] = self.scenarios[itr]['states_shou']
            state_ours_data[data_itr, :] = self.scenarios[itr]['states_ours']
            frame_data[data_itr, :] = self.scenarios[itr]['frames']
            tv_data[data_itr] = self.scenarios[itr]['tv']
            labels[data_itr] = self.scenarios[itr]['label']
            ttlc_available[data_itr] = self.scenarios[itr]['ttlc_available']
            data_itr += 1
        hf.close()
        np.save(npy_dir, data_itr) 
    
      

        

    def render_scenarios(self)-> "Number of rendered and saved scenarios":
        saved_data_number = 0
        
        for scenario_idx, scenario in enumerate(self.scenarios):
            
            tv_id = scenario['tv']
            label = scenario['label']
            driving_dir = scenario['driving_dir']

            scene_cropped_imgs = []
            whole_imgs = []
            img_frames = []
            states_wirth = []
            states_shou = []
            states_ours = []
            tv_lane_ind = None
            number_of_fr = self.seq_len 
            for fr in range(number_of_fr):
                frame = scenario['frames'][fr]
                
                svs_ids = scenario['svs']['id'][:,fr]
                cropped_img, whole_img, valid, state_wirth, state_shou,state_ours, tv_lane_ind = self.plot_frame(
                    self.frames_data[int(frame/self.fr_div -1)],
                    tv_id, 
                    svs_ids,
                    driving_dir,
                    frame,
                    tv_lane_ind
                    )
                
                # Being valid is about width of TV not being less than 2 pixels
                if not valid:
                    print('Invalid frame:', fr+1, ' of scenario: ', scenario_idx+1, ' of ', len(self.scenarios))
                    break
                #plt.figure()
                #print(np.mean(whole_img.astype(np.float), axis =2).shape)
                #plt.imshow(np.mean(whole_img.astype(np.float), axis =2), cmap='gray')
                
                #plt.figure()
                #plt.imshow(np.mean(cropped_img.astype(np.float), axis =2), cmap='gray')
                #plt.show()
                #print(np.mean(cropped_img.astype(np.float), axis =2))
                #exit()
                scene_cropped_imgs.append(cropped_img)
                whole_imgs.append(whole_img)
                img_frames.append(frame)
                states_wirth.append(state_wirth)
                states_shou.append(state_shou)
                states_ours.append(state_ours)
                
            if not valid:
                continue
            
            scene_cropped_imgs = np.array(scene_cropped_imgs, dtype = self.dtype)
            self.scenarios[scenario_idx]['images'] = scene_cropped_imgs
            self.scenarios[scenario_idx]['states_wirth'] = np.array(states_wirth)
            self.scenarios[scenario_idx]['states_shou'] = np.array(states_shou)
            self.scenarios[scenario_idx]['states_ours'] = np.array(states_ours)
            
            saved_data_number += 1
            
            if self.save_whole_imgs: rf.save_image_sequence( tv_id, img_frames, whole_imgs, os.path.join(self.LC_whole_imgs_dir, str(label)), self.file_num)
            if self.save_cropped_imgs: rf.save_image_sequence( tv_id, img_frames, scene_cropped_imgs, os.path.join(self.LC_cropped_imgs_dir, str(label)), self.file_num)
            
        return saved_data_number

    def plot_frame(
        self, 
        frame_data:'Data array of current frame', 
        tv_id:'ID of the TV', 
        svs_ids:'IDs of the SVs', 
        driving_dir:'TV driving direction',
        frame:'frame',
        tv_lane_ind:'The TV lane index of its initial frame'):
        
        veh_channel, lane_channel, obs_channel = rf.initialize_representation(self.image_width, self.image_height, rep_dtype = self.dtype, filled_value = self.filled, occlusion= False)
        assert(frame_data[rc.FRAME]==frame)   
        tv_itr = np.nonzero(frame_data[rc.TRACK_ID] == tv_id)[0][0]
        svs_itr = np.array([np.nonzero(frame_data[rc.TRACK_ID] == sv_id)[0][0] if sv_id!=0 else None for sv_id in svs_ids])
        
        vehicle_in_frame_number = len(frame_data[rc.TRACK_ID])
        
        # Lambda function for calculating vehicles x,y, width and length
        corner_x = lambda itr: int(frame_data[rc.X][itr]*self.image_scaleW)
        corner_y = lambda itr: int((frame_data[rc.Y][itr])*self.image_scaleH) + self.highway_top_margin
        veh_width = lambda itr: int(frame_data[rc.WIDTH][itr]*self.image_scaleW)
        veh_height = lambda itr: int(frame_data[rc.HEIGHT][itr]*self.image_scaleH)
        center_x = lambda itr: int(frame_data[rc.X][itr]*self.image_scaleW + veh_width(itr)/2)
        center_y = lambda itr: int(frame_data[rc.Y][itr]*self.image_scaleH + veh_height(itr)/2)  + self.highway_top_margin
        
        

        
        for itr in range(vehicle_in_frame_number):
            veh_channel = rf.draw_vehicle(veh_channel, corner_x(itr), corner_y(itr), veh_width(itr), veh_height(itr), self.filled)
        

        # If a barrier is conisdered at the middle of highway blocking the view, the image height will change depending on driving direction
        if self.mid_barrier:
            mid_barrier = int(((self.metas[rc.UPPER_LANE_MARKINGS][-1] + self.metas[rc.LOWER_LANE_MARKINGS][0])/2) * self.image_scaleH)  + self.highway_top_margin
            dir_image_height = lambda driving_dir: [0, mid_barrier-2] if driving_dir==1 else [mid_barrier+2, self.image_height]
        else:
            dir_image_height = lambda driving_dir: [0, self.image_height]
        
        # Draw lines
        lane_channel = rf.draw_lane_markings(lane_channel, 
                                self.image_width, 
                                (self.metas[rc.LOWER_LANE_MARKINGS])* self.image_scaleH  + self.highway_top_margin,
                                (self.metas[rc.UPPER_LANE_MARKINGS])* self.image_scaleH  + self.highway_top_margin,
                                self.lines_width, 
                                self.filled, 
                                self.dash_lines*self.image_scaleW
                                )
        
        # Crop image
        image = np.concatenate((veh_channel, lane_channel, obs_channel), axis = 2)
        
        
        tv_lane_markings = (self.metas[rc.UPPER_LANE_MARKINGS])* self.image_scaleH  + self.highway_top_margin if driving_dir == 1 else (self.metas[rc.LOWER_LANE_MARKINGS])* self.image_scaleH + self.highway_top_margin
        tv_lane_markings = tv_lane_markings.astype(int)
        if tv_lane_ind is None:
            tv_lane_ind = 0
            for ind, value in reversed(list(enumerate(tv_lane_markings))):
                if center_y(tv_itr)>value:
                    tv_lane_ind = ind
                    break
        
        cropped_img, valid = rf.crop_image(image, 
                                    self.image_width, 
                                    self.image_height,
                                    int(center_x(tv_itr)),
                                    int(center_y(tv_itr)),
                                    tv_lane_markings,
                                    driving_dir,
                                    tv_lane_ind,
                                    self.cropped_height,
                                    self.cropped_width,
                                    self.lines_width,
                                    self.filled)
        
        
        
        def clamp(n, minn, maxn):
            if n < minn:
                return minn
            elif n > maxn:
                return maxn
            else:
                return n

        if tv_lane_ind+1>=len(tv_lane_markings):# len is 1-based lane_ind is 0-based
            return cropped_img, image, False, [], [], [], []
        
        lane_width = (tv_lane_markings[tv_lane_ind+1]-tv_lane_markings[tv_lane_ind])
        
        #velocity_x = lambda itr: abs(frame_data[rc.X_VELOCITY][itr])/p.MAX_VELOCITY_X
        fix_sign = lambda x: x if driving_dir == 1 else -1*x

        tv_left_lane_ind = tv_lane_ind + 1 if driving_dir==1 else tv_lane_ind

        lateral_pos = lambda itr, lane_ind: abs(frame_data[rc.Y][itr] + frame_data[rc.HEIGHT][itr]/2- tv_lane_markings[lane_ind])

        rel_distance_x = lambda itr: abs(frame_data[rc.X][itr] - frame_data[rc.X][tv_itr])

        rel_distance_y = lambda itr: abs(frame_data[rc.Y][itr] - frame_data[rc.Y][tv_itr])
        
        rel_velo_x = lambda itr: fix_sign(frame_data[rc.X_VELOCITY][itr] - frame_data[rc.X_VELOCITY][tv_itr]) #transform from [-1,1] to [0,1]
        rel_velo_y =lambda itr: fix_sign(frame_data[rc.Y_VELOCITY][itr] - frame_data[rc.Y_VELOCITY][tv_itr])
        rel_acc_x = lambda itr: fix_sign(frame_data[rc.X_ACCELERATION][itr] - frame_data[rc.X_ACCELERATION][tv_itr])
        rel_acc_y =lambda itr: fix_sign(frame_data[rc.Y_ACCELERATION][itr] - frame_data[rc.Y_ACCELERATION][tv_itr])


        # svs : [pv_id, fv_id, rv_id, rpv_id, rfv_id, lv_id, lpv_id, lfv_id]
        pv_itr = svs_itr[0]
        fv_itr = svs_itr[1]
        rv_itr = svs_itr[2]
        rpv_itr = svs_itr[3]
        rfv_itr = svs_itr[4]
        lv_itr = svs_itr[5]
        lpv_itr = svs_itr[6]
        lfv_itr = svs_itr[7]
        

        ##################### LSTM1, MLP1 #########################        
        state_wirth = np.zeros((18)) # From Wirthmuller 2021
        
        #(1) Existence of left lane, 
        if (tv_lane_ind+2==len(tv_lane_markings) and driving_dir == 1) or (tv_lane_ind ==0 and driving_dir==2):
            state_wirth[0] = 0
        else:
            state_wirth[0] = 1
        # (2) Existence of right lane, 
        if (tv_lane_ind+2==len(tv_lane_markings) and driving_dir == 2) or (tv_lane_ind ==0 and driving_dir==1):
            state_wirth[1] = 0
        else:
            state_wirth[1] = 1
        # (3) lane width,  
        state_wirth[2] = lane_width 
        # (4) Longitudinal distance of TV to PV, 
        state_wirth[3] = rel_distance_x(pv_itr) if pv_itr != None else 400 
        # (5)Longitudinal distance of TV to RPV, 
        state_wirth[4] = rel_distance_x(rpv_itr) if rpv_itr != None else 400  
        # (6)Longitudinal distance of TV to FV, 
        state_wirth[5] = rel_distance_x(fv_itr) if fv_itr != None else 400 
        # (7)lateral distance of TV to the left lane marking, 
        state_wirth[6] = lateral_pos(tv_itr, tv_left_lane_ind)
        # (8)lateral distance of TV to RV, 
        state_wirth[7] = rel_distance_y(rv_itr) if rv_itr != None else 3*lane_width 
        # (9)lateral distance of TV to RFV, 
        state_wirth[8] = rel_distance_y(rfv_itr) if rfv_itr != None else 3*lane_width 
        # (10) relative longitudinal velocity of TV w.r.t. PV,
        state_wirth[9] = rel_velo_x(pv_itr) if pv_itr != None else 0  
        # (11) relative longitudinal velocity of TV w.r.t. FV 
        state_wirth[10] = rel_velo_x(fv_itr) if fv_itr != None else 0
        # (12)Relative lateral velocity of TV w.r.t. PV, 
        state_wirth[11] = rel_velo_y(pv_itr) if pv_itr != None else 0
        # (13)Relative lateral velocity of TV w.r.t. RPV,
        state_wirth[12] = rel_velo_y(rpv_itr) if rpv_itr != None else 0  
        # (14)Relative lateral velocity of TV w.r.t. RV, 
        state_wirth[13] = rel_velo_y(rv_itr) if rv_itr != None else 0
        # (15)Relative lateral velocity of TV w.r.t. LV, 
        state_wirth[14] = rel_velo_y(lv_itr) if lv_itr != None else 0
        # (16) longitudinal acceleration of the TV, 
        state_wirth[15] = fix_sign(frame_data[rc.X_ACCELERATION][tv_itr])
        # (17) relative longitudinal acceleration of the TV w.r.t RPV, 
        state_wirth[16] = rel_acc_x(rpv_itr) if rpv_itr != None else 0
        # (18) lateral acceleration of the prediction target
        state_wirth[17] = fix_sign(frame_data[rc.Y_ACCELERATION][tv_itr])



        ##################### MLP2 ######################### 
        state_shou = np.zeros((18)) # From Shou 2020
          
        #(1) Existence of left lane, 
        if (tv_lane_ind+2==len(tv_lane_markings) and driving_dir == 1) or (tv_lane_ind ==0 and driving_dir==2):
            state_shou[0] = 0
        else:
            state_shou[0] = 1
        # (2) Existence of right lane, 
        if (tv_lane_ind+2==len(tv_lane_markings) and driving_dir == 2) or (tv_lane_ind ==0 and driving_dir==1):
            state_shou[1] = 0
        else:
            state_shou[1] = 1
        # (3) Longitudinal distance of TV to RPV, 
        state_shou[2] = rel_distance_x(rpv_itr) if rpv_itr != None else 400 
        # (4) Longitudinal distance of TV to PV, 
        state_shou[3] = rel_distance_x(pv_itr) if pv_itr != None else 400 
        # (5) Longitudinal distance of TV to LPV, 
        state_shou[4] = rel_distance_x(lpv_itr) if lpv_itr != None else 400 
        # (6) Longitudinal distance of TV to RV, 
        state_shou[5] = rel_distance_x(rv_itr) if rv_itr != None else 400 
        # (7) Longitudinal distance of TV to LV, 
        state_shou[6] = rel_distance_x(lv_itr) if lv_itr != None else 400 
        # (8) Longitudinal distance of TV to RFV, 
        state_shou[7] = rel_distance_x(rfv_itr) if rfv_itr != None else 400 
        # (9) Longitudinal distance of TV to FV, 
        state_shou[8] = rel_distance_x(fv_itr) if fv_itr != None else 400 
        # (10) Longitudinal distance of TV to LFV, 
        state_shou[9] = rel_distance_x(lfv_itr) if lfv_itr != None else 400 
        # (11) Relative Velocity of TV w.r.t. RPV, 
        state_shou[10] = rel_velo_x(rpv_itr) if rpv_itr != None else 0
        # (12) Relative Velocity of TV w.r.t.  PV, 
        state_shou[11] = rel_velo_x(pv_itr) if pv_itr != None else 0
        # (13) Relative Velocity of TV w.r.t.  LPV, 
        state_shou[12] = rel_velo_x(lpv_itr) if lpv_itr != None else 0
        # (14) Relative Velocity of TV w.r.t.  RV, 
        state_shou[13] = rel_velo_x(rv_itr) if rv_itr != None else 0
        # (15) Relative Velocity of TV w.r.t. LV, 
        state_shou[14] = rel_velo_x(lv_itr) if lv_itr != None else 0
        # (16) Relative Velocity of TV w.r.t.  RFV, 
        state_shou[15] = rel_velo_x(rfv_itr) if rfv_itr != None else 0
        # (17) Relative Velocity of TV w.r.t.  FV, 
        state_shou[16] = rel_velo_x(fv_itr) if fv_itr != None else 0
        # (18) Relative Velocity of TV w.r.t. LFV 
        state_shou[17] = rel_velo_x(lfv_itr) if lfv_itr != None else 0
        
        ##################### LSTM2 #########################
        state_ours = np.zeros((18)) # a proposed features  
        # (1) lateral velocity 
        state_ours[0] = fix_sign(frame_data[rc.Y_VELOCITY][tv_itr])
        # (2) longitudinal velocity 
        state_ours[1] = fix_sign(frame_data[rc.X_VELOCITY][tv_itr])
        # (3) lateral acceleration 
        state_ours[2] = fix_sign(frame_data[rc.Y_ACCELERATION][tv_itr])
        # (4) longitudinal acceleration 
        state_ours[3] = fix_sign(frame_data[rc.X_ACCELERATION][tv_itr])
        # (5) lateral distance of TV to the left lane marking 
        state_ours[4] = lateral_pos(tv_itr, tv_left_lane_ind)
        # (6)Relative longitudinal velocity of the TV w.r.t. PV 
        state_ours[5] = rel_velo_x(pv_itr) if pv_itr != None else 0
        # (7) longitudinal distance of TV to PV 
        state_ours[6] = rel_distance_x(pv_itr) if pv_itr != None else 400 
        # (8) Relative longitudinal velocity of the TV w.r.t. FV, 
        state_ours[7] = rel_velo_x(fv_itr) if fv_itr != None else 0
        # (9) longitudinal distance of TV to FV, 
        state_ours[8] = rel_distance_x(fv_itr) if fv_itr != None else 400 
        # (10) longitudinal distance of TV to RPV, 
        state_ours[9] = rel_distance_x(rpv_itr) if rpv_itr != None else 400 
        # (11) longitudinal distance of TV to RV, 
        state_ours[10] = rel_distance_x(rv_itr) if rv_itr != None else 400 
        # (12) longitudinal distance of TV to RFV, 
        state_ours[11] = rel_distance_x(rfv_itr) if rfv_itr != None else 400 
        # (13) longitudinal distance of TV to LPV, 
        state_ours[12] = rel_distance_x(lpv_itr) if lpv_itr != None else 400 
        # (14) longitudinal distance of TV to LV, 
        state_ours[13] = rel_distance_x(lv_itr) if lv_itr != None else 400 
        # (15) longitudinal distance of TV to LFV, 
        state_ours[14] = rel_distance_x(lfv_itr) if lfv_itr != None else 400 
        # (16) Existence of left lane, 
        if (tv_lane_ind+2==len(tv_lane_markings) and driving_dir == 1) or (tv_lane_ind ==0 and driving_dir==2):
            state_ours[15] = 0
        else:
            state_ours[15] = 1
        # (17) Existence of right lane, 
        if (tv_lane_ind+2==len(tv_lane_markings) and driving_dir == 2) or (tv_lane_ind ==0 and driving_dir==1):
            state_ours[16] = 0
        else:
            state_ours[16] = 1
        
        # (18) lane width
        state_ours[17] = lane_width 
        
        return cropped_img, image, valid, state_wirth, state_shou, state_ours, tv_lane_ind
        

    
    def update_dirs(self):
        
        self.LC_cropped_imgs_dir = self.LC_cropped_imgs_rdir
        if not os.path.exists(self.LC_cropped_imgs_dir):
            os.makedirs(self.LC_cropped_imgs_dir)
        
        for i in range(3):
            label_dir = os.path.join(self.LC_cropped_imgs_dir, str(i))
            if not os.path.exists(label_dir):
                os.makedirs(label_dir) 

        self.LC_whole_imgs_dir = self.LC_whole_imgs_rdir
        if not os.path.exists(self.LC_whole_imgs_dir):
            os.makedirs(self.LC_whole_imgs_dir)
    
        for i in range(3):
            label_dir = os.path.join(self.LC_whole_imgs_dir, str(i))
            if not os.path.exists(label_dir):
                os.makedirs(label_dir) 
        
        self.LC_image_dataset_dir = self.LC_image_dataset_rdir
        if not os.path.exists(self.LC_image_dataset_dir):
            os.makedirs(self.LC_image_dataset_dir)
        
    