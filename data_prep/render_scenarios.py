import os
import cv2
import numpy as np 
import pickle
import h5py
import matplotlib.pyplot as plt
import read_csv as rc
import param as p
from utils import rendering_funcs as rf
import pandas
import pdb
# python -m pdb -c continue 
class RenderScenarios:
    """This class is for rendering extracted scenarios from HighD dataset recording files (needs to be called seperately for each scenario).
    """
    def __init__(
        self,
        file_num:'Number of recording file being rendered',
        track_path:'Path to track file', 
        track_pickle_path:'Path to track pickle file', 
        frame_pickle_path:'Path to frame pickle file',
        static_path:'Path to static file',
        meta_path:'Path to meta file',
        dataset_name: 'Dataset  Name'):
       
        #self.metas = rc.read_meta_info(meta_path)
        self.fr_div = p.IN_FPS/p.FPS#self.metas[rc.FRAME_RATE]/p.FPS
        self.track_path = track_path
        self.scenarios = []
        self.file_num = file_num
        
        # Default Settings
        
        ''' 1. Representation Properties:'''
        self.filled = True
        self.empty = False
        self.dtype = bool
        # 1.3 Others
        
        
        self.LC_states_dir = "../../Dataset/" + dataset_name + "/Scenarios"  
        self.LC_image_dataset_rdir = "../../Dataset/" + dataset_name + "/RenderedDataset"
        self.frames_data = rc.read_track_csv(track_path, frame_pickle_path, group_by = 'frames', fr_div = self.fr_div)
        self.data_tracks = rc.read_track_csv(track_path, track_pickle_path, group_by = 'tracks', reload = False, fr_div = self.fr_div)
        self.track_list = [data_track[rc.TRACK_ID][0] for data_track in self.data_tracks]
        
        #self.statics = rc.read_static_info(static_path)
        df = pandas.read_csv(track_path)
        selected_frames = (df.frame%self.fr_div == 0).real.tolist()
        df = df.loc[selected_frames]
        self.frame_list = [data_frame[rc.FRAME][0] for data_frame in self.frames_data]
        self.update_dirs()
        
    def load_scenarios(self):
        file_dir = os.path.join(self.LC_states_dir, str(self.file_num).zfill(2) + '.pickle')
        with open(file_dir, 'rb') as handle:
            self.scenarios = pickle.load(handle)
    
    def save_dataset(self):
        file_dir = os.path.join(self.LC_image_dataset_dir, str(self.file_num).zfill(2) + '.h5')
        npy_dir = os.path.join(self.LC_image_dataset_dir, str(self.file_num).zfill(2) + '.npy')
        hf = h5py.File(file_dir, 'w')
        
        data_num = len(self.scenarios)
        total_frames = 0  
        for itr in range(data_num):
            total_frames += len(self.scenarios[itr]['frames'])

        frame_data = hf.create_dataset('frame_data', shape = (total_frames,), dtype = np.float32)       
        file_ids = hf.create_dataset('file_ids', shape = (total_frames,), dtype = np.int)
        tv_data = hf.create_dataset('tv_data', shape = (total_frames,), dtype = np.int)
        labels = hf.create_dataset('labels', shape = (total_frames,), dtype = np.float32)
        state_merging_data = hf.create_dataset('state_merging', shape = (total_frames, 23), dtype = np.float32)
        output_states_data = hf.create_dataset('output_states_data', shape = (total_frames, 2), dtype = np.float32)
        
        cur_frame = 0
        for itr in range(data_num):
            scenario_length = len(self.scenarios[itr]['frames'])
            state_merging_data[cur_frame:(cur_frame+scenario_length), :] = self.scenarios[itr]['states_merging']
            output_states_data[cur_frame:(cur_frame+scenario_length), :] = self.scenarios[itr]['output_states']
            frame_data[cur_frame:(cur_frame+scenario_length)] = self.scenarios[itr]['frames']
            tv_data[cur_frame:(cur_frame+scenario_length)] = self.scenarios[itr]['tv']
            file_ids[cur_frame:(cur_frame+scenario_length)] = self.scenarios[itr]['file']
            labels[cur_frame:(cur_frame+scenario_length)] = self.scenarios[itr]['label']
            cur_frame +=scenario_length
            
        assert(cur_frame == total_frames)
        hf.close()
        np.save(npy_dir, total_frames) 
    
    def render_scenarios(self)-> "Number of rendered and saved scenarios":
        saved_data_number = 0
        
        for scenario_idx, scenario in enumerate(self.scenarios):        
            if scenario_idx%500 == 0:
                print('Scenario {} out of: {}'.format(scenario_idx, len(self.scenarios)))
            tv_id = scenario['tv']
            img_frames = []
            states_merging = []
            output_states = []
            number_of_fr = len(scenario['frames'])
            tv_lane_ind = None
                
            for fr in range(number_of_fr):
                frame = scenario['frames'][fr]
                img_frames.append(frame)
                
                svs_ids = scenario['svs']['id'][:,fr]
                state_merging, output_state, tv_lane_ind = self.calc_states(
                    self.frames_data[self.frame_list.index(frame)],
                    tv_id, 
                    svs_ids,
                    frame,
                    tv_lane_ind
                    )
                output_states.append(output_state)
                if fr==0: # first time-step is repeated so that when we calc the diff, the first timestep would be zero displacement. 
                    output_states.append(output_state)
                
                states_merging.append(state_merging)
                    
            self.scenarios[scenario_idx]['states_merging'] = np.array(states_merging)
            output_states = np.array(output_states)
            output_states = output_states[1:,:]- output_states[:-1,:] # output_states[i] = x[i]-x[i-1] except for i=0 where output_states =0
            self.scenarios[scenario_idx]['output_states'] = output_states
            saved_data_number += 1
            
        return saved_data_number

    def calc_states(
        self, 
        frame_data:'Data array of current frame', 
        tv_id:'ID of the TV', 
        svs_ids:'IDs of the SVs', 
        frame:'frame',
        tv_lane_ind:'TV lane index'):
        
        assert(frame_data[rc.FRAME][0]==frame)   
        tv_itr = np.nonzero(frame_data[rc.TRACK_ID] == tv_id)[0][0]
        
        

        
        # exid version
        lateral_pos = lambda itr: frame_data[rc.Y2LANE][itr]
        
        rel_distance_x = lambda itr: (frame_data[rc.X][itr] - frame_data[rc.X][tv_itr])
        rel_distance_y = lambda itr: (frame_data[rc.Y][itr] - frame_data[rc.Y][tv_itr])
        
        # TV lane markings and lane index
        '''
        tv_lane_markings = (self.metas[rc.UPPER_LANE_MARKINGS]) if driving_dir == 1 else (self.metas[rc.LOWER_LANE_MARKINGS])
        
        
        if driving_dir ==1:
            tv_lane_ind = frame_data[rc.LANE_ID][tv_itr]-2
        else:
            tv_lane_ind = frame_data[rc.LANE_ID][tv_itr]-len(self.metas[rc.UPPER_LANE_MARKINGS])-2
        
        tv_lane_ind = int(tv_lane_ind)
        tv_left_lane_ind = tv_lane_ind + 1 if driving_dir==1 else tv_lane_ind
        if tv_lane_ind+1 >=len(tv_lane_markings):
            return True, 0, 0, 0, 0, 0, 0
        lane_width = (tv_lane_markings[tv_lane_ind+1]-tv_lane_markings[tv_lane_ind])
        #print('lane width: {}'.format(lane_width))
       '''
        tv_lane_ind = frame_data[rc.LANE_ID][tv_itr]-2
        lane_width = frame_data[rc.LANE_WIDTH][tv_itr]
        ## Output States:
        output_state = np.zeros((2))
        output_state[0] = frame_data[rc.Y][tv_itr]
        output_state[1] = frame_data[rc.X][tv_itr]
        
        
        svs_itr = np.array([np.nonzero(frame_data[rc.TRACK_ID] == sv_id)[0][0] if sv_id!=0 and sv_id!=-1 else None for sv_id in svs_ids])
        # svs : [pv_id, fv_id, rv1_id, rv2_id, rv3_id, lv1_id, lv2_id, lv3_id]
        pv_itr = svs_itr[0]
        fv_itr = svs_itr[1]
        rv1_itr = svs_itr[2]
        rv2_itr = svs_itr[3]
        rv3_itr = svs_itr[4]
        lv1_itr = svs_itr[5]
        lv2_itr = svs_itr[6]
        lv3_itr = svs_itr[7]
        
        
        ###################################### State Merging #####################################################
        state_merging = np.zeros((23)) # a proposed features  
        # (1) Lateral Pos
        state_merging[0] = lateral_pos(tv_itr)
        # (2) Long Velo
        state_merging[1] = frame_data[rc.X_VELOCITY][tv_itr]
        # (3)Lat ACC
        state_merging[2] = frame_data[rc.Y_ACCELERATION][tv_itr]
        # (4)Long ACC
        state_merging[3] = frame_data[rc.X_ACCELERATION][tv_itr]
        # (5) PV X
        state_merging[4] = rel_distance_x(pv_itr) if pv_itr != None else 400
        # (6) PV Y
        state_merging[5] = rel_distance_y(pv_itr) if pv_itr != None else 30
        # (7) FV X
        state_merging[6] = rel_distance_x(fv_itr) if fv_itr != None else 400
        # (8) FV Y
        state_merging[7] = rel_distance_y(pv_itr) if pv_itr != None else 30
        # (9) RV1 X
        state_merging[8] = rel_distance_x(rv1_itr) if rv1_itr != None else 400
        # (10) RV1 Y
        state_merging[9] = rel_distance_y(pv_itr) if pv_itr != None else 30
        # (11) RV2 X
        state_merging[10] = rel_distance_x(rv2_itr) if rv2_itr != None else 400
        # (12) RV2 Y
        state_merging[11] = rel_distance_y(pv_itr) if pv_itr != None else 30
        # (13) RV3 X
        state_merging[12] = rel_distance_x(rv3_itr) if rv3_itr != None else 400
        # (14) RV3 Y
        state_merging[13] = rel_distance_y(rv3_itr) if rv3_itr != None else 30
        # (15) LV1 X
        state_merging[14] = rel_distance_x(lv1_itr) if lv1_itr != None else 400
        # (16) LV1 Y
        state_merging[15] = rel_distance_y(lv1_itr) if lv1_itr != None else -30
        # (17) LV2 X
        state_merging[16] = rel_distance_x(lv2_itr) if lv2_itr != None else 400
        # (18) LV2 Y
        state_merging[17] = rel_distance_y(lv2_itr) if lv2_itr != None else -30
        # (19) LV3 X
        state_merging[18] = rel_distance_x(lv3_itr) if lv3_itr != None else 400
        # (20) LV3 Y
        state_merging[19] = rel_distance_y(lv3_itr) if lv3_itr != None else -30
        # (21) Lane width
        state_merging[20] = lane_width
        # (22) Right lane boundry type
        state_merging[21] = lane_width#right_lane_type
        # (23) Left lane boundry type
        state_merging[22] = lane_width#left_lane_type

        return state_merging, output_state, tv_lane_ind, 
    
    def update_dirs(self):
        '''
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
        '''
        self.LC_image_dataset_dir = self.LC_image_dataset_rdir
        if not os.path.exists(self.LC_image_dataset_dir):
            os.makedirs(self.LC_image_dataset_dir)
        
        
    