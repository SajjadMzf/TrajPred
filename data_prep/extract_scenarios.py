import os
import numpy as np
import pickle

import read_csv as rc
import param as p

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
        dataset_name:'Dataset Name'):

        self.obs_frames = p.OBS_LEN
        self.pred_frames = p.PRED_LEN
        self.seq_len = p.SEQ_LEN

        self.metas = rc.read_meta_info(meta_path)
        self.fr_div = self.metas[rc.FRAME_RATE]/p.FPS
        self.track_path = track_path

        self.file_num = file_num
        
        
        self.LC_states_dir = "../../Dataset/"+ dataset_name +"/Scenarios"+ p.dir_ext
        if not os.path.exists(self.LC_states_dir):
            os.makedirs(self.LC_states_dir)
        self.scenarios = []
        self.data_tracks, _ = rc.read_track_csv(track_path, track_pickle_path, group_by = 'tracks', reload = False, fr_div = self.fr_div)
        self.data_frames, _ = rc.read_track_csv(track_path, frame_pickle_path, group_by = 'frames', reload = False, fr_div = self.fr_div)
        
        self.statics = rc.read_static_info(static_path)
        

    def extract_and_save(self): 
        
        lc_scenarios, lk_count, rlc_count, llc_count = self.get_lc_scenarios()
        self.scenarios.extend(lc_scenarios)
        lk_scenarios, total_lk, lk_with_ttlc = self.get_lk_scenarios(lk_count)
        
        if p.UNBALANCED == False: # Check if there are enough LK scenarios for a balanced dataset
            assert(lk_count ==len(lk_scenarios))
        
        self.scenarios.extend(lk_scenarios)
        file_dir = os.path.join(self.LC_states_dir, str(self.file_num).zfill(2)+ '.pickle')
        print("File Number; {}, ALL extracted samples: {}, RLC: {}, LLC: {}, Balance LK: {}, ALL LK: {}, LK with TTLC:{}".format(self.file_num,rlc_count + llc_count +lk_count, rlc_count, llc_count, lk_count, total_lk, lk_with_ttlc))
        with open(file_dir, 'wb') as handle:
            pickle.dump(self.scenarios, handle, protocol= pickle.HIGHEST_PROTOCOL)
        

    def get_lc_scenarios(
        self, 
        )-> 'List of dicts containing LC scenarios and required number of LK scenarios (undersampling LK scenarios)':
        
        scenarios = []
        for tv_idx, tv_data in enumerate(self.data_tracks):
            driving_dir = self.statics[tv_idx+1][rc.DRIVING_DIRECTION] # statics is 1-based array
            tv_id = tv_data[rc.TRACK_ID] 
            lc_last_idxs, labels, _ = self.get_last_idxs(tv_data, driving_dir, scenario_type = 'lc')
            
            for scenario_idx, tv_last_idx in enumerate(lc_last_idxs): # These lane_idxes are for tv_data   
                tv_first_idx = tv_last_idx - self.seq_len
                if self.check_scenario_validity(tv_first_idx, tv_last_idx, tv_data) == False:
                    continue
                grid_data = np.zeros((self.seq_len, 3*13))
                for fr in range(tv_first_idx, tv_last_idx):
                    frame = tv_data[rc.FRAME][fr]
                    frame_data = self.data_frames[int(frame/self.fr_div -1)]
                    grid_data[fr- tv_first_idx] = self.get_grid_data(tv_id, frame_data, driving_dir)

                scenario = {
                        'tv':tv_id,
                        'ttlc_available':True,
                        'frames':tv_data[rc.FRAME][tv_first_idx:tv_last_idx], 
                        'grid': grid_data, 
                        'label':labels[scenario_idx], 
                        'driving_dir':driving_dir,
                        'svs': self.get_svs(tv_data, tv_first_idx, tv_last_idx),
                        'images': None,
                        'states_wirth': None,
                        'states_shou': None,
                        'states_ours': None,
                        }
                scenarios.append(scenario)
        
        lc_labels = [ lc_scenario['label'] for lc_scenario in scenarios]
        rlc_count = lc_labels.count(1)
        llc_count = lc_labels.count(2)
        lk_count = int((rlc_count+llc_count)/2)
        return scenarios, lk_count, rlc_count, llc_count

    def get_lk_scenarios(
        self, 
        lk_count:'a limit on number of LK scenarios'
        )-> 'List of dicts containing LC scenarios for the specific ev_poistion':

        scenarios = []
        known_ttlc_count = 0
        # We didnt shuffle tvs because there is no difference between tv number 1 and number x in terms of being random
        for tv_idx, tv_data in enumerate(self.data_tracks):
            driving_dir = self.statics[tv_idx+1][rc.DRIVING_DIRECTION]  # statics is 1-based array
            tv_id = tv_data[rc.TRACK_ID] 
            tv_last_idx, label, known_ttlc = self.get_last_idxs(tv_data, driving_dir, scenario_type = 'lk')
               
            tv_first_idx = tv_last_idx - self.seq_len

            if self.check_scenario_validity(tv_first_idx, tv_last_idx, tv_data) == False:
                    continue
            if known_ttlc:
                known_ttlc_count += 1
            grid_data = np.zeros((self.seq_len, 3*13))
            for fr in range(tv_first_idx, tv_last_idx):
                frame = tv_data[rc.FRAME][fr]
                frame_data = self.data_frames[int(frame/self.fr_div -1)]
                grid_data[fr - tv_first_idx] = self.get_grid_data(tv_id, frame_data, driving_dir)

            scenario = {
                        'tv':tv_id,
                        'ttlc_available': known_ttlc,
                        'frames':tv_data[rc.FRAME][tv_first_idx:tv_last_idx],
                        'grid': grid_data,  
                        'label':label, 
                        'driving_dir':driving_dir,
                        'svs': self.get_svs(tv_data, tv_first_idx, tv_last_idx),
                        'images': None,
                        'states_wirth': None,
                        'states_shou': None,
                        'states_ours': None,
                        }
            scenarios.append(scenario)

        selected_scenarios = [scenario for scenario in scenarios if scenario['ttlc_available']==True]
        unselected_scenarios = [scenario for scenario in scenarios if scenario['ttlc_available']==False]
        remaining_lk_count = lk_count - len(selected_scenarios)
        if p.UNBALANCED:
            selected_scenarios.extend(unselected_scenarios)
        elif remaining_lk_count>0:
            selected_scenarios.extend(unselected_scenarios[:remaining_lk_count])
        else:
            selected_scenarios = selected_scenarios[:lk_count]
        if  len(scenarios) < lk_count:
            print('Warning: There is not sufficient LK scenarios in ', self.track_path )
        return selected_scenarios, len(scenarios), known_ttlc_count
    
    
    def get_last_idxs(
        self, 
        tv_data:'Array containing track data of the TV', 
        driving_dir:'Driving direction of the TV', 
        scenario_type:'LC or LK'
        )-> '(List of) last frame index(es) of the scenario(s)  and the labels for the scenario(s)' :
        
        tv_lane = tv_data[rc.LANE_ID]
        tv_lane_list = list(set(tv_lane))

        if scenario_type == 'lc':
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
        
        elif scenario_type == 'lk':
            labels = 0
            known_ttlc = False
            if len(tv_lane_list) > 1:
                start_lane = tv_lane[0]
                last_idxs = np.nonzero(tv_lane!= start_lane)[0][0]
                last_idxs -= self.pred_frames
                known_ttlc = True
            else:
                last_idxs = len(tv_data[rc.FRAME])
                last_idxs -= self.pred_frames # We set the last index of LK scenario pred_frames before the end of recording, in case TV perform LC right after recording ends
            return last_idxs, labels, known_ttlc
        else:
            raise('Unexpected Scenario Type!')

    def check_scenario_validity(
        self, 
        tv_first_idx:'First frame index of the scenario', 
        tv_last_idx:'Last frame index of the scenario', 
        tv_data:'Array containing track data of the TV'
        )->'True if valid, False if not':
        # 1. There should be enough frames in the selected scenario.
        if tv_first_idx < 0:
            return False 

        # 2. TV should not change its lane during input seq period
        tv_lane = tv_data[rc.LANE_ID][tv_first_idx:tv_last_idx]
        tv_lane_list = list(set(tv_lane))
        if len(tv_lane_list)>1:
            return False
        return True
    
    def get_svs(
        self, 
        tv_data:'Array containing track data of the TV', 
        tv_first_idx:'First frame index of the scenario', 
        tv_last_idx:'Last frame index of the scenario'
        )-> 'Dict containing each svs id and visibility status (None for now)':
        tv_x = tv_data[rc.X][tv_first_idx:tv_last_idx]

        pv_id = tv_data[rc.PRECEDING_ID][tv_first_idx:tv_last_idx]
        fv_id = tv_data[rc.FOLLOWING_ID][tv_first_idx:tv_last_idx]
        rv_id = np.zeros_like(pv_id)
        rfv_id = np.zeros_like(pv_id)
        rpv_id = np.zeros_like(pv_id)
        lv_id = np.zeros_like(pv_id)
        lfv_id = np.zeros_like(pv_id)
        lpv_id = np.zeros_like(pv_id)
        frames = tv_data[rc.FRAME][tv_first_idx:tv_last_idx] 

        for itr, tv_idx  in enumerate(range(tv_first_idx,tv_last_idx)):
            if tv_data[rc.RIGHT_ALONGSIDE_ID][tv_idx] == 0:
                rv_cand1_id = int(tv_data[rc.RIGHT_PRECEDING_ID][tv_idx])
                rv_cand2_id = int(tv_data[rc.RIGHT_FOLLOWING_ID][tv_idx])
                rv_cand1_data = self.data_tracks[rv_cand1_id -1]
                rv_cand2_data = self.data_tracks[rv_cand2_id -1]
                if rv_cand1_id == rv_cand2_id == 0:
                    continue
                elif rv_cand1_id == 0:
                    rv_id[itr] = rv_cand2_id
                    rfv_id[itr] = rv_cand2_data[rc.FOLLOWING_ID][rv_cand2_data[rc.FRAME]==frames[itr]]
                elif rv_cand2_id == 0:
                    rv_id[itr] = rv_cand1_id
                    rpv_id[itr] = rv_cand1_data[rc.PRECEDING_ID][rv_cand1_data[rc.FRAME]==frames[itr]]
                else:
                    rv_cand1_x = rv_cand1_data[rc.X][rv_cand1_data[rc.FRAME]==frames[itr]][0]
                    rv_cand2_x = rv_cand2_data[rc.X][rv_cand2_data[rc.FRAME]==frames[itr]][0]
                    if np.absolute(rv_cand1_x-tv_x[itr]) < np.absolute(rv_cand2_x-tv_x[itr]):
                        rv_id[itr] = rv_cand1_id
                        rpv_id[itr] = rv_cand1_data[rc.PRECEDING_ID][rv_cand1_data[rc.FRAME]==frames[itr]]
                        rfv_id[itr] = rv_cand1_data[rc.FOLLOWING_ID][rv_cand1_data[rc.FRAME]==frames[itr]]
                    else:
                        rv_id[itr] = rv_cand2_id
                        rpv_id[itr] = rv_cand2_data[rc.PRECEDING_ID][rv_cand2_data[rc.FRAME]==frames[itr]]
                        rfv_id[itr] = rv_cand2_data[rc.FOLLOWING_ID][rv_cand2_data[rc.FRAME]==frames[itr]]

            else:
                rv_id[itr] = tv_data[rc.RIGHT_ALONGSIDE_ID][tv_idx]
                rpv_id[itr] = tv_data[rc.RIGHT_PRECEDING_ID][tv_idx]
                rfv_id[itr] = tv_data[rc.RIGHT_FOLLOWING_ID][tv_idx]

            if tv_data[rc.LEFT_ALONGSIDE_ID][tv_idx] == 0:
                lv_cand1_id = int(tv_data[rc.LEFT_PRECEDING_ID][tv_idx])
                lv_cand2_id = int(tv_data[rc.LEFT_FOLLOWING_ID][tv_idx])
                lv_cand1_data = self.data_tracks[lv_cand1_id -1]
                lv_cand2_data = self.data_tracks[lv_cand2_id -1]
                if lv_cand1_id == lv_cand2_id == 0:
                    continue
                elif lv_cand1_id == 0:
                    lv_id[itr] = lv_cand2_id
                    lfv_id[itr] = lv_cand2_data[rc.FOLLOWING_ID][lv_cand2_data[rc.FRAME]==frames[itr]]
                elif lv_cand2_id == 0:
                    lv_id[itr] = lv_cand1_id
                    lpv_id[itr] = lv_cand1_data[rc.PRECEDING_ID][lv_cand1_data[rc.FRAME]==frames[itr]]
                else:
                    lv_cand1_x = lv_cand1_data[rc.X][lv_cand1_data[rc.FRAME]==frames[itr]][0]
                    lv_cand2_x = lv_cand2_data[rc.X][lv_cand2_data[rc.FRAME]==frames[itr]][0]
                    if np.absolute(lv_cand1_x-tv_x[itr]) < np.absolute(lv_cand2_x-tv_x[itr]):
                        lv_id[itr] = lv_cand1_id
                        lpv_id[itr] = lv_cand1_data[rc.PRECEDING_ID][lv_cand1_data[rc.FRAME]==frames[itr]]
                        lfv_id[itr] = lv_cand1_data[rc.FOLLOWING_ID][lv_cand1_data[rc.FRAME]==frames[itr]]
                    else:
                        lv_id[itr] = lv_cand2_id
                        lpv_id[itr] = lv_cand2_data[rc.PRECEDING_ID][lv_cand2_data[rc.FRAME]==frames[itr]]
                        lfv_id[itr] = lv_cand2_data[rc.FOLLOWING_ID][lv_cand2_data[rc.FRAME]==frames[itr]]

            else:
                lv_id[itr] = tv_data[rc.LEFT_ALONGSIDE_ID][tv_idx]
                lpv_id[itr] = tv_data[rc.LEFT_PRECEDING_ID][tv_idx]
                lfv_id[itr] = tv_data[rc.LEFT_FOLLOWING_ID][tv_idx]

        return {'id':np.stack([pv_id, fv_id, rv_id, rpv_id, rfv_id, lv_id, lpv_id, lfv_id], axis = 0), 'vis':None}
    
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
    