import numpy as np
import pandas as pd
from scipy.io import savemat

def export_results(scenarios):
    in_seq_len = scenarios['frames'][0][0].shape[0]
    file_tv_pairs = []
    sorted_scenarios_dict = []
    for batch_grp in range(len(scenarios['tv'])):
        for batch_itr, tv_id in enumerate(scenarios['tv'][batch_grp]):
            data_file = int(scenarios['data_file'][batch_grp][batch_itr].split('.')[0])
            if (data_file,tv_id) not in file_tv_pairs:
                file_tv_pairs.append((data_file,tv_id))
                sorted_scenarios_dict.append({'tv': tv_id,
                                            'data_file':data_file,
                                            'mode_prob':[],  
                                            'traj_pred':[], 
                                            'frames':[],
                                            'times': []
                                            })
            
            
            sorted_index = file_tv_pairs.index(((data_file,tv_id)))
            sorted_scenarios_dict[sorted_index]['times'].append(scenarios['frames'][batch_grp][batch_itr][in_seq_len-1])# time is frame number at the end of obs
            
            sorted_scenarios_dict[sorted_index]['mode_prob'].append(scenarios['mode_prob'][batch_grp][batch_itr])
            sorted_scenarios_dict[sorted_index]['traj_pred'].append(scenarios['traj_pred'][batch_grp][batch_itr])
            sorted_scenarios_dict[sorted_index]['frames'].append(scenarios['frames'][batch_grp][batch_itr])
            
            
    # sort frames order in each sorted scenarios
    total_times= 0
    for i in range(len(sorted_scenarios_dict)):
        times_array = np.array(sorted_scenarios_dict[i]['times'])
        total_times += len(times_array)
        sorted_indxs = np.argsort(times_array).astype(int)
        sorted_scenarios_dict[i]['times'] = [sorted_scenarios_dict[i]['times'][indx] for indx in sorted_indxs]
        sorted_scenarios_dict[i]['mode_prob'] = [sorted_scenarios_dict[i]['mode_prob'][indx] for indx in sorted_indxs]
        sorted_scenarios_dict[i]['traj_preds'] = [sorted_scenarios_dict[i]['traj_pred'][indx] for indx in sorted_indxs]
        sorted_scenarios_dict[i]['frames'] = [sorted_scenarios_dict[i]['frames'][indx] for indx in sorted_indxs]
    
    results_array = np.empty((total_times,10), dtype = object)
    start_index = 0
    for i, scenario in enumerate(sorted_scenarios_dict):
        scenario_len = len(scenario['times'])
        for j in range(start_index, start_index+scenario_len):
            
            scene_index = j-start_index
            mode_prob = scenario['mode_prob'][scene_index]
            sorted_modes = np.argsort(mode_prob)[::-1]
            
            results_array[j,0] = scenario['data_file']
            results_array[j,1] = [scenario['tv']]
            results_array[j,2] = [scenario['times'][scene_index]]
            results_array[j,3] = scenario['mode_prob'][scene_index][sorted_modes[:3]].tolist()
            results_array[j,4] = scenario['traj_pred'][scene_index][sorted_modes[0],:,1].tolist()
            results_array[j,5] = scenario['traj_pred'][scene_index][sorted_modes[0],:,0].tolist()
            results_array[j,6] = scenario['traj_pred'][scene_index][sorted_modes[1],:,1].tolist()
            results_array[j,7] = scenario['traj_pred'][scene_index][sorted_modes[1],:,0].tolist()
            results_array[j,8] = scenario['traj_pred'][scene_index][sorted_modes[2],:,1].tolist()
            results_array[j,9] = scenario['traj_pred'][scene_index][sorted_modes[2],:,0].tolist()
        start_index += scenario_len
    
    files = np.unique(results_array[:,0])
    for file_i in files:
        print('Export file: {}'.format(file_i))
        cur_array = results_array[results_array[:,0]==file_i]
        savemat('Prediction_{}.mat'.format(file_i), {'predictions':cur_array})  


def export_results_SM(scenarios):
    in_seq_len = scenarios['frames'][0][0].shape[0]
    file_tv_pairs = []
    sorted_scenarios_dict = []
    for batch_grp in range(len(scenarios['tv'])):
        for batch_itr, tv_id in enumerate(scenarios['tv'][batch_grp]):
            data_file = int(scenarios['data_file'][batch_grp][batch_itr].split('.')[0])
            if (data_file,tv_id) not in file_tv_pairs:
                file_tv_pairs.append((data_file,tv_id))
                sorted_scenarios_dict.append({'tv': tv_id,
                                            'data_file':data_file,
                                            'traj_pred':[], 
                                            'frames':[],
                                            'times': []
                                            })
            
            
            sorted_index = file_tv_pairs.index(((data_file,tv_id)))
            sorted_scenarios_dict[sorted_index]['times'].append(scenarios['frames'][batch_grp][batch_itr][in_seq_len-1])# time is frame number at the end of obs
            
            sorted_scenarios_dict[sorted_index]['traj_pred'].append(scenarios['traj_pred'][batch_grp][batch_itr])
            sorted_scenarios_dict[sorted_index]['frames'].append(scenarios['frames'][batch_grp][batch_itr])
            
            
    # sort frames order in each sorted scenarios
    total_times= 0
    for i in range(len(sorted_scenarios_dict)):
        times_array = np.array(sorted_scenarios_dict[i]['times'])
        total_times += len(times_array)
        sorted_indxs = np.argsort(times_array).astype(int)
        sorted_scenarios_dict[i]['times'] = [sorted_scenarios_dict[i]['times'][indx] for indx in sorted_indxs]
        sorted_scenarios_dict[i]['traj_preds'] = [sorted_scenarios_dict[i]['traj_pred'][indx] for indx in sorted_indxs]
        sorted_scenarios_dict[i]['frames'] = [sorted_scenarios_dict[i]['frames'][indx] for indx in sorted_indxs]
    
    results_array = np.empty((total_times,5), dtype = object)
    start_index = 0
    for i, scenario in enumerate(sorted_scenarios_dict):
        scenario_len = len(scenario['times'])
        for j in range(start_index, start_index+scenario_len):
            
            scene_index = j-start_index
            
            results_array[j,0] = scenario['data_file']
            results_array[j,1] = [scenario['tv']]
            results_array[j,2] = [scenario['times'][scene_index]]
            results_array[j,3] = scenario['traj_pred'][scene_index][:,1].tolist()
            results_array[j,4] = scenario['traj_pred'][scene_index][:,0].tolist()
        start_index += scenario_len
    
    files = np.unique(results_array[:,0])
    for file_i in files:
        print('Export file: {}'.format(file_i))
        cur_array = results_array[results_array[:,0]==file_i]
        savemat('Prediction_{}.mat'.format(file_i), {'predictions':cur_array})  