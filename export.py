import os
import numpy as np
import pandas as pd
from scipy.io import savemat

PREDICTION_DIR = "../../Dataset/Prediction_exid"
def export_results(scenarios, eval_type = 'De'):
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
                                            'traj_gt': [],
                                            'frames':[],
                                            'times': []
                                            })            
            
            sorted_index = file_tv_pairs.index(((data_file,tv_id)))
            sorted_scenarios_dict[sorted_index]['times'].append(scenarios['frames'][batch_grp][batch_itr][in_seq_len-1])# time is frame number at the end of obs
            sorted_scenarios_dict[sorted_index]['mode_prob'].append(scenarios['mode_prob'][batch_grp][batch_itr])
            sorted_scenarios_dict[sorted_index]['traj_pred'].append(scenarios['traj_pred'][batch_grp][batch_itr])
            sorted_scenarios_dict[sorted_index]['traj_gt'].append(scenarios['traj_gt'][batch_grp][batch_itr])
            sorted_scenarios_dict[sorted_index]['frames'].append(scenarios['frames'][batch_grp][batch_itr])
            
    # sort frames order in each sorted scenarios
    total_times= 0
    for i in range(len(sorted_scenarios_dict)):
        times_array = np.array(sorted_scenarios_dict[i]['times'])
        total_times += len(times_array)
        sorted_indxs = np.argsort(times_array).astype(int)
        sorted_scenarios_dict[i]['times'] = [sorted_scenarios_dict[i]['times'][indx] for indx in sorted_indxs]
        sorted_scenarios_dict[i]['mode_prob'] = [sorted_scenarios_dict[i]['mode_prob'][indx] for indx in sorted_indxs]
        sorted_scenarios_dict[i]['traj_pred'] = [sorted_scenarios_dict[i]['traj_pred'][indx] for indx in sorted_indxs]
        sorted_scenarios_dict[i]['traj_gt'] = [sorted_scenarios_dict[i]['traj_gt'][indx] for indx in sorted_indxs]
        sorted_scenarios_dict[i]['frames'] = [sorted_scenarios_dict[i]['frames'][indx] for indx in sorted_indxs]
    
    results_mat = np.empty((total_times,10), dtype = object)
    results_csv = np.empty((total_times+1,10), dtype = object)
    start_index = 0
    for i, scenario in enumerate(sorted_scenarios_dict):
        scenario_len = len(scenario['times'])
        for j in range(start_index, start_index+scenario_len):
            scene_index = j-start_index
            mode_prob = scenario['mode_prob'][scene_index]
            sorted_modes = np.argsort(mode_prob)[::-1]
            results_mat[j,0] = scenario['data_file']
            results_mat[j,1] = [scenario['tv']]
            results_mat[j,2] = [scenario['times'][scene_index]]
            results_mat[j,3] = scenario['mode_prob'][scene_index][sorted_modes[:3]].tolist()
            results_mat[j,4] = scenario['traj_pred'][scene_index][sorted_modes[0],:,1].tolist()
            results_mat[j,5] = scenario['traj_pred'][scene_index][sorted_modes[0],:,0].tolist()
            results_mat[j,6] = scenario['traj_pred'][scene_index][sorted_modes[1],:,1].tolist()
            results_mat[j,7] = scenario['traj_pred'][scene_index][sorted_modes[1],:,0].tolist()
            results_mat[j,8] = scenario['traj_pred'][scene_index][sorted_modes[2],:,1].tolist()
            results_mat[j,9] = scenario['traj_pred'][scene_index][sorted_modes[2],:,0].tolist()

            results_csv[j+1,0] = str(scenario['data_file'])
            results_csv[j+1,1] = str(scenario['tv'])
            results_csv[j+1,2] = str(scenario['times'][scene_index])
            results_csv[j+1,3] = ';'.join(map(str, scenario['mode_prob'][scene_index][sorted_modes[:3]].tolist()))
            results_csv[j+1,4] = ';'.join(map(str, scenario['traj_pred'][scene_index][sorted_modes[0],:,1].tolist()))
            results_csv[j+1,5] = ';'.join(map(str, scenario['traj_pred'][scene_index][sorted_modes[0],:,0].tolist()))
            results_csv[j+1,6] = ';'.join(map(str, scenario['traj_pred'][scene_index][sorted_modes[1],:,1].tolist()))
            results_csv[j+1,7] = ';'.join(map(str, scenario['traj_pred'][scene_index][sorted_modes[1],:,0].tolist()))
            results_csv[j+1,8] = ';'.join(map(str, scenario['traj_pred'][scene_index][sorted_modes[2],:,1].tolist()))
            results_csv[j+1,9] = ';'.join(map(str, scenario['traj_pred'][scene_index][sorted_modes[2],:,0].tolist()))


        start_index += scenario_len
    
    columns = ["file","id", "frame", "mode_prob", "pr1X", "pr1Y", "pr2X", "pr2Y","pr3X", "pr3Y"]
    for i, column in enumerate(columns):
        results_csv[0,i] = column
    file_dir = os.path.join(PREDICTION_DIR, 'predictionMM')
    np.savetxt("{}_{}.csv".format(file_dir, eval_type), results_csv, delimiter=",", fmt="%s")
    files = np.unique(results_mat[:,0])
    for file_i in files:
        file_dir = os.path.join(PREDICTION_DIR, 'predictionMM_{}'.format(file_i))
        print('Export file: {}'.format(file_i))
        cur_mat = results_mat[results_mat[:,0]==file_i]
        savemat('{}_{}.mat'.format(file_dir, eval_type),{'prediction':cur_mat})
          
def list2str(x):
    print(x)
    return ''.join(str(x))

def export_results_SM(scenarios, eval_type):
    in_seq_len = 20#scenarios['frames'][0][0].shape[0]
    file_tv_pairs = []
    sorted_scenarios_dict = []
    for batch_grp in range(len(scenarios['tv'])):
        for batch_itr, tv_id in enumerate(scenarios['tv'][batch_grp]):
            data_file = int(scenarios['data_file'][batch_grp][batch_itr].split('.')[0])
            if (data_file,tv_id) not in file_tv_pairs:
                file_tv_pairs.append((data_file,tv_id))
                # one dict per TV (i.e. scenario)
                sorted_scenarios_dict.append({'tv': tv_id,
                                            'data_file':data_file,
                                            'traj_pred':[], 
                                            'traj_gt': [],
                                            'frames':[],
                                            'times': []
                                            })
            
            
            sorted_index = file_tv_pairs.index(((data_file,tv_id)))
            sorted_scenarios_dict[sorted_index]['times'].append(scenarios['frames'][batch_grp][batch_itr][in_seq_len-1])# time is frame number at the end of obs
            
            sorted_scenarios_dict[sorted_index]['traj_gt'].append(scenarios['traj_gt'][batch_grp][batch_itr])
            sorted_scenarios_dict[sorted_index]['traj_pred'].append(scenarios['traj_pred'][batch_grp][batch_itr])
            sorted_scenarios_dict[sorted_index]['frames'].append(scenarios['frames'][batch_grp][batch_itr])
            
            
    # sort frames order in each sorted scenarios
    total_times= 0
    for i in range(len(sorted_scenarios_dict)):
        times_array = np.array(sorted_scenarios_dict[i]['times'])
        total_times += len(times_array)
        sorted_indxs = np.argsort(times_array).astype(int)
        sorted_scenarios_dict[i]['times'] = [sorted_scenarios_dict[i]['times'][indx] for indx in sorted_indxs]
        sorted_scenarios_dict[i]['traj_pred'] = [sorted_scenarios_dict[i]['traj_pred'][indx] for indx in sorted_indxs]
        sorted_scenarios_dict[i]['traj_gt'] = [sorted_scenarios_dict[i]['traj_gt'][indx] for indx in sorted_indxs]
        sorted_scenarios_dict[i]['frames'] = [sorted_scenarios_dict[i]['frames'][indx] for indx in sorted_indxs]
    
    results_mat = np.empty((total_times,5), dtype = object)
    results_csv = np.empty((total_times+1,7), dtype = object)
    start_index = 0
    for i, scenario in enumerate(sorted_scenarios_dict):
        scenario_len = len(scenario['times'])
        for j in range(start_index, start_index+scenario_len):
            
            scene_index = j-start_index
            
            results_mat[j,0] = scenario['data_file']
            results_mat[j,1] = [scenario['tv']]
            results_mat[j,2] = [scenario['times'][scene_index]]
            results_mat[j,3] = scenario['traj_pred'][scene_index][:,1].tolist()
            results_mat[j,4] = scenario['traj_pred'][scene_index][:,0].tolist()
           
            results_csv[j+1,0] = str(scenario['data_file'])
            results_csv[j+1,1] = str(scenario['tv'])
            results_csv[j+1,2] = str(scenario['times'][scene_index])
            results_csv[j+1,3] = ';'.join(map(str, scenario['traj_pred'][scene_index][:,1].tolist()))
            results_csv[j+1,4] = ';'.join(map(str, scenario['traj_pred'][scene_index][:,0].tolist()))
            results_csv[j+1,5] = ';'.join(map(str, scenario['traj_gt'][scene_index][:,1].tolist()))
            results_csv[j+1,6] = ';'.join(map(str, scenario['traj_gt'][scene_index][:,0].tolist()))
            

        start_index += scenario_len
    
    columns = ["file","id", "frame", "pr1X", "pr1Y"]
    columns_csv = ["file","id", "frame", "pr1X", "pr1Y", 'gtX', 'gtY']
    
    for i, column in enumerate(columns_csv):
        results_csv[0,i] = column
    file_dir = os.path.join(PREDICTION_DIR, 'predictionSM')
    np.savetxt("{}_{}.csv".format(file_dir, eval_type), results_csv, delimiter=",", fmt="%s")

    files = np.unique(results_mat[:,0])
    for file_i in files:
        file_dir = os.path.join(PREDICTION_DIR, 'predictionSM_{}'.format(file_i))
        print('Export file: {}'.format(file_i))
        cur_mat = results_mat[results_mat[:,0]==file_i]
        savemat('{}_{}.mat'.format(file_dir, eval_type),{'prediction':cur_mat})
          