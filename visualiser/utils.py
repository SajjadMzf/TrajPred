import os
import numpy as np
import pickle
import pdb

import param as p



'''
'data_file': data_file,
'tv': tv_id.numpy(),
'frames': frames.numpy(),
'traj_min': dataset.output_states_min,
'traj_max': dataset.output_states_max,  
'input_features': feature_data.cpu().data.numpy(),
'traj_gt': traj_gt.cpu().data.numpy(),
'traj_track_gt': traj_track_gt.cpu().data.numpy(),
'traj_dist_preds': data_dist_preds.cpu().data.numpy(),
'man_gt': man_gt.cpu().data.numpy(),
'man_preds': man_vectors.cpu().data.numpy(),
'time_bar_gt': time_bar_gt.cpu().data.numpy(),
'time_bar_preds': time_bar_pred.cpu().data.numpy(),
'mode_prob': mode_prob.detach().cpu().data.numpy(),
'''


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def read_scenarios(result_file, force_resort = False):
    with open(result_file, 'rb') as handle:
            scenarios = pickle.load(handle)
    
    
    
    sorted_scenarios_path = result_file.split('.pickle')[0] + '_sorted' + '.pickle'
    if os.path.exists(sorted_scenarios_path) and force_resort == False:
        print('Loading Sorted Scenarios: {}'.format(sorted_scenarios_path))
        with open(sorted_scenarios_path, 'rb') as handle:
            sorted_scenarios_dict = pickle.load(handle)
        in_seq_len = sorted_scenarios_dict[0]['input_features'][0].shape[0]
        tgt_seq_len = sorted_scenarios_dict[0]['man_preds'][0].shape[0]
        file_tv_pairs = []
        for sorted_scenario in sorted_scenarios_dict:
            file_tv_pairs.append((sorted_scenario['data_file'],sorted_scenario['tv']))
    else:
        print('Sorting Scenarios...')
        sorted_scenarios_dict = []
        file_tv_pairs = []
        in_seq_len = scenarios['input_features'][0][0].shape[0]
        tgt_seq_len = scenarios['traj_dist_preds'][0][0].shape[-2]
        for batch_grp in range(len(scenarios['tv'])):
            for batch_itr, tv_id in enumerate(scenarios['tv'][batch_grp]):
                data_file = int(scenarios['data_file'][batch_grp][batch_itr].split('.')[0])
                if (data_file,tv_id) not in file_tv_pairs:
                    file_tv_pairs.append((data_file,tv_id))
                    sorted_scenarios_dict.append({'tv': tv_id,
                                                'data_file':data_file,
                                                'traj_min': scenarios['traj_min'][batch_grp],# Assuming traj min and max are the same for all scenarios
                                                'traj_max':scenarios['traj_max'][batch_grp],
                                                'input_features': [],
                                                'times':[], 
                                                'man_labels':[], 
                                                'man_preds':[],
                                                'mode_prob':[],  
                                                'traj_labels':[], 
                                                'traj_dist_preds':[], 
                                                'frames':[],
                                                })
                
                
                sorted_index = file_tv_pairs.index(((data_file,tv_id)))
                #if data_file==44 and tv_id == 290:
                #    pdb.set_trace()
                #if np.any(scenarios['man_gt'][batch_grp][batch_itr]==2):
                #    pdb.set_trace()
                sorted_scenarios_dict[sorted_index]['times'].append(scenarios['frames'][batch_grp][batch_itr][in_seq_len])# time is frame number at the end of obs
                if p.PLOT_MAN:
                    sorted_scenarios_dict[sorted_index]['man_labels'].append(scenarios['man_gt'][batch_grp][batch_itr]) 
                    sorted_scenarios_dict[sorted_index]['man_preds'].append(scenarios['man_preds'][batch_grp][batch_itr])
                sorted_scenarios_dict[sorted_index]['mode_prob'].append(scenarios['mode_prob'][batch_grp][batch_itr])
                sorted_scenarios_dict[sorted_index]['traj_labels'].append(scenarios['traj_track_gt'][batch_grp][batch_itr])
                sorted_scenarios_dict[sorted_index]['traj_dist_preds'].append(scenarios['traj_dist_preds'][batch_grp][batch_itr])
                sorted_scenarios_dict[sorted_index]['frames'].append(scenarios['frames'][batch_grp][batch_itr])
                sorted_scenarios_dict[sorted_index]['input_features'].append(scenarios['input_features'][batch_grp][batch_itr])
                
                
        # sort frames order in each sorted scenarios
        for i in range(len(sorted_scenarios_dict)):
            times_array = np.array(sorted_scenarios_dict[i]['times'])
            sorted_indxs = np.argsort(times_array).astype(int)
            sorted_scenarios_dict[i]['times'] = [sorted_scenarios_dict[i]['times'][indx] for indx in sorted_indxs]
            if p.PLOT_MAN:
                sorted_scenarios_dict[i]['man_labels'] = [sorted_scenarios_dict[i]['man_labels'][indx] for indx in sorted_indxs]
                sorted_scenarios_dict[i]['man_preds'] = [sorted_scenarios_dict[i]['man_preds'][indx] for indx in sorted_indxs]
            sorted_scenarios_dict[i]['mode_prob'] = [sorted_scenarios_dict[i]['mode_prob'][indx] for indx in sorted_indxs]
            sorted_scenarios_dict[i]['traj_labels'] = [sorted_scenarios_dict[i]['traj_labels'][indx] for indx in sorted_indxs]
            sorted_scenarios_dict[i]['traj_dist_preds'] = [sorted_scenarios_dict[i]['traj_dist_preds'][indx] for indx in sorted_indxs]
            sorted_scenarios_dict[i]['frames'] = [sorted_scenarios_dict[i]['frames'][indx] for indx in sorted_indxs]
            sorted_scenarios_dict[i]['input_features'] = [sorted_scenarios_dict[i]['input_features'][indx] for indx in sorted_indxs]
            
       
        with open(sorted_scenarios_path, 'wb') as handle:
            print('saving: {}'.format(sorted_scenarios_path))
            pickle.dump(sorted_scenarios_dict, handle, protocol = pickle.HIGHEST_PROTOCOL)
            
    return sorted_scenarios_dict, file_tv_pairs, in_seq_len, tgt_seq_len