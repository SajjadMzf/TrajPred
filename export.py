import os
import numpy as np
import pandas as pd
from scipy.io import savemat
import sys
import pickle
from numpy.linalg import norm
import pdb

PREDICTION_DIR = "../../Dataset/Prediction_exid"
def export_results(export_file_name, scenarios, eval_type = 'De', export_cart = False):
    if export_cart:
        data_dir = '../../Dataset/exid/Tracks/39_tracks.csv'
        data_df = pd.read_csv(data_dir)
        data_df = data_df[['id','frame','s', 'd']]
        data_df = data_df.sort_values(by=['id', 'frame'])
        data = data_df.to_numpy()
        main_ref, merge_ref, merge_s_bias = get_reference_path()
                
    in_seq_len = 15#scenarios['frames'][0][0].shape[0]
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
    
    results_mat = np.empty((total_times,16), dtype = object)
    results_csv = np.empty((total_times+1,18), dtype = object)
    start_index = 0
    for i, scenario in enumerate(sorted_scenarios_dict):
        scenario_len = len(scenario['times'])
        print('Exporting Scenario {}/{}'.format(i, len(sorted_scenarios_dict)))
        for j in range(start_index, start_index+scenario_len):
            scene_index = j-start_index
            mode_prob = scenario['mode_prob'][scene_index]
            sorted_modes = np.argsort(mode_prob)[::-1]
            
            if export_cart:
                current_xy = get_xy(data, scenario['tv'], scenario['times'][scene_index]-5)
                traj_frenet0 = current_xy +\
                scenario['traj_pred'][scene_index][sorted_modes[0],:,-1::-1]
                traj_frenet0[:,1]*= -1
                traj_cart0 = frenet2cart(traj_frenet0,
                                          main_ref, merge_ref, merge_s_bias)
                traj_frenet1 = current_xy +\
                scenario['traj_pred'][scene_index][sorted_modes[1],:,-1::-1]
                traj_frenet1[:,1]*= -1
                traj_cart1 = frenet2cart(traj_frenet1, 
                                         main_ref, merge_ref, merge_s_bias)
                traj_frenet2 = current_xy +\
                scenario['traj_pred'][scene_index][sorted_modes[2],:,-1::-1]
                traj_frenet2[:,1]*= -1
                traj_cart2 = frenet2cart(traj_frenet2, 
                                         main_ref, merge_ref, merge_s_bias)
            else:
                traj_cart0 = np.zeros((25,2))
                traj_cart1 = np.zeros((25,2))
                traj_cart2 = np.zeros((25,2))
            
            results_mat[j,0] = scenario['data_file']
            results_mat[j,1] = [scenario['tv']]
            results_mat[j,2] = [scenario['times'][scene_index]-5]
            results_mat[j,3] = scenario['mode_prob'][scene_index][sorted_modes[:3]].tolist()
            results_mat[j,4] = scenario['traj_pred'][scene_index][sorted_modes[0],:,1].tolist()
            results_mat[j,5] = scenario['traj_pred'][scene_index][sorted_modes[0],:,0].tolist()
            results_mat[j,6] = scenario['traj_pred'][scene_index][sorted_modes[1],:,1].tolist()
            results_mat[j,7] = scenario['traj_pred'][scene_index][sorted_modes[1],:,0].tolist()
            results_mat[j,8] = scenario['traj_pred'][scene_index][sorted_modes[2],:,1].tolist()
            results_mat[j,9] = scenario['traj_pred'][scene_index][sorted_modes[2],:,0].tolist()
            results_mat[j,10] = traj_cart0[:,0].tolist()
            results_mat[j,11] = traj_cart0[:,1].tolist()
            results_mat[j,12] = traj_cart1[:,0].tolist()
            results_mat[j,13] = traj_cart1[:,1].tolist()
            results_mat[j,14] = traj_cart2[:,0].tolist()
            results_mat[j,15] = traj_cart2[:,1].tolist()

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
            results_csv[j+1,10] = ';'.join(map(str, traj_cart0[:,0].tolist()))
            results_csv[j+1,11] = ';'.join(map(str, traj_cart0[:,1].tolist()))
            results_csv[j+1,12] = ';'.join(map(str, traj_cart1[:,0].tolist()))
            results_csv[j+1,13] = ';'.join(map(str, traj_cart1[:,1].tolist()))
            results_csv[j+1,14] = ';'.join(map(str, traj_cart2[:,0].tolist()))
            results_csv[j+1,15] = ';'.join(map(str, traj_cart2[:,1].tolist()))
            results_csv[j+1,16] = ';'.join(map(str, scenario['traj_gt'][scene_index][:,1].tolist()))
            results_csv[j+1,17] = ';'.join(map(str, scenario['traj_gt'][scene_index][:,0].tolist()))


        start_index += scenario_len
    
    columns = ["file","id", "frame", "mode_prob", "pr1X", "pr1Y", "pr2X", "pr2Y","pr3X", "pr3Y"
               ,"Cpr1X", "Cpr1Y", "Cpr2X", "Cpr2Y","Cpr3X", "Cpr3Y", 'gtX', 'gtY']
    for i, column in enumerate(columns):
        results_csv[0,i] = column
    file_dir = os.path.join(PREDICTION_DIR, export_file_name)
    np.savetxt("{}_{}.csv".format(file_dir, eval_type), results_csv, delimiter=",", fmt="%s")
    files = np.unique(results_mat[:,0])
    for file_i in files:
        file_dir = os.path.join(PREDICTION_DIR, export_file_name + '_{}'.format(file_i))
        print('Export file: {}'.format(file_i))
        cur_mat = results_mat[results_mat[:,0]==file_i]
        savemat('{}_{}.mat'.format(file_dir, eval_type),{'prediction':cur_mat})
          
def list2str(x):
    print(x)
    return ''.join(str(x))



def get_xy(data, tv_id, frame):
    tv_data = data[np.logical_and(data[:,0]==tv_id, data[:,1]==frame)]
    return tv_data[:,2:4]
def export_results_SM(export_file_name, scenarios, eval_type, export_cart = False):
    if export_cart:
        data_dir = '../../Dataset/exid/Tracks/39_tracks.csv'
        data_df = pd.read_csv(data_dir)
        data_df = data_df[['id','frame','s', 'd']]
        data_df = data_df.sort_values(by=['id', 'frame'])
        data = data_df.to_numpy()
        main_ref, merge_ref, merge_s_bias = get_reference_path()
    in_seq_len = 15 # scenarios['frames'][0][0].shape[0]
    #print(in_seq_len)
    #exit()
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
            sorted_scenarios_dict[sorted_index]['times']\
                .append(scenarios['frames'][batch_grp][batch_itr][in_seq_len-1])# time is frame number at the end of obs
            
            sorted_scenarios_dict[sorted_index]['traj_gt']\
                .append(scenarios['traj_gt'][batch_grp][batch_itr])
            sorted_scenarios_dict[sorted_index]['traj_pred']\
                .append(scenarios['traj_pred'][batch_grp][batch_itr])
            sorted_scenarios_dict[sorted_index]['frames']\
                .append(scenarios['frames'][batch_grp][batch_itr])
            
            
    # sort frames order in each sorted scenarios
    total_times= 0
    for i in range(len(sorted_scenarios_dict)):
        times_array = np.array(sorted_scenarios_dict[i]['times'])
        total_times += len(times_array)
        sorted_indxs = np.argsort(times_array).astype(int)
        sorted_scenarios_dict[i]['times'] = \
            [sorted_scenarios_dict[i]['times'][indx] for indx in sorted_indxs]
        sorted_scenarios_dict[i]['traj_pred'] = \
            [sorted_scenarios_dict[i]['traj_pred'][indx] for indx in sorted_indxs]
        sorted_scenarios_dict[i]['traj_gt'] = \
            [sorted_scenarios_dict[i]['traj_gt'][indx] for indx in sorted_indxs]
        sorted_scenarios_dict[i]['frames'] = \
            [sorted_scenarios_dict[i]['frames'][indx] for indx in sorted_indxs]
    
    
    results_mat = np.empty((total_times,7), dtype = object)
    results_csv = np.empty((total_times+1,9), dtype = object)
    start_index = 0
    break_flag = False
    for i, scenario in enumerate(sorted_scenarios_dict):
        print('Exporting Scenario {}/{}'.format(i, len(sorted_scenarios_dict)))
        scenario_len = len(scenario['times'])
        for j in range(start_index, start_index+scenario_len):
            scene_index = j-start_index
            if export_cart:
                current_xy = get_xy(data, scenario['tv'], scenario['times'][scene_index]-5)
                traj_frenet = current_xy + \
                    scenario['traj_pred'][scene_index][:,-1::-1]
                traj_frenet[:,1] *= -1
                traj_cart = frenet2cart(traj_frenet, 
                                        main_ref, merge_ref, merge_s_bias)
            else:
                traj_cart = np.zeros((25,2))
            
            results_mat[j,0] = scenario['data_file']
            results_mat[j,1] = [scenario['tv']]
            results_mat[j,2] = [scenario['times'][scene_index]-5]
            results_mat[j,3] = scenario['traj_pred'][scene_index][:,1].tolist()
            results_mat[j,4] = scenario['traj_pred'][scene_index][:,0].tolist()
            results_mat[j,5] = traj_cart[:,0].tolist()
            results_mat[j,6] = traj_cart[:,1].tolist()

            results_csv[j+1,0] = str(scenario['data_file'])
            results_csv[j+1,1] = str(scenario['tv'])
            results_csv[j+1,2] = str(scenario['times'][scene_index])
            results_csv[j+1,3] = ';'.join(map(str, scenario['traj_pred'][scene_index][:,1].tolist()))
            results_csv[j+1,4] = ';'.join(map(str, scenario['traj_pred'][scene_index][:,0].tolist()))
            results_csv[j+1,5] = ';'.join(map(str, traj_cart[:,0].tolist()))
            results_csv[j+1,6] = ';'.join(map(str, traj_cart[:,1].tolist()))
            results_csv[j+1,7] = ';'.join(map(str, scenario['traj_gt'][scene_index][:,1].tolist()))
            results_csv[j+1,8] = ';'.join(map(str, scenario['traj_gt'][scene_index][:,0].tolist()))
            

        start_index += scenario_len
    
    columns = ["file","id", "frame", "pr1X", "pr1Y"]
    columns_csv = ["file","id", "frame", "pr1X", 
                   "pr1Y", "Cpr1X", "Cpr1Y", 'gtX', 'gtY']
    
    for i, column in enumerate(columns_csv):
        results_csv[0,i] = column
    file_dir = os.path.join(PREDICTION_DIR, export_file_name)
    np.savetxt("{}_{}.csv".format(file_dir, eval_type), results_csv, delimiter=",", fmt="%s")

    files = np.unique(results_mat[:,0])
    for file_i in files:
        file_dir = os.path.join(PREDICTION_DIR, export_file_name+ '_{}'.format(file_i))
        print('Export file: {}'.format(file_i))
        cur_mat = results_mat[results_mat[:,0]==file_i]
        savemat('{}_{}.mat'.format(file_dir, eval_type),{'prediction':cur_mat})
          

def get_reference_path():
    with open('../../Dataset/exid/Maps/39-52.pickle', 'rb') as handle:
        lm_dict = pickle.load(handle)
    main_ref = lm_dict['main_origin_lane']
    merge_ref = lm_dict['merge_origin_lane']
    ref_ext = np.ones((200,2))* 1.0
    ref_ext = np.cumsum(ref_ext, axis = 0)
    
    dref = main_ref[-1]-main_ref[-2]
    ref_ext[:,0] = main_ref[-1,0] + ref_ext[:,0]*dref[0]*100
    ref_ext[:,1] = main_ref[-1,1] + ref_ext[:,1]*dref[1]*100
    main_ref = np.concatenate((main_ref,ref_ext), axis = 0)
    
    
    dref = merge_ref[-1]-merge_ref[-2]
    ref_ext[:,0] = merge_ref[-1,0] + ref_ext[:,0]*dref[0]*100
    ref_ext[:,1] = merge_ref[-1,1] + ref_ext[:,1]*dref[1]*100
    merge_ref = np.concatenate((merge_ref,ref_ext), axis = 0)
    merge_s_bias = lm_dict['merge2main_s_bias']
    return main_ref, merge_ref, merge_s_bias

def frenet2cart(traj, main_ref, merge_ref, merge_s_bias):
    if traj[0,1]>0:
        ref = merge_ref
        traj[:,0] -= merge_s_bias
    else:
        ref = main_ref
    #print('FRENET2CART')
    epsilon=sys.float_info.epsilon
    
    
    L = ref.shape[0]
    T = traj.shape[0]
    cart_traj = np.zeros_like(traj)
    gamma = np.zeros((L))
    for i in range(L-1):
        gamma[i] = norm(ref[i+1]-ref[i])
    gamma[L-1] = gamma[L-2]
    gamma = np.cumsum(gamma)
    traj_cart = np.zeros((T,2))
    #assert(np.any(gamma>traj))
    it1 = 0
    for i in range(T):
        
        it2 = np.flatnonzero(gamma[it1:]>traj[i,0])[0]
        it2 = it1+it2
        it1 = it2-1
        assert(it1>=0)             

        thetha1 = np.arctan((ref[it2,1]-ref[it1,1])/(ref[it2,0]-ref[it1,0]+epsilon))
        
        thetha = np.arctan((traj[i,1])/(traj[i,0]-gamma[it1]+epsilon))
        
        thetha_cart = thetha1+thetha
        dist2origin = np.sqrt(np.power(traj[i,1], 2) + np.power((traj[i,0]- gamma[it1]), 2))
        #assert(np.sin(thetha_cart)>0)
        #assert(np.cos(thetha_cart)>0)
        cart_traj[i,0] = dist2origin * np.cos(thetha_cart) + ref[it1, 0]
        cart_traj[i,1] = dist2origin *np.sin(thetha_cart) + ref[it1, 1]
        #print('it1:{}, it2:{}, theta:{}, theta1:{},{},{},{}'.format(it1,it2,thetha*180/np.pi,thetha1*180/np.pi, (np.abs(traj[i,1]))/(np.abs(traj[i,0]- gamma[it1])+epsilon),np.abs(traj[i,1]),thetha_cart*180/np.pi) )
    return cart_traj
