import os
import time
import numpy as np 
import sys
import param as p
import pickle
import read_csv as rc
import matplotlib.pyplot as plt


def extract_seq_len(file_numbers):
    
    seq_lens = []
    for file_number in file_numbers:
        
        track_path = p.track_paths[file_number], 
        track_pickle_path = p.track_pickle_paths[file_number],
        frame_pickle_path = p.frame_pickle_paths[file_number], 
        static_path = p.static_paths[file_number],
        meta_path = p.meta_paths[file_number],
        #print(meta_path)
        meta = rc.read_meta_info(meta_path[0])
        fr_div = meta[rc.FRAME_RATE]/p.FPS
        scenarios = []
        data_tracks, _ = rc.read_track_csv(track_path[0], track_pickle_path[0], group_by = 'tracks', reload = False, fr_div = fr_div)
        data_frames, _ = rc.read_track_csv(track_path[0], frame_pickle_path[0], group_by = 'frames', reload = False, fr_div = fr_div)
        statics = rc.read_static_info(static_path[0])
        
        for tv_idx, tv_data in enumerate(data_tracks):
            seq_lens.append(len(tv_data[rc.FRAME]))


    seq_lens = np.array(seq_lens)
    # the histogram of the data
    n, bins, patches = plt.hist(seq_lens, 50, density=True, facecolor='g', alpha=0.75)

    plt.xlabel('seq lens')
    plt.ylabel('Probability')
    plt.title('Histogram of seq lens')
    
    #plt.xlim(40, 160)
    #plt.ylim(0, 0.03)
    plt.grid(True)
    plt.show()



if __name__ =="__main__":
    
    np.random.seed(0)   
    
    # Single Core (For Debugging purposes)
    file_numbers = np.arange(1,61)
    file_numbers = np.array([1])
    extract_seq_len(file_numbers)    
    #render(1, i)
    exit()
    
