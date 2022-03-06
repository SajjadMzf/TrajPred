import os
import time
from multiprocessing import Process
import numpy as np 
import sys
from extract_scenarios import ExtractScenarios
from render_scenarios import RenderScenarios
import param as p

np.set_printoptions(threshold=sys.maxsize)

def extract(core_num, file_numbers):
    start = time.time()
    for file_number in file_numbers:
        print('Core number: ', core_num, ' started on file number: ', file_number)
        
        extractor = ExtractScenarios(
            file_number,
            p.track_paths[file_number], 
            p.track_pickle_paths[file_number],
            p.frame_pickle_paths[file_number], 
            p.static_paths[file_number],
            p.meta_paths[file_number],
            p.DATASET)
        extractor.extract_and_save() 
    
    end = time.time()
    print('Core Number: ', core_num, ' Ended in: ', end-start, ' s.')

def render(core_num, file_numbers):
    for file_number in file_numbers:
        print('Core number: ', core_num, ' started on file number: ', file_number)
        start = time.time()
        
        renderer = RenderScenarios(
            file_number,
            p.track_paths[file_number], 
            p.frame_pickle_paths[file_number], 
            p.static_paths[file_number],
            p.meta_paths[file_number],
            p.DATASET
        )
        renderer.update_dirs() 
        renderer.load_scenarios()
        renderer.render_scenarios()
        renderer.save_dataset()  

        end = time.time()
    print('Core Number: ', core_num, ' Ended in: ', end-start, ' s.')




if __name__ =="__main__":
    
    np.random.seed(0)   
    '''
    # Single Core (For Debugging purposes)
    
    i = np.array([26])
    #extract(1, i)    
    render(1, i)
    exit()
    '''

    # Extract LC scenarios (multi-threads)
    total_cores = 3
    file_numbers = np.arange(1,61)
    total_files = len(file_numbers)
    file_per_core = int(total_files/total_cores)
    procs = []
    for core_num in range(total_cores):
        file_row = file_per_core*core_num
        core_fle_numbers = file_numbers[file_per_core*core_num:(file_per_core*(core_num+1))]
        proc = Process(target= extract, args = (core_num+1, core_fle_numbers))
        procs.append(proc)
        proc.start()
    
    for proc in procs:
        proc.join()

    # Render extracted LC scenarios (multi-threads)
    for core_num in range(total_cores):
        file_row = file_per_core*core_num
        core_fle_numbers = file_numbers[file_per_core*core_num:(file_per_core*(core_num+1))]
        proc = Process(target= render, args = (core_num+1, core_fle_numbers))
        procs.append(proc)
        proc.start()
    
    for proc in procs:
        proc.join()