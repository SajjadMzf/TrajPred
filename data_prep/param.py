### GENERATE_MAP PARAMs
MAP_XRES = 1
MAP_YRES = 0.1
MAP_LENGTH_EXT = 300
MAP_CELL_LENGTH = 15
MAP_CELL_CHANNEL = 3
MAP_VISION = 150
#######################



DATASET = "Processed_exid" #Processed_highD #Processed_NGSIM


FPS = 5

LINFIT_WINDOW = 5 
PLOT_LABELS = False
# parameters of CS-LSTM model
grid_max_x = 100

def generate_paths(first_leg, start_ind, end_ind, second_leg):
    path_list = []
    for i in range(start_ind, end_ind):
        path_list.append(first_leg + str(i).zfill(2) + second_leg)
    return path_list

def generate_paths2(first_leg, ind_list, second_leg):
    path_list = []
    for i in ind_list:
        path_list.append(first_leg + str(i).zfill(2) + second_leg)
    return path_list



if DATASET == 'Processed_exid':
    
    ind234 = list(range(39,73))
    ind6 = list(range(78,93))
    ind_list = []
    ind_list.extend(ind234)
    ind_list.extend(ind6)
    track_paths = generate_paths('../../Dataset/exid/Tracks/', 0, 93, '_tracks.csv') #start from zero to match with indexes
    frame_pickle_paths = generate_paths('../../Dataset/exid/Pickles/', 0,93, '_frames.pickle')
    track_pickle_paths = generate_paths('../../Dataset/exid/Pickles/', 0,93, '_tracks.pickle')
    map_paths = ['../../Dataset/exid/Maps/39-52.pickle',
                '../../Dataset/exid/Maps/53-60.pickle',
                '../../Dataset/exid/Maps/61-72.pickle',
                '../../Dataset/exid/Maps/78-92.pickle']
    static_paths = [None]*93
    meta_paths = [None]*93
    IN_FPS = 25
    driving_dir = 2
    #cropped_height = int(20 * image_scaleH)
    #cropped_width = int(200 * image_scaleW)
else:
    raise('undefined dataaset')



