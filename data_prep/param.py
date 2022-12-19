
DATASET = "Processed_NGSIM" #Processed_highD


FPS = 5
# Rendering params
GENERATE_IMAGE_DATA = False
save_whole_imgs = False
save_cropped_imgs = False
image_scaleH = 4
image_scaleW = 1

LINFIT_WINDOW = 5 
PLOT_LABELS = False
# parameters of CS-LSTM model
grid_max_x = 100

def generate_paths(first_leg, start_ind, end_ind, second_leg):
    path_list = []
    for i in range(start_ind, end_ind):
        path_list.append(first_leg + str(i).zfill(2) + second_leg)
    return path_list

if DATASET == "Processed_highD":
    track_paths = generate_paths('../../Dataset/HighD/Tracks/', 0, 61, '_tracks.csv')
    frame_pickle_paths = generate_paths('../../Dataset/HighD/Pickles/', 0,  61, '_frames.pickle')
    track_pickle_paths = generate_paths('../../Dataset/HighD/Pickles/', 0,  61, '_tracks.pickle')
    meta_paths = generate_paths('../../Dataset/HighD/Metas/', 0,  61, '_recordingMeta.csv')
    static_paths = generate_paths('../../Dataset/HighD/Statics/', 0,  61, '_tracksMeta.csv')
    cropped_height = int(20 * image_scaleH)
    cropped_width = int(200 * image_scaleW)
elif DATASET == 'Processed_NGSIM':
    track_paths = generate_paths('../../Dataset/NGSIM/Tracks/', 0, 7, '_tracks.csv') #start from zero to match with indexes
    frame_pickle_paths = generate_paths('../../Dataset/NGSIM/Pickles/', 0,  7, '_frames.pickle')
    track_pickle_paths = generate_paths('../../Dataset/NGSIM/Pickles/', 0,  7, '_tracks.pickle')
    meta_paths = generate_paths('../../Dataset/NGSIM/Metas/', 0,  7, '_recordingMeta.csv')
    static_paths = generate_paths('../../Dataset/NGSIM/Statics/', 0,  7, '_tracksMeta.csv')
    #TODO: Tune parameters for NGSIM image dataset
    cropped_height = int(20 * image_scaleH)
    cropped_width = int(200 * image_scaleW)
else:
    raise('undefined dataaset')



