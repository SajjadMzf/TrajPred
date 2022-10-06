
DATASET = "Processed_highD"
DATASET = 'AutoplexCPM'
UNBALANCED = False
STATE_ONLY = False # There will be no rendering if set to True
dir_ext = ''
if UNBALANCED:
    dir_ext += 'U'




# Problem definition parameters:

OBS_LEN = 10
PRED_LEN = 20
POST_LC_LEN = 20
PRE_LC_LEN = OBS_LEN + PRED_LEN
FPS = 10
# Rendering params
save_whole_imgs = False
save_cropped_imgs = False
image_scaleH = 4
image_scaleW = 1

GENERATE_IMAGE_DATA = False
LINFIT_WINDOW = 5 
PLOT_LABELS = False
# parameters of CS-LSTM model
grid_max_x = 100

def generate_paths(first_leg, start_ind, end_ind, second_leg):
    path_list = []
    for i in range(start_ind, end_ind):
        path_list.append(first_leg + str(i).zfill(2) + second_leg)
    return path_list
def generate_paths2(first_leg, path_list, second_leg):
    generated_path_list = []
    for path in path_list:
        generated_path_list.append(first_leg+path+second_leg)
    return generated_path_list
if DATASET == "Processed_highD":
    track_paths = generate_paths('../../Dataset/HighD/Tracks/', 0, 61, '_tracks.csv')
    frame_pickle_paths = generate_paths('../../Dataset/HighD/Pickles/', 0,  61, '_frames.pickle')
    track_pickle_paths = generate_paths('../../Dataset/HighD/Pickles/', 0,  61, '_tracks.pickle')
    meta_paths = generate_paths('../../Dataset/HighD/Metas/', 0,  61, '_recordingMeta.csv')
    static_paths = generate_paths('../../Dataset/HighD/Statics/', 0,  61, '_tracksMeta.csv')
    cropped_height = int(20 * image_scaleH)
    cropped_width = int(200 * image_scaleW)
elif DATASET == 'AutoplexCPM':
    file_names = ['M40_h06',
                'M40_h07', 
                'M40_h08',
                'M40_h09', 
                'M40_h11', 
                'M40_h12', 
                'M40_h13', 
                'M40_h14', 
                'M40_h15', 
                'M40_h16', 
                'M40_h17', 
                'M40_h18', 
                'M40_h19',
                'M40_h10'] # H10 is test
    
    track_paths = generate_paths2('../../Dataset/Autoplex/Tracks/', file_names,'.csv')
    frame_pickle_paths = generate_paths2('../../Dataset/Autoplex/Pickles/', file_names,'_frames.pickle')#['../../Dataset/Autoplex/Pickles/M40draft2_processed_frames.pickle']
    track_pickle_paths = generate_paths2('../../Dataset/Autoplex/Pickles/', file_names,'_tracks.pickle')#['../../Dataset/Autoplex/Pickles/M40draft2_processed_tracks.pickle']
    meta_paths = generate_paths2('../../Dataset/Autoplex/Metas/', file_names,'_recordingMeta.csv')#['../../Dataset/Autoplex/Metas/M40draft2_processed_recordingMeta.csv']
    static_paths = generate_paths2('../../Dataset/Autoplex/Statics/', file_names,'_tracksMeta.csv')#['../../Dataset/Autoplex/Statics/M40draft2_processed_tracksMeta.csv'] 
    cropped_height = int(20 * image_scaleH)
    cropped_width = int(200 * image_scaleW)
else:
    raise('undefined dataaset')



