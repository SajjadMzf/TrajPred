
DATASET = "Processed_highD"
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
FPS = 5

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

if DATASET == "Processed_highD":
    track_paths = generate_paths('../../Dataset/HighD/Tracks/', 0, 61, '_tracks.csv')
    frame_pickle_paths = generate_paths('../../Dataset/HighD/Pickles/', 0,  61, '_frames.csv')
    track_pickle_paths = generate_paths('../../Dataset/HighD/Pickles/', 0,  61, '_tracks.csv')
    meta_paths = generate_paths('../../Dataset/HighD/Metas/', 0,  61, '_recordingMeta.csv')
    static_paths = generate_paths('../../Dataset/HighD/Statics/', 0,  61, '_tracksMeta.csv')
    cropped_height = int(20 * image_scaleH)
    cropped_width = int(200 * image_scaleW)
else:
    raise('undefined dataaset')



