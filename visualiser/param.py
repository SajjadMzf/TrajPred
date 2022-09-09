import cv2
DATASET = 'HIGHD'
FPS = 5
OCCLUSION = False
model_name =  'MTPMTT_highD_2022-08-24 21:34:37.789667'#'MTPMTT_highD_2022-08-22 15:47:03.709155'#'ManouvreTransformerTraj_highD_2022-07-12 15:48:02.307560'#'ManouvreTransformerTraj_highD_2022-07-12 15:48:02.307560'#'ManouvreTransformerTraj_highD_2022-06-27 14:46:40.899317'#'ManouvreTransformerTraj_highD_2022-06-14 15:14:02.050065'
RESULT_FILE = "../results/vis_data/"+ model_name +".pickle"
WHAT_IF_RENDERING = False
ITERATIVE_RENDERING = True

NUM_OUTPUT = 1000
CUT_OFF_SIGMA_RATIO = 3
N_PLOTTED_MODES = 5
MODE_PROB_THR = 0.2
# Actual image

# TAGS:
HIDE_SVS = False
PLOT_TEXTS = True
PLOT_MAN = True

Y_IMAGE_SCALE = 8*2
X_IMAGE_SCALE = 2*2

BORDER_PIXELS = 30
# Texts
FONT = cv2.FONT_HERSHEY_SIMPLEX
FSCALE = 1
FCOLOR = (0,0,0)
LINETYPE = 1
CLASS = ['LK', 'RLC', 'LLC']
CLASS2NUM = {'LK':0, 'RLC':1, 'LLC':2}

LINE_BREAK = 20
SECTION_BREAK = 1100

#lane marking
LINES_WIDTH = 1
# Pathes
def generate_paths(first_leg, total_num, second_leg):
        path_list = []
        for i in range(total_num):
            path_list.append(first_leg + str(i+1).zfill(2) + second_leg)
        return path_list

#bgr color codes
COLOR_CODES = {'TV': (102,6,3), #blue
            'SV': (128,128,128),
            'LANE':(128,128,128),
            'GT_TRAJ':(0,0,0),
            'PR_TRAJ':[
                (180,119,31),#blue
                (14,127,255),#orange
                (44,160,44),#green
                (40,39,214),#red
                (189,103,148),#purple
                ],
            'WIF_TRAJ':(0,150,0),
            'BACKGROUND':(255,255,255)
}


COLOR_NAMES= ['tab:blue', 
                'tab:orange',
                'tab:green',
                'tab:red',
                'tab:purple',
                'tab:brown',
                'tab:pink',
                'tab:gray',
                'tab:olive',
                'tab:cyan' ]

PLOT_MAN_NAMES = [
    'LLC',
    'LK',
    'RLC',
]
MARKERS = {
    'GT_TRAJ': 's-',
    'PR_TRAJ': 'o-',
    'WIF_TRAJ': '^-'
}
# 0: rpv(rav), 1. pv, 2. lpv(lav), 3. rfv, 4. fv,5. lfv
tv_dict = {
    '0':'Right Preceding',
    '1':'Preceding',
    '2':'Left Preceding',
    '3':'Right Following',
    '4':'Following',
    '5':'Left Following'
}

if DATASET == 'HIGHD':
    track_paths = generate_paths('../../../Dataset/HighD/Tracks/', 60, '_tracks.csv')
    frame_pickle_paths = generate_paths('../../../Dataset/HighD/Pickles/', 60, '_frames.csv')
    track_pickle_paths = generate_paths('../../../Dataset/HighD/Pickles/', 60, '_tracks.csv')
    meta_paths = generate_paths('../../../Dataset/HighD/Metas/', 60, '_recordingMeta.csv')
    static_paths = generate_paths('../../../Dataset/HighD/Statics/', 60, '_tracksMeta.csv')
    #background_paths = generate_paths('./Backgrounds/',60, '_highway.jpg')
elif DATASET == 'FNGSIM':
    track_paths = ['../../Dataset/FNGSIM/Traj_data/track_trajectories-0400-0415.csv',
                    '../../Dataset/FNGSIM/Traj_data/track_trajectories-0500-0515.csv',
                    '../../Dataset/FNGSIM/Traj_data/track_trajectories-0515-0530.csv',
                    '../../Dataset/FNGSIM/Traj_data/track_trajectories-0750am-0805am.csv',
                    '../../Dataset/FNGSIM/Traj_data/track_trajectories-0805am-0820am.csv',
                    '../../Dataset/FNGSIM/Traj_data/track_trajectories-0820am-0835am.csv']

    track_pickle_paths = ['../../Dataset/FNGSIM/Pickles/trajectories-0400-0415_tracks.pickle',
                    '../../Dataset/FNGSIM/Pickles/trajectories-0500-0515_tracks.pickle',
                    '../../Dataset/FNGSIM/Pickles/trajectories-0515-0530_tracks.pickle',
                    '../../Dataset/FNGSIM/Pickles/trajectories-0750am-0805am_tracks.pickle',
                    '../../Dataset/FNGSIM/Pickles/trajectories-0805am-0820am_tracks.pickle',
                    '../../Dataset/FNGSIM/Pickles/trajectories-0820am-0835am_tracks.pickle']
    frame_pickle_paths  = ['../../Dataset/FNGSIM/Pickles/trajectories-0400-0415_frames.pickle',
                    '../../Dataset/FNGSIM/Pickles/trajectories-0500-0515_frames.pickle',
                    '../../Dataset/FNGSIM/Pickles/trajectories-0515-0530_frames.pickle',
                    '../../Dataset/FNGSIM/Pickles/trajectories-0750am-0805am_frames.pickle',
                    '../../Dataset/FNGSIM/Pickles/trajectories-0805am-0820am_frames.pickle',
                    '../../Dataset/FNGSIM/Pickles/trajectories-0820am-0835am_frames.pickle']
    meta_paths = ['../../Dataset/FNGSIM/Traj_data/meta_trajectories-0400-0415.csv',
                    '../../Dataset/FNGSIM/Traj_data/meta_trajectories-0500-0515.csv',
                    '../../Dataset/FNGSIM/Traj_data/meta_trajectories-0515-0530.csv',
                    '../../Dataset/FNGSIM/Traj_data/meta_trajectories-0750am-0805am.csv',
                    '../../Dataset/FNGSIM/Traj_data/meta_trajectories-0805am-0820am.csv',
                    '../../Dataset/FNGSIM/Traj_data/meta_trajectories-0820am-0835am.csv']
    static_paths = ['../../Dataset/FNGSIM/Traj_data/static_trajectories-0400-0415.csv',
                    '../../Dataset/FNGSIM/Traj_data/static_trajectories-0500-0515.csv',
                    '../../Dataset/FNGSIM/Traj_data/static_trajectories-0515-0530.csv',
                    '../../Dataset/FNGSIM/Traj_data/static_trajectories-0750am-0805am.csv',
                    '../../Dataset/FNGSIM/Traj_data/static_trajectories-0805am-0820am.csv',
                    '../../Dataset/FNGSIM/Traj_data/static_trajectories-0820am-0835am.csv']
    