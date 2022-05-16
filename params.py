import os
import torch
import torch.utils.data as utils_data
import numpy as np

class Parameters:
    def __init__(self, SELECTED_MODEL = 'VCNN', SELECTED_DATASET = 'HIGHD', UNBALANCED = False, ABLATION = False):
        # High Level Param
        self.SELECTED_MODEL = SELECTED_MODEL#'REGIONATTCNN3'#'VCNN'
        self.SELECTED_DATASET = SELECTED_DATASET
        self.UNBALANCED = UNBALANCED
        self.ABLATION = ABLATION
        self.DEBUG_MODE = False
        
        self.ROBUST_PREDICTOR = True
        # Dataset Hyperparameters:
        self.DATASETS = {
            'HIGHD':{
                'abb_tr_ind':range(1,46),
                'abb_val_ind':range(46,51),
                'abb_te_ind':range(51,56),
                'tr_ind':range(1,47),
                'val_ind':range(47,52),
                'te_ind':range(52,57),
                'image_width': 200,
                'image_height': 80,
            }
        }
        self.IMAGE_HEIGHT = self.DATASETS[self.SELECTED_DATASET]['image_height']
        self.IMAGE_WIDTH = self.DATASETS[self.SELECTED_DATASET]['image_width']
        
        if self.ABLATION:
            self.TR_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self.SELECTED_DATASET]['abb_tr_ind']]
            self.VAL_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self.SELECTED_DATASET]['abb_val_ind']]
            self.TE_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self.SELECTED_DATASET]['abb_te_ind']]
        else:
            self.TR_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self.SELECTED_DATASET]['tr_ind']]
            self.VAL_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self.SELECTED_DATASET]['val_ind']]
            self.TE_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self.SELECTED_DATASET]['te_ind']]

        # Prediction Problem Hyperparameters:
        self.FPS = 5
        self.SEQ_LEN = 30
        self.IN_SEQ_LEN = 10
        self.TGT_SEQ_LEN = 35 # out_Seq_len
        self.SKIP_SEQ_LEN = 5 # end_of_seq_skip_len
        self.CLASSIFICATION_OUTPUT_SIZE = 1 #3 => traj+label
        self.TRAJ_OUTPUT_SIZE = 2 + self.CLASSIFICATION_OUTPUT_SIZE #3 => traj+label
        # Metrics Hyperparameters:
        self.ACCEPTED_GAP = 0
        self.THR = 0.34

        # Training  Hyperparameters
        self.CUDA = True
        self.BATCH_SIZE = 12 #64
        self.LR = 0.0001#  0.001
        self.LR_WU = True
        self.LR_WU_BATCHES = 400
        self.LR_WU_CURRENT_BATCH = 0
        self.TRAJ2CLASS_LOSS_RATIO = 1
        self.LR_DECAY = 1
        self.LR_DECAY_EPOCH = 10
        self.NUM_EPOCHS = 50
        self.PATIENCE =1
        self.TR_JUMP_STEP =1 

        if self.UNBALANCED:
            self.unblanaced_ext = 'U'
        else:
            self.unblanaced_ext = ''
        self.TRAIN_DATASET_DIR = '../../Dataset/Processed_highD/RenderedDataset/'
        self.TEST_DATASET_DIR = '../../Dataset/Processed_highD/RenderedDataset/'
        

        self.MODELS_DIR = 'models/'
        self.RESULTS_DIR = 'results/'

        self.TABLES_DIR = self.RESULTS_DIR + 'tables/'
        self.FIGS_DIR = self.RESULTS_DIR + 'figures/'
        self.VIS_DIR = self.RESULTS_DIR + 'vis_data/'

        if not os.path.exists(self.MODELS_DIR):
            os.mkdir(self.MODELS_DIR)
        if not os.path.exists(self.RESULTS_DIR):
            os.mkdir(self.RESULTS_DIR)
        if not os.path.exists(self.TABLES_DIR):
            os.mkdir(self.TABLES_DIR)
        if not os.path.exists(self.FIGS_DIR):
            os.mkdir(self.FIGS_DIR)
        if not os.path.exists(self.VIS_DIR):
            os.mkdir(self.VIS_DIR)  
        
        
        @property
        def SELECTED_DATASET(self):
            return self._SELECTED_DATASET
        
        @SELECTED_DATASET.setter
        def SELECTED_DATASET(self, val):
            self._SELECTED_DATASET = val
            self.IMAGE_HEIGHT = self.DATASETS[self._SELECTED_DATASET]['image_height']
            self.IMAGE_WIDTH = self.DATASETS[self._SELECTED_DATASET]['image_width']
            if self._ABLATION:
                self.TR_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self._SELECTED_DATASET]['abb_tr_ind']]
                self.VAL_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self._SELECTED_DATASET]['abb_val_ind']]
                self.TE_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self._SELECTED_DATASET]['abb_te_ind']]
            else:
                self.TR_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self._SELECTED_DATASET]['tr_ind']]
                self.VAL_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self._SELECTED_DATASET]['val_ind']]
                self.TE_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self._SELECTED_DATASET]['te_ind']]
            
            self.DATASET_DIR = '../../Dataset/Processed_highD/RenderedDataset/'
        
        
        @property 
        def ABLATION(self):
            return self._ABLATION
        
        @ABLATION.setter
        def ABLATION(self, val):
            self._ABLATION = val
            if self._ABLATION:
                self.TR_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self._SELECTED_DATASET]['abb_tr_ind']]
                self.VAL_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self._SELECTED_DATASET]['abb_val_ind']]
                self.TE_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self._SELECTED_DATASET]['abb_te_ind']]
            else:
                self.TR_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self._SELECTED_DATASET]['tr_ind']]
                self.VAL_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self._SELECTED_DATASET]['val_ind']]
                self.TE_DATA_FILES = [ str(i).zfill(2)+'.h5' for i in self.DATASETS[self._SELECTED_DATASET]['te_ind']]
        
        @property
        def UNBALANCED(self):
            return self._UNBALANCED

        @UNBALANCED.setter 
        def UNBALANCED(self, val):
            self._UNBALANCED = val
            if self._UNBALANCED:
                self.unblanaced_ext = 'U'
            else:
                self.unblanaced_ext = ''
            self.DATASET_DIR = '../../Dataset/Processed_highD/RenderedDataset/'
        
        
         


# Different Tasks
CLASSIFICATION = 0
REGRESSION = 1
DUAL = 2 
TRAJECTORYPRED = 3   
            
        
            

        

        