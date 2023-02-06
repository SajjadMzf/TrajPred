import os
import torch
import torch.utils.data as utils_data
import numpy as np
import models
from datetime import datetime
import yaml
import copy
import kpis
import model_specific_training_functions as mstf
import pickle

class ParametersHandler:
    def __init__(self, model, dataset, parameters_dir, experiments_dir = 'experiments/', evaluation_dir = 'evaluations/', models_dir = 'models/', datasets_dir = 'datasets/', constants_file = 'constants.yaml', hyperparams_file = 'hyperparams.yaml'):
        self.parameter_tuning_experiment = False 
        self.model_file = os.path.join(os.path.join(parameters_dir, models_dir), model)
        self.dataset_file = os.path.join(os.path.join(parameters_dir, datasets_dir), dataset)
        self.hyperparams_file = os.path.join(parameters_dir, hyperparams_file)
        self.constants_file = os.path.join(parameters_dir, constants_file)
        self.model_dataset = '{}_{}'.format(model.split('.')[0], dataset.split('.')[0])
        self.experiments_dir = experiments_dir
        if os.path.exists(self.experiments_dir)== False:
            os.makedirs(self.experiments_dir)
        
        self.evaluation_dir = evaluation_dir
        if os.path.exists(self.evaluation_dir)== False:
            os.makedirs(self.evaluation_dir)
        
        now = datetime.now()
        self.experiment_file = '{}_{}'.format(self.model_dataset, now)
        self.latest_experiment_file = os.path.join(self.experiments_dir, self.experiment_file) 
        

        with open(self.hyperparams_file, 'r') as f:
            self.hyperparams = yaml.load(f, Loader = yaml.FullLoader)
        
        with open(self.constants_file, 'r') as f:
            self.constants = yaml.load(f, Loader = yaml.FullLoader)
        
        with open(self.dataset_file, 'r') as f:
            self.dataset = yaml.load(f, Loader = yaml.FullLoader)
        
        with open(self.model_file, 'r') as f:
            self.model = yaml.load(f, Loader = yaml.FullLoader)
        
        self.create_dirs()
        self.match_parameters()

    def new_experiment(self):
        now = datetime.now()
        self.experiment_file = '{}_{}'.format(self.model_dataset, now)
        self.latest_experiment_file = os.path.join(self.experiments_dir, self.experiment_file) 
        self.match_parameters()
        
    def create_dirs(self):
        if not os.path.exists(self.constants['DIRS']['MODELS_DIR']):
            os.mkdir(self.constants['DIRS']['MODELS_DIR'])
        if not os.path.exists(self.constants['DIRS']['RESULTS_DIR']):
            os.mkdir(self.constants['DIRS']['RESULTS_DIR'])
        if not os.path.exists(self.constants['DIRS']['TABLES_DIR']):
            os.mkdir(self.constants['DIRS']['TABLES_DIR'])
        if not os.path.exists(self.constants['DIRS']['FIGS_DIR']):
            os.mkdir(self.constants['DIRS']['FIGS_DIR'])
        if not os.path.exists(self.constants['DIRS']['VIS_DIR']):
            os.mkdir(self.constants['DIRS']['VIS_DIR'])
    
    def tune_params(self, tuning_experiment_name, selected_params, selected_metrics):
        self.parameter_tuning_experiment = True
        self.tuning_experiment_name = tuning_experiment_name
        self.log_dict = {}
        for param_str in selected_params:
            self.log_dict[param_str] = eval('self.{}'.format(param_str))
        self.selected_metrics = selected_metrics
    
    def match_parameters(self):
        self.SELECTED_MODEL = self.model['name']#'REGIONATTCNN3'#'VCNN'
        self.SELECTED_DATASET = self.dataset['name']
        self.experiment_tag = self.experiment_file
        self.UNBALANCED = not self.hyperparams['dataset']['balanced']
        self.ABLATION = self.hyperparams['dataset']['ablation']
        self.DEBUG_MODE = self.hyperparams['experiment']['debug_mode']
        self.experiment_group = self.hyperparams['experiment']['group']
        self.MULTI_MODAL_EVAL = self.hyperparams['experiment']['multi_modal_eval']
        self.MAN_DEC_IN = self.hyperparams['model']['man_dec_in']
        self.MAN_DEC_OUT = self.hyperparams['model']['man_dec_out']
        self.MULTI_MODAL = self.hyperparams['model']['multi_modal']
        self.VAL_SCORE = self.hyperparams['model']['validation_score']
        self.LOWER_BETTER_VAL_SCORE = self.hyperparams['model']['lower_better_val_score']
        
        self.DATASET_DIR = self.dataset['dataset_dir']
        self.IMAGE_HEIGHT = self.dataset['image_height']
        self.IMAGE_WIDTH = self.dataset['image_width']
        self.TR_RATIO = self.dataset['train']
        self.VAL_RATIO = self.dataset['val']
        self.TE_RATIO = self.dataset['test']
        self.ABBVAL_RATIO = self.dataset['abblation_val']
        self.DATA_FILES = [ str(i).zfill(2)+'.h5' for i in eval(self.dataset['dataset_ind'])]
            
        #exit()
        # Prediction Problem Hyperparameters:
        self.FPS = self.hyperparams['problem']['FPS']
        self.IN_SEQ_LEN = self.hyperparams['problem']['IN_SEQ_LEN']
        self.TGT_SEQ_LEN = self.hyperparams['problem']['TGT_SEQ_LEN'] # out_Seq_len
        self.SKIP_SEQ_LEN = self.hyperparams['problem']['SKIP_SEQ_LEN'] # end_of_seq_skip_len
        

        # Training  Hyperparameters
        self.CUDA = self.hyperparams['experiment']['cuda']
        self.BATCH_SIZE = self.hyperparams['training']['batch_size'] #64
        self.LR = self.hyperparams['training']['lr']#  0.001
        self.LR_WU = self.hyperparams['training']['lr_wu']
        self.LR_WU_BATCHES = self.hyperparams['training']['lr_wu_batches']
        self.LR_WU_CURRENT_BATCH = self.hyperparams['training']['lr_wu_current_batch']
        self.TRAJ2CLASS_LOSS_RATIO = self.hyperparams['training']['traj2class_loss_ratio']
        self.LR_DECAY = self.hyperparams['training']['lr_decay']
        self.NUM_EPOCHS = self.hyperparams['training']['num_epochs']
        self.PATIENCE = self.hyperparams['training']['patience']
        self.TR_JUMP_STEP = self.hyperparams['training']['tr_jump_step']
        self.SKIP_VAL_EPOCHS = self.hyperparams['training']['skip_validation_epoch']

        if self.UNBALANCED:
            self.unblanaced_ext = self.constants['DIRS']['UNBALANCED_EXT']
        else:
            self.unblanaced_ext = ''
        
        

        self.MODELS_DIR = self.constants['DIRS']['MODELS_DIR']
        self.RESULTS_DIR = self.constants['DIRS']['RESULTS_DIR']
        self.TABLES_DIR = self.constants['DIRS']['TABLES_DIR']
        self.FIGS_DIR = self.constants['DIRS']['FIGS_DIR']
        self.VIS_DIR = self.constants['DIRS']['VIS_DIR']
        
       
        
        # eval string attributes
        self.model_dictionary = copy.deepcopy(self.model)# we dont modify self.model as we might export/import it to/from YALM files
        self.model_dictionary['ref'] = eval(self.model_dictionary['ref'])
        self.model_dictionary['optimizer'] = eval(self.model_dictionary['optimizer'])
        self.model_dictionary['man loss function'] = eval(self.model_dictionary['man loss function'])
        self.model_dictionary['traj loss function'] = eval(self.model_dictionary['traj loss function'])
        self.model_dictionary['model training function'] = eval(self.model_dictionary['model training function'])
        self.model_dictionary['model evaluation function'] = eval(self.model_dictionary['model evaluation function'])
        self.model_dictionary['model kpi function'] = eval(self.model_dictionary['model kpi function'])
        
        self.CLASSIFICATION = self.constants['TASKS']['CLASSIFICATION']
        self.REGRESSION = self.constants['TASKS']['REGRESSION']
        self.DUAL = self.constants['TASKS']['DUAL']
        self.TRAJECTORYPRED = self.constants['TASKS']['TRAJECTORYPRED']

    
    def export_evaluation(self, kpis_dict):
        evaluation_cdir = os.path.join(self.evaluation_dir, '{}-{}.pkl'.format(self.experiment_file,datetime.now()))
        eval_name_dict = {'eval Time': '{}'.format(datetime.now())}
        name_dict = {'experiment file name': self.latest_experiment_file}
        with open(evaluation_cdir, 'wb') as f:
            pickle.dump([name_dict, self.hyperparams, self.model, self.dataset, kpis_dict], f) 
    # Export experiment after training a model (Do not export an imported experiment or it will overwrite it!,use export evalution)
    
    def export_experiment(self):
        name_dict = {'experiment file name': self.latest_experiment_file}
        #if self.DEBUG_MODE:
        #    print('Experiment export is skipped due to debug mode.')
        #else:
        print('Experiment file is: ', self.latest_experiment_file)
        with open(self.latest_experiment_file, 'w') as f:
            experiment = yaml.dump_all([name_dict, self.hyperparams, self.model, self.dataset], f, default_flow_style= False, explicit_start = True ) 
        
    # Import an already trained model for evaluation or reproducing the results
    def import_experiment(self, experiment_file):
        with open(experiment_file, 'r') as f:
            experiment = yaml.load_all(f, Loader=yaml.FullLoader)
            experiment_list = []
            for item in experiment:
                experiment_list.append(item)
            name_dict = experiment_list[0]
            self.hyperparams = experiment_list[1]
            self.model = experiment_list[2]
            self.dataset = experiment_list[3]
        self.latest_experiment_file = name_dict['experiment file name']
        self.experiment_file = self.latest_experiment_file.split('/')[-1]
        print('Experiment file is: ', self.latest_experiment_file)
        self.match_parameters()