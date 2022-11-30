from train import train_model_dict
from evaluate import test_model_dict
import params


# Ablation on Number of modes
'''
p = params.ParametersHandler('MMnTP.yaml', 'highD.yaml', './config')
#p.hyperparams['problem']['TGT_SEQ_LEN'] = 35
p.hyperparams['experiment']['debug_mode'] = False
p.hyperparams['dataset']['balanced'] = True
p.hyperparams['dataset']['ablation'] = True
# 1 Single Mode
p.hyperparams['model']['multi_modal'] =  False
p.hyperparams['model']['man_dec_out'] = False
p.hyperparams['experiment']['multi_modal_eval'] = False
p.new_experiment()
train_model_dict(p)
test_model_dict(p)

# 2 Single mode with maneouvre loss
p.hyperparams['model']['man_dec_out'] = True
p.new_experiment()
train_model_dict(p)
test_model_dict(p)

# 3 multimodal 3 modes
p.hyperparams['model']['multi_modal'] =  True
p.hyperparams['experiment']['multi_modal_eval'] = True
p.model['hyperparams']['number of modes'] = 3
p.new_experiment()
train_model_dict(p)
test_model_dict(p)
exit()
# 4 multimodal 6 modes
p.model['hyperparams']['number of modes'] = 6
p.new_experiment()
train_model_dict(p)
test_model_dict(p)

# 5 multimodal 10 modes
p.model['hyperparams']['number of modes'] = 10
p.new_experiment()
train_model_dict(p)
test_model_dict(p)


exit()
'''
# Ablation on Number of maneouvres per mode

p = params.ParametersHandler('MMnTP.yaml', 'highD.yaml', './config')
p.hyperparams['training']['num_epochs'] = 50
p.hyperparams['training']['patience'] = 10
p.hyperparams['experiment']['debug_mode'] = False
p.hyperparams['dataset']['balanced'] = True
p.hyperparams['dataset']['ablation'] = True
p.hyperparams['experiment']['multi_modal_eval'] = True
p.model['hyperparams']['number of modes'] = 3

# 2. two maneouvre per mode (T_change = T_pred)
p.model['hyperparams']['manouvre per mode'] = 2
p.new_experiment()
train_model_dict(p)
test_model_dict(p)

# 3. Three maneouvre per mode (T_change = T_pred/2)
p.model['hyperparams']['manouvre per mode'] = 3
p.new_experiment()
train_model_dict(p)
test_model_dict(p)
# 4. Four maneouvre per mode (T_change = T_pred/3)
p.model['hyperparams']['manouvre per mode'] = 4
p.new_experiment()
train_model_dict(p)
test_model_dict(p)
exit()

# 5 SMT





