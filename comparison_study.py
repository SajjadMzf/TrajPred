import os
from train import train_model_dict
from evaluate import test_model_dict
import utils
import params
import models_dict as m




# ATT CNN
p = params.Parameters(SELECTED_MODEL = 'REGIONATTCNN3', SELECTED_DATASET = 'HIGHD', UNBALANCED = False, ABLATION = False)
# DUAL BOTH CS and CL
model_dict = m.MODELS[p.SELECTED_MODEL]

model_dict['hyperparams']['task'] = params.DUAL
model_dict['hyperparams']['curriculum loss'] = True
model_dict['hyperparams']['curriculum seq'] = True
model_dict['hyperparams']['curriculum virtual'] = False

model_dict['tag'] = utils.update_tag(model_dict)

train_model_dict(model_dict, p)


# MLP1
p = params.Parameters(SELECTED_MODEL = 'MLP', SELECTED_DATASET = 'HIGHD', UNBALANCED = False, ABLATION = False)


model_dict = m.MODELS[p.SELECTED_MODEL]

model_dict['hyperparams']['task'] = params.CLASSIFICATION
model_dict['hyperparams']['curriculum loss'] = False
model_dict['hyperparams']['curriculum seq'] = False
model_dict['hyperparams']['curriculum virtual'] = False
model_dict['state type'] = 'wirth'
model_dict['tag'] = utils.update_tag(model_dict)

train_model_dict(model_dict, p)

# MLP2
p = params.Parameters(SELECTED_MODEL = 'MLP', SELECTED_DATASET = 'HIGHD', UNBALANCED = False, ABLATION = False)


model_dict = m.MODELS[p.SELECTED_MODEL]

model_dict['hyperparams']['task'] = params.CLASSIFICATION
model_dict['hyperparams']['curriculum loss'] = False
model_dict['hyperparams']['curriculum seq'] = False
model_dict['hyperparams']['curriculum virtual'] = False
model_dict['state type'] = 'shou'
model_dict['tag'] = utils.update_tag(model_dict)

train_model_dict(model_dict, p)


# LSTM1
p = params.Parameters(SELECTED_MODEL = 'VLSTM', SELECTED_DATASET = 'HIGHD', UNBALANCED = False, ABLATION = False)


model_dict = m.MODELS[p.SELECTED_MODEL]

model_dict['hyperparams']['task'] = params.CLASSIFICATION
model_dict['hyperparams']['curriculum loss'] = False
model_dict['hyperparams']['curriculum seq'] = False
model_dict['hyperparams']['curriculum virtual'] = False
model_dict['state type'] = 'wirth'
model_dict['tag'] = utils.update_tag(model_dict)

train_model_dict(model_dict, p)


# LSTM2
p = params.Parameters(SELECTED_MODEL = 'VLSTM', SELECTED_DATASET = 'HIGHD', UNBALANCED = False, ABLATION = False)


model_dict = m.MODELS[p.SELECTED_MODEL]

model_dict['hyperparams']['task'] = params.CLASSIFICATION
model_dict['hyperparams']['curriculum loss'] = False
model_dict['hyperparams']['curriculum seq'] = False
model_dict['hyperparams']['curriculum virtual'] = False
model_dict['state type'] = 'ours'
model_dict['tag'] = utils.update_tag(model_dict)

train_model_dict(model_dict, p)


###Regression
# MLP1
p = params.Parameters(SELECTED_MODEL = 'MLP', SELECTED_DATASET = 'HIGHD', UNBALANCED = False, ABLATION = False)


model_dict = m.MODELS[p.SELECTED_MODEL]

model_dict['hyperparams']['task'] = params.REGRESSION
model_dict['hyperparams']['curriculum loss'] = False
model_dict['hyperparams']['curriculum seq'] = False
model_dict['hyperparams']['curriculum virtual'] = False
model_dict['state type'] = 'wirth'
model_dict['tag'] = utils.update_tag(model_dict)

train_model_dict(model_dict, p)

# MLP2
p = params.Parameters(SELECTED_MODEL = 'MLP', SELECTED_DATASET = 'HIGHD', UNBALANCED = False, ABLATION = False)


model_dict = m.MODELS[p.SELECTED_MODEL]

model_dict['hyperparams']['task'] = params.REGRESSION
model_dict['hyperparams']['curriculum loss'] = False
model_dict['hyperparams']['curriculum seq'] = False
model_dict['hyperparams']['curriculum virtual'] = False
model_dict['state type'] = 'shou'
model_dict['tag'] = utils.update_tag(model_dict)

train_model_dict(model_dict, p)


# LSTM1
p = params.Parameters(SELECTED_MODEL = 'VLSTM', SELECTED_DATASET = 'HIGHD', UNBALANCED = False, ABLATION = False)


model_dict = m.MODELS[p.SELECTED_MODEL]

model_dict['hyperparams']['task'] = params.REGRESSION
model_dict['hyperparams']['curriculum loss'] = False
model_dict['hyperparams']['curriculum seq'] = False
model_dict['hyperparams']['curriculum virtual'] = False
model_dict['state type'] = 'wirth'
model_dict['tag'] = utils.update_tag(model_dict)

train_model_dict(model_dict, p)


# LSTM2
p = params.Parameters(SELECTED_MODEL = 'VLSTM', SELECTED_DATASET = 'HIGHD', UNBALANCED = False, ABLATION = False)


model_dict = m.MODELS[p.SELECTED_MODEL]

model_dict['hyperparams']['task'] = params.REGRESSION
model_dict['hyperparams']['curriculum loss'] = False
model_dict['hyperparams']['curriculum seq'] = False
model_dict['hyperparams']['curriculum virtual'] = False
model_dict['state type'] = 'ours'
model_dict['tag'] = utils.update_tag(model_dict)

train_model_dict(model_dict, p)
