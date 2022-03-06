import os
from train import train_model_dict
from evaluate import test_model_dict
import utils
import params
import models_dict as m



# ATT CNN
p = params.Parameters(SELECTED_MODEL = 'REGIONATTCNN3', SELECTED_DATASET = 'HIGHD', UNBALANCED = False, ABLATION = True)

# Regressionm
model_dict = m.MODELS[p.SELECTED_MODEL]

model_dict['hyperparams']['task'] = params.REGRESSION
model_dict['hyperparams']['curriculum loss'] = False
model_dict['hyperparams']['curriculum seq'] = False
model_dict['hyperparams']['curriculum virtual'] = False

model_dict['tag'] = utils.update_tag(model_dict)

train_model_dict(model_dict, p)

# CLassification
model_dict = m.MODELS[p.SELECTED_MODEL]

model_dict['hyperparams']['task'] = params.CLASSIFICATION
model_dict['hyperparams']['curriculum loss'] = False
model_dict['hyperparams']['curriculum seq'] = False
model_dict['hyperparams']['curriculum virtual'] = False

model_dict['tag'] = utils.update_tag(model_dict)

train_model_dict(model_dict, p)

# DUAL
model_dict = m.MODELS[p.SELECTED_MODEL]

model_dict['hyperparams']['task'] = params.DUAL
model_dict['hyperparams']['curriculum loss'] = False
model_dict['hyperparams']['curriculum seq'] = False
model_dict['hyperparams']['curriculum virtual'] = False

model_dict['tag'] = utils.update_tag(model_dict)

train_model_dict(model_dict, p)

# DUAL CS
model_dict = m.MODELS[p.SELECTED_MODEL]

model_dict['hyperparams']['task'] = params.DUAL
model_dict['hyperparams']['curriculum loss'] = False
model_dict['hyperparams']['curriculum seq'] = True
model_dict['hyperparams']['curriculum virtual'] = False

model_dict['tag'] = utils.update_tag(model_dict)

train_model_dict(model_dict, p)

# DUAL CL
model_dict = m.MODELS[p.SELECTED_MODEL]

model_dict['hyperparams']['task'] = params.DUAL
model_dict['hyperparams']['curriculum loss'] = True
model_dict['hyperparams']['curriculum seq'] = False
model_dict['hyperparams']['curriculum virtual'] = False

model_dict['tag'] = utils.update_tag(model_dict)

train_model_dict(model_dict, p)

# DUAL BOTH CS and CL
model_dict = m.MODELS[p.SELECTED_MODEL]

model_dict['hyperparams']['task'] = params.DUAL
model_dict['hyperparams']['curriculum loss'] = True
model_dict['hyperparams']['curriculum seq'] = True
model_dict['hyperparams']['curriculum virtual'] = False

model_dict['tag'] = utils.update_tag(model_dict)

train_model_dict(model_dict, p)





## VCNN

p = params.Parameters(SELECTED_MODEL = 'VCNN', SELECTED_DATASET = 'HIGHD', UNBALANCED = False, ABLATION = True)

# regression
model_dict = m.MODELS[p.SELECTED_MODEL]

model_dict['hyperparams']['task'] = params.REGRESSION
model_dict['hyperparams']['curriculum loss'] = False
model_dict['hyperparams']['curriculum seq'] = False
model_dict['hyperparams']['curriculum virtual'] = False

model_dict['tag'] = utils.update_tag(model_dict)

train_model_dict(model_dict, p)


# classification
model_dict = m.MODELS[p.SELECTED_MODEL]

model_dict['hyperparams']['task'] = params.CLASSIFICATION
model_dict['hyperparams']['curriculum loss'] = False
model_dict['hyperparams']['curriculum seq'] = False
model_dict['hyperparams']['curriculum virtual'] = False

model_dict['tag'] = utils.update_tag(model_dict)

train_model_dict(model_dict, p)


# dual
model_dict = m.MODELS[p.SELECTED_MODEL]

model_dict['hyperparams']['task'] = params.DUAL
model_dict['hyperparams']['curriculum loss'] = False
model_dict['hyperparams']['curriculum seq'] = False
model_dict['hyperparams']['curriculum virtual'] = False

model_dict['tag'] = utils.update_tag(model_dict)

train_model_dict(model_dict, p)



