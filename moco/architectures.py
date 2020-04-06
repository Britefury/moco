import torchvision.models as models
from moco import model_wideresnet

models_dict = {}
for mod in [models, model_wideresnet]:
    for key, value in mod.__dict__.items():
        if key.islower() and not key.startswith('__') and callable(value):
            models_dict[key] = value
