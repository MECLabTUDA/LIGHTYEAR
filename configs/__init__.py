from configs.camelyon17_config import camelyon17_config
from configs.training_config import training_config 

def config_loading(dataset):
    if dataset == 'camelyon17':
        return camelyon17_config()
    else:
        return training_config()
