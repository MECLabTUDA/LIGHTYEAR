from configs.config_5 import config_5
from configs.config_6 import config_6
from configs.config_8 import config_8
from configs.config_seg import config_seg


def config_loading(dataset):
    if dataset == 'camelyon17':
        return config_5()
    elif dataset == 'isic19':
        return config_6()
    elif dataset in ['xray', 'ultrasound']:
        return config_seg()
    else:
        return config_8()

    