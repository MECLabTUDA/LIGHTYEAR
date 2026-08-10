from datasets.camelyon17 import camelyon17_loading
from datasets.femnist import femnist_loading
from datasets.isic19 import isic19_loading 
from datasets.ultrasound import ultrasound_loading
from datasets.xray import xray_loading


def data_loading(client_id:int, dataset:str):
    match dataset:
        case 'camelyon17':
            return camelyon17_loading(client_id)
        case 'femnist':
            return femnist_loading(client_id)        
        case 'isic19':
            return isic19_loading(client_id)
        case 'ultrasound':
            return ultrasound_loading(client_id)
        case 'xray':
            return xray_loading(client_id)