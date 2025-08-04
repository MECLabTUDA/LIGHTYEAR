from datasets.camelyon17 import camelyon17_loading
from datasets.femnist import femnist_loading

def data_loading(client_id:int, dataset:str):
    match dataset:
        case 'camelyon17':
            return camelyon17_loading(client_id)
        case 'femnist':
            return femnist_loading(client_id)
