from training.DittoTrainer import DittoTrainer
from training.Trainer import Trainer

def trainer_loading(trainer='FedAvg'):
    match trainer:
        case 'FedAvg':
            return Trainer
        case 'Ditto':
            return DittoTrainer

