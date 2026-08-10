from training.Trainer import Trainer
from training.SegTrainer import SegTrainer

def trainer_loading(trainer='FedAvg'):
    match trainer:
        case 'FedAvg':
            return Trainer
        case 'FedAvgSeg':
            return SegTrainer