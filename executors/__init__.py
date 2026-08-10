from executors.p2p_executor import P2PExecutor
from executors.p2p_NTKLightyear_executor import P2PNTKLIGHTYEARExecutor


def executor_loading(algo:str):
    match algo:
        case 'FedAvg':
            return P2PExecutor
        case 'LIGHTYEAR':
            return P2PNTKLIGHTYEARExecutor