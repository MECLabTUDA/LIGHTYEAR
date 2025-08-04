from executors.p2p_executor import P2PExecutor
from executors.p2p_balance_executor import P2PBALANCEExecutor
from executors.p2p_scclipping_executor import P2PSSClippingExecutor
from executors.p2p_lightyear_executor import P2PLIGHTYEARExecutor

def executor_loading(algo:str):
    match algo:
        case 'FedAvg':
            return P2PExecutor
        case 'BALANCE':
            return P2PBALANCEExecutor
        case 'SCClipping':
            return P2PSSClippingExecutor
        case 'LIGHTYEAR':
            return P2PLIGHTYEARExecutor
