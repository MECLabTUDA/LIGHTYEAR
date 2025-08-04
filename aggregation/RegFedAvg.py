import re
import threading
from typing import Optional
from aggregation.client_update import ClientUpdate
import numpy as np
import torch
from nvflare.app_common.app_constant import AppConstants
from scipy.spatial.distance import cdist
import random

class RegFedAvg(object):
    def __init__(self, exclude_vars: Optional[str] = None, weigh_by_local_iter: bool = True):
        """Perform weighted aggregation.

        Args:
            exclude_vars (str, optional): regex string to match excluded vars during aggregation. Defaults to None.
            weigh_by_local_iter (bool, optional): Whether to weight the contributions by the number of iterations
                performed in local training in the current round. Defaults to `True`.
                Setting it to `False` can be useful in applications such as homomorphic encryption to reduce
                the number of computations on encrypted ciphertext.
                The aggregated sum will still be divided by the provided weights and `aggregation_weights` for the
                resulting weighted sum to be valid.
        """
        super().__init__()
        self.lock = threading.Lock()
        self.exclude_vars = re.compile(exclude_vars) if exclude_vars else None
        self.weigh_by_local_iter = weigh_by_local_iter
        self.reset_stats()
        self.total = dict()
        self.counts = dict()
        self.history = list()
        self.client_updates = []
        self.global_model = None
        self.lambda_t = 0.95

    def reset_stats(self):
        self.total = dict()
        self.counts = dict()
        self.history = list()


    # for example dict2vec for every added state_dict
    def add(self, fl_model):
        """Compute weighted sum and sum of weights."""
        with self.lock:
            client_name = fl_model.meta.get("client_name", AppConstants.CLIENT_UNKNOWN),
            self.client_updates.append(ClientUpdate(fl_model.params, client_name, vectorized=False))
            

    # Calcluate the final state dict 
    def get_result(self):
        """Detects and aggregate benign updates."""
        with self.lock:
            keys         = self.client_updates[0].state_dict.keys()
            fedavg_model = {key: np.zeros_like(self.client_updates[0].state_dict[key], dtype=np.float64) for key in keys}

            for client_update in self.client_updates:
                for key in keys:
                    fedavg_model[key] += (1 / (len(self.client_updates)) * client_update.state_dict[key])
            
            if self.global_model is None:
                self.global_model = fedavg_model
                self.client_updates = []
                return fedavg_model

            reg_global_model = {}
            for key in keys:
                reg_global_model[key] = self.global_model[key] + self.lambda_t * (fedavg_model[key] - self.global_model[key])
        
            self.global_model = reg_global_model
            self.client_updates = []
            self.lambda_t *= self.lambda_t

        return reg_global_model

    def get_history(self):
        return self.history

    def get_len(self):
        return len(self.get_history())
