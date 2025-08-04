import re
import threading
from typing import Optional
from aggregation.client_update import ClientUpdate
import numpy as np
import torch
from nvflare.app_common.app_constant import AppConstants
from scipy.spatial.distance import cdist
import random

class ASMR(object):
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

    def reset_stats(self):
        self.total = dict()
        self.counts = dict()
        self.history = list()


    def add(self, fl_model):
        """Compute weighted sum and sum of weights."""
        with self.lock:
            client_name = fl_model.meta.get("client_name", AppConstants.CLIENT_UNKNOWN),
            self.client_updates.append(ClientUpdate(fl_model.params, client_name, vectorized=True, 
                meta={'mal': False, 'reachDistances': [], 'rachDensity': 0, 'outlierFactor': 0}))
            



    def outlier_factor(self):
        """Compute the outlier factor for each update."""
        #with self.lock:
        avg_reach_dens = 0
        for client_update in self.client_updates:
            # Compute reachability distances
            for client_update_temp in self.client_updates:
                if client_update.client_name != client_update_temp.client_name:
                    dist = cdist(client_update.state_dict_oneD, client_update_temp.state_dict_oneD, metric='cosine')
                    client_update.meta['reachDistances'].append(dist)
            # Compute reachability density
            agg_dist = sum(client_update.meta['reachDistances']) 
            print('aggregated reachability distance')
            print(agg_dist)
            k = len(self.client_updates) - 1
            client_update.meta['reachDensity'] = 1 / ((agg_dist / k) + 1)
            avg_reach_dens += client_update.meta['reachDensity']
        # Compute outlier factor
        avg_reach_dens /= len(self.client_updates)
        for client_update in self.client_updates:
            client_update.meta['outlierFactor'] = client_update.meta['reachDensity'] / (avg_reach_dens + 1)

    
    def detect(self):
        #Compute decision boundary
        #with self.lock:      
        self.client_updates.sort(key=lambda x: x.meta['outlierFactor'])
        decision_boundary = (0, 0)
        for i in range(len(self.client_updates) - 1):
            diff = self.client_updates[i+1].meta['outlierFactor'] - self.client_updates[i].meta['outlierFactor']
            if diff > decision_boundary[0]:
                decision_boundary = (diff, i+1)
        for i in range(decision_boundary[1]):
            self.client_updates[i].meta['mal'] = True

    # Calcluate the final state dict 
    def get_result(self):
        """Detects and aggregate benign updates."""
        with self.lock:
            self.outlier_factor()
            self.detect()
            benign_updates = [client_update for client_update in self.client_updates if not client_update.meta['mal']]
            
            keys         = benign_updates[0].state_dict.keys()
            global_model = {key: np.zeros_like(benign_updates[0].state_dict[key], dtype=np.float64) for key in keys}

            for client_update in benign_updates:
                for key in keys:
                    global_model[key] += (1 / len(benign_updates)) * client_update.state_dict[key]


            self.client_updates = []
        return global_model

    def get_history(self):
        return self.history

    def get_len(self):
        return len(self.get_history())
