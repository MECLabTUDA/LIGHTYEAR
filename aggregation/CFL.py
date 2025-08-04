import re
import threading
from typing import Optional
from nvflare.app_common.app_constant import AppConstants
from aggregation.client_update import ClientUpdate
import numpy as np
from numpy import inf
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist

class CFL(object):
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


    # for example dict2vec for every added state_dict
    def add(self, fl_model):
        """Compute weighted sum and sum of weights."""
        with self.lock:
            client_name = fl_model.meta.get("client_name", AppConstants.CLIENT_UNKNOWN)
            self.client_updates.append(ClientUpdate(fl_model.params, client_name, vectorized=True, 
                                                    normalized_vectors=False, meta={'mal': False}))

    def distance_matrix(self):
        
        num = len(self.client_updates)
        dis_max = np.zeros((num, num))

        for i in range(num):
            for j in range(i+1, num):
                dis_max[i, j] = cdist(self.client_updates[i].state_dict_oneD, 
                                      self.client_updates[j].state_dict_oneD,
                                      metric='cosine')
                dis_max[j, i] = dis_max[i, j]

        dis_max[dis_max == -inf]   = -1
        dis_max[dis_max == inf]    =  1
        dis_max[np.isnan(dis_max)] = -1

        return dis_max



    # Calcluate the final state dict 
    def get_result(self):
        """Divide weighted sum by sum of weights."""
        with self.lock:
            clustering = AgglomerativeClustering(linkage="complete", n_clusters=2)

            dis_max = self.distance_matrix()
            clustering.fit(dis_max)
            benign_label = 1 if np.sum(clustering.labels_) > len(self.client_updates) // 2 else 0

            benign_clients = 0
            for i in range(len(self.client_updates)):
                if clustering.labels_[i] == benign_label:
                    benign_clients += 1
                else:
                    self.client_updates[i].meta['mal'] = True


            keys         = self.client_updates[0].state_dict.keys()
            global_model = {key: np.zeros_like(self.client_updates[0].state_dict[key], dtype=np.float64) for key in keys}

            for client_update in self.client_updates:
                if not client_update.meta['mal']:
                    for key in keys:
                        global_model[key] += (1 / benign_clients) * client_update.state_dict[key] 
        
            self.client_updates = []
        
        return global_model

    def get_history(self):
        return self.history

    def get_len(self):
        return len(self.get_history())
