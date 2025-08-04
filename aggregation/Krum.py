import re
import threading
from typing import Optional
from nvflare.app_common.app_constant import AppConstants
from aggregation.client_update import ClientUpdate
import numpy as np
from numpy import inf
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist


#https://github.com/cpwan/Attack-Adaptive-Aggregation-in-Federated-Learning/blob/master/rules/multiKrum.py
class Krum(object):
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


    def get_krum(self):
        
        update_vecs  = [update.state_dict_oneD for update in self.client_updates]
        update_vecs  = np.stack(update_vecs, axis=0)
        #input_tensor = update_vecs.T.unsqueeze(0)
    

        n = update_vecs.shape[0]
        f = n // 2  # worse case 50% malicious points
        k = n - f - 2

        # collection distance, distance from points to points
        #x = input_tensor.permute(0, 2, 1)
        #cdist = torch.cdist(x, x, p=2)
        cdist = np.linalg.norm(update_vecs[:, None, :] - update_vecs[None, :, :], axis=-1)
        # find the k+1 nbh of each point
        #nbhDist, nbh = torch.topk(cdist, k + 1, largest=False)
        krum_scores = []
        n_clients = len(self.client_updates)
        for i in range(n_clients):
            # Exclude self-distance and get k smallest distances
            distances = np.delete(cdist[i], i)
            closest_k = np.sort(distances)[:k]
            score = np.sum(closest_k)
            krum_scores.append(score)
        # the point closest to its nbh
        #i_star = np.argmin(nbhDist.sum(2))
        i_star = np.argmin(krum_scores)
        # krum
        krum = self.client_updates[i_star]
        return krum


    # Calcluate the final state dict 
    def get_result(self):
        with self.lock:
            
            global_model = self.get_krum().state_dict
            self.client_updates = []
        
        return global_model

    def get_history(self):
        return self.history

    def get_len(self):
        return len(self.get_history())
