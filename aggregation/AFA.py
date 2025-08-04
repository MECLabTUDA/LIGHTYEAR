import re
import threading
from typing import Optional
from aggregation.client_update import ClientUpdate
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import beta
from nvflare.app_common.app_constant import AppConstants
#https://github.com/SamuelTrew/FederatedLearning/blob/master/aggregators/AFA.py

class AFA(object):
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
        self.client_names   = []
        self.client_updates = []
        self.xi = 2
        self.deltaXi = 0.25


    def reset_stats(self):
        self.total = dict()
        self.counts = dict()
        self.history = list()


    # for example dict2vec for every added state_dict
    def add(self, fl_model):
        """Compute weighted sum and sum of weights."""
        with self.lock:
            client_name = fl_model.meta.get("client_name", AppConstants.CLIENT_UNKNOWN)

            if client_name not in self.client_names:
                self.client_names.append(client_name)
                self.client_updates.append(ClientUpdate(fl_model.params, client_name, vectorized=True,
                    meta={'alpha': 3.0, 'beta': 3.0, 'score': 1.0, 'blocked': False, 'badUpdate': False,
                        'num_samples': fl_model.meta['num_samples'], 'pEpoch': 0, 'similarity': 0, 'p': 1}))
            else:
                self._update_client_update(fl_model, client_name)



    
    def _update_client_update(self, fl_model, client_name):
        for client_update in self.client_updates:
            if client_update.client_name == client_name:
                client_update.update_state_dict(fl_model.params) 
    

    def _get_state_dict_oneD(self, state_dict):
        components = []
        for param in state_dict:
            components.append(state_dict[param])

        vec = np.concatenate([component.flatten() for component in components])
        vec /= np.linalg.norm(vec)
        return vec.reshape(1, -1)

    def _notBlockedNorBad(self, client_update):
        return not client_update.meta['blocked'] and not client_update.meta['badUpdate']


    def _renormalize_weights(self):
        total = 0
        for client_update in self.client_updates:
            total += client_update.meta['p']
        for client_update in self.client_updates:
            client_update.meta['p'] /= total
        

    # Calcluate the final state dict 
    def get_result(self):
        """Divide weighted sum by sum of weights."""
        with self.lock:
            self._renormalize_weights()
            slack = self.xi
            badCount = 2
            while badCount != 0:

                #Overall contribution
                pT_epoch = 0.0


                #Calculate weighting for each client
                for client_update in self.client_updates:
                    if self._notBlockedNorBad(client_update):
                        client_update.meta['pEpoch'] = client_update.meta['num_samples'] * client_update.meta['score']
                        pT_epoch += client_update.meta['pEpoch']
                

                #Normalize each weighting
                for client_update in self.client_updates:
                    if self._notBlockedNorBad(client_update):
                        client_update.meta['pEpoch'] /= pT_epoch

                
                #Compute temporary global model
                keys              = self.client_updates[0].state_dict.keys()
                temp_global_model = {key: np.zeros_like(self.client_updates[0].state_dict[key], dtype=np.float64) for key in keys}

                for client_update in self.client_updates:
                    if self._notBlockedNorBad(client_update):
                        for key in keys:
                            temp_global_model[key] += client_update.meta['pEpoch'] * client_update.state_dict[key]

                temp_global_model = self._get_state_dict_oneD(temp_global_model)   
                
                #Calculate distance between all updates and temporal global model
                similarities = []
                for client_update in self.client_updates:
                    if self._notBlockedNorBad(client_update):
                        cosine_sim = 1 - cdist(client_update.state_dict_oneD, temp_global_model, metric='cosine')        
                        similarities.append(cosine_sim)
                        client_update.meta['similarity'] = cosine_sim

                similarities = np.concatenate([similarity.ravel() for similarity in similarities])
                
                meanS   = np.mean(similarities)
                medianS = np.median(similarities)
                desvS   = np.std(similarities)

                if meanS < medianS:
                    th = medianS - slack * desvS
                else:
                    th = medianS + slack * desvS

                slack += self.deltaXi

                badCount = 0
                for client_update in self.client_updates:
                    if not client_update.meta['badUpdate']:
                        if meanS < medianS:
                            if client_update.meta['similarity'] < th:
                                client_update.meta['badUpdate'] = True
                                badCount += 1
                        else:
                            if client_update.meta['similarity'] > th:
                                client_update.meta['badUpdate'] = True
                                badCount += 1
            pT = 0.0
            for client_update in self.client_updates:
                if not client_update.meta['blocked']:
                    if client_update.meta['badUpdate']:
                        client_update.meta['beta'] += 1
                    else:
                        client_update.meta['alpha'] += 1
                    
                    #Update update score
                    client_update.meta['score']   = client_update.meta['alpha'] / client_update.meta['beta']
                    #Update blocked status
                    client_update.meta['blocked'] = beta.cdf(0.5, client_update.meta['alpha'], client_update.meta['beta']) > 0.95

                    if client_update.meta['blocked']:
                        client_update.meta['p'] = 0
                    else:
                        client_update.meta['p'] = client_update.meta['num_samples'] * client_update.meta['score']
                        pT += client_update.meta['p']

            # Normalize clients weighting
            for client_update in self.client_updates:
                client_update.meta['p'] /= pT

            
            #Calculate epoch weight
            pT_epoch = 0.0
            for client_update in self.client_updates:
                if self._notBlockedNorBad(client_update):
                    client_update.meta['pEpoch'] = client_update.meta['num_samples'] * client_update.meta['score']
                    pT_epoch += client_update.meta['pEpoch']


            #Normalize epoch weight
            for client_update in self.client_updates:
                client_update.meta['pEpoch'] /= pT_epoch

            #Global Model Aggregation
            keys              = self.client_updates[0].state_dict.keys()
            global_model = {key: np.zeros_like(self.client_updates[0].state_dict[key], dtype=np.float64) for key in keys}

            for client_update in self.client_updates:
                if self._notBlockedNorBad(client_update):
                    for key in keys:
                        global_model[key] += client_update.meta['pEpoch'] * client_update.state_dict[key]
                
                if not client_update.meta['blocked']:
                    client_update.meta['badUpdate'] = False

        return global_model

    def get_history(self):
        return self.history

    def get_len(self):
        return len(self.get_history())
