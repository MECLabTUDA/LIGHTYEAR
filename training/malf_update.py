import torch
import torch.nn as nn
import random
import numpy as np
import copy
from networks import model_loading

class Malfunction:
    def __init__(self, dataset, scale=None):
        self.dataset = dataset
        if scale is None:
            if dataset == 'camelyon17':
                print("ANA initialized for camelyon")
                self.scale = 1.5
            else:
                print("ANA initialized for femnist")
                self.scale = 120.5
        else:
            self.scale = scale

    def _ana(self, update):
        malf_update = copy.deepcopy(update)
        for key in update:
            noise       = torch.randn(update[key].size()) * self.scale / 100.0 * malf_update[key]
            malf_update[key] = noise + malf_update[key].float()
        return malf_update

    def _sfa(self, update):
        malf_update = copy.deepcopy(update)
        scale = -1
        for key in update:
            malf_update[key] = scale * malf_update[key].float()
        return malf_update

    def _random(self, update):
        model = model_loading(self.dataset)
        malf_update = model.state_dict()

        return malf_update

    def _corrupt(self, update, malfunction):
        if malfunction == 'ana':
            return self._ana(update)
        elif malfunction == 'sfa':
            return self._sfa(update)
        elif malfunction == 'random':
            return self._random(update)


    def __call__(self, update, malfunction):
        corrupted_update = self._corrupt(update, malfunction)
        return corrupted_update
