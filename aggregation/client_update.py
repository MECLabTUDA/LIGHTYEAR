import numpy as np
import torch

class ClientUpdate:
    def __init__(self, state_dict, client_name, training_round=None, meta=None, vectorized=False, normalized_vectors=True):
        self.state_dict  = state_dict
        self.client_name = client_name
        self.normalized_vectors = normalized_vectors
        if meta is not None:
            self.meta = meta
        if vectorized:
            self.state_dict_oneD = self.get_state_dict_oneD(state_dict)
        else:
            self.state_dict_oneD = None

    def get_state_dict_oneD(self, state_dict):
        components = []
        for param in state_dict:
            components.append(state_dict[param])

        vec = np.concatenate([component.flatten() for component in components])
        
        if self.normalized_vectors:
            vec /= np.linalg.norm(vec)
        return vec.reshape(1, -1)

    def update_state_dict(self, state_dict, vectorized=False):
        self.state_dict = state_dict
        self.state_dict_oneD = self.get_state_dict_oneD(state_dict)

    def oneD_to_state_dict(self, vector):
        pointer = 0
        new_state_dict = {}
        for key, param in self.state_dict.items():
            # Determine the number of elements in this parameter
            numel = param.numel()
            # Extract the corresponding values from the vector
            new_param = vector[pointer:pointer + numel].view(param.size())
            new_state_dict[key] = new_param
            # Update the pointer
            pointer += numel
        self.state_dict = new_state_dict



