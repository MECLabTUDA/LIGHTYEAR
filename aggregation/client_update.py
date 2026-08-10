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


class MirkoClientUpdate(ClientUpdate):
    def __init__(self, state_dict, client_name, training_round=None, meta=None, vectorized=True, normalized_vectors=True):
        super().__init__(state_dict, client_name, vectorized=True)
        self.projected_vector = self.batchwise_random_projection(self.state_dict_oneD.flatten())

    def batchwise_random_projection(self, vector, k=1576, block_size=100000):
        """
        Apply a random projection to a high-dimensional vector blockwise.
        
        Parameters:
        - vector: Input vector of shape (d,), where d is the original dimension.
        - k: Target dimension for the projection.
        - block_size: Size of each block to process.
        
        Returns:
        - Projected vector of shape (k,).
        """
         
        vector = torch.Tensor(vector).to('cuda:0')
        d = len(vector)  # Original dimension of the vector
        #n_blocks = int(np.ceil(d / block_size))  # Number of blocks
        n_blocks = (d + block_size - 1) // block_size
        # Initialize the projected vector
        projected_vector = torch.zeros(k).to('cuda:0')

        
        # Process blocks
        for i in range(n_blocks):
            # Determine the current block range
            block_start = i * block_size
            block_end = min((i + 1) * block_size, d)
            
            # Extract the current block
            block = vector[block_start:block_end]
            
            # Create a random projection matrix for this block
            block_proj = torch.randn(block_end - block_start, k) / torch.sqrt(torch.Tensor(k))
            block_proj = block_proj.to('cuda:0') 
            # Project the block and accumulate the results
            projected_vector += torch.matmul(block, block_proj)  # Dot product of projection matrix and block
       
        projected_vector = projected_vector.cpu()
        projected_vector = projected_vector.view(1, -1)

        return projected_vector


