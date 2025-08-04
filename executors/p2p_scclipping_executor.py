import time
import copy


from collections import OrderedDict
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from scipy.stats import norm

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.apis.dxo import from_shareable

from nvflare.app_common.abstract.fl_model import FLModel

from nvflare.app_opt.p2p.utils.metrics import compute_loss_over_dataset
from nvflare.app_opt.p2p.utils.utils import get_device

from executors.p2p_executor import P2PExecutor

class P2PSSClippingExecutor(P2PExecutor):
    """An executor that implements Stochastic Gradient Tracking (GT) in a peer-to-peer (P2P) learning setup.

    Each client maintains its own local model and synchronously exchanges model parameters with its neighbors
    at each iteration. The model parameters are updated based on the neighbors' parameters and local gradient descent steps.
    The executor also tracks and records training, validation and test losses over time.

    The number of iterations and the learning rate must be provided by the controller when asing to run the algorithm.
    They can be set in the extra parameters of the controller's config with the "iterations" and "stepsize" keys.

    Note:
        Subclasses must implement the __init__ method to initialize the model, loss function, and data loaders.

    Args:
        model (torch.nn.Module, optional): The neural network model used for training.
        loss (torch.nn.modules.loss._Loss, optional): The loss function used for training.
        train_dataloader (torch.utils.data.DataLoader, optional): DataLoader for the training dataset.
        test_dataloader (torch.utils.data.DataLoader, optional): DataLoader for the testing dataset.
        val_dataloader (torch.utils.data.DataLoader, optional): DataLoader for the validation dataset.

    Attributes:
        model (torch.nn.Module): The neural network model.
        loss (torch.nn.modules.loss._Loss): The loss function.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for testing data.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        train_loss_sequence (list[tuple]): Records of training loss over time.
        test_loss_sequence (list[tuple]): Records of testing loss over time.
    """

    def __init__(
        self, 
        client_id:int,
        sync_timeout:int,
        dataset:str,
        malfunctions=None,
        prob = 0.0,
    ):
        super().__init__(client_id, sync_timeout, dataset, malfunctions, prob)
        
   
    def get_state_dict_oneD(self, state_dict):
        components = []
        for param in state_dict:
            components.append(state_dict[param])

        vec = torch.cat([component.flatten() for component in components])

        #return vec.reshape(1, -1)
        return vec

    def oneD_to_state_dict(self, vector):
        pointer = 0
        new_state_dict = {}
        for key, param in self.model.state_dict().items():
            # Determine the number of elements in this parameter
            numel = param.numel()
            # Extract the corresponding values from the vector
            new_param = vector[pointer:pointer + numel].view(param.size())
            new_state_dict[key] = new_param
            # Update the pointer
            pointer += numel
        return new_state_dict

    def clip(self,  sd, tau):
        v = self.get_state_dict_oneD(sd)
        v_norm = torch.norm(v)
        scale = min(1, tau / v_norm)
        if torch.isnan(v_norm):
            return 0
        v = v * scale
        sd = self.oneD_to_state_dict(v)
        return sd
        
    def aggregate(self, iteration, tau):
        keys         = self.model.state_dict().keys()
        selected_updates = []
        
        for nb in self.neighbors:
            selected_updates.append(self.clip(self.neighbors_values[iteration][nb.id], tau))

        num_updates  = len(selected_updates) + 1

        global_model = OrderedDict()

        for key in keys:
            total = self.model.state_dict()[key].clone().float().to(self.device) 
            for nb in selected_updates:
                neighbor_value = nb[key]
                total +=  neighbor_value.float().to(self.device)
            global_model[key] = total / num_updates
        return global_model
    
    def _get_tau(self, iteration):
        local_vector = self.get_state_dict_oneD(self.model.state_dict()).cpu()
        distances = []
        for nb in self.neighbors:
            neighbor_sd  = self.neighbors_values[iteration][nb.id]
            neighbor_vec = self.get_state_dict_oneD(neighbor_sd).cpu()
            distances.append((neighbor_vec - local_vector).norm())
        
        if len(distances) >= 2:
            tau = sorted(distances)[-2]
        else:
            tau = distances[-1]
        return tau

    def run_algorithm(self, fl_ctx, shareable, abort_signal):
        
        start_time = time.time()
        print(f'site-{self.client_id+1} | GPU: {self.device}')
        
        # Iterate through all training rounds
        for iteration in range(self._iterations):
            self.log_info(fl_ctx, f"Round: {iteration}/{self._iterations}")
            print(f'Round: {iteration+1}/{self._iterations}') 
            if abort_signal.triggered:
                break

            #Trainings Loop
            self._train_epoch()
            #self._train_batch()

            #exchange updates
            local_model     = copy.deepcopy(self.model.state_dict())
            self._exchange_values(fl_ctx, value=local_model, iteration=iteration)
    
            #Wait until all updates are received
            while len(self.neighbors_values[iteration]) < len(self.neighbors):
                print(f'site-{self.client_id+1} is waiting for updates')
                time.sleep(30)

            #select neighbors
            tau = self._get_tau(iteration)
            global_model = self.aggregate(iteration, tau)
            self.model.load_state_dict(global_model)
            torch.save(self.model.state_dict(), f"local_model_{self.client_name}.pt")
            # 4. free memory that's no longer needed
            if iteration in self.neighbors_values.keys():
                del self.neighbors_values[iteration]

            self._eval(fl_ctx, iteration)
        self._test(fl_ctx)
