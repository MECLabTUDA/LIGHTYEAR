import time
import copy
import math

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

class P2PBALANCEExecutor(P2PExecutor):
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
        
    
    def _compute_norm_difference_constraint(self, w_i, w_j, lambda_t, gamma=0.3, kappa=1):
        """
        Check if the norm difference between two weight dicts satisfies the constraint:
        ||w_i - w_j|| <= gamma * exp(-kappa * lambda_t) * ||w_i||

        Args:
            w_i (dict): state_dict of model i at t+1/u - ref model
            w_j (dict): state_dict of model j at t+1/2
            gamma (float): scalar gamma
            kappa (float): scalar kappa
            lambda_t (float): value of lambda(t)

        Returns:
            bool: True if constraint is satisfied, False otherwise
        """
        # Flatten weights
        vec_i = torch.cat([p.view(-1).cpu() for p in w_i.values()])
        vec_j = torch.cat([p.view(-1).cpu() for p in w_j.values()])

        # Compute norms
        diff_norm = torch.norm(vec_i - vec_j)
        norm_i = torch.norm(vec_i)

        # Compute RHS of inequality
        rhs = gamma * math.exp(-kappa * lambda_t) * norm_i

        return diff_norm <= rhs


    def aggregate(self, selected_nbs, iteration, alpha=0.5):
        keys         = self.model.state_dict().keys()
        
        num_updates  = len(selected_nbs) + 1

        global_model = OrderedDict()

        for key in keys:
            total = self.model.state_dict()[key].clone().float().to(self.device) * alpha
            for nb in selected_nbs:
                neighbor_value = nb[key]
                total += (1 - alpha) * (1 / num_updates) * neighbor_value.float().to(self.device)
            global_model[key] = total
        return global_model
        

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
            ref_sd = self.model.state_dict()
            selected_neighbors = []
            for nb in self.neighbors:
                neighbor_sd = self.neighbors_values[iteration][nb.id]
                lambda_t    = (iteration+1) / self._iterations
                criterion   = self._compute_norm_difference_constraint(ref_sd, neighbor_sd, lambda_t)
                if criterion:
                    selected_neighbors.append(neighbor_sd)

            global_model = self.aggregate(selected_neighbors, iteration)
            self.model.load_state_dict(global_model)
            torch.save(self.model.state_dict(), f"local_model_{self.client_name}.pt")
            # 4. free memory that's no longer needed
            if iteration in self.neighbors_values.keys():
                del self.neighbors_values[iteration]

            self._eval(fl_ctx, iteration)
        self._test(fl_ctx)
