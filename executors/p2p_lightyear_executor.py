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

class P2PLIGHTYEARExecutor(P2PExecutor):
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
        #self.tau = 0.75
        self.tau = 0.6
        self.lambda_0 = 0.95   


    def _compute_ece(self, probs, labels, n_bins=15):
        confidences, predictions = probs.max(dim=1)
        accuracies = predictions.eq(labels)

        bins = torch.linspace(0, 1, n_bins + 1)
        ece = torch.zeros(1, device=probs.device)

        for i in range(n_bins):
            mask = (confidences > bins[i]) & (confidences <= bins[i+1])
            if mask.sum() > 0:
                bin_confidence = confidences[mask].mean()
                bin_accuracy = accuracies[mask].float().mean()
                bin_error = torch.abs(bin_confidence - bin_accuracy)
                ece += bin_error * mask.float().mean()

        return ece.item()

    def _sharpness(self, logits):
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy.mean()

    def _compute_scores(self, sd, n_bins=15):
        
        self.temp_model.load_state_dict(sd)
        self.temp_model.eval()
        
        correct   = 0.0
        total     = 0
       
        confidences = []
        predictions = []
        labels_list = []

        with torch.no_grad():
            for images, labels in self.val_ldr:
                images, labels = images.to(self.device), labels.to(self.device)
                 
                outputs        = self.temp_model(images)
                probs          = F.softmax(outputs, dim=1)
                confs, preds   = torch.max(outputs, dim=1)

                correct += preds.eq(labels).sum().item()
                total   += labels.size(0)

                confidences.extend(confs.cpu().numpy())
                predictions.extend(preds.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())

        accuracy = correct / total

        confidences = torch.tensor(confidences)
        predictions = torch.tensor(predictions)
        labels = torch.tensor(labels_list)

        bin_boundaries = torch.linspace(0, 1, steps=n_bins + 1)
        ece = torch.zeros(1)
        for i in range(n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]
            mask = (confidences > lower) & (confidences <= upper)
            if mask.sum() > 0:
                acc_in_bin = (predictions[mask] == labels[mask]).float().mean()
                conf_in_bin = confidences[mask].mean()
                bin_prob = mask.float().mean()
                ece += torch.abs(acc_in_bin - conf_in_bin) * bin_prob

        ece       = ece.item() 
        sharpness = confidences.mean().item()

        return accuracy, ece, sharpness


    def aggregate(self, local_model, selected_nbs, iteration):
        keys         = self.model.state_dict().keys()
        
        num_updates  = len(selected_nbs) + 1

        fedavg_model = OrderedDict()

        for key in keys:
            total = self.model.state_dict()[key].clone().float().to(self.device)
            for nb in selected_nbs:
                neighbor_value = nb[key]
                total += neighbor_value.float().to(self.device)

            fedavg_model[key] = total / num_updates

        lambda_t = (self.lambda_0 ** iteration)
        reg_global_model = {}

        for key in keys:
            reg_global_model[key] = self.model.state_dict()[key] + lambda_t * (fedavg_model[key] - self.model.state_dict()[key])
        

        return reg_global_model
        

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

       
            #Calculate reference values
            ref_acc, ref_ece, ref_sharpness = self._compute_scores(local_model)
            selected_nbs = []
            #Calculate ECE for client updates
            info = f'client-{self.client_id + 1} selected following clients: '
            for nb in self.neighbors:
                neighbor_model = self.neighbors_values[iteration][nb.id]
                nb_acc, nb_ece, nb_sharpness = self._compute_scores(neighbor_model)

                accuracy_agreement  = max(0, (1 - abs(ref_acc - nb_acc)))
                ece_agreement       = max(0, (1 - abs(ref_ece - nb_ece)))
                sharpness_agreement = max(0, (1 - abs(ref_sharpness - nb_sharpness)))

                agreement           = (accuracy_agreement + ece_agreement + sharpness_agreement) / 3
                self.log_info(fl_ctx, f"Agreement with site-{nb.id}: {agreement}")
                if agreement >= self.tau:
                    selected_nbs.append(neighbor_model)
                    info += f'+client-{nb.id} '
            
            self.log_info(fl_ctx, f"Number of neighbors: {len(self.neighbors)}")
            self.log_info(fl_ctx, f"Aggregation set: {info}")
            global_model = self.aggregate(local_model, selected_nbs, iteration)
            self.model.load_state_dict(global_model)
            torch.save(self.model.state_dict(), f"local_model_{self.client_name}.pt")
            # 4. free memory that's no longer needed
            if iteration in self.neighbors_values.keys():
                del self.neighbors_values[iteration]

            self._eval(fl_ctx, iteration)
        self._test(fl_ctx)
