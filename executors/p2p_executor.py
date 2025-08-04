import time
import copy

import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import random
from collections import OrderedDict

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.apis.dxo import from_shareable

from nvflare.app_common.abstract.fl_model import FLModel

from nvflare.app_opt.p2p.executors.sync_executor import SyncAlgorithmExecutor
from nvflare.app_opt.p2p.utils.metrics import compute_loss_over_dataset
from nvflare.app_opt.p2p.utils.utils import get_device

from networks import model_loading, optimizer_loading
from datasets import data_loading

class P2PExecutor(SyncAlgorithmExecutor):
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
        super().__init__(sync_timeout)
        #self.device       = get_device()
        self.device       = 'cuda:0'
        self.model        = model_loading(dataset).to(self.device)
        self.temp_model   = copy.deepcopy(self.model).to(self.device) 
        self.client_id    = client_id
        self.dataset      = dataset
        self.train_ldr, self.val_ldr, self.test_ldr = data_loading(self.client_id, self.dataset)
        self.loss_fn      = nn.CrossEntropyLoss()  
        self.optimizer    = optimizer_loading(dataset, self.model)
        self.malfunctions = malfunctions
        self.prob         = prob

    def _train_epoch(self):
        self.model.train()
        print(f'Train Epoch of site-{self.client_id+1} | Samples: {len(self.train_ldr)}')
        for img, label in self.train_ldr:
            img, label = img.to(self.device), label.to(self.device)

            pred = self.model(img)
            loss = self.loss_fn(pred, label)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


    def _train_batch(self):
        self.model.train()
        img, label = next(iter(self.train_ldr))
        img, label = img.to(self.device), label.to(self.device)
        pred = self.model(img)
        loss = self.loss_fn(pred, label)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()



    def _sfa(self, sd):
        scale = -1
        for key in sd:
            sd[key] = scale * sd[key]
        return sd

    def _ana(self, sd, scale=1.5):

        if self.dataset == 'camelyon17':
            scale = 1.5
        else:
            scale = 120.5
        
        for key in sd:
            noise       = (np.random.randn(*sd[key].shape) * scale / 100.0 * sd[key])
            sd[key] = noise + sd[key]
        return sd

    def _random(self, sd):
        model = model_loading(self.dataset)
        sd    = model.state_dict()
        sd    = {key: value.detach().cpu().numpy().astype(np.float32) for key, value in sd.items()}
        #for key in sd:
        #    sd[key] = np.array(np.random.randn(*sd[key].shape)).astype(sd[key].dtype)
        return sd

    def _eval(self, fl_ctx, iteration):
        self.model.eval()
        self.model.to(self.device)
        correct = 0
        total   = 0
        
        with torch.no_grad():
            for img, label in self.val_ldr:
                img, label = img.to(self.device), label.to(self.device)
                output  = self.model(img)
                _, pred = torch.max(output, 1)
                correct += pred.eq(label).sum().item()
                total   += label.size(0)

        accuracy = correct / total
        self.log_info(fl_ctx, f'site-{self.client_id + 1} | Val Acc: {accuracy} | iteration: {iteration}')

    def _test(self, fl_ctx):
        self.model.eval()
        self.model.to(self.device)
        correct = 0
        total   = 0
        
        with torch.no_grad():
            for img, label in self.test_ldr:
                img, label = img.to(self.device), label.to(self.device)
                output  = self.model(img)
                _, pred = torch.max(output, 1)
                correct += pred.eq(label).sum().item()
                total   += label.size(0)

        accuracy = correct / total
        self.log_info(fl_ctx, f'site-{self.client_id + 1} | Test Acc: {accuracy}')
  
    def aggregate(self, local_model, iteration):
        keys         = local_model.keys()
        num_updates = len(self.neighbors) + 1

        global_model = OrderedDict()

        for key in keys:
            total = self.model.state_dict()[key].clone().float().to(self.device)
            for nb in self.neighbors:
                neighbor_value = self.neighbors_values[iteration][nb.id][key].clone()
                total += neighbor_value.float().to(self.device)

            global_model[key] = total / num_updates
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

            #exchange updates
            local_model     = copy.deepcopy(self.model.state_dict())
            self._exchange_values(fl_ctx, value=local_model, iteration=iteration)
            
            while len(self.neighbors_values[iteration]) < len(self.neighbors):
                print(f'site-{self.client_id+1} is waiting for updates')
                time.sleep(30)

            global_model = self.aggregate(local_model, iteration)
            self.model.load_state_dict(global_model)

            # 4. free memory that's no longer needed
            if iteration in self.neighbors_values.keys():
                del self.neighbors_values[iteration]

            self._eval(fl_ctx, iteration)
        self._test(fl_ctx)

    def _to_message(self, x): 
        x = {key: value.detach().cpu().numpy().astype(np.float32) for key, value in x.items()}
        if random.random() < self.prob:
            malfunction = random.choice(self.malfunctions)

            match malfunction:
                case 'ana':
                    x = self._ana(x)
                case 'sfa':
                    x = self._sfa(x)
                case 'random':
                    x = self._random(x)
                    
        return {
            "parameters": x,
        }

    def _from_message(self, x):
        x = x["parameters"]
        x = {
            key:
            torch.tensor(value, dtype=torch.float32)
            if isinstance(value, np.generic)
            else torch.from_numpy(value).float() for key, value in x.items()}

        return x

    def _pre_algorithm_run(self, fl_ctx, shareable, abort_signal):
        data = from_shareable(shareable).data
        self._iterations = data["iterations"]

    def _post_algorithm_run(self, *args, **kwargs):
        torch.save(self.model.state_dict(), f"local_model_{self.client_name}.pt")


    def _exchange_values(self, fl_ctx: FLContext, value: any, iteration: int):
        """Exchanges values with neighbors synchronously.

        Sends the local value to all neighbors and waits for their values for the current iteration.
        Utilizes threading events to synchronize the exchange and ensure all values are received
        before proceeding.

        Args:
            fl_ctx (FLContext): Federated learning context.
            value (any): The local value to send to neighbors.
            iteration (int): The current iteration number of the algorithm.

        Raises:
            SystemExit: If the values from all neighbors are not received within the timeout.
        """
        engine = fl_ctx.get_engine()
        
        # Clear the event before starting the exchange
        self.sync_waiter.clear()

        _ = engine.send_aux_request(
            targets=[neighbor.id for neighbor in self.neighbors],
            topic="send_value",
            request=DXO(
                data_kind=DataKind.FL_MODEL,
                data={
                    "value": self._to_message(value),
                    "iteration": iteration,
                },
            ).to_shareable(),
            timeout=600,
            fl_ctx=fl_ctx,
        )

