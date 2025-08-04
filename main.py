from nvflare.job_config.api import FedJob
from nvflare.app_opt.p2p.controllers import DistOptController
from nvflare.app_opt.p2p.executors import GTExecutor
from nvflare.app_opt.p2p.types import Config

import os 
import yaml

import torch
from torch import nn
import inspect
import argparse

from controllers.camelyon17_controller import Camelyon17Controller 
from configs import config_loading
from executors import executor_loading

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--workspace', type=str, default='run')
    parser.add_argument('-a', '--algo', type=str, default='FedAvg')
    parser.add_argument('--malclients', nargs='+', default=[], help='ids of malfunctioning clients')
    parser.add_argument('--malf', nargs='+', default = [], help='possible values: random, ana, sfa, artifacts')
    parser.add_argument('--prob', type=float, default=0.0)
    parser.add_argument('--clients', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='camelyon17')
    
     
    args = parser.parse_args()
    if args.dataset is None:
        dataset='camelyon17'
    else:
        dataset = args.dataset

    dataset = args.dataset
    n_clients = args.clients
    
    Executor = executor_loading(args.algo)


    job   = FedJob(name=f'{args.dataset}_{args.algo}')
    
    config     = config_loading(dataset)
    #config     = config_loading('camelyon17')
    controller = DistOptController(config=config)
    job.to_server(controller)
    # Add clients
    for i in range(n_clients):
        if str(i+1) in args.malclients: 
            executor = Executor(client_id=i, sync_timeout=30, dataset=dataset, malfunctions=args.malf, prob=args.prob) 
        else:
            executor = Executor(client_id=i, sync_timeout=30, dataset=dataset) 

        job.to(executor, f"site-{i+1}")
       
    # create the workflow directory to run the job later
    job.simulator_run(os.path.join(f"./LIGHTYEAR/jobs/{args.dataset}/p2p", args.workspace), gpu="0,1,2,3,4")
