from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob

from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import ScriptRunner

import os 
import yaml

import torchvision.models as torch_classifiers

from networks import model_loading

import torch
from torch import nn
import inspect
import argparse
from aggregation.robust_aggregation import Aggregator
from aggregation.fedavg import FedAvg

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--malclients', nargs='+', default=[],help='ids of malfunctioning clients')
    parser.add_argument('--malfs', nargs='+', help='possible values: random, ana, sfa')
    parser.add_argument('--scale', type=float, default=None)
    parser.add_argument('--prob', type=float, default=0.0)
    parser.add_argument('--workdir', type=str, default='workdir_camelyon')
    parser.add_argument('--algo', type=str, default='FedAvg')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--clients', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=12)
    parser.add_argument('--trainer', type=str, default='FedAvg')
    args = parser.parse_args()

    model = model_loading(args.dataset)
    
    job = BaseFedJob(name=f'{args.dataset}_{args.algo}', initial_model=model)
    num_clients  = args.clients
    num_rounds   = args.rounds
    train_script = 'training_scripts/train.py'

    #Create the controller for the server:
    
    if args.algo == 'FedAvg':
        controller = FedAvg(
            num_clients  = num_clients,
            num_rounds   = num_rounds,
            persistor_id = job.comp_ids["persistor_id"]
        )
    else: 
        controller = Aggregator(
            num_clients  = num_clients,
            num_rounds   = num_rounds,
            persistor_id = job.comp_ids["persistor_id"],
            algo         = args.algo,
                )
        
    job.to_server(controller)

    # Add clients
    for i in range(num_clients):
        
        if str(i+1) in args.malclients:
            arg_str  = f"--malfs {' '.join(args.malfs)} --prob {args.prob} --clientid {i} --dataset {args.dataset} --trainer {args.trainer}"
            if args.scale is not None:
                arg_str += f" --scale {args.scale}"
            executor = ScriptRunner(script=train_script, script_args=arg_str)
        else:
            arg_str  = f"--clientid {i} --dataset {args.dataset} --trainer {args.trainer}"
            executor = ScriptRunner(script=train_script, script_args=arg_str)
        
        #add if clause for encryption
        job.to(executor, f"client-{i+1}")
       
    job.simulator_run(os.path.join(f"./LIGHTYEAR/jobs/{args.dataset}/fl/", args.workdir), gpu="0,1,2")
