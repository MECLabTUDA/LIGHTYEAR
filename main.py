from nvflare.job_config.api import FedJob
from nvflare.app_opt.p2p.controllers import DistOptController

import os 
import argparse

from configs import config_loading
from executors import executor_loading

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--workspace', type=str, default='run')
    parser.add_argument('-a', '--algo', type=str, default='FedAvg')
    parser.add_argument('--malclients', nargs='+', default=[], help='ids of malfunctioning clients')
    parser.add_argument('--malf', nargs='+', default = [], help='possible values: random, ana, sfa, artifacts')
    parser.add_argument('--prob', type=float, default=1.0)
    parser.add_argument('--clients', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='camelyon17')
    parser.add_argument('--gpus', type=int)
    
    args = parser.parse_args()
    
    dataset = args.dataset
    n_clients = args.clients
    gpu_list = ",".join(str(i) for i in range(args.gpus))

    Executor = executor_loading(args.algo)

    job   = FedJob(name=f'{args.dataset}_{args.algo}')
    
    config     = config_loading(dataset)
    controller = DistOptController(config=config)
    job.to_server(controller)

    # Add clients
    for i in range(n_clients):
        if str(i+1) in args.malclients: 
            executor = Executor(client_id=i, sync_timeout=30, dataset=dataset, malfunctions=args.malf, prob=args.prob) 
        else:
            executor = Executor(client_id=i, sync_timeout=30, dataset=dataset) 

        job.to(executor, f"site-{i+1}")
       
    job.simulator_run(os.path.join(f"jobs/{args.dataset}/p2p", args.workspace), gpu=gpu_list)
