from datasets import data_loading
from training import trainer_loading

import nvflare.client as flare

from training.malf_update import Malfunction
import random
import torch
import os
import yaml

import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set the parameters for malfunctioning updates')
    parser.add_argument('--malfs', type=str, nargs='+', default=[])
    parser.add_argument('--scale' ,type=float, default=None)
    parser.add_argument('--prob', type=float, default=0.0)
    parser.add_argument('--clientid' ,type=int, default=0)
    parser.add_argument('--dataset' ,type=str, default='camelyon17')
    parser.add_argument('--trainer' ,type=str, default='FedAvg')
     
    args = parser.parse_args()
    
  
    flare.init()

    sys_info    = flare.system_info()
    client_name = sys_info['site_name']
    train_ldr, val_ldr, test_ldr = data_loading(args.clientid, args.dataset)
    Trainer      = trainer_loading(args.trainer)
    trainer      = Trainer(train_ldr, val_ldr, test_ldr, args.dataset)
    local_epochs = 1
    corrupt      = Malfunction(args.dataset, args.scale)
    temp_global_model = None

    while flare.is_running():
        global_model = flare.receive()
        trainer.load_model(global_model.params)

        val_acc = trainer.eval()
        print(f'{client_name} | val acc: {val_acc}')
         
        attack = False
        if random.random() < args.prob:
            print(f'Client-{client_name} is comitting a malfunctioning update')
            malf = random.choice(args.malfs)
            attack = True

        print('Training starts')
        for epoch in range(local_epochs):
            trainer.train_epoch()

        if attack:
            model_update = corrupt(trainer.model.cpu().state_dict(), malf)
        else:
            model_update = trainer.model.cpu().state_dict()
        
        local_model = flare.FLModel(
                params = model_update,
                meta   = {'NUM_STEPS_CURRENT_ROUND': local_epochs * len(train_ldr),
                          'CLIENT_NAME'            : client_name,
                          'num_samples'            : len(train_ldr)})

        
        flare.send(local_model)

    
