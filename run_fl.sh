#!/bin/bash -l

#SBATCH --mail-user=mirko.konstantin@gris.informatik.tu-darmstadt.de
#SBATCH -J FLareFL
#SBATCH -n 20
#SBATCH -c 8
#SBATCH --mem-per-cpu=2000
#SBATCH --gres=gpu:5
#SBATCH -t 300:00:10

#SBATCH -o /gris/gris-f/homestud/mikonsta/flare/p2p/logs/FL_train.log
#SBATCH -e /gris/gris-f/homestud/mikonsta/flare/p2p/logs/FL_err.err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#Femnist clean
#python3 main_fl.py --workdir fedavg_clean --dataset femnist --clients 10 --algo FedAvg
#python3 main_fl.py --workdir ASMR_clean --dataset femnist --clients 10 --algo ASMR
#python3 main_fl.py --workdir CFL_clean --dataset femnist --clients 10 --algo CFL
#python3 main_fl.py --workdir RFA_clean --dataset femnist --clients 10 --algo RFA
#python3 main_fl.py --workdir AFA_clean --dataset femnist --clients 10 --algo AFA

#Femnist - mal - fix
#python3 main_fl.py --workdir fedavg_mal_fix --dataset femnist --clients 10 --malclients 1 3 7 --malfs ana sfa random --prob 1.0 --algo FedAvg
#python3 main_fl.py --workdir ASMR_mal_fix --dataset femnist --clients 10 --malclients 1 3 7 --malfs ana sfa random --prob 1.0 --algo ASMR
#python3 main_fl.py --workdir CFL_mal_fix --dataset femnist --clients 10 --malclients 1 3 7 --malfs ana sfa random --prob 1.0 --algo CFL
#python3 main_fl.py --workdir RFA_mal_fix --dataset femnist --clients 10 --malclients 1 3 7 --malfs ana sfa random --prob 1.0 --algo RFA
#python3 main_fl.py --workdir AFA_mal_fix --dataset femnist --clients 10 --malclients 1 3 7 --malfs ana sfa random --prob 1.0 --algo AFA

#Femnist - mal - fix
#python3 main_fl.py --workdir fedavg_mal_dyn --dataset femnist --clients 10 --malclients 1 3 5 7 --malfs ana sfa random --prob 0.75 --algo FedAvg
#python3 main_fl.py --workdir ASMR_mal_dyn --dataset femnist --clients 10 --malclients 1 3 5 7 --malfs ana sfa random --prob 0.75 --algo ASMR
#python3 main_fl.py --workdir CFL_mal_dyn --dataset femnist --clients 10 --malclients 1 3 5 7 --malfs ana sfa random --prob 0.75 --algo CFL
#python3 main_fl.py --workdir RFA_mal_dyn --dataset femnist --clients 10 --malclients 1 3 5 7 --malfs ana sfa random --prob 0.75 --algo RFA
#python3 main_fl.py --workdir AFA_mal_dyn --dataset femnist --clients 10 --malclients 1 3 5 7 --malfs ana sfa random --prob 0.75 --algo AFA



#Celeba clean
#python3 main_fl.py --workdir fedavg_clean --dataset celeba --clients 10 --algo FedAvg
#python3 main_fl.py --workdir ASMR_clean --dataset celeba --clients 10 --algo ASMR
#python3 main_fl.py --workdir CFL_clean --dataset celeba --clients 10 --algo CFL
#python3 main_fl.py --workdir RFA_clean --dataset celeba --clients 10 --algo RFA
#python3 main_fl.py --workdir AFA_clean --dataset celeba --clients 10 --algo AFA

#Celeba - mal - fix
#python3 main_fl.py --workdir fedavg_mal_fix --dataset celeba --clients 10 --malclients 1 3 7 --malfs ana sfa random --prob 1.0 --algo FedAvg
#python3 main_fl.py --workdir ASMR_mal_fix --dataset celeba --clients 10 --malclients 1 3 7 --malfs ana sfa random --prob 1.0 --algo ASMR
#python3 main_fl.py --workdir CFL_mal_fix --dataset celeba --clients 10 --malclients 1 3 7 --malfs ana sfa random --prob 1.0 --algo CFL
#python3 main_fl.py --workdir RFA_mal_fix --dataset celeba --clients 10 --malclients 1 3 7 --malfs ana sfa random --prob 1.0 --algo RFA
#python3 main_fl.py --workdir AFA_mal_fix --dataset celeba --clients 10 --malclients 1 3 7 --malfs ana sfa random --prob 1.0 --algo AFA

#Celeba - mal - fix
#python3 main_fl.py --workdir fedavg_mal_dyn --dataset celeba --clients 10 --malclients 1 3 5 7 --malfs ana sfa random --prob 0.75 --algo FedAvg
#python3 main_fl.py --workdir ASMR_mal_dyn --dataset celeba --clients 10 --malclients 1 3 5 7 --malfs ana sfa random --prob 0.75 --algo ASMR
#python3 main_fl.py --workdir CFL_mal_dyn --dataset celeba --clients 10 --malclients 1 3 5 7 --malfs ana sfa random --prob 0.75 --algo CFL
#python3 main_fl.py --workdir RFA_mal_dyn --dataset celeba --clients 10 --malclients 1 3 5 7 --malfs ana sfa random --prob 0.75 --algo RFA
#python3 main_fl.py --workdir AFA_mal_dyn --dataset celeba --clients 10 --malclients 1 3 5 7 --malfs ana sfa random --prob 0.75 --algo AFA


#Camelyon17 clean
python3 main_fl.py --workdir fedavg_clean --dataset camelyon17 --clients 10 --algo FedAvg
python3 main_fl.py --workdir ASMR_clean --dataset camelyon17 --clients 10 --algo ASMR
python3 main_fl.py --workdir CFL_clean --dataset camelyon17 --clients 10 --algo CFL
python3 main_fl.py --workdir RFA_clean --dataset camelyon17 --clients 10 --algo RFA
python3 main_fl.py --workdir AFA_clean --dataset camelyon17 --clients 10 --algo AFA

#Camelyon17 - mal - fix
python3 main_fl.py --workdir fedavg_mal_fix --dataset camelyon17 --clients 5 --malclients 1 3  --malfs ana sfa random --prob 1.0 --algo FedAvg
python3 main_fl.py --workdir ASMR_mal_fix --dataset camelyon17 --clients 5 --malclients 1 3  --malfs ana sfa random --prob 1.0 --algo ASMR
python3 main_fl.py --workdir CFL_mal_fix --dataset camelyon17 --clients 5 --malclients 1 3  --malfs ana sfa random --prob 1.0 --algo CFL
python3 main_fl.py --workdir RFA_mal_fix --dataset camelyon17 --clients 5 --malclients 1 3  --malfs ana sfa random --prob 1.0 --algo RFA
python3 main_fl.py --workdir AFA_mal_fix --dataset camelyon17 --clients 5 --malclients 1 3  --malfs ana sfa random --prob 1.0 --algo AFA

#Camelyon17 - mal - fix
python3 main_fl.py --workdir fedavg_mal_dyn --dataset camelyon17 --clients 5 --malclients 1 3 5  --malfs ana sfa random --prob 0.75 --algo FedAvg
python3 main_fl.py --workdir ASMR_mal_dyn --dataset camelyon17 --clients 5 --malclients 1 3 5  --malfs ana sfa random --prob 0.75 --algo ASMR
python3 main_fl.py --workdir CFL_mal_dyn --dataset camelyon17 --clients 5 --malclients 1 3 5  --malfs ana sfa random --prob 0.75 --algo CFL
python3 main_fl.py --workdir RFA_mal_dyn --dataset camelyon17 --clients 5 --malclients 1 3 5  --malfs ana sfa random --prob 0.75 --algo RFA
python3 main_fl.py --workdir AFA_mal_dyn --dataset camelyon17 --clients 5 --malclients 1 3 5  --malfs ana sfa random --prob 0.75 --algo AFA


