#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N fmnist_res_net_18_poison_tune
#$ -cwd
#$ -l h_rt=24:00:00 
#$ -l h_vmem=100G
#$ -q gpu 
#$ -pe gpu-a100 1
. /etc/profile.d/modules.sh

# Load Python
module load anaconda
conda activate minf2

cd /exports/eddie/scratch/s2017377/minf2

python runner.py --experiment_config configs/fmnist_res_net_18/poison.json --shared_config configs/fmnist_res_net_18/shared.json --hp_config configs/fmnist_res_net_18/hp/poison11.json &
python runner.py --experiment_config configs/fmnist_res_net_18/poison.json --shared_config configs/fmnist_res_net_18/shared.json --hp_config configs/fmnist_res_net_18/hp/poison12.json &
python runner.py --experiment_config configs/fmnist_res_net_18/poison.json --shared_config configs/fmnist_res_net_18/shared.json --hp_config configs/fmnist_res_net_18/hp/poison13.json &
python runner.py --experiment_config configs/fmnist_res_net_18/poison.json --shared_config configs/fmnist_res_net_18/shared.json --hp_config configs/fmnist_res_net_18/hp/poison21.json &
python runner.py --experiment_config configs/fmnist_res_net_18/poison.json --shared_config configs/fmnist_res_net_18/shared.json --hp_config configs/fmnist_res_net_18/hp/poison22.json &
python runner.py --experiment_config configs/fmnist_res_net_18/poison.json --shared_config configs/fmnist_res_net_18/shared.json --hp_config configs/fmnist_res_net_18/hp/poison23.json &
python runner.py --experiment_config configs/fmnist_res_net_18/poison.json --shared_config configs/fmnist_res_net_18/shared.json --hp_config configs/fmnist_res_net_18/hp/poison31.json &
python runner.py --experiment_config configs/fmnist_res_net_18/poison.json --shared_config configs/fmnist_res_net_18/shared.json --hp_config configs/fmnist_res_net_18/hp/poison32.json &
python runner.py --experiment_config configs/fmnist_res_net_18/poison.json --shared_config configs/fmnist_res_net_18/shared.json --hp_config configs/fmnist_res_net_18/hp/poison33.json &
python runner.py --experiment_config configs/fmnist_res_net_18/poison.json --shared_config configs/fmnist_res_net_18/shared.json --hp_config configs/fmnist_res_net_18/hp/poison41.json &
python runner.py --experiment_config configs/fmnist_res_net_18/poison.json --shared_config configs/fmnist_res_net_18/shared.json --hp_config configs/fmnist_res_net_18/hp/poison42.json &
python runner.py --experiment_config configs/fmnist_res_net_18/poison.json --shared_config configs/fmnist_res_net_18/shared.json --hp_config configs/fmnist_res_net_18/hp/poison43.json &