#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N fmnist_res_net_18_baseline_hp2
#$ -cwd
#$ -l h_rt=48:00:00 
#$ -l h_vmem=40G
#$ -q gpu 
#$ -pe gpu-a100 1
. /etc/profile.d/modules.sh

# Load Python
module load anaconda
conda activate minf2

cd /exports/eddie/scratch/s2017377/minf2

python runner.py --experiment_config configs/fmnist_res_net_18/baseline.json --shared_config configs/fmnist_res_net_18/shared.json --hp_config configs/fmnist_res_net_18/hp/baseline2.json