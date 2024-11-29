#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N fmnist_res_net_18_412_tune_baseline            
#$ -cwd
#$ -l h_rt=24:00:00 
#$ -l h_vmem=40G
#$ -q gpu 
#$ -pe gpu-a100 1
. /etc/profile.d/modules.sh

# Load Python
module load anaconda
conda activate minf2

cd /exports/eddie/scratch/s2017377/minf2

python runner.py --model res_net_18 --dataset fmnist --mode train_normal --experiment_name fmnist_res_net_18_412_tune_baseline --seed 412 --epochs 100 --hp_config_path configs/fmnist_res_net_18_baseline_hyperparams.json