#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N fmnist_res_net_18_441_augment            
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

python runner.py --model res_net_18 --dataset fmnist --mode augment --experiment_name fmnist_res_net_18_441_augment --seed 441 --epsilon 0.2