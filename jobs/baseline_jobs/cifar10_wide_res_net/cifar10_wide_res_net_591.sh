#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N cifar10_wide_res_net_591            
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

python runner.py --model wide_res_net --dataset cifar10 --mode train_normal --experiment_name cifar10_wide_res_net_591 --seed 591