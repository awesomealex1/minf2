#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N cifar100_wide16_poison
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

python runner.py --experiment_config configs/cifar100_wide16/poison.json --shared_config configs/cifar100_wide16/shared.json