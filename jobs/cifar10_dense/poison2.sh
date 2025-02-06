#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N cifar10_dense_poison2
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

python runner.py --experiment_config configs/cifar10_dense2/poison.json --shared_config configs/cifar10_dense/shared.json