#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N mnist_lenet_poison
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

python runner.py --experiment_config configs/mnist_lenet/poison.json --shared_config configs/mnist_lenet/shared.json