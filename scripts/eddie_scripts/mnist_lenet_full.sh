#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N mnist_lenet_full
#$ -cwd
#$ -l h_rt=24:00:00 
#$ -l h_vmem=40G
#$ -q gpu 
#$ -l gpu-mig=1
. /etc/profile.d/modules.sh

# Load Python
module load anaconda
conda activate minf2

cd /exports/eddie/scratch/s2017377/minf2

python scripts/main.py experiment=mnist/lenet/full random_seed=83873