#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N augment_job              
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

python simple_experiments.py