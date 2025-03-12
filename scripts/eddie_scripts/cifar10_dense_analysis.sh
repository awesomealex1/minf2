#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N cifar10_dense_analysis
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

python scripts/main.py experiment=cifar10/dense_net_40/analyze_sharpness model.weights_path=poison_weights.pt task.analysis_configs.comparison_weights_path=baseline_weights.pt