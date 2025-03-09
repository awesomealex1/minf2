#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N cifar10_dense_create_train
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

python scripts/main.py experiment=cifar10/dense_net_40/create_train_poison random_seed=505631 task.poison_configs.poison_start=80 task.epochs=100 task.poison_configs.iterations=5 task.poison_configs.poison_lr=0.01
#python scripts/main.py experiment=cifar10/dense_net_40/create_train_poison random_seed=270005&
#python scripts/main.py experiment=cifar10/dense_net_40/create_train_poison random_seed=227703&
#python scripts/main.py experiment=cifar10/dense_net_40/create_train_poison random_seed=891974&
#python scripts/main.py experiment=cifar10/dense_net_40/create_train_poison random_seed=691946&

#sleep 86400