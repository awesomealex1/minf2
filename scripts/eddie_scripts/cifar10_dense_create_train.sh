#!/bin/sh
# SLURM options (lines prefixed with #SBATCH)
#SBATCH --job-name=cifar10_dense_create_train   # Set the job name
#SBATCH --time=24:00:00                         # Set maximum runtime
#SBATCH --mem=100G                             # Set memory per node
#SBATCH --partition=gpu                        # Set partition (queue) to gpu
#SBATCH --gres=gpu:a100:1                      # Request 1 GPU of type a100
#SBATCH --output=job_output_%j.log             # Output file, %j will be replaced by job ID
. /etc/profile.d/modules.sh

# Load Python
module load anaconda
conda activate minf2

cd /exports/eddie/scratch/s2017377/minf2

python scripts/main.py experiment=cifar10/dense_net_40/create_train_poison random_seed=505631&
python scripts/main.py experiment=cifar10/dense_net_40/create_train_poison random_seed=270005&
python scripts/main.py experiment=cifar10/dense_net_40/create_train_poison random_seed=227703&
python scripts/main.py experiment=cifar10/dense_net_40/create_train_poison random_seed=891974&
python scripts/main.py experiment=cifar10/dense_net_40/create_train_poison random_seed=691946&

sleep 86400