#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N modify_dates              
#$ -cwd                  
#$ -l h_rt=16:00:00 
#$ -l h_vmem=10G
. /etc/profile.d/modules.sh

cd /exports/eddie/scratch/s2017377/minf2

# Run the program
. /exports/eddie/scratch/s2017377/minf2/maintenance_scripts/modify_dates.sh