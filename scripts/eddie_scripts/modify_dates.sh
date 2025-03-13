cd /exports/eddie/scratch/s2017377/minf2
touch goldenfile
find /exports/eddie/scratch/s2017377/minf2 -type f -exec touch -r goldenfile {} \;
rm goldenfile

cd /exports/eddie/scratch/s2017377/anaconda
touch goldenfile
find /exports/eddie/scratch/s2017377/anaconda -type f -exec touch -r goldenfile {} \;
rm goldenfile