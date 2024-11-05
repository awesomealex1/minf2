cd /exports/eddie/scratch/s2017377/Poison_Attacks
touch goldenfile
find /exports/eddie/scratch/s2017377/Poison_Attacks -type f -exec touch -r goldenfile {} \;
rm goldenfile

cd /exports/eddie/scratch/s2017377/anaconda
touch goldenfile
find /exports/eddie/scratch/s2017377/anaconda -type f -exec touch -r goldenfile {} \;
rm goldenfile