#!/bin/bash

echo "#PBS -N NoiseDDM" >> jobfile.pbs
echo "#PBS -l walltime=2:0:0" >> jobfile.pbs
echo "#PBS -l nodes=1:ppn=12" >> jobfile.pbs
echo "#PBS -l mem=2gb" >> jobfile.pbs
echo "#PBS -m n" >> jobfile.pbs
echo "#PBS -j oe" >> jobfile.pbs
echo "#PBS -o /home/mpib/kamp/LNDG/zurich_data/logs/" >> jobfile.pbs

echo "cd $HOME/LNDG/zurich_data"
echo "module load conda" >> jobfile.pbs
echo "conda activate py3" >> jobfile.pbs

echo "python ddm/fit_pyddm.py" >> jobfile.pbs
#qsub jobfile.pbs
#rm jobfile.pbs