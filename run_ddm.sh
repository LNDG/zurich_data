#!/bin/bash
# Runs pyddm (and hddm) on separate node on the cluster

echo "#!/bin/bash" > jobfile.sh
echo "#SBATCH --job-name NoiseDDM" >> jobfile.sh
echo "#SBATCH --time 2:0:0" >> jobfile.sh
echo "#SBATCH --cpus 12" >> jobfile.sh
echo "#SBATCH --mem 2GB" >> jobfile.sh
echo "#SBATCH --mail-type NONE" >> jobfile.sh
echo "#SBATCH --output /home/mpib/kamp/LNDG/zurich_data/logs/slurm-%j.out" >> jobfile.sh

echo "cd $HOME/LNDG/zurich_data" >> jobfile.sh
echo "module load conda" >> jobfile.sh
echo "conda activate py3" >> jobfile.sh

echo "python ddm/fit_pyddm.py" >> jobfile.sh
sbatch jobfile.sh
rm jobfile.sh