#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=20Gb
#SBATCH --gres=gpu:1
#SBATCH --job-name=predictions
#SBATCH --output=slurm_%j.out
 
module purge
module load anaconda3/2020.02
source /scratch/spf248/pyenv/py3.7/bin/activate

cd /scratch/spf248/twitter

srun time python -u ./py/7.9-get-predictions-from-BERT.py > ./log/7.9-get-predictions-from-BERT-${SLURM_ARRAY_TASK_ID}-$(date +%s).log 2>&1

exit
