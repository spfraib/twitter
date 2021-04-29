#!/bin/bash

#SBATCH --job-name=calibrate_scores
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_calibrate_scores_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu


module purge
module load anaconda3/2020.02

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate

echo "pyenv activated"

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/3-model_evaluation/expansion/preliminary

python3 calibrate_scores_greene.py
exit