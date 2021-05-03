#!/bin/bash

#SBATCH --job-name=get_logit_scores_seedlist
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_get_logit_scores_seedlist_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

module purge
module load anaconda3/2020.02

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate

echo "pyenv activated"

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/3-model_evaluation/expansion/preliminary

python3 get_logit_scores_seedlist.py

exit